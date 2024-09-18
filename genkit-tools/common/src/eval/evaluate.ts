/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { randomUUID } from 'crypto';
import { EvalFlowInput } from '.';
import { Runner } from '../runner';
import { Action, EvalInput, EvalRun, RunActionResponse } from '../types';
import {
  evaluatorName,
  getEvalExtractors,
  isEvaluator,
  logger,
} from '../utils';
import { enrichResultsWithScoring, extractMetricsMetadata } from './parser';

interface BulkRunResponse {
  traceId?: string;
  hasErrored: boolean;
  response: any;
}

/** Handles the Inference part of Inference-Evaluation cycle */
export async function runInference(
  runner: Runner,
  actionRef: string,
  evalFlowInput: EvalFlowInput,
  auth?: string
): Promise<EvalInput[]> {
  let inputs: any[] = Array.isArray(evalFlowInput)
    ? (evalFlowInput as any[])
    : evalFlowInput.samples.map((c) => c.input);

  const runResponses = await bulkRunAction(runner, actionRef, inputs, auth);
  const runStates: BulkRunResponse[] = runResponses.map((r) => {
    return {
      traceId: r.telemetry?.traceId,
      // TODO(ssbushi): Track errors from the trace
      hasErrored: !r.telemetry?.traceId,
      response: r.result,
    } as BulkRunResponse;
  });
  if (runStates.some((s) => s.hasErrored)) {
    logger.debug('Some flows failed with errors');
  }

  const evalDataset = await fetchDataSet(
    runner,
    actionRef,
    runStates,
    evalFlowInput
  );
  return evalDataset;
}

/** Handles the Evaluation part of Inference-Evaluation cycle */
export async function runEvaluation(
  runner: Runner,
  filteredEvaluatorActions: Action[],
  evalDataset: EvalInput[],
  actionRef?: string
): Promise<EvalRun> {
  const evalRunId = randomUUID();
  const scores: Record<string, any> = {};
  for (const action of filteredEvaluatorActions) {
    const name = evaluatorName(action);
    const response = await runner.runAction({
      key: name,
      input: {
        dataset: evalDataset.filter((row) => !row.error),
        evalRunId,
      },
    });
    scores[name] = response.result;
  }

  const scoredResults = enrichResultsWithScoring(scores, evalDataset);
  const metadata = extractMetricsMetadata(filteredEvaluatorActions);

  const evalRun = {
    key: {
      actionRef,
      evalRunId,
      createdAt: new Date().toISOString(),
    },
    results: scoredResults,
    metricsMetadata: metadata,
  };
  return evalRun;
}

export async function getMatchingEvaluators(
  runner: Runner,
  evaluators?: string
): Promise<Action[]> {
  const allActions = await runner.listActions();
  const allEvaluatorActions = [];
  for (const key in allActions) {
    if (isEvaluator(key)) {
      allEvaluatorActions.push(allActions[key]);
    }
  }
  const filteredEvaluatorActions = allEvaluatorActions.filter(
    (action) => !evaluators || evaluators.split(',').includes(action.name)
  );
  if (filteredEvaluatorActions.length === 0) {
    if (allEvaluatorActions.length == 0) {
      throw new Error('No evaluators installed');
    } else {
      const availableActions = allEvaluatorActions
        .map((action) => action.name)
        .join(',');
      throw new Error(
        `No evaluators matched your specifed filter: ${evaluators}. All available evaluators: '${availableActions}'`
      );
    }
  }
  return filteredEvaluatorActions;
}

async function bulkRunAction(
  runner: Runner,
  actionRef: string,
  inputs: any[],
  auth?: string
): Promise<RunActionResponse[]> {
  let responses: RunActionResponse[] = [];
  for (const d of inputs) {
    logger.info(`Running '${actionRef}' ...`);
    let response = await runner.runAction({
      key: actionRef,
      input: {
        start: {
          input: d,
        },
        auth: auth ? JSON.parse(auth) : undefined,
      },
    });
    responses.push(response);
  }

  return responses;
}

async function fetchDataSet(
  runner: Runner,
  flowName: string,
  states: BulkRunResponse[],
  parsedData: EvalFlowInput
): Promise<EvalInput[]> {
  let references: any[] | undefined = undefined;
  if (!Array.isArray(parsedData)) {
    const maybeReferences = parsedData.samples.map((c: any) => c.reference);
    if (maybeReferences.length === states.length) {
      references = maybeReferences;
    } else {
      logger.warn(
        'The input size does not match the flow states generated. Ignoring reference mapping...'
      );
    }
  }
  const extractors = await getEvalExtractors(flowName);
  return await Promise.all(
    states.map(async (s, i) => {
      const traceId = s.traceId;
      if (!traceId) {
        logger.warn('No traceId available...');
        return {
          // TODO Replace this with unified trace class
          testCaseId: randomUUID(),
          traceIds: [],
        };
      }

      const trace = await runner.getTrace({
        // TODO: We should consider making this a argument and using it to
        // to control which tracestore environment is being used when
        // running a flow.
        env: 'dev',
        traceId,
      });

      let inputs: string[] = [];
      let outputs: string[] = [];
      let contexts: string[] = [];

      inputs.push(extractors.input(trace));
      outputs.push(extractors.output(trace));
      contexts.push(extractors.context(trace));

      if (s.hasErrored) {
        return {
          testCaseId: randomUUID(),
          input: inputs[0],
          error: 'Inference error',
          reference: references?.at(i),
          traceIds: [traceId],
        };
      }

      return {
        // TODO Replace this with unified trace class
        testCaseId: randomUUID(),
        input: inputs[0],
        output: outputs[0],
        context: JSON.parse(contexts[0]) as string[],
        reference: references?.at(i),
        traceIds: [traceId],
      };
    })
  );
}
