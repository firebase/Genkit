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

import { BaseDataPointSchema, defineEvaluator } from '@genkit-ai/ai/evaluator';
import { ModelArgument } from '@genkit-ai/ai/model';
import { Criteria, loadEvaluator } from 'langchain/evaluation';
import { genkitModel } from './model.js';
import { GenkitTracer } from './tracing.js';

export function langchainEvaluator(
  type: 'labeled_criteria' | 'criteria',
  criteria: Criteria,
  judgeLlm: ModelArgument,
  judgeConfig?: any
) {
  return defineEvaluator(
    {
      name: `langchain/${criteria}`,
      displayName: `${criteria}`,
      definition: `${criteria}: refer to https://js.langchain.com/docs/guides/evaluation`,
      dataPointType: BaseDataPointSchema,
    },
    async (datapoint) => {
      try {
        switch (type) {
          case 'labeled_criteria':
          case 'criteria':
            const evaluator = await loadEvaluator(
              type as 'labeled_criteria' | 'criteria',
              {
                criteria,
                llm: genkitModel(judgeLlm, judgeConfig),
                chainOptions: {
                  callbacks: [new GenkitTracer()],
                },
              }
            );
            const lcData = {
              input: datapoint.input as string,
              prediction: datapoint.output as string,
            };
            if (datapoint.reference) {
              lcData['reference'] = datapoint.reference as string;
            }
            const res = await evaluator.evaluateStrings({ ...lcData });
            return {
              testCaseId: datapoint.testCaseId,
              evaluation: {
                score: res.score,
                details: {
                  reasoning: res.reasoning,
                  value: res.value,
                },
              },
            };
          default:
            throw new Error(`unsupported evaluator type ${type}`);
        }
      } catch (e) {
        return {
          testCaseId: datapoint.testCaseId,
          evaluation: {
            error: `${e}`,
          },
        };
      }
    }
  );
}
