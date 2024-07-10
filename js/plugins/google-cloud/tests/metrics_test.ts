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

import { generate } from '@genkit-ai/ai';
import { defineModel } from '@genkit-ai/ai/model';
import {
  configureGenkit,
  defineAction,
  FlowState,
  FlowStateQuery,
  FlowStateQueryResponse,
  FlowStateStore,
} from '@genkit-ai/core';
import { registerFlowStateStore } from '@genkit-ai/core/registry';
import { defineFlow, run, runAction, runFlow } from '@genkit-ai/flow';
import {
  __getMetricExporterForTesting,
  GcpOpenTelemetry,
  googleCloud,
} from '@genkit-ai/google-cloud';
import {
  Counter,
  DataPoint,
  Histogram,
  ScopeMetrics,
  SumMetricData,
} from '@opentelemetry/sdk-metrics';
import assert from 'node:assert';
import { before, beforeEach, describe, it } from 'node:test';
import { z } from 'zod';

describe('GoogleCloudMetrics', () => {
  before(async () => {
    process.env.GENKIT_ENV = 'dev';
    const config = configureGenkit({
      // Force GCP Plugin to use in-memory metrics exporter
      plugins: [
        googleCloud({
          projectId: 'test',
          telemetryConfig: {
            forceDevExport: false,
            metricExportIntervalMillis: 100,
            metricExportTimeoutMillis: 100,
          },
        }),
      ],
      enableTracingAndMetrics: true,
      telemetry: {
        instrumentation: 'googleCloud',
      },
    });
    registerFlowStateStore('dev', async () => new NoOpFlowStateStore());
    // Wait for the telemetry plugin to be initialized
    await config.getTelemetryConfig();
  });
  beforeEach(async () => {
    __getMetricExporterForTesting().reset();
  });

  it('writes flow metrics', async () => {
    const testFlow = createFlow('testFlow');

    await runFlow(testFlow);
    await runFlow(testFlow);

    const requestCounter = await getCounterMetric('genkit/flow/requests');
    const latencyHistogram = await getHistogramMetric('genkit/flow/latency');
    assert.equal(requestCounter.value, 2);
    assert.equal(requestCounter.attributes.name, 'testFlow');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'success');
    assert.ok(requestCounter.attributes.sourceVersion);
    assert.equal(latencyHistogram.value.count, 2);
    assert.equal(latencyHistogram.attributes.name, 'testFlow');
    assert.equal(latencyHistogram.attributes.source, 'ts');
    assert.equal(latencyHistogram.attributes.status, 'success');
    assert.ok(latencyHistogram.attributes.sourceVersion);
  });

  it('writes flow failure metrics', async () => {
    const testFlow = createFlow('testFlow', async () => {
      const nothing = null;
      nothing.something;
    });

    assert.rejects(async () => {
      await runFlow(testFlow);
    });

    const requestCounter = await getCounterMetric('genkit/flow/requests');
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.name, 'testFlow');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.error, 'TypeError');
    assert.equal(requestCounter.attributes.status, 'failure');
  });

  it('writes action metrics', async () => {
    const testAction = createAction('testAction');
    const testFlow = createFlow('testFlowWithActions', async () => {
      await Promise.all([
        runAction(testAction),
        runAction(testAction),
        runAction(testAction),
      ]);
    });

    await runFlow(testFlow);
    await runFlow(testFlow);

    const requestCounter = await getCounterMetric('genkit/action/requests');
    const latencyHistogram = await getHistogramMetric('genkit/action/latency');
    assert.equal(requestCounter.value, 6);
    assert.equal(requestCounter.attributes.name, 'testAction');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'success');
    assert.ok(requestCounter.attributes.sourceVersion);
    assert.equal(latencyHistogram.value.count, 6);
    assert.equal(latencyHistogram.attributes.name, 'testAction');
    assert.equal(latencyHistogram.attributes.source, 'ts');
    assert.equal(latencyHistogram.attributes.status, 'success');
    assert.ok(latencyHistogram.attributes.sourceVersion);
  });

  it('truncates metric dimensions', async () => {
    const testFlow = createFlow('anExtremelyLongFlowNameThatIsTooBig');

    await runFlow(testFlow);

    const requestCounter = await getCounterMetric('genkit/flow/requests');
    const latencyHistogram = await getHistogramMetric('genkit/flow/latency');
    assert.equal(
      requestCounter.attributes.name,
      'anExtremelyLongFlowNameThatIsToo'
    );
    assert.equal(
      latencyHistogram.attributes.name,
      'anExtremelyLongFlowNameThatIsToo'
    );
  });

  it('writes action failure metrics', async () => {
    const testAction = createAction('testActionWithFailure', async () => {
      const nothing = null;
      nothing.something;
    });
    const testFlow = createFlow('testFlowWithFailingActions', async () => {
      await runAction(testAction);
    });

    assert.rejects(async () => {
      await runFlow(testFlow);
    });

    const requestCounter = await getCounterMetric('genkit/action/requests');
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.name, 'testActionWithFailure');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'failure');
    assert.equal(requestCounter.attributes.error, 'TypeError');
  });

  it('writes generate metrics', async () => {
    const testModel = createModel('testModel', async () => {
      return {
        candidates: [
          {
            index: 0,
            finishReason: 'stop',
            message: {
              role: 'user',
              content: [
                {
                  text: 'response',
                },
              ],
            },
          },
        ],
        usage: {
          inputTokens: 10,
          outputTokens: 14,
          inputCharacters: 8,
          outputCharacters: 16,
          inputImages: 1,
          outputImages: 3,
        },
      };
    });

    const response = await generate({
      model: testModel,
      prompt: 'test prompt',
      config: {
        temperature: 1.0,
        topK: 3,
        topP: 5,
        maxOutputTokens: 7,
      },
    });

    const requestCounter = await getCounterMetric(
      'genkit/ai/generate/requests'
    );
    const inputTokenCounter = await getCounterMetric(
      'genkit/ai/generate/input/tokens'
    );
    const outputTokenCounter = await getCounterMetric(
      'genkit/ai/generate/output/tokens'
    );
    const inputCharacterCounter = await getCounterMetric(
      'genkit/ai/generate/input/characters'
    );
    const outputCharacterCounter = await getCounterMetric(
      'genkit/ai/generate/output/characters'
    );
    const inputImageCounter = await getCounterMetric(
      'genkit/ai/generate/input/images'
    );
    const outputImageCounter = await getCounterMetric(
      'genkit/ai/generate/output/images'
    );
    const latencyHistogram = await getHistogramMetric(
      'genkit/ai/generate/latency'
    );
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.maxOutputTokens, 7);
    assert.equal(inputTokenCounter.value, 10);
    assert.equal(outputTokenCounter.value, 14);
    assert.equal(inputCharacterCounter.value, 8);
    assert.equal(outputCharacterCounter.value, 16);
    assert.equal(inputImageCounter.value, 1);
    assert.equal(outputImageCounter.value, 3);
    assert.equal(latencyHistogram.value.count, 1);
    for (metric of [
      requestCounter,
      inputTokenCounter,
      outputTokenCounter,
      inputCharacterCounter,
      outputCharacterCounter,
      inputImageCounter,
      outputImageCounter,
      latencyHistogram,
    ]) {
      assert.equal(metric.attributes.modelName, 'testModel');
      assert.equal(metric.attributes.temperature, 1.0);
      assert.equal(metric.attributes.topK, 3);
      assert.equal(metric.attributes.topP, 5);
      assert.equal(metric.attributes.source, 'ts');
      assert.equal(metric.attributes.status, 'success');
      assert.ok(metric.attributes.sourceVersion);
    }
  });

  it('writes generate failure metrics', async () => {
    const testModel = createModel('failingTestModel', async () => {
      const nothing = null;
      nothing.something;
    });

    assert.rejects(async () => {
      return await generate({
        model: testModel,
        prompt: 'test prompt',
        config: {
          temperature: 1.0,
          topK: 3,
          topP: 5,
          maxOutputTokens: 7,
        },
      });
    });

    const requestCounter = await getCounterMetric(
      'genkit/ai/generate/requests'
    );
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.modelName, 'failingTestModel');
    assert.equal(requestCounter.attributes.temperature, 1.0);
    assert.equal(requestCounter.attributes.topK, 3);
    assert.equal(requestCounter.attributes.topP, 5);
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'failure');
    assert.equal(requestCounter.attributes.error, 'TypeError');
    assert.ok(requestCounter.attributes.sourceVersion);
  });

  it('writes flow label to action metrics when running inside flow', async () => {
    const testAction = createAction('testAction');
    const flow = createFlow('flowNameLabelTestFlow', async () => {
      return await runAction(testAction);
    });

    await runFlow(flow);

    const requestCounter = await getCounterMetric('genkit/action/requests');
    const latencyHistogram = await getHistogramMetric('genkit/action/latency');
    assert.equal(requestCounter.attributes.flowName, 'flowNameLabelTestFlow');
    assert.equal(latencyHistogram.attributes.flowName, 'flowNameLabelTestFlow');
  });

  it('writes flow label to generate metrics when running inside flow', async () => {
    const testModel = createModel('testModel', async () => {
      return {
        candidates: [
          {
            index: 0,
            finishReason: 'stop',
            message: {
              role: 'user',
              content: [
                {
                  text: 'response',
                },
              ],
            },
          },
        ],
        usage: {
          inputTokens: 10,
          outputTokens: 14,
          inputCharacters: 8,
          outputCharacters: 16,
          inputImages: 1,
          outputImages: 3,
        },
      };
    });
    const flow = createFlow('testFlow', async () => {
      return await generate({
        model: testModel,
        prompt: 'test prompt',
      });
    });

    await runFlow(flow);

    const metrics = [
      await getCounterMetric('genkit/ai/generate/requests'),
      await getCounterMetric('genkit/ai/generate/input/tokens'),
      await getCounterMetric('genkit/ai/generate/output/tokens'),
      await getCounterMetric('genkit/ai/generate/input/characters'),
      await getCounterMetric('genkit/ai/generate/output/characters'),
      await getCounterMetric('genkit/ai/generate/input/images'),
      await getCounterMetric('genkit/ai/generate/output/images'),
      await getHistogramMetric('genkit/ai/generate/latency'),
    ];
    for (metric of metrics) {
      assert.equal(metric.attributes.flowName, 'testFlow');
    }
  });

  it('writes flow paths metrics', async () => {
    const flow = createFlow('pathTestFlow', async () => {
      const step1Result = await run('step1', async () => {
        return await run('substep_a', async () => {
          return await run('substep_b', async () => 'res1');
        });
      });
      const step2Result = await run('step2', async () => 'res2');
      return step1Result + step2Result;
    });

    await runFlow(flow);

    const expectedPaths = new Set([
      '/{pathTestFlow,t:flow}/{step2,t:flowStep}',
      '/{pathTestFlow,t:flow}/{step1,t:flowStep}/{substep_a,t:flowStep}/{substep_b,t:flowStep}',
    ]);
    const pathCounterPoints = await getCounterDataPoints(
      'genkit/flow/path/requests'
    );
    const pathLatencyPoints = await getHistogramDataPoints(
      'genkit/flow/path/latency'
    );
    const paths = new Set(
      pathCounterPoints.map((point) => point.attributes.path)
    );
    assert.deepEqual(paths, expectedPaths);
    pathCounterPoints.forEach((point) => {
      assert.equal(point.value, 1);
      assert.equal(point.attributes.flowName, 'pathTestFlow');
      assert.equal(point.attributes.source, 'ts');
      assert.equal(point.attributes.status, 'success');
      assert.ok(point.attributes.sourceVersion);
    });
    pathLatencyPoints.forEach((point) => {
      assert.equal(point.value.count, 1);
      assert.equal(point.attributes.flowName, 'pathTestFlow');
      assert.equal(point.attributes.source, 'ts');
      assert.equal(point.attributes.status, 'success');
      assert.ok(point.attributes.sourceVersion);
    });
  });

  it('writes flow path failure metrics in root', async () => {
    const flow = createFlow('testFlow', async () => {
      const subPath = await run('sub-action', async () => {
        return 'done';
      });
      return Promise.reject(new Error('failed'));
    });

    assert.rejects(async () => {
      await runFlow(flow);
    });

    const reqPoints = await getCounterDataPoints('genkit/flow/path/requests');
    const reqStatuses = reqPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(reqStatuses, [
      ['/{testFlow,t:flow}/{sub-action,t:flowStep}', 'success'],
      ['/{testFlow,t:flow}', 'failure'],
    ]);
    const latencyPoints = await getHistogramDataPoints(
      'genkit/flow/path/latency'
    );
    const latencyStatuses = latencyPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(latencyStatuses, [
      ['/{testFlow,t:flow}/{sub-action,t:flowStep}', 'success'],
      ['/{testFlow,t:flow}', 'failure'],
    ]);
  });

  it('writes flow path failure metrics in subaction', async () => {
    const flow = createFlow('testFlow', async () => {
      const subPath1 = await run('sub-action-1', async () => {
        const subPath2 = await run('sub-action-2', async () => {
          return Promise.reject(new Error('failed'));
        });
        return 'done';
      });
      return 'done';
    });

    assert.rejects(async () => {
      await runFlow(flow);
    });

    const reqPoints = await getCounterDataPoints('genkit/flow/path/requests');
    const reqStatuses = reqPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(reqStatuses, [
      [
        '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
        'failure',
      ],
    ]);
    const latencyPoints = await getHistogramDataPoints(
      'genkit/flow/path/latency'
    );
    const latencyStatuses = latencyPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(latencyStatuses, [
      [
        '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
        'failure',
      ],
    ]);
  });

  it('writes flow path failure metrics in subaction', async () => {
    const flow = createFlow('testFlow', async () => {
      const subPath1 = await run('sub-action-1', async () => {
        const subPath2 = await run('sub-action-2', async () => {
          return 'done';
        });
        return Promise.reject(new Error('failed'));
      });
      return 'done';
    });

    assert.rejects(async () => {
      await runFlow(flow);
    });

    const reqPoints = await getCounterDataPoints('genkit/flow/path/requests');
    const reqStatuses = reqPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(reqStatuses, [
      [
        '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
        'success',
      ],
      ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'failure'],
    ]);
    const latencyPoints = await getHistogramDataPoints(
      'genkit/flow/path/latency'
    );
    const latencyStatuses = latencyPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(latencyStatuses, [
      [
        '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
        'success',
      ],
      ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'failure'],
    ]);
  });

  it('writes flow path failure in sub-action metrics', async () => {
    const flow = createFlow('testFlow', async () => {
      const subPath1 = await run('sub-action-1', async () => {
        return 'done';
      });
      const subPath2 = await run('sub-action-2', async () => {
        return Promise.reject(new Error('failed'));
      });
      return 'done';
    });

    assert.rejects(async () => {
      await runFlow(flow);
    });

    const reqPoints = await getCounterDataPoints('genkit/flow/path/requests');
    const reqStatuses = reqPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(reqStatuses, [
      ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'success'],
      ['/{testFlow,t:flow}/{sub-action-2,t:flowStep}', 'failure'],
    ]);
    const latencyPoints = await getHistogramDataPoints(
      'genkit/flow/path/latency'
    );
    const latencyStatuses = latencyPoints.map((p) => [
      p.attributes.path,
      p.attributes.status,
    ]);
    assert.deepEqual(latencyStatuses, [
      ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'success'],
      ['/{testFlow,t:flow}/{sub-action-2,t:flowStep}', 'failure'],
    ]);
  });

  describe('Configuration', () => {
    it('should export only traces', async () => {
      const telemetry = new GcpOpenTelemetry({
        telemetryConfig: {
          forceDevExport: true,
          disableMetrics: true,
        },
      });
      assert.equal(telemetry['shouldExportTraces'](), true);
      assert.equal(telemetry['shouldExportMetrics'](), false);
    });

    it('should export only metrics', async () => {
      const telemetry = new GcpOpenTelemetry({
        telemetryConfig: {
          forceDevExport: true,
          disableTraces: true,
          disableMetrics: false,
        },
      });
      assert.equal(telemetry['shouldExportTraces'](), false);
      assert.equal(telemetry['shouldExportMetrics'](), true);
    });
  });

  /** Polls the in memory metric exporter until the genkit scope is found. */
  async function getGenkitMetrics(
    name: string = 'genkit',
    maxAttempts: number = 100
  ): promise<ScopeMetrics> {
    var attempts = 0;
    while (attempts++ < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      const found = __getMetricExporterForTesting()
        .getMetrics()
        .find((e) => e.scopeMetrics.map((sm) => sm.scope.name).includes(name));
      if (found) {
        return found.scopeMetrics.find((e) => e.scope.name === name);
      }
    }
    assert.fail(`Waiting for metric ${name} but it has not been written.`);
  }

  /** Finds all datapoints for a counter metric with the given name in the in memory exporter */
  async function getCounterDataPoints(
    metricName: string
  ): Promise<List<DataPoint<Counter>>> {
    const genkitMetrics = await getGenkitMetrics();
    const counterMetric: SumMetricData = genkitMetrics.metrics.find(
      (e) => e.descriptor.name === metricName && e.descriptor.type === 'COUNTER'
    );
    if (counterMetric) {
      return counterMetric.dataPoints;
    }
    assert.fail(
      `No counter metric named ${metricName} was found. Only found: ${genkitMetrics.metrics.map((e) => e.descriptor.name)}`
    );
  }

  /** Finds a counter metric with the given name in the in memory exporter */
  async function getCounterMetric(
    metricName: string
  ): Promise<DataPoint<Counter>> {
    return getCounterDataPoints(metricName).then((points) => points.at(-1));
  }

  /**
   * Finds all datapoints for a histogram metric with the given name in the in
   * memory exporter.
   */
  async function getHistogramDataPoints(
    metricName: string
  ): Promise<List<DataPoint<Histogram>>> {
    const genkitMetrics = await getGenkitMetrics();
    const histogramMetric: HistogramMetricData = genkitMetrics.metrics.find(
      (e) =>
        e.descriptor.name === metricName && e.descriptor.type === 'HISTOGRAM'
    );
    if (histogramMetric) {
      return histogramMetric.dataPoints;
    }
    assert.fail(
      `No histogram metric named ${metricName} was found. Only found: ${genkitMetrics.metrics.map((e) => e.descriptor.name)}`
    );
  }

  /** Finds a histogram metric with the given name in the in memory exporter */
  async function getHistogramMetric(
    metricName: string
  ): Promise<DataPoint<Histogram>> {
    return getHistogramDataPoints(metricName).then((points) => points.at(-1));
  }

  /** Helper to create a flow with no inputs or outputs */
  function createFlow(name: string, fn: () => Promise<void> = async () => {}) {
    return defineFlow(
      {
        name,
        inputSchema: z.void(),
        outputSchema: z.void(),
      },
      fn
    );
  }

  /** Helper to create an action with no inputs or outputs */
  function createAction(
    name: string,
    fn: () => Promise<void> = async () => {}
  ) {
    return defineAction(
      {
        name,
        actionType: 'test',
      },
      fn
    );
  }

  /** Helper to create a model that returns the value produced by the givne
   * response function. */
  function createModel(
    name: string,
    respFn: () => Promise<GenerateResponseData>
  ) {
    return defineModel({ name }, (req) => respFn());
  }
});

class NoOpFlowStateStore implements FlowStateStore {
  state: Record<string, string> = {};

  load(id: string): Promise<FlowState | undefined> {
    return Promise.resolve(undefined);
  }

  save(id: string, state: FlowState): Promise<void> {
    return Promise.resolve();
  }

  async list(
    query?: FlowStateQuery | undefined
  ): Promise<FlowStateQueryResponse> {
    return {};
  }
}
