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
import { defineModel, GenerateResponseData } from '@genkit-ai/ai/model';
import { configureGenkit } from '@genkit-ai/core';
import { defineFlow, run } from '@genkit-ai/flow';
import {
  __addTransportStreamForTesting,
  __forceFlushSpansForTesting,
  __getSpanExporterForTesting,
  googleCloud,
} from '@genkit-ai/google-cloud';
import { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import assert from 'node:assert';
import { before, beforeEach, describe, it } from 'node:test';
import { Writable } from 'stream';
import { z } from 'zod';

describe('GoogleCloudLogs', () => {
  let logLines = '';
  const logStream = new Writable();
  logStream._write = (chunk, encoding, next) => {
    logLines = logLines += chunk.toString();
    next();
  };

  before(async () => {
    process.env.GENKIT_ENV = 'dev';
    __addTransportStreamForTesting(logStream);
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
        logger: 'googleCloud',
      },
    });
    // Wait for the telemetry plugin to be initialized
    await config.getTelemetryConfig();
    await waitForLogsInit();
  });
  beforeEach(async () => {
    logLines = '';
    __getSpanExporterForTesting().reset();
  });

  it('writes path logs', async () => {
    const testFlow = createFlow('testFlow');

    await testFlow();

    await getExportedSpans();

    const logMessages = await getLogs();
    assert.equal(logMessages.includes('[info] Paths[testFlow]'), true);
  });

  it('writes error logs', async () => {
    const testFlow = createFlow('testFlow', async () => {
      const nothing: any = null;
      nothing.something;
    });

    assert.rejects(async () => {
      await testFlow();
    });

    await getExportedSpans();

    const logMessages = await getLogs();
    assert.equal(
      logMessages.includes(
        '[error] Error[testFlow, TypeError] Cannot read properties of null ' +
          "(reading 'something')"
      ),
      true
    );
  });

  it('writes generate logs', async () => {
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
    const testFlow = createFlow('testFlow', async () => {
      return await run('sub1', async () => {
        return await run('sub2', async () => {
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
      });
    });

    await testFlow();

    await getExportedSpans();

    const logMessages = await getLogs();
    assert.equal(
      logMessages.includes(
        '[info] Config[testFlow > sub1 > sub2 > generate > testModel, testModel]'
      ),
      true
    );
    assert.equal(
      logMessages.includes(
        '[info] Input[testFlow > sub1 > sub2 > generate > testModel, testModel]'
      ),
      true
    );
    assert.equal(
      logMessages.includes(
        '[info] Output[testFlow > sub1 > sub2 > generate > testModel, testModel]'
      ),
      true
    );
  });

  /** Helper to create a flow with no inputs or outputs */
  function createFlow(name: string, fn: () => Promise<any> = async () => {}) {
    return defineFlow(
      {
        name,
        inputSchema: z.void(),
        outputSchema: z.void(),
      },
      fn
    );
  }

  /**
   * Helper to create a model that returns the value produced by the given
   * response function.
   */
  function createModel(
    name: string,
    respFn: () => Promise<GenerateResponseData>
  ) {
    return defineModel({ name }, (req) => respFn());
  }

  async function waitForLogsInit() {
    await import('winston');
    const testFlow = createFlow('testFlow');
    await testFlow();
    await getLogs(1);
  }

  async function getLogs(
    logCount: number = 1,
    maxAttempts: number = 100
  ): Promise<String[]> {
    var attempts = 0;
    while (attempts++ < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      const found = logLines
        .trim()
        .split('\n')
        .map((l) => l.trim());
      if (found.length >= logCount) {
        return found;
      }
    }
    assert.fail(`Waiting for logs, but none have been written.`);
  }
});

/** Polls the in memory metric exporter until the genkit scope is found. */
async function getExportedSpans(
  maxAttempts: number = 200
): Promise<ReadableSpan[]> {
  __forceFlushSpansForTesting();
  var attempts = 0;
  while (attempts++ < maxAttempts) {
    await new Promise((resolve) => setTimeout(resolve, 50));
    const found = __getSpanExporterForTesting().getFinishedSpans();
    if (found.length > 0) {
      return found;
    }
  }
  assert.fail(`Timed out while waiting for spans to be exported.`);
}
