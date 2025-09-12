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

import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { genkit, z, type GenerateResponseData, type Genkit } from 'genkit';
import { SPAN_TYPE_ATTR, appendSpan } from 'genkit/tracing';
import { Writable } from 'stream';
import {
  __addTransportStreamForTesting,
  __forceFlushSpansForTesting,
  __getSpanExporterForTesting,
  __useJsonFormatForTesting,
  enableGoogleCloudTelemetry,
} from '../src/index.js';

jest.mock('../src/auth.js', () => {
  const original = jest.requireActual('../src/auth.js');
  return {
    ...(original || {}),
    resolveCurrentPrincipal: jest.fn().mockImplementation(() => {
      return Promise.resolve({
        projectId: 'test',
        serviceAccountEmail: 'test@test.com',
      });
    }),
    credentialsFromEnvironment: jest.fn().mockImplementation(() => {
      return Promise.resolve({
        projectId: 'test',
        credentials: {
          client_email: 'test@genkit.com',
          private_key: '-----BEGIN PRIVATE KEY-----',
        },
      });
    }),
  };
});

describe('GoogleCloudLogs', () => {
  let logLines = '';
  const logStream = new Writable();
  logStream._write = (chunk, encoding, next) => {
    logLines = logLines += chunk.toString();
    next();
  };

  let ai: Genkit;

  beforeAll(async () => {
    process.env.GCLOUD_PROJECT = 'test';
    process.env.GENKIT_ENV = 'dev';
    __useJsonFormatForTesting();
    __addTransportStreamForTesting(logStream);
    await enableGoogleCloudTelemetry({
      projectId: 'test',
      forceDevExport: false,
      metricExportIntervalMillis: 100,
      metricExportTimeoutMillis: 100,
    });
    ai = genkit({
      // Force GCP Plugin to use in-memory metrics exporter
      plugins: [],
    });
    await waitForLogsInit(ai, logLines);
  });
  beforeEach(async () => {
    logLines = '';
    __getSpanExporterForTesting().reset();
  });
  afterAll(async () => {
    await ai.stopServers();
  });

  describe('with truncation', () => {
    it('truncates large output logs', async () => {
      const testModel = createModel(ai, 'testModel', async () => {
        return {
          message: {
            role: 'user',
            content: [
              {
                text: 'r'.repeat(130_000),
              },
            ],
          },
          finishReason: 'stop',
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
      const testFlow = createFlowWithInput(ai, 'testFlow', async (input) => {
        return await ai.run('sub1', async () => {
          return await ai.run('sub2', async () => {
            return await ai.generate({
              model: testModel,
              prompt: `${input} prompt`,
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

      await testFlow('test');
      await getExportedSpans();

      const logMessages = await getLogs(1, 100, logLines);
      const logObjects = logMessages.map((l) => JSON.parse(l as string));
      const logObjectMessages = logObjects.map(
        (structuredLog) => structuredLog.message
      );

      expect(logObjectMessages).toContain('Output[testFlow, testFlow]');

      logObjects.map((structuredLog) => {
        if (structuredLog.message === 'Output[testFlow, testFlow]') {
          expect(structuredLog.content.length).toBe(128_000);
        }
      });
    });

    it('truncates large input logs', async () => {
      const testModel = createModel(ai, 'testModel', async () => {
        return {
          message: {
            role: 'user',
            content: [
              {
                text: 'response',
              },
            ],
          },
          finishReason: 'stop',
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
      const testFlow = createFlowWithInput(ai, 'testFlow', async (input) => {
        return await ai.run('sub1', async () => {
          return await ai.run('sub2', async () => {
            return await ai.generate({
              model: testModel,
              prompt: `${input} prompt`,
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

      await testFlow('t'.repeat(130_000));
      await getExportedSpans();

      const logMessages = await getLogs(1, 100, logLines);
      const logObjects = logMessages.map((l) => JSON.parse(l as string));
      const logObjectMessages = logObjects.map(
        (structuredLog) => structuredLog.message
      );

      expect(logObjectMessages).toContain('Input[testFlow, testFlow]');

      logObjects.map((structuredLog) => {
        if (structuredLog.message === 'Input[testFlow, testFlow]') {
          expect(structuredLog.content.length).toBe(128_000);
        }
      });
    });

    it('truncates large model names', async () => {
      const testModel = createModel(ai, 'm'.repeat(2046), async () => {
        return {
          message: {
            role: 'user',
            content: [
              {
                text: 'response',
              },
            ],
          },
          finishReason: 'stop',
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
      const testFlow = createFlowWithInput(ai, 'testFlow', async (input) => {
        return await ai.run('sub1', async () => {
          return await ai.run('sub2', async () => {
            return await ai.generate({
              model: testModel,
              prompt: `${input} prompt`,
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

      await testFlow('test');
      await getExportedSpans();

      const logMessages = await getLogs(1, 100, logLines);
      const logObjects = logMessages.map((l) => JSON.parse(l as string));
      const logObjectModels = logObjects.map(
        (structuredLog) => structuredLog.model
      );

      expect(logObjectModels).toContain('m'.repeat(1024));
    });
  });

  describe('path logs', () => {
    it('writes error log for failed path', async () => {
      const testFlow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub1', async () => {
          return 'not failing';
        });
        await ai.run('sub2', async () => {
          return explode();
        });
        return 'never reached';
      });

      await assert.rejects(async () => {
        await testFlow();
      });

      await getExportedSpans();

      const logs = await getLogs(1, 100, logLines);
      const logObjectMessages = getStructuredLogMessages(logs);
      expect(logObjectMessages).toContain(
        "Error[testFlow > sub2, TypeError] Cannot read properties of undefined (reading 'explode')"
      );
      const errorLogs = logObjectMessages.filter(
        (m) => m.indexOf('Error[') >= 0
      );
      expect(errorLogs).toHaveLength(1); // Only 1 error log
    }, 10000); //timeout

    it('writes error log for failed root', async () => {
      const testFlow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub1', async () => {
          return 'not failing';
        });
        await ai.run('sub2', async () => {
          return 'not failing';
        });
        return explode();
      });

      await assert.rejects(async () => {
        await testFlow();
      });

      await getExportedSpans();

      const logs = await getLogs(1, 100, logLines);
      const logObjectMessages = getStructuredLogMessages(logs);
      expect(logObjectMessages).toContain(
        "Error[testFlow, TypeError] Cannot read properties of undefined (reading 'explode')"
      );
      const errorLogs = logObjectMessages.filter(
        (m) => m.indexOf('Error[') >= 0
      );
      expect(errorLogs).toHaveLength(1); // Only 1 error log
    }, 10000); //timeout

    it('writes error log for multiple failing spans', async () => {
      const testFlow = createFlow(ai, 'testFlow', async () => {
        await Promise.all([
          ai.run('sub1', async () => {
            return explode();
          }),
          ai.run('sub2', async () => {
            return explode();
          }),
        ]);
        return 'not failing';
      });

      await assert.rejects(async () => {
        await testFlow();
      });

      await getExportedSpans();

      const logs = await getLogs(1, 100, logLines);
      const logObjectMessages = getStructuredLogMessages(logs);
      expect(logObjectMessages).toContain(
        "Error[testFlow > sub1, TypeError] Cannot read properties of undefined (reading 'explode')"
      );
      expect(logObjectMessages).toContain(
        "Error[testFlow > sub2, TypeError] Cannot read properties of undefined (reading 'explode')"
      );
      const errorLogs = logObjectMessages.filter(
        (m) => m.indexOf('Error[') >= 0
      );
      expect(errorLogs).toHaveLength(2); // Only 2 error log
    }, 10000); //timeout
  });

  it('writes generate logs', async () => {
    const testModel = createModel(ai, 'testModel', async () => {
      return {
        message: {
          role: 'model',
          content: [
            {
              text: 'response',
            },
          ],
        },
        finishReason: 'stop',
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
    const testFlow = createFlowWithInput(ai, 'testFlow', async (input) => {
      return await ai.run('sub1', async () => {
        return await ai.run('sub2', async () => {
          return await ai.generate({
            model: testModel,
            prompt: `${input} prompt`,
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

    await testFlow('test');

    await getExportedSpans();

    const logs = await getLogs(1, 100, logLines);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain(
      'Config[testFlow > sub1 > sub2 > generate > testModel, testModel]'
    );
    expect(logObjectMessages).toContain(
      'Input[testFlow > sub1 > sub2 > generate > testModel, testModel] '
    );
    expect(logObjectMessages).toContain(
      'Output[testFlow > sub1 > sub2 > generate > testModel, testModel] '
    );
    expect(logObjectMessages).toContain('Input[testFlow, testFlow]');
    expect(logObjectMessages).toContain('Output[testFlow, testFlow]');
    expect(logObjectMessages).toContain(
      'Input[testFlow > sub1 > sub2 > generate, testFlow]'
    );
    expect(logObjectMessages).toContain(
      'Output[testFlow > sub1 > sub2 > generate, testFlow]'
    );
    // Ensure the model input/output has an associated role
    logs.forEach((log) => {
      const structuredLog = JSON.parse(log as string);
      if (
        structuredLog.message ===
        'Input[testFlow > sub1 > sub2 > generate > testModel, testModel] '
      ) {
        expect(structuredLog.role).toBe('user');
      }
      if (
        structuredLog.message ===
        'Output[testFlow > sub1 > sub2 > generate > testModel, testModel]'
      ) {
        expect(structuredLog.role).toBe('model');
      }
    });
  });

  it('writes feature logs for generate without flow', async () => {
    const testModel = createModel(ai, 'testModel', async () => {
      return {
        message: {
          role: 'model',
          content: [
            {
              text: 'response',
            },
          ],
        },
        finishReason: 'stop',
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

    await ai.generate({
      model: testModel,
      prompt: `a test prompt`,
      config: {
        temperature: 1.0,
        topK: 3,
        topP: 5,
        maxOutputTokens: 7,
      },
    });

    await getExportedSpans();

    const logs = await getLogs(1, 100, logLines);
    expect(logs.length).toEqual(6);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain(
      'Config[generate > testModel, testModel]'
    );
    expect(logObjectMessages).toContain(
      'Input[generate > testModel, testModel] '
    );
    expect(logObjectMessages).toContain(
      'Output[generate > testModel, testModel] '
    );
    expect(logObjectMessages).toContain('Input[generate, generate]');
    expect(logObjectMessages).toContain('Output[generate, generate]');
  });

  it('writes user feedback log', async () => {
    await appendSpan(
      'trace1',
      'parent1',
      {
        name: 'user-feedback',
        path: '/{flowName}',
        metadata: {
          subtype: 'userFeedback',
          feedbackValue: 'negative',
          textFeedback: 'terrible',
        },
      },
      { [SPAN_TYPE_ATTR]: 'userEngagement' }
    );

    await getExportedSpans();
    const logs = await getLogs(1, 100, logLines);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain('UserFeedback[flowName]');
  });

  it('writes user acceptance log', async () => {
    await appendSpan(
      'trace1',
      'parent1',
      {
        name: 'user-acceptance',
        path: '/{flowName}',
        metadata: { subtype: 'userAcceptance', acceptanceValue: 'rejected' },
      },
      { [SPAN_TYPE_ATTR]: 'userEngagement' }
    );

    await getExportedSpans();
    const logs = await getLogs(1, 100, logLines);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain('UserAcceptance[flowName]');
  });

  it('writes prompt template input and output logs', async () => {
    const testPrompt = ai.definePrompt(
      {
        name: 'testPrompt',
        input: {
          schema: z.object({ name: z.string() }),
        },
      },
      async (input) => {
        return {
          messages: [
            { role: 'user', content: [{ text: `Hello, ${input.name}` }] },
          ],
        };
      }
    );
    const testFlow = createFlowWithInput(ai, 'testFlow', async (input) => {
      return await testPrompt.render({ name: input });
    });

    await testFlow('test');

    await getExportedSpans();
    const logs = await getLogs(1, 100, logLines);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain('Input[testFlow > render, testFlow]');
    expect(logObjectMessages).toContain('Output[testFlow > render, testFlow]');
    // Ensure the structured data is as expected
    logs.map((log) => {
      const structuredLog = JSON.parse(log as string);
      if (structuredLog.message === 'Input[testFlow > render, testFlow]') {
        expect(JSON.parse(structuredLog.content)).toEqual({ name: 'test' });
      }
      if (structuredLog.message === 'Output[testFlow > render, testFlow]') {
        expect(JSON.parse(structuredLog.content)).toEqual({
          messages: [
            {
              content: [{ text: 'Hello, test' }],
              role: 'user',
            },
          ],
        });
      }
    });
  });

  it('writes tool input and output logs', async () => {
    const echoTool = ai.defineTool(
      { name: 'echoTool', description: 'echo' },
      async (input) => input
    );
    await echoTool('Helllooooo!');
    await getExportedSpans();
    const logs = await getLogs(1, 100, logLines);
    const logObjectMessages = getStructuredLogMessages(logs);
    expect(logObjectMessages).toContain('Input[echoTool, echoTool]');
    expect(logObjectMessages).toContain('Output[echoTool, echoTool]');
  });
});

/** Helper to create a flow with no inputs or outputs */
function createFlow(
  ai: Genkit,
  name: string,
  fn: () => Promise<any> = async () => {}
) {
  return ai.defineFlow(
    {
      name,
      inputSchema: z.void(),
      outputSchema: z.void(),
    },
    fn
  );
}

function createFlowWithInput(
  ai: Genkit,
  name: string,
  fn: (input: string) => Promise<any>
) {
  return ai.defineFlow(
    {
      name,
      inputSchema: z.string(),
      outputSchema: z.any(),
    },
    fn
  );
}

/**
 * Helper to create a model that returns the value produced by the given
 * response function.
 */
function createModel(
  genkit: Genkit,
  name: string,
  respFn: () => Promise<GenerateResponseData>
) {
  return genkit.defineModel({ name }, (req) => respFn());
}

async function waitForLogsInit(genkit: Genkit, logLines: any) {
  await import('winston');
  const testFlow = createFlow(genkit, 'testLogsInitFlow');
  await testFlow();
  await getLogs(1, 100, logLines);
}

async function getLogs(
  logCount: number,
  maxAttempts: number,
  logLines: string
): Promise<string[]> {
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

/** Polls the in memory metric exporter until the genkit scope is found. */
async function getExportedSpans(maxAttempts = 200): Promise<ReadableSpan[]> {
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

function getStructuredLogMessages(logs: string[]): string[] {
  const logObjects = logs.map((l) => JSON.parse(l as string));
  return logObjects.map((log) => log.message);
}

function explode() {
  const nothing: { missing?: any } = { missing: 1 };
  delete nothing.missing;
  return nothing.missing.explode;
}
