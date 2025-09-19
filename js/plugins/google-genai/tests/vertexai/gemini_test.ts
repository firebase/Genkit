/**
 * Copyright 2025 Google LLC
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

import * as assert from 'assert';
import { z } from 'genkit';
import { GenerateRequest, ModelAction, ModelReference } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { AsyncLocalStorage } from 'node:async_hooks';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { FinishReason } from '../../src/common/types.js';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import {
  GeminiConfigSchema,
  createModelRef,
  defineModel,
} from '../../src/vertexai/gemini.js';
import {
  ClientOptions,
  GenerateContentRequest,
  GenerateContentResponse,
  HarmBlockThreshold,
  HarmCategory,
  isFunctionDeclarationsTool,
  isGoogleSearchRetrievalTool,
  isRetrievalTool,
} from '../../src/vertexai/types.js';

describe('Vertex AI Gemini', () => {
  let fetchStub: sinon.SinonStub;
  let mockAsyncStore: sinon.SinonStubbedInstance<AsyncLocalStorage<any>>;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const defaultRegionalClientOptions: ClientOptions = {
    kind: 'regional',
    projectId: 'test-project',
    location: 'us-central1',
    authClient: {} as any,
  };

  const defaultGlobalClientOptions: ClientOptions = {
    kind: 'global',
    projectId: 'test-project',
    location: 'global',
    authClient: {} as any,
    apiKey: 'test-api-key',
  };

  const defaultExpressClientOptions: ClientOptions = {
    kind: 'express',
    apiKey: 'test-express-api-key',
  };

  beforeEach(() => {
    mockAsyncStore = sinon.createStubInstance(AsyncLocalStorage);
    authMock = sinon.createStubInstance(GoogleAuth);

    authMock.getAccessToken.resolves('test-token');
    defaultRegionalClientOptions.authClient = authMock as unknown as GoogleAuth;
    defaultGlobalClientOptions.authClient = authMock as unknown as GoogleAuth;

    mockAsyncStore.getStore.returns(undefined);
    mockAsyncStore.run.callsFake((_, callback) => callback());

    fetchStub = sinon.stub(global, 'fetch');
  });

  afterEach(() => {
    sinon.restore();
  });

  function mockFetchResponse(body: any, status = 200) {
    const response = new Response(JSON.stringify(body), {
      status: status,
      statusText: status === 200 ? 'OK' : 'Error',
      headers: { 'Content-Type': 'application/json' },
    });
    fetchStub.resolves(Promise.resolve(response));
  }

  function mockFetchStreamResponse(responses: GenerateContentResponse[]) {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        for (const response of responses) {
          const chunk = `data: ${JSON.stringify(response)}\n\n`;
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });

    const response = new Response(stream, {
      status: 200,
      statusText: 'OK',
      headers: { 'Content-Type': 'text/event-stream' },
    });
    fetchStub.resolves(Promise.resolve(response));
  }

  const minimalRequest: GenerateRequest<typeof GeminiConfigSchema> = {
    messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
    config: {},
  };

  const mockCandidate = {
    index: 0,
    content: { role: 'model', parts: [{ text: 'Hi there' }] },
    finishReason: FinishReason.STOP,
  };

  const defaultApiResponse: GenerateContentResponse = {
    candidates: [mockCandidate],
  };

  describe('gemini() function', () => {
    it('returns a ModelReference for a known model string', () => {
      const name = 'gemini-2.0-flash';
      const modelRef: ModelReference<typeof GeminiConfigSchema> =
        createModelRef(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('returns a ModelReference for an unknown model string', () => {
      const name = 'gemini-new-model';
      const modelRef: ModelReference<typeof GeminiConfigSchema> =
        createModelRef(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('applies options to the ModelReference', () => {
      const options = { temperature: 0.9, topK: 20 };
      const modelRef: ModelReference<typeof GeminiConfigSchema> =
        createModelRef('gemini-2.0-flash', options);
      assert.deepStrictEqual(modelRef.config, options);
    });
  });

  function runCommonTests(clientOptions: ClientOptions) {
    describe(`Model Action Callback ${clientOptions.kind}`, () => {
      let modelAction: ModelAction;

      beforeEach(() => {
        modelAction = defineModel('gemini-2.5-flash', clientOptions);
      });

      function getExpectedHeaders(
        configApiKey?: string
      ): Record<string, string | undefined> {
        const headers: Record<string, string | undefined> = {
          'Content-Type': 'application/json',
          'X-Goog-Api-Client': getGenkitClientHeader(),
          'User-Agent': getGenkitClientHeader(),
        };
        if (clientOptions.kind !== 'express') {
          headers['Authorization'] = 'Bearer test-token';
          headers['x-goog-user-project'] = clientOptions.projectId;
        } else {
          headers['x-goog-api-key'] = clientOptions.apiKey as string;
        }
        if (configApiKey || clientOptions.apiKey) {
          headers['x-goog-api-key'] =
            configApiKey || clientOptions.apiKey || '';
        }
        return headers;
      }

      function getExpectedUrl(
        modelName: string,
        method: string,
        queryParams: string[] = []
      ): string {
        let baseUrl: string;
        let projectAndLocation = '';
        if (clientOptions.kind != 'express') {
          projectAndLocation = `projects/${clientOptions.projectId}/locations/${clientOptions.location}`;
        }

        if (clientOptions.kind === 'regional') {
          baseUrl = `https://${clientOptions.location}-aiplatform.googleapis.com/v1beta1/${projectAndLocation}`;
        } else if (clientOptions.kind === 'global') {
          baseUrl = `https://aiplatform.googleapis.com/v1beta1/${projectAndLocation}`;
        } else {
          // express
          baseUrl = `https://aiplatform.googleapis.com/v1beta1`;
        }

        let url = `${baseUrl}/publishers/google/models/${modelName}:${method}`;
        const params = [...queryParams];

        if (params.length > 0) {
          url += '?' + params.join('&');
        }
        return url;
      }

      it('throws if no messages are provided', async () => {
        await assert.rejects(
          modelAction({ messages: [], config: {} }, {}),
          /No messages provided/
        );
      });

      it('calls fetch for non-streaming requests', async () => {
        mockFetchResponse(defaultApiResponse);
        const result = await modelAction(minimalRequest, {});

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const options = fetchArgs[1];

        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'generateContent'
        );
        assert.strictEqual(url, expectedUrl);
        assert.strictEqual(options.method, 'POST');
        const body = JSON.parse(options.body);
        assert.deepStrictEqual(body.contents, [
          { role: 'user', parts: [{ text: 'Hello' }] },
        ]);
        assert.strictEqual(body.labels, undefined);

        assert.deepStrictEqual(options.headers, getExpectedHeaders());

        assert.strictEqual(result.candidates?.length, 1);
        assert.strictEqual(
          result.candidates[0].message.content[0].text,
          'Hi there'
        );
      });

      it('calls fetch for streaming requests', async () => {
        mockFetchStreamResponse([defaultApiResponse]);

        const sendChunkSpy = sinon.spy();
        await modelAction(minimalRequest, {
          onChunk: sendChunkSpy,
        });

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'streamGenerateContent',
          ['alt=sse']
        );
        assert.strictEqual(url, expectedUrl);

        await new Promise((resolve) => setTimeout(resolve, 10));

        sinon.assert.calledOnce(sendChunkSpy);
        const chunkArg = sendChunkSpy.lastCall.args[0];
        assert.deepStrictEqual(chunkArg, {
          index: 0,
          content: [{ text: 'Hi there' }],
        });
      });

      it('passes AbortSignal to fetch', async () => {
        mockFetchResponse(defaultApiResponse);
        const controller = new AbortController();
        const abortSignal = controller.signal;
        await modelAction(minimalRequest, {
          abortSignal,
        });
        sinon.assert.calledOnce(fetchStub);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.ok(fetchOptions.signal, 'Fetch options should have a signal');
        assert.notStrictEqual(
          fetchOptions.signal,
          abortSignal,
          'Fetch signal should be a new signal, not the original'
        );

        const fetchSignal = fetchOptions.signal;
        const abortSpy = sinon.spy();
        fetchSignal.addEventListener('abort', abortSpy);
        controller.abort();
        sinon.assert.calledOnce(abortSpy);
      });

      it('handles system instructions', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'system', content: [{ text: 'Be concise' }] },
            { role: 'user', content: [{ text: 'Hello' }] },
          ],
          config: {},
        };
        await modelAction(request, {});

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.systemInstruction, {
          role: 'user',
          parts: [{ text: 'Be concise' }],
        });
        assert.deepStrictEqual(apiRequest.contents, [
          { role: 'user', parts: [{ text: 'Hello' }] },
        ]);
      });

      it('merges config from request', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { temperature: 0.1, topP: 0.8, maxOutputTokens: 100 },
        };
        await modelAction(request, {});

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.generationConfig?.temperature, 0.1);
        assert.strictEqual(apiRequest.generationConfig?.topP, 0.8);
        assert.strictEqual(apiRequest.generationConfig?.maxOutputTokens, 100);
      });

      it('sends labels when provided in config', async () => {
        mockFetchResponse(defaultApiResponse);
        const myLabels = { env: 'test', version: '1' };
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { labels: myLabels },
        };
        await modelAction(request, {});

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.labels, myLabels);
      });

      it('constructs tools array with functionDeclarations', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          tools: [
            {
              name: 'myFunc',
              description: 'Does something',
              inputSchema: z.object({ foo: z.string() }),
              outputSchema: z.string(),
            },
          ],
          config: {},
        };
        await modelAction(request, {});

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        const tool = apiRequest.tools![0];
        assert.ok(
          isFunctionDeclarationsTool(tool),
          'Expected FunctionDeclarationsTool'
        );
        if (isFunctionDeclarationsTool(tool)) {
          assert.ok(tool.functionDeclarations);
          assert.strictEqual(tool.functionDeclarations?.length, 1);
          assert.strictEqual(tool.functionDeclarations?.[0].name, 'myFunc');
        }
      });

      it('handles googleSearchRetrieval tool (as googleSearch)', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            googleSearchRetrieval: {},
          },
        };
        await modelAction(request, {});
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        const searchTool = apiRequest.tools?.find(isGoogleSearchRetrievalTool);
        assert.ok(searchTool, 'Expected GoogleSearchRetrievalTool');
        if (searchTool) {
          assert.ok(searchTool.googleSearch, 'Expected googleSearch property');
          assert.deepStrictEqual(searchTool, { googleSearch: {} });
        }
      });

      if (clientOptions.kind === 'regional') {
        it('handles vertexRetrieval tool', async () => {
          mockFetchResponse(defaultApiResponse);
          const request: GenerateRequest<typeof GeminiConfigSchema> = {
            ...minimalRequest,
            config: {
              vertexRetrieval: {
                datastore: { dataStoreId: 'my-store' },
                disableAttribution: true,
              },
            },
          };
          await modelAction(request, {});
          const apiRequest: GenerateContentRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          const retrievalTool = apiRequest.tools?.find(isRetrievalTool);
          assert.ok(retrievalTool, 'Expected RetrievalTool');
          if (retrievalTool) {
            assert.ok(retrievalTool.retrieval, 'Expected retrieval property');
            assert.deepStrictEqual(retrievalTool, {
              retrieval: {
                vertexAiSearch: {
                  datastore:
                    'projects/test-project/locations/us-central1/collections/default_collection/dataStores/my-store',
                },
                disableAttribution: true,
              },
            });
          }
        });
      }

      it('applies safetySettings', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_ONLY_HIGH',
              },
            ],
          },
        };
        await modelAction(request, {});
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.safetySettings, [
          {
            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
          },
        ]);
      });

      it('handles JSON output mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          output: { format: 'json' },
          config: {},
        };
        await modelAction(request, {});
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.generationConfig?.responseMimeType,
          'application/json'
        );
      });

      it('throws if no candidates are returned', async () => {
        mockFetchResponse({ candidates: [] });
        await assert.rejects(
          modelAction(minimalRequest, {}),
          /No valid candidates returned/
        );
      });

      it('handles API call error', async () => {
        mockFetchResponse({ error: { message: 'API Error' } }, 400);
        await assert.rejects(
          modelAction(minimalRequest, {}),
          /Error fetching from .*?: \[400 Error\] API Error/
        );
      });

      it('handles config.apiKey override', async () => {
        mockFetchResponse(defaultApiResponse);
        const overrideKey = 'override-api-key';
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { apiKey: overrideKey },
        };
        await modelAction(request, {});
        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];

        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'generateContent',
          []
        );
        assert.strictEqual(url, expectedUrl);
        assert.deepStrictEqual(
          fetchArgs[1].headers,
          getExpectedHeaders(overrideKey)
        );
      });
    });
  }

  describe('defineModel - Regional Client', () => {
    it('defines a model with the correct name', () => {
      const modelAction = defineModel(
        'gemini-2.0-flash',
        defaultRegionalClientOptions
      );
      assert.strictEqual(
        modelAction.__action.name,
        'vertexai/gemini-2.0-flash'
      );
    });

    runCommonTests(defaultRegionalClientOptions);

    it('handles googleSearchRetrieval tool for gemini-1.5', async () => {
      const modelAction = defineModel(
        'gemini-1.5-pro',
        defaultRegionalClientOptions
      );
      mockFetchResponse(defaultApiResponse);
      const request: GenerateRequest<typeof GeminiConfigSchema> = {
        ...minimalRequest,
        config: {
          googleSearchRetrieval: {},
        },
      };
      await modelAction(request, {});
      const apiRequest: GenerateContentRequest = JSON.parse(
        fetchStub.lastCall.args[1].body
      );
      const searchTool = apiRequest.tools?.find(isGoogleSearchRetrievalTool);
      assert.ok(searchTool, 'Expected GoogleSearchRetrievalTool');
      if (searchTool) {
        assert.ok(
          searchTool.googleSearchRetrieval,
          'Expected googleSearchRetrieval property'
        );
        assert.deepStrictEqual(searchTool, { googleSearchRetrieval: {} });
      }
    });
  });

  describe('defineModel - Global Client', () => {
    it('defines a model with the correct name', () => {
      const modelAction = defineModel(
        'gemini-2.0-flash',
        defaultGlobalClientOptions
      );
      assert.strictEqual(
        modelAction.__action.name,
        'vertexai/gemini-2.0-flash'
      );
    });

    runCommonTests(defaultGlobalClientOptions);
  });

  describe('defineModel - Express Client', () => {
    it('defines a model with the correct name', () => {
      const modelAction = defineModel(
        'gemini-2.0-flash',
        defaultExpressClientOptions
      );
      assert.strictEqual(
        modelAction.__action.name,
        'vertexai/gemini-2.0-flash'
      );
    });

    runCommonTests(defaultExpressClientOptions);
  });
});
