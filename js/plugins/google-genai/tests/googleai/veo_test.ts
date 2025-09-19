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
import { Operation } from 'genkit';
import { GenerateRequest } from 'genkit/model';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';

import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getGoogleAIUrl } from '../../src/googleai/client.js';
import { VeoOperation, VeoPredictRequest } from '../../src/googleai/types.js';
import {
  TEST_ONLY,
  VeoConfig,
  VeoConfigSchema,
  createModelRef,
  defineModel,
} from '../../src/googleai/veo.js';

const { toVeoParameters, fromVeoOperation, GENERIC_MODEL, KNOWN_MODELS } =
  TEST_ONLY;

describe('Google AI Veo', () => {
  describe('KNOWN_MODELS', () => {
    it('should contain non-zero number of models', () => {
      assert.ok(Object.keys(KNOWN_MODELS).length > 0);
    });
  });

  describe('model()', () => {
    it('should return a ModelReference for a known model', () => {
      const modelName = 'veo-2.0-generate-001';
      const ref = createModelRef(modelName);
      const supports: any = { ...ref.info?.supports };

      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.ok(supports?.media);
      assert.ok(supports?.longRunning);
    });

    it('should return a ModelReference for an unknown model using generic info', () => {
      const modelName = 'veo-unknown-model';
      const ref = createModelRef(modelName);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.info, GENERIC_MODEL.info);
    });

    it('should apply config to a known model', () => {
      const modelName = 'veo-2.0-generate-001';
      const config: VeoConfig = { aspectRatio: '16:9' };
      const ref = createModelRef(modelName, config);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should apply config to an unknown model', () => {
      const modelName = 'veo-unknown-model';
      const config: VeoConfig = { durationSeconds: 6 };
      const ref = createModelRef(modelName, config);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should handle model name with prefix', () => {
      const modelName = 'models/veo-2.0-generate-001';
      const ref = createModelRef(modelName);
      assert.strictEqual(ref.name, 'googleai/veo-2.0-generate-001');
    });
  });

  describe('toVeoParameters', () => {
    const baseRequest: GenerateRequest<typeof VeoConfigSchema> = {
      messages: [],
    };

    it('should include config parameters', () => {
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        ...baseRequest,
        config: {
          aspectRatio: '16:9',
          personGeneration: 'allow_adult',
          durationSeconds: 7,
        },
      };
      const result = toVeoParameters(request);
      assert.strictEqual(result.aspectRatio, '16:9');
      assert.strictEqual(result.personGeneration, 'allow_adult');
      assert.strictEqual(result.durationSeconds, 7);
    });

    it('should omit null but keep false config parameters', () => {
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        ...baseRequest,
        config: {
          enhancePrompt: false,
          negativePrompt: null as any,
        },
      };
      const result = toVeoParameters(request);
      assert.strictEqual(result.hasOwnProperty('enhancePrompt'), true);
      assert.strictEqual(result.hasOwnProperty('negativePrompt'), false);
    });
  });

  describe('fromVeoOperation', () => {
    it('should convert pending VeoOperation', () => {
      const apiOp: VeoOperation = {
        name: 'operations/123',
        done: false,
      };
      const result = fromVeoOperation(apiOp);
      assert.deepStrictEqual(result, {
        id: 'operations/123',
        done: false,
      });
    });

    it('should convert completed VeoOperation with video', () => {
      const apiOp: VeoOperation = {
        name: 'operations/123',
        done: true,
        response: {
          generateVideoResponse: {
            generatedSamples: [
              { video: { uri: 'https://example.com/video.mp4' } },
            ],
          },
        },
      };
      const result = fromVeoOperation(apiOp);
      assert.deepStrictEqual(result, {
        id: 'operations/123',
        done: true,
        output: {
          finishReason: 'stop',
          raw: apiOp.response,
          message: {
            role: 'model',
            content: [
              {
                media: {
                  url: 'https://example.com/video.mp4',
                },
              },
            ],
          },
        },
      });
    });

    it('should convert VeoOperation with error', () => {
      const apiOp: VeoOperation = {
        name: 'operations/123',
        done: true,
        error: { message: 'Something went wrong' },
      };
      const result = fromVeoOperation(apiOp);
      assert.deepStrictEqual(result, {
        id: 'operations/123',
        done: true,
        error: { message: 'Something went wrong' },
      });
    });
  });

  describe('defineModel()', () => {
    let fetchStub: sinon.SinonStub;
    let envStub: sinon.SinonStub;

    const modelName = 'veo-test-model';
    const defaultApiKey = 'default-api-key';

    beforeEach(() => {
      fetchStub = sinon.stub(global, 'fetch');
      envStub = sinon.stub(process, 'env').value({});
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

    function captureModelRunner(
      defineOptions: {
        name?: string;
        apiKey?: string | false;
        apiVersion?: string;
        baseUrl?: string;
      } = {}
    ) {
      const name = defineOptions.name || modelName;
      const apiVersion = defineOptions.apiVersion;
      const baseUrl = defineOptions.baseUrl;
      const apiKey = defineOptions.apiKey;

      const modelAction = defineModel(name, { apiKey, apiVersion, baseUrl });
      assert.strictEqual(modelAction.__action.name, `googleai/${name}`);
      return modelAction;
    }

    describe('start()', () => {
      const prompt = 'A dancing cat';
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
        config: {
          aspectRatio: '16:9',
        },
      };

      it('should call fetch for veoPredict and return operation', async () => {
        const mockOp: VeoOperation = {
          name: 'operations/start123',
          done: false,
        };
        mockFetchResponse(mockOp);

        const modelAction = captureModelRunner({ apiKey: defaultApiKey });
        const result = await modelAction.start(request);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;

        const expectedUrl = getGoogleAIUrl({
          resourcePath: `models/${modelName}`,
          resourceMethod: 'predictLongRunning',
          clientOptions: { apiVersion: undefined, baseUrl: undefined },
        });
        assert.strictEqual(fetchArgs[0], expectedUrl);

        const expectedHeaders = {
          'Content-Type': 'application/json',
          'x-goog-api-key': defaultApiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        };
        assert.deepStrictEqual(fetchArgs[1].headers, expectedHeaders);
        assert.strictEqual(fetchArgs[1].method, 'POST');

        const expectedVeoPredictRequest: VeoPredictRequest = {
          instances: [{ prompt: prompt }], // NOTE: image key is omitted
          parameters: toVeoParameters(request),
        };
        assert.deepStrictEqual(
          JSON.parse(fetchArgs[1].body),
          expectedVeoPredictRequest
        );

        const expected = fromVeoOperation(mockOp);
        assert.strictEqual(result.id, expected.id);
        assert.strictEqual(result.done, expected.done);
      });

      it('should handle custom apiVersion and baseUrl', async () => {
        mockFetchResponse({ name: 'operations/start456', done: false });
        const apiVersion = 'v1test';
        const baseUrl = 'https://test.example.com';

        const modelAction = captureModelRunner({
          apiKey: defaultApiKey,
          apiVersion,
          baseUrl,
        });
        await modelAction.start(request);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const expectedUrl = getGoogleAIUrl({
          resourcePath: `models/${modelName}`,
          resourceMethod: 'predictLongRunning',
          clientOptions: { apiVersion, baseUrl },
        });
        assert.strictEqual(fetchArgs[0], expectedUrl);
      });

      it('should use key from env if not provided in init', async () => {
        const envKey = 'env-api-key';
        envStub.value({ GOOGLE_API_KEY: envKey });
        mockFetchResponse({ name: 'operations/start789', done: false });

        const modelAction = captureModelRunner({ apiKey: undefined });
        await modelAction.start(request);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        assert.strictEqual(fetchArgs[1].headers['x-goog-api-key'], envKey);
      });

      it('should propagate API errors', async () => {
        const errorBody = { error: { message: 'Invalid argument', code: 400 } };
        mockFetchResponse(errorBody, 400);

        const modelAction = captureModelRunner({ apiKey: defaultApiKey });
        await assert.rejects(
          modelAction.start(request),
          /Error fetching from .*models\/veo-test-model:predictLongRunning.* Invalid argument/
        );
      });
    });

    describe('check()', () => {
      const operationId = 'operations/check123';
      const pendingOp: Operation = { id: operationId, done: false };

      it('should call fetch for veoCheckOperation and return updated operation', async () => {
        const mockResponse: VeoOperation = {
          name: operationId,
          done: true,
          response: {
            generateVideoResponse: {
              generatedSamples: [{ video: { uri: 'https://video.url' } }],
            },
          },
        };
        mockFetchResponse(mockResponse);

        const modelAction = captureModelRunner({ apiKey: defaultApiKey });
        const result = await modelAction.check(pendingOp);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;

        const expectedUrl = getGoogleAIUrl({
          resourcePath: operationId,
          clientOptions: { apiVersion: undefined, baseUrl: undefined },
        });
        assert.strictEqual(fetchArgs[0], expectedUrl);

        const expectedHeaders = {
          'Content-Type': 'application/json',
          'x-goog-api-key': defaultApiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        };
        assert.deepStrictEqual(fetchArgs[1].headers, expectedHeaders);
        assert.strictEqual(fetchArgs[1].method, 'GET');

        const expected = fromVeoOperation(mockResponse);
        assert.strictEqual(result.id, expected.id);
        assert.strictEqual(result.done, expected.done);
        assert.deepStrictEqual(result.output, expected.output);
      });

      it('should handle custom apiVersion and baseUrl for check', async () => {
        mockFetchResponse({ name: operationId, done: true });
        const apiVersion = 'v1test';
        const baseUrl = 'https://test.example.com';

        const modelAction = captureModelRunner({
          apiKey: defaultApiKey,
          apiVersion,
          baseUrl,
        });
        await modelAction.check(pendingOp);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const expectedUrl = getGoogleAIUrl({
          resourcePath: operationId,
          clientOptions: { apiVersion, baseUrl },
        });
        assert.strictEqual(fetchArgs[0], expectedUrl);
      });

      it('should propagate API errors for check', async () => {
        const errorBody = { error: { message: 'Not found', code: 404 } };
        mockFetchResponse(errorBody, 404);

        const modelAction = captureModelRunner({ apiKey: defaultApiKey });
        await assert.rejects(
          modelAction.check(pendingOp),
          /Error fetching from .*operations\/check123.* Not found/
        );
      });
    });
  });
});
