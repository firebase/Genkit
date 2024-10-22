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

import { Genkit } from 'genkit';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import {
  SUPPORTED_MODELS as EMBEDDER_MODELS,
  textEmbeddingGecko001,
  textEmbeddingGeckoEmbedder,
} from './embedder.js';
import {
  SUPPORTED_V15_MODELS,
  SUPPORTED_V1_MODELS,
  gemini15Flash,
  gemini15Pro,
  geminiPro,
  geminiProVision,
  googleAIModel,
} from './gemini.js';
export {
  gemini15Flash,
  gemini15Pro,
  geminiPro,
  geminiProVision,
  textEmbeddingGecko001,
};

export interface PluginOptions {
  apiKey?: string;
  apiVersion?: string | string[];
  baseUrl?: string;
}

export function googleAI(options?: PluginOptions): GenkitPlugin {
  return genkitPlugin('googleai', async (ai: Genkit) => {
    let apiVersions = ['v1'];

    if (options?.apiVersion) {
      if (Array.isArray(options?.apiVersion)) {
        apiVersions = options?.apiVersion;
      } else {
        apiVersions = [options?.apiVersion];
      }
    }
    if (apiVersions.includes('v1beta')) {
      Object.keys(SUPPORTED_V15_MODELS).map((name) =>
        googleAIModel(ai, name, options?.apiKey, 'v1beta', options?.baseUrl)
      );
    }
    if (apiVersions.includes('v1')) {
      Object.keys(SUPPORTED_V1_MODELS).map((name) =>
        googleAIModel(ai, name, options?.apiKey, undefined, options?.baseUrl)
      );
      Object.keys(SUPPORTED_V15_MODELS).map((name) =>
        googleAIModel(ai, name, options?.apiKey, undefined, options?.baseUrl)
      );
      Object.keys(EMBEDDER_MODELS).map((name) =>
        textEmbeddingGeckoEmbedder(ai, name, { apiKey: options?.apiKey })
      );
    }
  });
}
export default googleAI;
