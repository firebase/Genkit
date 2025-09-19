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

import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';
import type { CommonPluginOptions } from '../common/types.js';
import type { RerankerOptions } from './types.js';

import { getDerivedParams } from '../common/index.js';

import { vertexAiRerankers } from './reranker.js';

export interface PluginOptions extends CommonPluginOptions, RerankerOptions {}

/**
 * Add Google Cloud Vertex AI Rerankers API to Genkit.
 */
export function vertexAIRerankers(options: PluginOptions): GenkitPluginV2 {
  return genkitPluginV2({
    name: 'vertexAIRerankers',
    init: async () => {
      const { projectId, location, authClient } =
        await getDerivedParams(options);

      return vertexAiRerankers({
        projectId,
        location,
        authClient,
        rerankOptions: options.rerankers.map((o) =>
          typeof o === 'string' ? { model: o } : o
        ),
      });
    },
  });
}
