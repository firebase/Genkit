/**
 * Copyright 2024 The Fire Company
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

// import { defineEmbedder, embedderRef } from '@genkit-ai/ai/embedder';

import type { EmbedderAction, EmbedderReference, Genkit } from 'genkit';
import OpenAI from 'openai';

/**
 * Method to define a new Genkit Embedder that is compatibale with the Open AI
 * Embeddings API. 
 *
 * @param params An object containing parameters for defining the OpenAI embedder.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the embedder.
 * @param params.client The OpenAI client instance.
 * @param params.embedderRef Optional reference to the embedder's configuration and
 * custom options.

 * @returns the created {@link EmbedderAction}
 */
export function defineCompatOpenAIEmbedder(params: {
  ai: Genkit;
  name: string;
  client: OpenAI;
  embedderRef?: EmbedderReference;
}): EmbedderAction {
  const { ai, name, client, embedderRef } = params;
  const model = name.split('/').pop();
  return ai.defineEmbedder(
    {
      name,
      configSchema: embedderRef?.configSchema,
      ...embedderRef?.info,
    },
    async (input, options) => {
      const { encodingFormat: encoding_format, ...restOfConfig } = options;
      const embeddings = await client.embeddings.create({
        model: model!,
        input: input.map((d) => d.text),
        encoding_format,
        ...restOfConfig,
      });
      return {
        embeddings: embeddings.data.map((d) => ({ embedding: d.embedding })),
      };
    }
  );
}
