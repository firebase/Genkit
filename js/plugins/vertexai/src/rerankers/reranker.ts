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

import { ResolvableAction, reranker } from 'genkit/plugin';
import { RankedDocument, rerankerRef } from 'genkit/reranker';
import { DEFAULT_MODEL, getRerankEndpoint } from './constants.js';
import {
  VertexAIRerankerOptionsSchema,
  type VertexRerankOptions,
} from './types.js';

/**
 * Creates Vertex AI rerankers.
 *
 * This function creates and registers rerankers for the specified models.
 *
 * @param {VertexRerankOptions<EmbedderCustomOptions>} options - The parameters for creating the rerankers.
 * @returns {Promise<void>}
 */
export async function vertexAiRerankers(
  options: VertexRerankOptions
): Promise<ResolvableAction[]> {
  const rerankOptions = options.rerankOptions;

  if (rerankOptions.length === 0) {
    throw new Error('Provide at least one reranker configuration.');
  }

  const auth = options.authClient;
  const client = await auth.getClient();
  const projectId = options.projectId;

  const rerankerActions: ResolvableAction[] = [];
  for (const rerankOption of rerankOptions) {
    if (!rerankOption.name && !rerankOption.model) {
      throw new Error('At least one of name or model must be provided.');
    }
    const rerankerAction = reranker(
      {
        name: `vertexai/${rerankOption.name || rerankOption.model}`,
        configSchema: VertexAIRerankerOptionsSchema.optional(),
      },
      async (query, documents, _options) => {
        const response = await client.request({
          method: 'POST',
          url: getRerankEndpoint(projectId, options.location ?? 'us-central1'),
          data: {
            model: rerankOption.model || DEFAULT_MODEL, // Use model from config or default
            query: query.text,
            records: documents.map((doc, idx) => ({
              id: `${idx}`,
              content: doc.text,
            })),
          },
        });

        const rankedDocuments: RankedDocument[] = (
          response.data as any
        ).records.map((record: any) => {
          const doc = documents[record.id];
          return new RankedDocument({
            content: doc.content,
            metadata: {
              ...doc.metadata,
              score: record.score,
            },
          });
        });

        return { documents: rankedDocuments };
      }
    );

    rerankerActions.push(rerankerAction);
  }

  return rerankerActions;
}

/**
 * Creates a reference to a Vertex AI reranker.
 *
 * @param {Object} params - The parameters for the reranker reference.
 * @param {string} [params.displayName] - An optional display name for the reranker.
 * @returns {Object} - The reranker reference object.
 */
export const vertexAiRerankerRef = (params: {
  rerankerName: string;
  displayName?: string;
}) => {
  return rerankerRef({
    name: `vertexai/${params.rerankerName}`,
    info: {
      label: params.displayName ?? `Vertex AI Reranker`,
    },
    configSchema: VertexAIRerankerOptionsSchema.optional(),
  });
};
