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

import { GoogleAuth } from 'google-auth-library';
import { PluginOptions } from '..';
import { GenkitError, z } from 'genkit';
import { IndexerAction, RetrieverAction } from 'genkit/retriever';

export {
  DocumentIndexer,
  DocumentRetriever,
  Neighbor,
  VectorSearchOptions,
  VertexAIVectorIndexerOptions,
  VertexAIVectorIndexerOptionsSchema,
  VertexAIVectorRetrieverOptions,
  VertexAIVectorRetrieverOptionsSchema,
} from './types';

let getBigQueryDocumentIndexer;
let getBigQueryDocumentRetriever;
let getFirestoreDocumentIndexer;
let getFirestoreDocumentRetriever;
let vertexAiIndexerRef;
let vertexAiIndexers;
let vertexAiRetrieverRef;
let vertexAiRetrievers;

export default async function vertexAiVectorSearch(
  projectId: string,
  location: string,
  options: PluginOptions | undefined,
  authClient: GoogleAuth,
  embedders: any[]
): Promise<{
  indexers: IndexerAction<any>[];
  retrievers: RetrieverAction<any>[];
}> {
  // Embedders are required for vector search
  if (options?.excludeEmbedders !== true) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'Vector search requires embedders. Please disable the exclusion of embedders.',
    });
  }

  await initalizeDependencies();

  let indexers: IndexerAction<z.ZodTypeAny>[] = [];
  let retrievers: RetrieverAction<z.ZodTypeAny>[] = [];

  const defaultEmbedder = embedders[0];

  indexers = vertexAiIndexers({
    pluginOptions: options,
    authClient,
    defaultEmbedder,
  });

  retrievers = vertexAiRetrievers({
    pluginOptions: options,
    authClient,
    defaultEmbedder,
  });

  return { indexers, retrievers };
}

async function initalizeDependencies() {
  const {
    getBigQueryDocumentIndexer: getBigQueryDocumentIndexerImport,
    getBigQueryDocumentRetriever: getBigQueryDocumentRetrieverImport
  } = await import('./bigquery.js');

  const {
    getFirestoreDocumentIndexer: getFirestoreDocumentIndexerImport,
    getFirestoreDocumentRetriever: getFirestoreDocumentRetrieverImport
  } = await import('./firestore.js');

  const {
    vertexAiIndexerRef: vertexAiIndexerRefImport,
    vertexAiIndexers: vertexAiIndexersImport
  } = await import('./indexers.js');

  const {
    vertexAiRetrieverRef: vertexAiRetrieverRefImport,
    vertexAiRetrievers: vertexAiRetrieversImport
  } = await import('./retrievers.js');

  getBigQueryDocumentIndexer = getBigQueryDocumentIndexerImport;
  getBigQueryDocumentRetriever = getBigQueryDocumentRetrieverImport;
  getFirestoreDocumentIndexer = getFirestoreDocumentIndexerImport;
  getFirestoreDocumentRetriever = getFirestoreDocumentRetrieverImport;
  vertexAiIndexerRef = vertexAiIndexerRefImport;
  vertexAiIndexers = vertexAiIndexersImport;
  vertexAiRetrieverRef = vertexAiRetrieverRefImport;
  vertexAiRetrievers = vertexAiRetrieversImport;
}

export {
  getBigQueryDocumentIndexer,
  getBigQueryDocumentRetriever,
  getFirestoreDocumentIndexer,
  getFirestoreDocumentRetriever,
  vertexAiIndexerRef,
  vertexAiIndexers,
  vertexAiRetrieverRef,
  vertexAiRetrievers,
};
