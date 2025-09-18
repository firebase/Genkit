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
  ChromaClient,
  IncludeEnum,
  type Collection,
  type CollectionMetadata,
  type Embeddings,
  type IEmbeddingFunction,
  type Metadata,
  type ChromaClientParams as NativeChromaClientParams,
  type Where,
  type WhereDocument,
} from 'chromadb';
import {
  Document,
  indexerRef,
  retrieverRef,
  z,
  type EmbedderAction,
  type EmbedderArgument,
  type Embedding,
  type Genkit,
} from 'genkit';
import {
  genkitPluginV2,
  indexer,
  retriever,
  type GenkitPluginV2,
  type ResolvableAction,
} from 'genkit/plugin';
import type { ActionType } from 'genkit/registry';
import { CommonRetrieverOptionsSchema } from 'genkit/retriever';
import { Md5 } from 'ts-md5';

export { IncludeEnum };

const WhereSchema: z.ZodType<Where> = z.any();
const WhereDocumentSchema: z.ZodType<WhereDocument> = z.any();

const IncludeOptionSchema = z
  .array(z.enum(['documents', 'embeddings', 'metadatas', 'distances']))
  .optional();
type IncludeOption = z.infer<typeof IncludeOptionSchema>;

const ChromaRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  include: IncludeOptionSchema,
  where: WhereSchema.optional(),
  whereDocument: WhereDocumentSchema.optional(),
});

export const ChromaIndexerOptionsSchema = z.null().optional();

type ChromaClientParams =
  | NativeChromaClientParams
  | (() => Promise<NativeChromaClientParams>);

type ChromaPluginParams<
  EmbedderCustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  clientParams?: ChromaClientParams;
  collectionName: string;
  createCollectionIfMissing?: boolean;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}[];

/**
 * Chroma plugin that provides the Chroma retriever and indexer
 */

export function chroma<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: ChromaPluginParams<EmbedderCustomOptions>
): GenkitPluginV2 {
  return genkitPluginV2({
    name: 'chroma',
    async init() {
      const actions: ResolvableAction[] = [];
      for (const param of params) {
        actions.push(createChromaRetriever(param));
        actions.push(createChromaIndexer(param));
      }
      return actions;
    },
    async resolve(actionType: ActionType, name: string) {
      // Find the matching param by collection name
      const collectionName = name.replace('chroma/', '');
      const param = params.find((p) => p.collectionName === collectionName);
      if (!param) return undefined;

      switch (actionType) {
        case 'retriever':
          return createChromaRetriever(param);
        case 'indexer':
          return createChromaIndexer(param);
        default:
          return undefined;
      }
    },
    async list() {
      return params.flatMap((param) => [
        {
          name: `chroma/${param.collectionName}`,
          type: 'retriever' as const,
          info: {
            label: `Chroma DB - ${param.collectionName}`,
          },
        },
        {
          name: `chroma/${param.collectionName}`,
          type: 'indexer' as const,
          info: {
            label: `Chroma DB - ${param.collectionName}`,
          },
        },
      ]);
    },
  });
}

export const chromaRetrieverRef = (params: {
  collectionName: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `chroma/${params.collectionName}`,
    info: {
      label: params.displayName ?? `Chroma DB - ${params.collectionName}`,
    },
    configSchema: ChromaRetrieverOptionsSchema.optional(),
  });
};

export const chromaIndexerRef = (params: {
  collectionName: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `chroma/${params.collectionName}`,
    info: {
      label: params.displayName ?? `Chroma DB - ${params.collectionName}`,
    },
    configSchema: ChromaIndexerOptionsSchema.optional(),
  });
};

/**
 * Configures a Chroma vector store retriever.
 */
export function chromaRetriever<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: {
    clientParams?: ChromaClientParams;
    collectionName: string;
    createCollectionIfMissing?: boolean;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const { embedder, collectionName, embedderOptions } = params;
  return ai.defineRetriever(
    {
      name: `chroma/${collectionName}`,
      configSchema: ChromaRetrieverOptionsSchema.optional(),
    },
    async (content, options) => {
      const clientParams = await resolve(params.clientParams);
      const client = new ChromaClient(clientParams);
      let collection: Collection;
      if (params.createCollectionIfMissing) {
        collection = await client.getOrCreateCollection({
          name: collectionName,
        });
      } else {
        collection = await client.getCollection({
          name: collectionName,
        });
      }

      const queryEmbeddings = await ai.embed({
        embedder,
        content,
        options: embedderOptions,
      });
      const results = await collection.query({
        nResults: options?.k,
        include: getIncludes(options?.include),
        where: options?.where,
        whereDocument: options?.whereDocument,
        queryEmbeddings: queryEmbeddings[0].embedding,
      });

      const documents = results.documents[0];
      const metadatas = results.metadatas;
      const embeddings = results.embeddings;
      const distances = results.distances;

      const combined = documents
        .map((d, i) => {
          if (d !== null) {
            return {
              document: d,
              metadata: constructMetadata(i, metadatas, embeddings, distances),
            };
          }
          return undefined;
        })
        .filter(
          (r): r is { document: string; metadata: Record<string, any> } => !!r
        );

      return {
        documents: combined.map((result) => {
          const data = result.document;
          const metadata = result.metadata.metadata[0];
          const dataType = metadata.dataType;
          const docMetadata = metadata.docMetadata
            ? JSON.parse(metadata.docMetadata)
            : undefined;
          return Document.fromData(data, dataType, docMetadata).toJSON();
        }),
      };
    }
  );
}

/**
 * Helper method to compute effective Include enum. It always
 * includes documents
 */
function getIncludes(includes: IncludeOption): IncludeEnum[] | undefined {
  if (!includes) {
    // Default behaviour
    return undefined;
  }

  // Always include documents
  let effectiveIncludes = [IncludeEnum.Documents];
  effectiveIncludes = effectiveIncludes.concat(includes as IncludeEnum[]);
  const includesSet = new Set(effectiveIncludes);
  return Array.from(includesSet);
}

/**
 * Helper method to construct metadata, including the optional {@link IncludeEnum} passed in config.
 */
function constructMetadata(
  i: number,
  metadatas: (Metadata | null)[][],
  embeddings: Embeddings[] | null,
  distances: number[][] | null
): unknown {
  var fullMetadata: Record<string, unknown> = {};
  if (metadatas && metadatas[i]) {
    fullMetadata.metadata = metadatas[i];
  }
  if (embeddings) {
    fullMetadata.embedding = embeddings[i];
  }
  if (distances) {
    fullMetadata.distances = distances[i];
  }
  return fullMetadata;
}

/**
 * Configures a Chroma indexer.
 */
export function chromaIndexer<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: {
    clientParams?: ChromaClientParams;
    collectionName: string;
    createCollectionIfMissing?: boolean;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const { collectionName, embedder, embedderOptions } = {
    ...params,
  };

  return ai.defineIndexer(
    {
      name: `chroma/${params.collectionName}`,
      configSchema: ChromaIndexerOptionsSchema,
    },
    async (docs) => {
      const clientParams = await resolve(params.clientParams);
      const client = new ChromaClient(clientParams);

      let collection: Collection;
      if (params.createCollectionIfMissing) {
        collection = await client.getOrCreateCollection({
          name: collectionName,
        });
      } else {
        collection = await client.getCollection({
          name: collectionName,
        });
      }

      const embeddings = await Promise.all(
        docs.map((doc) =>
          ai.embed({
            embedder,
            content: doc,
            options: embedderOptions,
          })
        )
      );

      const entries = embeddings
        .map((value, i) => {
          const doc = docs[i];
          // The array of embeddings for this document
          const docEmbeddings: Embedding[] = value;
          const embeddingDocs = doc.getEmbeddingDocuments(docEmbeddings);
          return docEmbeddings.map((docEmbedding, j) => {
            const metadata: Metadata = {
              docMetadata: JSON.stringify(embeddingDocs[j].metadata),
              dataType: embeddingDocs[j].dataType || '',
            };

            const data = embeddingDocs[j].data;
            const id = Md5.hashStr(JSON.stringify(embeddingDocs[j]));
            return {
              id,
              value: docEmbedding.embedding,
              document: data,
              metadata,
            };
          });
        })
        .reduce((acc, val) => {
          return acc.concat(val);
        }, []);

      await collection.add({
        ids: entries.map((e) => e.id),
        embeddings: entries.map((e) => e.value),
        metadatas: entries.map((e) => e.metadata),
        documents: entries.map((e) => e.document),
      });
    }
  );
}

/**
 * Helper function for creating Chroma collections.
 * Currently only available for text
 * https://docs.trychroma.com/docs/embeddings/multimodal
 */
export async function createChromaCollection<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    name: string;
    clientParams?: ChromaClientParams;
    metadata?: CollectionMetadata;
    embedder?: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  let chromaEmbedder: IEmbeddingFunction | undefined = undefined;
  const embedder = params.embedder;
  if (!!embedder) {
    chromaEmbedder = {
      generate(texts: string[]) {
        return Promise.all(
          texts.map((text) =>
            ai.embed({
              embedder,
              content: text,
              options: params.embedderOptions,
            })
          )
        ).then((results: Embedding[][]) => {
          return results.map((result: Embedding[]) => result[0].embedding);
        });
      },
    };
  }
  const clientParams = await resolve(params.clientParams);
  const client = new ChromaClient(clientParams);
  return await client.createCollection({
    ...params,
    embeddingFunction: chromaEmbedder,
  });
}

/**
 * Helper function for deleting Chroma collections.
 */
export async function deleteChromaCollection(params: {
  name: string;
  clientParams?: ChromaClientParams;
}) {
  const clientParams = await resolve(params.clientParams);
  const client = new ChromaClient(clientParams);
  return await client.deleteCollection({
    ...params,
  });
}

/**
 * Standalone Chroma retriever action (v2 API)
 */
function createChromaRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: {
  clientParams?: ChromaClientParams;
  collectionName: string;
  createCollectionIfMissing?: boolean;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}) {
  const { embedder, collectionName, embedderOptions } = params;

  return retriever(
    {
      name: `chroma/${collectionName}`,
      configSchema: ChromaRetrieverOptionsSchema.optional(),
    },
    async (content, options) => {
      const clientParams = await resolve(params.clientParams);
      const client = new ChromaClient(clientParams);
      let collection: Collection;
      if (params.createCollectionIfMissing) {
        collection = await client.getOrCreateCollection({
          name: collectionName,
        });
      } else {
        collection = await client.getCollection({
          name: collectionName,
        });
      }

      // For v2 API, we need to handle embedding differently
      // The embedder will be resolved at runtime
      const queryEmbeddings = await resolveEmbedder(embedder, {
        content,
        options: embedderOptions,
      });

      const results = await collection.query({
        nResults: options?.k,
        include: getIncludes(options?.include),
        where: options?.where,
        whereDocument: options?.whereDocument,
        queryEmbeddings: queryEmbeddings[0].embedding,
      });

      const documents = results.documents[0];
      const metadatas = results.metadatas;
      const embeddings = results.embeddings;
      const distances = results.distances;

      const combined = documents
        .map((d, i) => {
          if (d !== null) {
            return {
              document: d,
              metadata: constructMetadata(i, metadatas, embeddings, distances),
            };
          }
          return undefined;
        })
        .filter(
          (r): r is { document: string; metadata: Record<string, any> } => !!r
        );

      return {
        documents: combined.map((result) => {
          const data = result.document;
          const metadata = result.metadata.metadata[0];
          const dataType = metadata.dataType;
          const docMetadata = metadata.docMetadata
            ? JSON.parse(metadata.docMetadata)
            : undefined;
          return Document.fromData(data, dataType, docMetadata).toJSON();
        }),
      };
    }
  );
}

/**
 * Standalone Chroma indexer action (v2 API)
 */
function createChromaIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: {
  clientParams?: ChromaClientParams;
  collectionName: string;
  createCollectionIfMissing?: boolean;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}) {
  const { collectionName, embedder, embedderOptions } = {
    ...params,
  };

  return indexer(
    {
      name: `chroma/${params.collectionName}`,
      configSchema: ChromaIndexerOptionsSchema,
    },
    async (docs) => {
      const clientParams = await resolve(params.clientParams);
      const client = new ChromaClient(clientParams);

      let collection: Collection;
      if (params.createCollectionIfMissing) {
        collection = await client.getOrCreateCollection({
          name: collectionName,
        });
      } else {
        collection = await client.getCollection({
          name: collectionName,
        });
      }

      const embeddings = await Promise.all(
        docs.map((doc) =>
          resolveEmbedder(embedder, {
            content: doc,
            options: embedderOptions,
          })
        )
      );

      const entries = embeddings
        .map((value, i) => {
          const doc = docs[i];
          // The array of embeddings for this document
          const docEmbeddings: Embedding[] = value;
          const embeddingDocs = doc.getEmbeddingDocuments(docEmbeddings);
          return docEmbeddings.map((docEmbedding, j) => {
            const metadata: Metadata = {
              docMetadata: JSON.stringify(embeddingDocs[j].metadata),
              dataType: embeddingDocs[j].dataType || '',
            };

            const data = embeddingDocs[j].data;
            const id = Md5.hashStr(JSON.stringify(embeddingDocs[j]));
            return {
              id,
              value: docEmbedding.embedding,
              document: data,
              metadata,
            };
          });
        })
        .reduce((acc, val) => {
          return acc.concat(val);
        }, []);

      await collection.add({
        ids: entries.map((e) => e.id),
        embeddings: entries.map((e) => e.value),
        metadatas: entries.map((e) => e.metadata),
        documents: entries.map((e) => e.document),
      });
    }
  );
}

/**
 * Helper function to resolve embedder and get embeddings
 * Call embedder actions directly
 */
async function resolveEmbedder<EmbedderCustomOptions extends z.ZodTypeAny>(
  embedder: EmbedderArgument<EmbedderCustomOptions>,
  params: {
    content: Document;
    options?: z.infer<EmbedderCustomOptions>;
  }
): Promise<Embedding[]> {
  // If embedder is an action (function with __action property), call it directly
  if (typeof embedder === 'function' && '__action' in embedder) {
    const embedderAction = embedder as EmbedderAction<EmbedderCustomOptions>;
    const response = await embedderAction({
      input: [params.content],
      options: params.options,
    });
    return response.embeddings;
  }

  // If embedder is a string reference, we need to resolve it
  // throw an error as this requires registry access
  throw new Error(
    `Embedder resolution for string references not supported in v2 API: ${embedder}`
  );
}

async function resolve(
  params?: ChromaClientParams
): Promise<NativeChromaClientParams | undefined> {
  if (!params) {
    return undefined;
  }
  if (typeof params === 'function') {
    return await params();
  }
  return params;
}
