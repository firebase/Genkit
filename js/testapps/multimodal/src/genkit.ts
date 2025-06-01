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

import { devLocalVectorstore } from '@genkit-ai/dev-local-vectorstore';
import {
  gemini15Flash,
  multimodalEmbedding001,
  vertexAI,
} from '@genkit-ai/vertexai';
import { genkit, type Genkit } from 'genkit';
import { chroma } from 'genkitx-chromadb';
import { pinecone } from 'genkitx-pinecone';
import { GoogleAuth, type IdTokenClient } from 'google-auth-library';

const auth = new GoogleAuth();
let authClient: IdTokenClient | undefined = undefined;

/** Helper method to cache {@link IdTokenClient} instance */
async function getCloudRunAuthClient(aud: string) {
  if (!authClient) {
    authClient = await auth.getIdTokenClient(aud);
  }
  return authClient;
}

export const ai: Genkit = genkit({
  plugins: [
    vertexAI({
      location: 'us-central1',
    }),
    pinecone([
      {
        indexId: 'pinecone-multimodal-index',
        embedder: multimodalEmbedding001,
      },
    ]),
    chroma([
      {
        collectionName: 'multimodal_collection',
        embedder: multimodalEmbedding001,
        createCollectionIfMissing: true,
        clientParams: async () => {
          // Replace this with your Cloud Run Instance URL
          const host = 'https://<my-cloud-run-url>.run.app';
          const client = await getCloudRunAuthClient(host);
          const idToken = await client.idTokenProvider.fetchIdToken(host);
          return {
            path: host,
            fetchOptions: {
              headers: {
                Authorization: 'Bearer ' + idToken,
              },
            },
          };
        },
      },
    ]),
    devLocalVectorstore([
      {
        indexName: 'localMultiModalIndex',
        embedder: multimodalEmbedding001,
      },
    ]),
  ],
  model: gemini15Flash,
});
