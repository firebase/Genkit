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
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { defineOllamaEmbedder } from '../src/embeddings.js'; // Adjust the import path as necessary
import type { OllamaPluginParams } from '../src/types.js'; // Adjust the import path as necessary
// Utility function to parse command-line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const serverAddress =
    args.find((arg) => arg.startsWith('--server-address='))?.split('=')[1] ||
    'http://localhost:11434';
  const modelName =
    args.find((arg) => arg.startsWith('--model-name='))?.split('=')[1] ||
    'nomic-embed-text';
  return { serverAddress, modelName };
}
const { serverAddress, modelName } = parseArgs();
describe('defineOllamaEmbedder - Live Tests', () => {
  const options: OllamaPluginParams = {
    models: [{ name: modelName }],
    serverAddress,
  };
  it('should successfully return embeddings', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder',
      modelName: 'nomic-embed-text',
      dimensions: 768,
      options,
    });
    const result = await embedder({
      input: [{ content: [{ text: 'Hello, world!' }] }],
    });
    assert.strictEqual(result.embeddings[0].embedding.length, 768);
  });
});
