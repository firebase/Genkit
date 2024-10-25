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

import { GenkitError } from '@genkit-ai/core';
import JSON5 from 'json5';
import { extractJson } from '../extract';
import type { Formatter } from './types';

function objectLines(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('{'));
}

export const jsonlParser: Formatter = (request) => {
  if (
    request.output?.schema &&
    (request.output?.schema.type !== 'array' ||
      request.output?.schema.items?.type !== 'object')
  ) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Must supply an 'array' schema type containing 'object' items when using the 'jsonl' parser format.`,
    });
  }

  let instructions: boolean | string = false;
  if (request.output?.schema?.items) {
    instructions = `Output should be JSONL format, a sequence of JSON objects (one per line). Each line should conform to the following schema:

\`\`\`
${JSON.stringify(request.output.schema.items)}
\`\`\`
    `;
  }

  let cursor = 0;

  return {
    parseChunk: (chunk, emit) => {
      const jsonLines = objectLines(chunk.accumulatedText);

      for (let i = cursor; i < jsonLines.length; i++) {
        try {
          const result = JSON5.parse(jsonLines[i]);
          if (result) {
            emit(result);
          }
        } catch (e) {
          cursor = i;
          return;
        }
      }

      cursor = jsonLines.length;
    },

    parseResponse: (response) => {
      const items = objectLines(response.text)
        .map((l) => extractJson(l))
        .filter((l) => !!l);

      return items;
    },

    instructions,
  };
};
