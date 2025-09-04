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

module.exports = {
  evaluators: [
    {
      flowName: 'generateDraftFlow',
      extractors: {
        context: (trace) => {
          const rootSpan = Object.values(trace.spans).find(
            (s) =>
              s.attributes['genkit:type'] === 'flow' &&
              s.attributes['genkit:name'] === 'generateDraftFlow'
          );
          if (!rootSpan) return JSON.stringify([]);

          const input = JSON.parse(rootSpan.attributes['genkit:input']);
          return JSON.stringify([JSON.stringify(input)]);
        },
        // Keep the default extractors for input and output
      },
    },
    {
      flowName: 'classifyInquiryFlow',
      extractors: {
        context: (trace) => {
          const rootSpan = Object.values(trace.spans).find(
            (s) =>
              s.attributes['genkit:type'] === 'flow' &&
              s.attributes['genkit:name'] === 'classifyInquiryFlow'
          );
          if (!rootSpan) return JSON.stringify([]);

          const input = JSON.parse(rootSpan.attributes['genkit:input']);
          return JSON.stringify([JSON.stringify(input)]);
        },
        // Keep the default extractors for input and output
      },
    },
  ],
};
