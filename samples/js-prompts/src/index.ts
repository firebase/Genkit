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

// Import the Genkit core libraries and plugins.
import { startFlowServer } from '@genkit-ai/express';
import { googleAI } from '@genkit-ai/googleai';
import { genkit, z } from 'genkit';

const ai = genkit({
  plugins: [
    googleAI(), // Provide the key via the GOOGLE_GENAI_API_KEY environment variable or arg { apiKey: 'yourkey'}
  ],
});

const simplePrompt = ai.defineFlow('simplePrompt', () =>
  ai.generate({
    model: googleAI.model('gemini-2.0-flash'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  })
);

const simpleTemplate = ai.defineFlow('simpleTemplate', () => {
  const name = 'Fred';
  return ai.generate({
    model: googleAI.model('gemini-2.0-flash'),
    prompt: `You are a helpful AI assistant named Walt. Say hello to ${name}.`,
  });
});

const helloDotprompt = ai.definePrompt(
  {
    name: 'helloPrompt',
    model: googleAI.model('gemini-2.0-flash'),
    input: {
      schema: z.object({ name: z.string() }),
    },
  },
  `You are a helpful AI assistant named Walt. Say hello to {{name}}`
);

const simpleDotprompt = ai.defineFlow('simpleDotprompt', () => {
  return helloDotprompt({ name: 'Fred' });
});

const outputSchema = z.object({
  short: z.string(),
  friendly: z.string(),
  likeAPirate: z.string(),
});

const threeGreetingsPrompt = ai.definePrompt(
  {
    name: 'threeGreetingsPrompt',
    model: googleAI.model('gemini-2.0-flash'),
    input: {
      schema: z.object({ name: z.string() }),
    },
    output: {
      format: 'json',
      schema: outputSchema,
    },
  },
  `You are a helpful AI assistant named Walt. Say hello to {{name}}, write a response for each of the styles requested`
);

const threeGreetings = ai.defineFlow('threeGreetingsPrompt', async () => {
  const response = await threeGreetingsPrompt({ name: 'Fred' });
  return response.output;
});

const configuredPrompt = ai.definePrompt(
  {
    name: 'configuredPrompt',
    model: googleAI.model('gemini-2.0-flash'),
    config: {
      temperature: 0.9,
      topK: 50,
      topP: 0.4,
      maxOutputTokens: 400,
      stopSequences: ['<end>', '<fin>'],
    },
    input: {
      schema: z.object({ topic: z.string() }),
    },
  },
  `Tell me a short story about {{topic}}.`
);

const multiMessagePrompt = ai.definePrompt(
  {
    name: 'multiMessagePrompt',
    model: googleAI.model('gemini-2.0-flash'),
    input: {
      schema: z.object({ userQuestion: z.string() }),
    },
    messages: async (input) => [
      {
        role: 'system',
        content: 'You are a helpful AI assistant that really loves to talk about food. Try to work food items into all of your conversations.',
      },
      {
        role: 'user',
        content: input.userQuestion,
      },
    ],
  }
);

const multiModalPrompt = ai.definePrompt(
  {
    name: 'multiModalPrompt',
    model: googleAI.model('gemini-2.0-flash'),
    input: {
      schema: z.object({ photoUrl: z.string() }),
    },
    messages: async (input) => [
      {
        role: 'user',
        content: [
          { text: 'Describe this image in a detailed paragraph:' },
          { media: { url: input.photoUrl } },
        ],
      },
    ],
  }
);

ai.defineHelper('shout', (text: string) => text.toUpperCase());

const customHelperPrompt = ai.definePrompt(
  {
    name: 'customHelperPrompt',
    model: googleAI.model('gemini-2.0-flash'),
    input: {
      schema: z.object({ name: z.string() }),
    },
  },
  `HELLO, {{shout name}}!!!`
);

// Start a flow server, which exposes your flows as HTTP endpoints. This call
// must come last, after all of your plug-in configuration and flow definitions.
// You can optionally specify a subset of flows to serve, and configure some
// HTTP server options, but by default, the flow server serves all defined flows.
startFlowServer({
  flows: [threeGreetings, simpleTemplate, simpleDotprompt, simplePrompt],
});
