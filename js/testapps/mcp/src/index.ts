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

import { googleAI } from '@genkit-ai/googleai';
import { defineMcpHost } from '@genkit-ai/mcp';
import { genkit, z } from 'genkit';
import { logger } from 'genkit/logging';
import path from 'path';

// Turn off safety checks for evaluation so that the LLM as an evaluator can
// respond appropriately to potentially harmful content without error.
export const PERMISSIVE_SAFETY_SETTINGS: any = {
  safetySettings: [
    {
      category: 'HARM_CATEGORY_HATE_SPEECH',
      threshold: 'BLOCK_NONE',
    },
    {
      category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
      threshold: 'BLOCK_NONE',
    },
    {
      category: 'HARM_CATEGORY_HARASSMENT',
      threshold: 'BLOCK_NONE',
    },
    {
      category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
      threshold: 'BLOCK_NONE',
    },
  ],
};

export const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.5-flash'),
});

logger.setLogLevel('debug'); // Set the logging level to debug for detailed output

export const mcpHost = defineMcpHost(ai, {
  name: 'test-mcp-manager',
  mcpServers: {
    'git-client': {
      command: 'uvx',
      args: ['mcp-server-git'],
    },
    fs: {
      command: 'npx',
      args: [
        '-y',
        '@modelcontextprotocol/server-filesystem',
        `${process.cwd()}/test-workspace`,
      ],
    },
    everything: {
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-everything'],
    },
  },
});

ai.defineFlow('git-commits', async (q) => {
  const { text } = await ai.generate({
    prompt: `summarize last 5 commits in '${path.resolve(process.cwd(), '../../..')}'`,
    tools: await mcpHost.getActiveTools(ai),
  });

  return text;
});

ai.defineFlow('dynamic-git-commits', async (q) => {
  const { text } = await ai.generate({
    prompt: `summarize last 5 commits in '${path.resolve(process.cwd(), '../../..')}'`,
    tools: ['test-mcp-manager:tool/*'], // All the tools
  });

  return text;
});

ai.defineFlow('get-file', async (q) => {
  const { text } = await ai.generate({
    prompt: `summarize contexts of hello-world.txt (in '${process.cwd()}/test-workspace')`,
    tools: await mcpHost.getActiveTools(ai),
  });

  return text;
});

ai.defineFlow('dynamic-get-file', async (q) => {
  const { text } = await ai.generate({
    prompt: `summarize contexts of hello-world.txt (in '${process.cwd()}/test-workspace')`,
    tools: ['test-mcp-manager:tool/fs/read_file'], // Just this one tool
  });

  return text;
});

ai.defineFlow('test-resource', async (q) => {
  const { text } = await ai.generate({
    prompt: [
      { text: 'analyze this: ' },
      { resource: { uri: 'test://static/resource/1' } },
    ],
    resources: await mcpHost.getActiveResources(ai),
  });

  return text;
});

// TODO(ifielker): Make resources work too
// ai.defineFlow('dynamic-test-resource', async (q) => {
//   const { text } = await ai.generate({
//     prompt: [
//       { text: 'analyze this: ' },
//       { resource: { uri: 'test://static/resource/1' } },
//     ],
//     resources: ['test-mcp-manager:resource/*'],
//   });

//   return text;
// });

ai.defineFlow('update-file', async (q) => {
  const { text } = await ai.generate({
    prompt: `Improve hello-world.txt (in '${process.cwd()}/test-workspace') by rewriting the text, making it longer, just do it, use your imagination.`,
    tools: await mcpHost.getActiveTools(ai),
  });

  return text;
});

// MCP Controls
export const controlMcp = ai.defineFlow(
  {
    name: 'controlMcp',
    inputSchema: z.object({
      action: z.enum(['RECONNECT', 'ENABLE', 'DISABLE', 'DISCONNECT'] as const),
      clientId: z.string().optional(),
    }),
    outputSchema: z.string(),
  },
  async ({ action, clientId }) => {
    const id = clientId ?? 'git-client';
    switch (action) {
      case 'DISABLE':
        await mcpHost.disable(id);
        break;
      case 'DISCONNECT':
        await mcpHost.disconnect(id);
        break;
      case 'RECONNECT':
        await mcpHost.reconnect(id);
        break;
      case 'ENABLE':
        await mcpHost.enable(id);
        break;
    }
    return action;
  }
);
