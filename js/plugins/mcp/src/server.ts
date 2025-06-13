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
  GenkitError,
  Message,
  type Genkit,
  type MessageData,
  type PromptAction,
} from 'genkit';
import type { McpServerOptions } from './index.js';

import { toJsonSchema } from '@genkit-ai/core/schema';
import type { Server } from '@modelcontextprotocol/sdk/server/index.js' with { 'resolution-mode': 'import' };
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js' with { 'resolution-mode': 'import' };
import type {
  CallToolRequest,
  CallToolResult,
  GetPromptRequest,
  GetPromptResult,
  ListPromptsRequest,
  ListPromptsResult,
  ListToolsRequest,
  ListToolsResult,
  Prompt,
  PromptMessage,
  Tool,
} from '@modelcontextprotocol/sdk/types.js' with { 'resolution-mode': 'import' };
import { logger } from 'genkit/logging';
import { toToolDefinition, type ToolAction } from 'genkit/tool';
export class GenkitMcpServer {
  ai: Genkit;
  options: McpServerOptions;
  server?: Server;
  actionsResolved = false;
  toolActions: ToolAction[] = [];
  promptActions: PromptAction[] = [];

  constructor(ai: Genkit, options: McpServerOptions) {
    this.ai = ai;
    this.options = options;
    this.setup();
  }

  async setup(): Promise<void> {
    if (this.actionsResolved) return;
    const { Server } = await import(
      '@modelcontextprotocol/sdk/server/index.js'
    );

    this.server = new Server(
      { name: this.options.name, version: this.options.version || '1.0.0' },
      {
        capabilities: {
          prompts: {},
          tools: {},
        },
      }
    );

    const {
      CallToolRequestSchema,
      GetPromptRequestSchema,
      ListPromptsRequestSchema,
      ListToolsRequestSchema,
    } = await import('@modelcontextprotocol/sdk/types.js');

    this.server.setRequestHandler(
      ListToolsRequestSchema,
      this.listTools.bind(this)
    );
    this.server.setRequestHandler(
      CallToolRequestSchema,
      this.callTool.bind(this)
    );
    this.server.setRequestHandler(
      ListPromptsRequestSchema,
      this.listPrompts.bind(this)
    );
    this.server.setRequestHandler(
      GetPromptRequestSchema,
      this.getPrompt.bind(this)
    );

    const allActions = await this.ai.registry.listActions();
    const toolList: ToolAction[] = [];
    const promptList: PromptAction[] = [];
    for (const k in allActions) {
      if (k.startsWith('/tool/')) {
        toolList.push(allActions[k] as ToolAction);
      } else if (k.startsWith('/prompt/')) {
        promptList.push(allActions[k] as PromptAction);
      }
    }
    this.toolActions = toolList;
    this.promptActions = promptList;
    this.actionsResolved = true;
  }

  async listTools(req: ListToolsRequest): Promise<ListToolsResult> {
    await this.setup();
    return {
      tools: this.toolActions.map((t): Tool => {
        const def = toToolDefinition(t);
        return {
          name: def.name,
          inputSchema: (def.inputSchema as any) || { type: 'object' },
          description: def.description,
        };
      }),
    };
  }

  async callTool(req: CallToolRequest): Promise<CallToolResult> {
    await this.setup();
    const tool = this.toolActions.find(
      (t) => t.__action.name === req.params.name
    );
    if (!tool)
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `Tried to call tool '${req.params.name}' but it could not be found.`,
      });

    const clientContext = req.params._meta?.context;

    let result: any;

    if (clientContext) {
      const currentContext = this.ai.currentContext() || {};
      result = await tool(req.params.arguments, {
        context: { ...currentContext, ...clientContext },
      });
    } else {
      result = await tool(req.params.arguments);
    }
    return { content: [{ type: 'text', text: JSON.stringify(result) }] };
  }

  async listPrompts(req: ListPromptsRequest): Promise<ListPromptsResult> {
    await this.setup();
    return {
      prompts: this.promptActions.map((p): Prompt => {
        return {
          name: p.__action.name,
          description: p.__action.description,
          arguments: toMcpPromptArguments(p),
        };
      }),
    };
  }

  async getPrompt(req: GetPromptRequest): Promise<GetPromptResult> {
    await this.setup();
    const prompt = this.promptActions.find(
      (p) => p.__action.name === req.params.name
    );
    if (!prompt)
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `[@genkit-ai/mcp] Tried to call prompt '${req.params.name}' but it could not be found.`,
      });
    const result = await prompt(req.params.arguments);
    return {
      description: prompt.__action.description,
      messages: result.messages.map(toMcpPromptMessage),
    };
  }

  async start(transport?: Transport) {
    if (!transport) {
      const { StdioServerTransport } = await import(
        '@modelcontextprotocol/sdk/server/stdio.js'
      );
      transport = new StdioServerTransport();
    }
    await this.setup();
    await this.server!.connect(transport);
    logger.info(
      `[@genkit-ai/mcp] MCP server '${this.options.name}' started successfully.`
    );
  }
}

function toMcpPromptArguments(
  p: PromptAction
): Prompt['arguments'] | undefined {
  const jsonSchema = toJsonSchema({
    schema: p.__action.inputSchema,
    jsonSchema: p.__action.inputJsonSchema,
  });

  if (!jsonSchema) return undefined;
  if (!jsonSchema.properties)
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        '[@genkit-ai/mcp] MCP prompts must take objects as input schema.',
    });

  const args: Prompt['arguments'] = [];
  for (const k in jsonSchema.properties) {
    const { type, description } = jsonSchema.properties[k];
    if (
      type !== 'string' &&
      (!Array.isArray(type) || !type.includes('string'))
    ) {
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message: `[@genkit-ai/mcp] MCP prompts may only take string arguments, but ${p.__action.name} has property '${k}' of type '${type}'.`,
      });
    }
    args.push({
      name: k,
      description,
      required: jsonSchema.required?.includes(k),
    });
  }
  return args;
}

const ROLE_MAP = { model: 'assistant', user: 'user' } as const;

function toMcpPromptMessage(messageData: MessageData): PromptMessage {
  if (messageData.role !== 'model' && messageData.role !== 'user') {
    throw new GenkitError({
      status: 'UNIMPLEMENTED',
      message: `[@genkit-ai/mcp] MCP prompt messages do not support role '${messageData.role}'. Only 'user' and 'model' messages are supported.`,
    });
  }
  const message = new Message(messageData);
  const common = { role: ROLE_MAP[messageData.role] };
  if (message.media) {
    const { url, contentType } = message.media;
    if (!url.startsWith('data:'))
      throw new GenkitError({
        status: 'UNIMPLEMENTED',
        message: `[@genkit-ai/mcp] MCP prompt messages only support base64 data images.`,
      });
    const mimeType =
      contentType || url.substring(url.indexOf(':')! + 1, url.indexOf(';'));
    const data = url.substring(url.indexOf(',') + 1);
    return { ...common, content: { type: 'image', mimeType, data } };
  } else {
    return { ...common, content: { type: 'text', text: message.text } };
  }
}
