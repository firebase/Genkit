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

import type { StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js' with { 'resolution-mode': 'import' };
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js' with { 'resolution-mode': 'import' };
import { GenkitError, type Genkit } from 'genkit';
import { genkitPlugin } from 'genkit/plugin';
import { registerAllPrompts } from './client/prompts.js';
import { registerResourceTools } from './client/resources.js';
import { registerAllTools } from './client/tools.js';
import { GenkitMcpServer } from './server.js';

export interface McpClientOptions {
  /** Provide a name for this client which will be its namespace for all tools and prompts. */
  name: string;
  /** Provide a version number for this client (defaults to 1.0.0). */
  version?: string;
  /** If you already have an MCP transport you'd like to use, pass it here to connect to the server. */
  transport?: Transport;
  /** Start a local server process using the stdio MCP transport. */
  serverProcess?: StdioServerParameters;
  /** Connect to a remote server process using the SSE MCP transport. */
  serverUrl?: string;
  /** Connect to a remote server process using the WebSocket MCP transport. */
  serverWebsocketUrl?: string | URL;
  /** Return tool responses in raw MCP form instead of processing them for Genkit compatibility. */
  rawToolResponses?: boolean;
  /** Send the Genkit context to the server. */
  sendGenkitContext?: boolean;
}

async function transportFrom(params: McpClientOptions): Promise<Transport> {
  if (params.transport) return params.transport;
  if (params.serverUrl) {
    const { SSEClientTransport } = await import(
      '@modelcontextprotocol/sdk/client/sse.js'
    );
    return new SSEClientTransport(new URL(params.serverUrl));
  }
  if (params.serverProcess) {
    const { StdioClientTransport } = await import(
      '@modelcontextprotocol/sdk/client/stdio.js'
    );
    return new StdioClientTransport(params.serverProcess);
  }
  if (params.serverWebsocketUrl) {
    const { WebSocketClientTransport } = await import(
      '@modelcontextprotocol/sdk/client/websocket.js'
    );
    let url = params.serverWebsocketUrl;
    if (typeof url === 'string') url = new URL(url);
    return new WebSocketClientTransport(url);
  }

  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message:
      'Unable to create a server connection with supplied options. Must provide transport, stdio, or sseUrl.',
  });
}

export function mcpClient(params: McpClientOptions) {
  return genkitPlugin(params.name, async (ai: Genkit) => {
    const { Client } = await import(
      '@modelcontextprotocol/sdk/client/index.js'
    );

    const transport = await transportFrom(params);
    ai.options.model;
    const client = new Client(
      { name: params.name, version: params.version || '1.0.0' },
      { capabilities: {} }
    );
    await client.connect(transport);
    const capabilties = await client.getServerCapabilities();
    const promises: Promise<any>[] = [];
    if (capabilties?.tools) promises.push(registerAllTools(ai, client, params));
    if (capabilties?.prompts)
      promises.push(registerAllPrompts(ai, client, params));
    if (capabilties?.resources)
      promises.push(registerResourceTools(ai, client, params));
    await Promise.all(promises);
  });
}

export interface McpServerOptions {
  /** The name you want to give your server for MCP inspection. */
  name: string;
  /** The version you want the server to advertise to clients. Defaults to 1.0.0. */
  version?: string;
}

export function mcpServer(ai: Genkit, options: McpServerOptions) {
  return new GenkitMcpServer(ai, options);
}
