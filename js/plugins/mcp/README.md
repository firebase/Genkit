# Genkit MCP

> [!WARNING]  
> This plugin is experimental, meaning it may not be supported long-term and APIs are subject to more often breaking changes.

This plugin provides integration between Genkit and the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). MCP is an open standard allowing developers to build "servers" which provide tools, resources, and prompts to clients. Genkit MCP allows Genkit developers to both consume MCP tools, prompts, and resources as a client and provide tools and prompts as a server.

## Installation

To get started, you'll need Genkit and the MCP plugin:

```bash
npm i genkit genkitx-mcp
```

## MCP Client

To create an MCP client, you call the `mcpClient` function to generate a Genkit plugin for an MCP server. For example, to use MCP's example [filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem):

```ts
import { genkit } from 'genkit';
import { mcpClient } from 'genkitx-mcp';

// the filesystem server requires one or more allowed directories
const ALLOWED_DIRS = ['/Users/yourusername/Desktop'];

const filesystemClient = mcpClient({
  name: 'filesystem',
  serverProcess: {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-everything', ...ALLOWED_DIRS],
  },
});

const ai = genkit({
  plugins: [
    filesystemClient /* ... other plugins such as model providers ...*/,
  ],
});
```

Most MCP servers are built to run as spawned processes on the same machine using the `stdio` transport. When you supply the `serverProcess` option, you are specifying the command, arguments, and environment variables for spawning the server as a subprocess.

### mcpClient() Options

- **`name`**: (required) The name for this client, which namespaces its tools and prompts.
- **`version`**: (optional) The client's version number. Defaults to "1.0.0".
- You must supply one of:
  - **`serverProcess`**: Parameters for launching a local server process using the stdio MCP transport.
    - **`command`**: Shell command path for launching the MCP server. Can be e.g. `npx` or `uvx` to download and run the server from a package manager.
    - **`args`**: (optional) Array of string arguments to pass to the command.
    - **`env`**: (optional) Key value object of environment variables to pass to the command.
  - **`serverUrl`**: The URL of a remote server to connect to using the SSE MCP transport.
  - **`serverWebsocketUrl`: The URL of a remote server to connect to using the WebSocket MCP transport.
  - **`transport`**: An existing MCP transport object for connecting to the server.
- **`rawToolResponses`**: (optional) A boolean flag. If `true`, tool responses are returned in their raw MCP format; otherwise, they are processed for Genkit compatibility.
- **`sendGenkitContext`**: (optional) A boolean flag. If `true`, tool calls include context data from the current execution.

### Using MCP Actions

The Genkit MCP client automatically discovers available tools and prompts and registers them with Genkit making them available anywhere other tools and prompts can be used. To access resources, special `list_resources` and `read_resource` tools are registered that will access resources for the server.

All MCP actions are namespaced under the name you supply, so a client called `filesystem` will register tools such as `filesystem/read_file`.

### Tool Responses

MCP tools return a `content` array as opposed to a structured response like most Genkit tools. The Genkit MCP plugin attempts to parse and coerce returned content:

1. If content is text and valid JSON, the JSON is parsed and returned.
2. If content is text and not valid JSON, the text is returned.
3. If content has a single non-text part, it is returned.
4. If content has multiple/mixed parts, the full content response is returned.

## MCP Server

You can also expose all of the tools and prompts from a Genkit instance as an MCP server:

```ts
import { genkit, z } from 'genkit';
import { mcpServer } from 'genkitx-mcp';

const ai = genkit({});

ai.defineTool(
  {
    name: 'add',
    description: 'add two numbers together',
    inputSchema: z.object({ a: z.number(), b: z.number() }),
    outputSchema: z.number(),
  },
  async ({ a, b }) => {
    return a + b;
  }
);

ai.definePrompt(
  {
    name: "happy",
    description: "everybody together now",
    input: {
      schema: z.object({
        action: z.string().default("clap your hands").optional(),
      }),
    },
  },
  `If you're happy and you know it, {{action}}.`
);

mcpServer(ai, { name: 'example_server', version: '0.0.1' }).start();
```

The above will start up an MCP server with the stdio transport that exposes a tool called `add` and a prompt called `happy`. To start the server with a different transport, use `mcpServer(...).start(otherTransport)`.

### Known Limitations

- MCP prompts are only able to take string parameters, so inputs to schemas must be objects with only string property values.
- MCP prompts only support `user` and `model` messages. `system` messages are not supported.
- MCP prompts only support a single "type" within a message so you can't mix media and text in the same message.

### Testing your MCP server

You can test your MCP server using the official inspector. For example, if your server code compiled into `dist/index.js`, you could run:

    npx @modelcontextprotocol/inspector dist/index.js

Once you start the inspector, you can list prompts and actions and test them out manually.
