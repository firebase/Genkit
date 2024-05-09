# Ollama plugin

The Ollama plugin provides interfaces to any of the local LLMs supported by
[Ollama](https://ollama.com/).

## Installation

```posix-terminal
npm i --save genkitx-ollama
```

## Configuration

This plugin requires that you first install and run ollama server. You can follow
the instructions on: [https://ollama.com/download](https://ollama.com/download)

You can use the Ollama CLI to download the model you are interested in. For
example:

```posix-terminal
ollama pull gemma
```

To use this plugin, specify it when you call `configureGenkit()`.

```js
import { ollama } from 'genkitx-ollama';

export default configureGenkit({
  plugins: [
    ollama({
      models: [
        {
          name: 'gemma',
          type: 'generate', // type: 'chat' | 'generate' | undefined
        },
      ],
      serverAddress: 'http://127.0.0.1:11434', // default local address
    }),
  ],
});
```

### Authentication

If you would like to access remote deployments of ollama that require custom headers (static,
such as API keys, or dynamic, such as auth headers), you can specify those in the ollama config plugin:

Static headers:

```js
ollama({
  models: [{ name: 'gemma'}],
  requestHeaders: {
    'api-key': 'API Key goes here'
  },
  serverAddress: 'https://my-deployment',
}),
```

You can also dynamically set headers per request. Here's an example of how to set an ID token using
the Google Auth library:

```js
import { GoogleAuth } from 'google-auth-library';
import { ollama, OllamaPluginParams } from 'genkitx-ollama';
import { configureGenkit, isDevEnv } from '@genkit-ai/core';

const ollamaCommon = {models: [{name: "gemma:2b"}]};
const ollamaDev = {
  ...ollamaCommon,
  serverAddress: 'http://127.0.0.1:11434',
} as OllamaPluginParams;
const ollamaProd = {
  ...ollamaCommon,
  serverAddress: 'https://my-deployment',
  requestHeaders: async (params) => ({
    Authorization: `Bearer ${await getIdToken(params.serverAddress)}`,
  }),
} as OllamaPluginParams;

export default configureGenkit({
  plugins: [
    ollama(isDevEnv() ? ollamaDev: ollamaProd),
  ],
});

export async function getIdToken(url: string): Promise<string> {
  const auth = getAuthClient();
  const client = await auth.getIdTokenClient(url);
  return client.idTokenProvider.fetchIdToken(url);
}

let auth: GoogleAuth;
function getAuthClient() {
  // Lazy load GoogleAuth client.
  if (!auth) {
    auth = new GoogleAuth();
  }
  return auth;
}
```

## Usage

This plugin doesn't statically export model references. Specify one of the
models you configured using a string identifier:

```js
const llmResponse = await generate({
  model: 'ollama/gemma',
  prompt: 'Tell me a joke.',
});
```
