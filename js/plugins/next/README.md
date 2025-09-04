# Genkit Next.js Plugin

This plugin provides utilities for conveninetly exposing Genkit flows and actions via Next.js app routs for REST APIs.

```ts
// /genkit/simpleFlow.ts
const simpleFlow = ai.defineFlow(
  'simpleFlow',
  async (input, streamingCallback) => {
    const { text } = await ai.generate({
      model: gemini15Flash,
      prompt: input,
      streamingCallback: (chunk) => streamingCallback(chunk.text),
    });
    return text;
  }
);
```

```ts
// /app/api/simpleFlow/route.ts
import { simpleFlow } from '@/genkit/simpleFlow';
import { appRoute } from '@genkit-ai/next';

export const POST = appRoute(simpleFlow);
```

APIs can be called with the generic `genkit/beta/client` library, or `@genkit-ai/next/client`

```ts
import { runFlow, streamFlow } from '@genkit-ai/next/client';
import { simpleFlow } from '@/genkit/simpleFlow';

const result = await runFlow<typeof simpleFlow>({
  url: '/api/simpleFlow',
  input: 'say hello',
});

console.log(result); // hello

// set auth headers (when using auth policies)
const result = await runFlow<typeof simpleFlow>({
  url: `/api/simpleFlow`,
  headers: {
    Authorization: 'open sesame',
  },
  input: 'say hello',
});

console.log(result); // hello

// and streamed
const result = streamFlow<typeof simpleFlow>({
  url: '/api/simpleFlow',
  input: 'say hello',
});
for await (const chunk of result.stream()) {
  console.log(chunk.output);
}
console.log(await result.output());
```

The sources for this package are in the main [Genkit](https://github.com/firebase/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/docs/get-started).

License: Apache 2.0
