# Google Gemini Developer API plugin for Genkit

## Installing the plugin

```bash
npm i --save @genkit-ai/googleai
```

## Using the plugin

### Basic Setup

```ts
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/googleai';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.0-flash'),
});
```

### Text Generation

```ts
// Simple text generation
const { text } = await ai.generate('Explain quantum computing');
console.log(text);

// With a specific model
const response = await ai.generate({
  model: googleAI.model('gemini-1.5-pro'),
  prompt: 'Write a haiku about coding',
});
```

### Multimodal Generation

```ts
// With images
const response = await ai.generate({
  model: googleAI.model('gemini-2.0-flash'),
  prompt: [
    { text: 'What is in this image?' },
    { media: { url: 'data:image/jpeg;base64,...' } }
  ],
});
```

### Image Generation

```ts
// Using Imagen
const imageResponse = await ai.generate({
  model: googleAI.model('imagen-3.0-generate-002'),
  prompt: 'A serene mountain landscape at sunset',
});

// Using Imagen 4 (Preview)
const imagen4Response = await ai.generate({
  model: googleAI.model('imagen-4.0-generate-preview-06-06'),
  prompt: 'A futuristic city with flying cars',
});
```

### Video Generation

```ts
// Using Veo 2
const videoResponse = await ai.generate({
  model: googleAI.model('veo-2.0-generate-001'),
  prompt: 'A time-lapse of clouds moving over a city skyline',
  config: {
    aspectRatio: '16:9',
    durationSeconds: 8,
  }
});

// Using Veo 3 (if available)
const veo3Response = await ai.generate({
  model: googleAI.model('veo-3.0-generate-003'),
  prompt: 'Ocean waves crashing on a beach at sunset',
  config: {
    aspectRatio: '9:16',
    durationSeconds: 5,
  }
});
```

### Text-to-Speech

```ts
// Using Gemini TTS
const audioResponse = await ai.generate({
  model: googleAI.model('gemini-2.5-flash-preview-tts'),
  prompt: 'Hello, welcome to our presentation.',
});
```

### Native Audio (Conversational)

```ts
// Using native audio models
const audioDialogResponse = await ai.generate({
  model: googleAI.model('gemini-2.5-flash-preview-native-audio-dialog'),
  prompt: 'Tell me a story about a brave knight',
});
```

### Embeddings

```ts
// Text embeddings
const embedding = await ai.embed({
  embedder: googleAI.embedder('gemini-embedding-exp'),
  content: 'The quick brown fox jumps over the lazy dog',
});
```

### Using Fine-tuned Models

```ts
// Use your fine-tuned model
const response = await ai.generate({
  model: googleAI.model('tunedModels/your-model-id'),
  prompt: 'Your prompt here',
});
```

## Supported Models

For a comprehensive list of all supported models with their capabilities and specifications, see [SUPPORTED_MODELS.md](./SUPPORTED_MODELS.md).

The plugin uses dynamic model discovery, so new models released through the Gemini API are often supported automatically without requiring plugin updates.

## Documentation

The sources for this package are in the main [Genkit](https://github.com/firebase/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/docs/plugins/google-genai/).

License: Apache 2.0
