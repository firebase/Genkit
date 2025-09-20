# Supported Models - Google AI Plugin

The `@genkit-ai/googleai` plugin connects to the Gemini API and is designed to be highly flexible. Because of this, Genkit supports nearly any generative or embedding model available through the API, including new and fine-tuned models, often without needing a plugin update.

The following table lists many of the available models to help you get started. However, as the Gemini API evolves rapidly, this list may not be exhaustive. **For the most current and complete list of models, always refer to the official [Google AI Models documentation](https://ai.google.dev/gemini-api/docs/models).**

## Text, Multimodal, and Live Models

| Model Name | Code Reference | Capabilities | Notes |
| :--- | :--- | :--- | :--- |
| **Gemini 2.5 Pro** | `googleAI.model('gemini-2.5-pro')` | Text, Vision, Audio, PDF | Enhanced thinking and reasoning. |
| **Gemini 2.5 Flash** | `googleAI.model('gemini-2.5-flash')` | Text, Vision, Audio | Fast and versatile. |
| **Gemini 2.5 Flash-Lite** | `googleAI.model('gemini-2.5-flash-lite-preview-06-17')` | Text, Vision, Audio | Cost-efficient, high throughput. | `Preview` |
| **Gemini 2.5 Flash Live** | `googleAI.model('gemini-live-2.5-flash-preview')` | Bidirectional Voice & Video | For low-latency interactive sessions. | `Preview` |
| **Gemini 2.0 Flash** | `googleAI.model('gemini-2.0-flash')` | Text, Vision, Audio | Next-gen features and speed. |
| **Gemini 2.0 Flash Image Gen** | `googleAI.model('gemini-2.0-flash-preview-image-generation')` | Text, Vision, Image Generation | Conversational image generation. | `Preview` |
| **Gemini 2.0 Flash-Lite** | `googleAI.model('gemini-2.0-flash-lite')` | Text, Vision, Audio | Cost-efficient and low latency. |
| **Gemini 2.0 Flash Live** | `googleAI.model('gemini-2.0-flash-live-001')` | Bidirectional Voice & Video | For low-latency interactive sessions. |
| **Gemini 1.5 Pro** | `googleAI.model('gemini-1.5-pro')` | Text, Vision, Audio | Complex reasoning tasks. |
| **Gemini 1.5 Flash** | `googleAI.model('gemini-1.5-flash')` | Text, Vision, Audio | Fast performance for diverse tasks. |
| **Gemini 1.5 Flash-8B** | `googleAI.model('gemini-1.5-flash-8b')` | Text, Vision, Audio | High-volume, lower intelligence tasks. |

## Specialized Models (Audio, Image, Video, Embeddings)

| Model Type | Model Name | Code Reference | Capabilities | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Native Audio** | Gemini 2.5 Flash Native Audio (Dialog) | `googleAI.model('gemini-2.5-flash-preview-native-audio-dialog')` | Interleaved Text & Audio | Natural conversational audio. | `Preview` |
| **Native Audio** | Gemini 2.5 Flash Native Audio (Thinking) | `googleAI.model('gemini-2.5-flash-exp-native-audio-thinking-dialog')` | Interleaved Text & Audio | Includes thinking audio cues. | `Experimental` |
| **Text-to-Speech** | Gemini 2.5 Flash TTS | `googleAI.model('gemini-2.5-flash-preview-tts')` | Text-to-Speech | Low-latency audio generation. | `Preview` |
| **Text-to-Speech** | Gemini 2.5 Pro TTS | `googleAI.model('gemini-2.5-pro-preview-tts')` | Text-to-Speech | High-quality audio generation. | `Preview` |
| **Image Gen** | Imagen 4 | `googleAI.model('imagen-4.0-generate-preview-06-06')` | Image Generation | Latest image generation. | `Preview` |
| **Image Gen** | Imagen 4 Ultra | `googleAI.model('imagen-4.0-ultra-generate-preview-06-06')` | Image Generation | Highest quality image generation. | `Preview` |
| **Image Gen** | Imagen 3 | `googleAI.model('imagen-3.0-generate-002')` | Image Generation | High-quality images. |
| **Video Gen** | Veo 2 | `googleAI.model('veo-2.0-generate-001')` | Video Generation | High-quality video generation. |
| **Video Gen** | Veo 3 | `googleAI.model('veo-3.0-generate-003')` | Video Generation | Latest video generation model. | `If available` |
| **Embedding** | Gemini Embedding | `googleAI.embedder('gemini-embedding-exp')` | Text Embedding | Measures relatedness of text. | `Experimental` |

## Notes

- Models marked as `Preview` or `Experimental` may have limited availability or be subject to changes.
- Video generation models (Veo) return long-running operations that need to be polled for completion.
- Some models have specific configuration options. Check the model's documentation for details.
- The plugin uses dynamic model discovery, so new models are often supported automatically without plugin updates.
- You can also use fine-tuned models by passing their ID: `googleAI.model('tunedModels/your-model-id')`

For usage examples and code samples, see the [README](./README.md).

For the most up-to-date information about model capabilities, limitations, and pricing, please refer to the [official Google AI documentation](https://ai.google.dev/gemini-api/docs/models).