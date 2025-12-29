# FunctionGemma Web Runtime

`@fg-toolkit/lib-web-runtime` is a client-side library for integrating Google's FunctionGemma model into web applications using ONNX Runtime (via `transformers.js`).

## Features

- **In-Browser Inference**: Run FunctionGemma directly in the browser without server-side dependencies.
- **Prompt Templating**: Automatically format chat messages into the specific prompt structure required by FunctionGemma.
- **Function Calling Support**: (Coming Soon) High-level API for defining and executing tools.

## Installation

```bash
npm install @fg-toolkit/lib-web-runtime
```

## Usage

### Basic Chat

```typescript
import { FunctionGemmaWeb } from '@fg-toolkit/lib-web-runtime';

async function main() {
  // Initialize the runtime
  const runtime = new FunctionGemmaWeb({
    modelId: 'google/gemma-1.1-2b-it', // Replace with your ONNX converted model path / ID
    quantized: true,
  });

  await runtime.init();

  // Chat conversation
  const messages = [
    { role: 'user', content: 'Hello! Who are you?' }
  ];

  // Generate response
  const response = await runtime.chat(messages);
  console.log(response);
}

main();
```

### Prompt Templating Only

If you just need to format prompts but use your own inference backend:

```typescript
import { TemplateEngine } from '@fg-toolkit/lib-web-runtime/prompts/TemplateEngine'; // Note: check exports if deep import is allowed or import from main

const engine = new TemplateEngine();
const prompt = engine.applyTemplate([
  { role: 'user', content: 'Turn on the lights' }
]);
```

## Configuration

- `modelId`: The Hugging Face model ID or path to the ONNX model.
- `quantized`: Whether to use the quantized version (default: `true`).

## License

MIT
