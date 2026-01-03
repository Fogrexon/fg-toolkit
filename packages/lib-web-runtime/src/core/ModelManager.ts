import { pipeline, env, TextStreamer, PreTrainedTokenizer } from '@huggingface/transformers';
import { Message, TransformersToolDefinition } from '../types';

// Skip local checks for browser environment compatibility
env.allowLocalModels = false;
env.useBrowserCache = true;
env.allowRemoteModels = true; // Default

/**
 * Configuration options for model loading
 */
/**
 * Configuration options for model loading
 */
export interface ModelLoadOptions {
  modelPath?: string;
  quantized?: boolean;
  progressCallback?: (progress: any) => void;
  // Intentionally omitting modelId to restrict arbitrary HF model loading
}

// Define a minimal interface for the pipeline to avoid 'any' if types are not fully available
interface TextGenerationPipeline {
  tokenizer: PreTrainedTokenizer;
  (input: any, options?: any): Promise<any>;
}

/**
 * Manages the loading and inference of the FunctionGemma model.
 */
export class ModelManager {
  private pipe: TextGenerationPipeline | null = null;
  // Default model constant - immutable from outside
  private readonly DEFAULT_MODEL_ID = 'onnx-community/functiongemma-270m-it-ONNX';

  /**
   * Loads the model.
   * @param options Load options
   */
  async load(options: ModelLoadOptions = {}): Promise<void> {
    if (this.pipe) {
      return;
    }

    let modelToLoad = this.DEFAULT_MODEL_ID;

    // Strict model loading policy:
    // 1. If modelPath is provided, load from that path/URL.
    // 2. Otherwise/Default, load the specific FunctionGemma model.
    // Arbitrary HF model IDs are not supported.

    if (options.modelPath) {
      modelToLoad = options.modelPath;
      console.log(`Loading custom model from path: ${modelToLoad}`);

      // If it looks like a local path or URL, we might need to adjust env settings or just pass it to pipeline.
      // transformers.js handles URLs/Paths in place of model ID.
      // For local files served by web server, it is treated as a path/URL.

      // Disable remote models if loading locally to avoid confusion?
      // Actually, if we pass a path starting with ./ or /, it is treated as local/URL resource.
      if (modelToLoad.startsWith('/') || modelToLoad.startsWith('./') || modelToLoad.startsWith('http')) {
        env.allowRemoteModels = false;
      }
    } else {
      console.log(`Loading default model: ${modelToLoad}`);
      // Ensure remote models are allowed for the default usage
      env.allowRemoteModels = true;
    }

    // Pipeline call
    this.pipe = await pipeline('text-generation', modelToLoad, {
      progress_callback: options.progressCallback,
      dtype: options.quantized ? 'q4' : 'fp32',
    }) as unknown as TextGenerationPipeline;
  }

  /**
   * Generates text based on the provided messages.
   * @param messages Conversation history
   * @param tools Optional list of tools to include in the prompt
   * @param maxNewTokens Maximum new tokens to generate
   * @param onUpdate Optional callback for streaming updates
   * @returns The generated text
   */
  async generate(messages: Message[], tools?: TransformersToolDefinition[], maxNewTokens: number = 128, onUpdate?: (token: string) => void): Promise<string> {
    if (!this.pipe) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Use the tokenizer's chat template
    const prompt = this.pipe.tokenizer.apply_chat_template(messages, {
      tools: tools,
      tokenize: false,
      add_generation_prompt: true
    }) as string;

    const streamer = onUpdate ? new TextStreamer(this.pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: onUpdate,
    }) : undefined;

    const output = await this.pipe(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_full_text: false,
      streamer,
    });

    // When using chat input (array), the output is typically the last generated message object or text depending on version
    // transformers.js v3 usually returns [{ generated_text: "..." }] where text is the full generated content (or assistant response).
    // Let's inspect typical behavior: if input is array, output[0].generated_text is usually the latest response if return_full_text=false? 
    // Actually pipeline text-generation with chat inputs often returns the generated assistant message.

    const generated = output[0].generated_text;

    // If generated is an array (conversation history), take the last message
    if (Array.isArray(generated)) {
      const lastMessage = generated[generated.length - 1];
      return lastMessage.content;
    }

    // Check if it's a single object (Message) with content
    if (typeof generated === 'object' && generated && generated.content) {
      return generated.content;
    }

    // Fallback: assume it's a string
    return String(generated);
  }
}
