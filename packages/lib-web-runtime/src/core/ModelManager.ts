import { pipeline, env, TextStreamer, PreTrainedTokenizer } from '@huggingface/transformers';
import { Message, TransformersToolDefinition } from '../types';
import { FUNCTION_GEMMA_CHAT_TEMPLATE } from './templates';

// In transformers.js v3 (browser), models are fetched via fetch(). 
// env.allowLocalModels refers to the local file system (Node.js/Electron).
// env.allowRemoteModels refers to fetching from HF Hub or a URL.
// When using a local server (like Vite's public folder), transformers.js 
// treats the URL as "remote" but within the browser environment.
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

// Tune ONNX Runtime environment for performance and stability
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1; // Single thread often more stable for very large models in workers
  env.backends.onnx.wasm.proxy = false;     // We are likely already in a worker
}

// Specifically for local model serving, we can set the localModelPath 
// if we use a specific directory, but usually absolute URLs are passed to pipeline().

/**
 * Configuration options for model loading
 */
export interface ModelLoadOptions {
  /** Custom model path or URL. Must be a complete URL (e.g., 'http://localhost:5173/model' or 'https://example.com/models/my-model'). If not specified, uses the default FunctionGemma model from HuggingFace Hub. */
  modelPath?: string;
  quantized?: boolean;
  device?: 'wasm' | 'webgpu' | 'auto';
  progressCallback?: (progress: any) => void;
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
  private modelId: string = 'onnx-community/functiongemma-270m-it-ONNX'; // Default model

  /**
   * Loads the model.
   * @param options Load options
   */
  async load(options: ModelLoadOptions = {}): Promise<void> {
    if (this.pipe) {
      return;
    }

    const resolvedModelPath = options.modelPath || this.modelId;

    console.log(`[ModelManager] Loading model from: ${resolvedModelPath}`);
    console.log(`[ModelManager] env.allowLocalModels: ${env.allowLocalModels}, env.allowRemoteModels: ${env.allowRemoteModels}`);
    console.log(`[ModelManager] Requested dtype: ${options.quantized ? 'q4' : 'fp32'}`);

    try {
      this.pipe = await pipeline('text-generation', this.modelId, {
        progress_callback: options.progressCallback,
        dtype: options.quantized ? 'q4' : 'fp32',
        device: options.device || 'wasm',
      }) as unknown as TextGenerationPipeline;
      console.log(`[ModelManager] Pipeline loaded successfully.`);

      // Log model inputs/outputs if available (ONNX specific)
      const model = (this.pipe as any).model;
      if (model && model.session) {
        console.log(`[ModelManager] ONNX Input Names:`, model.session.inputNames);
        console.log(`[ModelManager] ONNX Output Names:`, model.session.outputNames);
      } else if (model && model.model && model.model.session) {
        // Some versions wrap it
        console.log(`[ModelManager] ONNX Input Names:`, model.model.session.inputNames);
        console.log(`[ModelManager] ONNX Output Names:`, model.model.session.outputNames);
      }
    } catch (error: any) {
      console.error(`[ModelManager] Failed to load pipeline:`, error);

      let extraMessage = "";
      if (typeof error === 'number' || (error && error.message && error.message.includes('out of memory'))) {
        extraMessage = "\n\nCRITICAL: This often indicates a WASM memory limit or large file allocation failure. " +
          "For models > 1GB, please use a quantized version (e.g., Q4) or try WebGPU if available.";
      }

      if (typeof error === 'number') {
        throw new Error(`Model loading failed with numeric code: ${error}.${extraMessage}`);
      }
      throw new Error(`${error.message || error}${extraMessage}`);
    }
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

    // Use the tokenizer's chat template, or fallback if not set
    const chatTemplate = this.pipe.tokenizer.chat_template || FUNCTION_GEMMA_CHAT_TEMPLATE;

    // Generate the prompt string
    const prompt = this.pipe.tokenizer.apply_chat_template(messages, {
      tools: tools,
      tokenize: false,
      add_generation_prompt: true,
      chat_template: chatTemplate
    }) as string;

    console.log(`[ModelManager] Generated prompt length: ${prompt?.length}`);

    const streamer = onUpdate ? new TextStreamer(this.pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: onUpdate,
    }) : undefined;

    // Pass the prompt string directly to the pipeline. 
    // Transformers.js will handle tokenization and input tensor creation (including attention_mask/position_ids).
    const output = await this.pipe(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_full_text: false,
      streamer,
      use_cache: true,
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
