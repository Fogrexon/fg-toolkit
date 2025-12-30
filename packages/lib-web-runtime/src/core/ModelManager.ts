import { pipeline, env, TextStreamer, PreTrainedTokenizer } from '@huggingface/transformers';
import { Message, TransformersToolDefinition } from '../types';

// Skip local checks for browser environment compatibility
env.allowLocalModels = false;
env.useBrowserCache = true;

/**
 * Configuration options for model loading
 */
export interface ModelLoadOptions {
  modelId?: string;
  quantized?: boolean;
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

    if (options.modelId) {
      this.modelId = options.modelId;
    }

    // TODO: Ideally we should use a specific FunctionGemma ONNX model here when available
    // For now we use a placeholder or a text-generation pipeline.
    this.pipe = await pipeline('text-generation', this.modelId, {
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
