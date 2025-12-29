
import { pipeline, env } from '@huggingface/transformers';

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

/**
 * Manages the loading and inference of the FunctionGemma model.
 */
export class ModelManager {
  private pipe: any = null;
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
    } as any);
  }

  /**
   * Generates text based on the provided prompt.
   * @param prompt The input prompt
   * @param maxNewTokens Maximum new tokens to generate
   * @returns The generated text
   */
  async generate(input: string | any[], maxNewTokens: number = 128): Promise<string> {
    if (!this.pipe) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const output = await this.pipe(input, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_full_text: false,
    });

    // When using chat input (array), the output is typically the last generated message object or text depending on version
    // transformers.js v3 usually returns [{ generated_text: "..." }] where text is the full generated content (or assistant response).
    // Let's inspect typical behavior: if input is array, output[0].generated_text is usually the latest response if return_full_text=false? 
    // Actually pipeline text-generation with chat inputs often returns the generated assistant message.

    const generated = output[0].generated_text;
    // Check if it's an object (Message) or string
    if (typeof generated === 'object' && generated.content) {
      return generated.content;
    }
    return generated;
  }
}
