
import { pipeline, env } from '@xenova/transformers';

// Skip local checks for browser environment compatibility
env.allowLocalModels = false;
env.useBrowserCache = true;

/**
 * Configuration options for model loading
 */
export interface ModelLoadOptions {
  modelId?: string;
  quantized?: boolean;
  progressCallback?: (progress: number) => void;
}

/**
 * Manages the loading and inference of the FunctionGemma model.
 */
export class ModelManager {
  private pipe: any = null;
  private modelId: string = 'google/gemma-1.1-2b-it'; // Default model (subject to change to specific FunctionGemma ONNX)

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
      quantized: options.quantized ?? true,
    });
  }

  /**
   * Generates text based on the provided prompt.
   * @param prompt The input prompt
   * @param maxNewTokens Maximum new tokens to generate
   * @returns The generated text
   */
  async generate(prompt: string, maxNewTokens: number = 128): Promise<string> {
    if (!this.pipe) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const output = await this.pipe(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false, // Greedy decoding for function calling usually works better, or low temp
      return_full_text: false,
    });

    return output[0].generated_text;
  }
}
