// src/core/ModelManager.ts
import { pipeline, env } from "@xenova/transformers";
env.allowLocalModels = false;
env.useBrowserCache = true;
var ModelManager = class {
  constructor() {
    this.pipe = null;
    this.modelId = "google/gemma-1.1-2b-it";
  }
  // Default model (subject to change to specific FunctionGemma ONNX)
  /**
   * Loads the model.
   * @param options Load options
   */
  async load(options = {}) {
    if (this.pipe) {
      return;
    }
    if (options.modelId) {
      this.modelId = options.modelId;
    }
    this.pipe = await pipeline("text-generation", this.modelId, {
      progress_callback: options.progressCallback,
      quantized: options.quantized ?? true
    });
  }
  /**
   * Generates text based on the provided prompt.
   * @param prompt The input prompt
   * @param maxNewTokens Maximum new tokens to generate
   * @returns The generated text
   */
  async generate(prompt, maxNewTokens = 128) {
    if (!this.pipe) {
      throw new Error("Model not loaded. Call load() first.");
    }
    const output = await this.pipe(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      // Greedy decoding for function calling usually works better, or low temp
      return_full_text: false
    });
    return output[0].generated_text;
  }
};

// src/prompts/TemplateEngine.ts
var TemplateEngine = class {
  /**
   * Applies the template to a list of messages.
   * @param messages List of chat conversation messages.
   * @param tools Optional list of available tools.
   * @returns The formatted prompt string.
   */
  applyTemplate(messages, tools) {
    let prompt = "";
    if (tools && tools.length > 0) {
    }
    for (const msg of messages) {
      prompt += `<start_of_turn>${msg.role}
${msg.content}<end_of_turn>
`;
    }
    prompt += "<start_of_turn>model\n";
    return prompt;
  }
};

// src/index.ts
var FunctionGemmaWeb = class {
  constructor(config = {}) {
    this.config = config;
    this.modelManager = new ModelManager();
    this.templateEngine = new TemplateEngine();
  }
  /**
   * Initializes the library and loads the model.
   */
  async init() {
    await this.modelManager.load(this.config);
  }
  /**
   * Generates a response based on chat messages.
   * @param messages Conversation history
   * @param tools Available tools
   * @returns Generated output string
   */
  async chat(messages, tools) {
    const prompt = this.templateEngine.applyTemplate(messages, tools);
    return await this.modelManager.generate(prompt);
  }
};
export {
  FunctionGemmaWeb
};
