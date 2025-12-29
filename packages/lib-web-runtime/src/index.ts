
import { ModelManager, ModelLoadOptions } from './core/ModelManager';
import { TemplateEngine, Message, FunctionTool } from './prompts/TemplateEngine';

export { Message, FunctionTool };

/**
 * Configuration for the FunctionGemmaWeb instance.
 */
export interface Config extends ModelLoadOptions {
    // Add other config options here
}

/**
 * Main class for the FunctionGemma Web Runtime.
 */
export class FunctionGemmaWeb {
    private modelManager: ModelManager;
    private templateEngine: TemplateEngine;
    private config: Config;

    constructor(config: Config = {}) {
        this.config = config;
        this.modelManager = new ModelManager();
        this.templateEngine = new TemplateEngine();
    }

    /**
     * Initializes the library and loads the model.
     */
    async init(): Promise<void> {
        await this.modelManager.load(this.config);
    }

    /**
     * Generates a response based on chat messages.
     * @param messages Conversation history
     * @param tools Available tools
     * @returns Generated output string
     */
    async chat(messages: Message[], tools?: FunctionTool[]): Promise<string> {
        const prompt = this.templateEngine.applyTemplate(messages, tools);
        return await this.modelManager.generate(prompt);
    }
}
