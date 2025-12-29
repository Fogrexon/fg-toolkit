/**
 * Configuration options for model loading
 */
interface ModelLoadOptions {
    modelId?: string;
    quantized?: boolean;
    progressCallback?: (progress: number) => void;
}

/**
 * Represents a chat message.
 */
interface Message {
    role: 'user' | 'model' | 'function';
    content: string;
}
/**
 * Definition of a function tool.
 * (Simplified schema for demonstration, can be expanded to full JSON schema)
 */
interface FunctionTool {
    name: string;
    description: string;
    parameters: Record<string, any>;
}

/**
 * Configuration for the FunctionGemmaWeb instance.
 */
interface Config extends ModelLoadOptions {
}
/**
 * Main class for the FunctionGemma Web Runtime.
 */
declare class FunctionGemmaWeb {
    private modelManager;
    private templateEngine;
    private config;
    constructor(config?: Config);
    /**
     * Initializes the library and loads the model.
     */
    init(): Promise<void>;
    /**
     * Generates a response based on chat messages.
     * @param messages Conversation history
     * @param tools Available tools
     * @returns Generated output string
     */
    chat(messages: Message[], tools?: FunctionTool[]): Promise<string>;
}

export { type Config, FunctionGemmaWeb, type FunctionTool, type Message };
