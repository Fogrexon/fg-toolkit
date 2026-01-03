/**
 * Represents a chat message.
 */
export interface Message {
    role: 'user' | 'model' | 'function' | 'system';
    content: string;
}

/**
 * Definition of a function tool.
 */
export interface FunctionTool {
    name: string;
    description: string;
    parameters: Record<string, any>;
    /**
     * Optional implementation of the function.
     * If provided, the library can automatically execute this function when called by the model.
     * The implementation should return a JSON stringifiable object or a string.
     */
    implementation?: (args: any) => Promise<any> | any;
}

/**
 * Tool definition format expected by transformers.js
 */
export interface TransformersToolDefinition {
    type: 'function';
    function: {
        name: string;
        description: string;
        parameters: Record<string, any>;
    };
}
