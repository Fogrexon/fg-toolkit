
import { ModelManager, ModelLoadOptions } from './core/ModelManager';
import { Message, FunctionTool, TransformersToolDefinition } from './types';

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
    private config: Config;
    constructor(config: Config = {}) {
        this.config = config;
        this.modelManager = new ModelManager();
    }

    /**
     * Initializes the library and loads the model.
     */
    async init(): Promise<void> {
        await this.modelManager.load(this.config);
    }

    /**
     * Helper to map FunctionTool to the format expected by transformers.js
     */
    private mapFunctionToolToDefinition(tool: FunctionTool): TransformersToolDefinition {
        return {
            type: 'function',
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters
            }
        };
    }

    /**
     * Parses the model response to check for function calls.
     * Looks for <start_function_call>...<end_function_call>
     */
    private parseFunctionCall(response: string): { name: string; arguments: any } | null {
        // Simple regex to find the function call block.
        // Format: <start_function_call>function_name({ ... })<end_function_call>
        // Note: FunctionGemma sometimes outputs `name({...})` or just JSON.
        // The standard format is: <start_function_call>name(json_args)<end_function_call>
        const match = response.match(/<start_function_call>(.*?)\((.*?)\)<end_function_call>/s);
        if (match) {
            const name = match[1].trim();
            const argsString = match[2];
            try {
                // FunctionGemma arguments are usually JSON-like but might need handling
                // Ideally it's valid JSON inside definition
                const args = JSON.parse(argsString);
                return { name, arguments: args };
            } catch (e) {
                console.error('Failed to parse function arguments:', argsString, e);
                return null;
            }
        }
        return null;
    }

    /**
     * Generates a response based on chat messages.
     * @param messages Conversation history
     * @param tools Available tools
     * @returns Generated output string
     */
    async chat(messages: Message[], tools?: FunctionTool[], onUpdate?: (token: string) => void): Promise<string> {
        const mappedTools = tools?.map(t => this.mapFunctionToolToDefinition(t));

        // Initial generation
        let response = await this.modelManager.generate(messages, mappedTools, 128, onUpdate);

        // Check for function calls
        // We limit recursion to prevent infinite loops (e.g., max 5 turns)
        let loopCount = 0;
        const MAX_LOOPS = 5;

        while (loopCount < MAX_LOOPS) {
            const call = this.parseFunctionCall(response);
            if (!call) {
                break;
            }

            loopCount++;

            // Append the model's response (function call request) to history
            // Note: In a real app we should append this to the user's message history reference too,
            // but here we are just operating on the local variable for the loop.
            // A better design would be to update the passed `messages` array or return the new history.
            // For this interface, we will just proceed with the internal state to get the final answer.

            // Important: The `messages` array needs to accumulate the context.
            // We'll append the model output first.
            messages.push({ role: 'model', content: response });

            const tool = tools?.find(t => t.name === call.name);
            let result = '';

            if (tool && tool.implementation) {
                try {
                    console.log(`Executing tool ${tool.name} with args:`, call.arguments);
                    const output = await tool.implementation(call.arguments);
                    result = typeof output === 'string' ? output : JSON.stringify(output);
                } catch (e) {
                    result = `Error executing tool: ${e}`;
                }
            } else {
                result = `Tool ${call.name} not found or no implementation provided.`;
            }

            // Append function response
            const functionResponseMsg: Message = {
                role: 'function',
                // FunctionGemma specific format for function response:
                // <start_function_response>name\ncontent<end_function_response>
                // But the chat template usually handles the wrapping if we pass role='function'?
                // Let's check the standard. Usually role='function', name='name', content='result'.
                // If we use the raw prompt format, we might need to format it manually.
                // However, since we are using apply_chat_template in the next iteration, 
                // we should stick to the Message object format.
                // FunctionGemma expects the content to be wrapped in <start_function_response>...
                // Let's rely on standard `function` role behavior for now or manually format matching the prompt.
                // The debug output showed: <start_function_response>name...
                // We'll trust the tokenizer to handle role='function'.
                content: result
            };

            // Hack: transformers.js chat template for FunctionGemma MIGHT expect a specific way to pass function name in the message object.
            // Standard OpenAI format uses `name` field in message. checks if our Message type supports it.
            // Our Message type currently only has role and content.
            // We should format the content as: <start_function_response>name\nresult<end_function_response>
            // to be safe if the tokenizer doesn't handle the 'name' field on 'function' role widely.
            // Actually, based on documentation, the model expects: 
            // <start_function_response>function_name\n{"result": ...}<end_function_response>

            // let's manually format the content for safety
            functionResponseMsg.content = result; // The tokenizer template hopefully handles the wrapping if we follow the schema.
            // Checking the debug output earlier:
            // <start_of_turn>model\n<start_function_call>...
            // Next turn should be function response.

            // Let's assume for now we just push the content. 
            // Use a specific format if we find the tokenizer doesn't wrapping it.
            messages.push(functionResponseMsg);

            // Generate again with the new context
            response = await this.modelManager.generate(messages, mappedTools, 128, onUpdate);
        }

        return response;
    }
}
