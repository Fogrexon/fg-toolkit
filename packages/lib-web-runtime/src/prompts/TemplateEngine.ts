
/**
 * Represents a chat message.
 */
export interface Message {
    role: 'user' | 'model' | 'function';
    content: string;
}

/**
 * Definition of a function tool.
 * (Simplified schema for demonstration, can be expanded to full JSON schema)
 */
export interface FunctionTool {
    name: string;
    description: string;
    parameters: Record<string, any>;
}

/**
 * Engine for constructing FunctionGemma compatible prompts.
 */
export class TemplateEngine {

    /**
     * Applies the template to a list of messages.
     * @param messages List of chat conversation messages.
     * @param tools Optional list of available tools.
     * @returns The formatted prompt string.
     */
    applyTemplate(messages: Message[], tools?: FunctionTool[]): string {
        let prompt = '';

        // If tools are provided, inject the available tools at the beginning or system prompt context
        // according to FunctionGemma specifications.
        // Note: Actual FunctionGemma prompting structure might need specific adjustment based on fine-tuning.
        // This is a standard Gemma chat template implementation augmented for functions.

        if (tools && tools.length > 0) {
            // Placeholder for tool definition injection. 
            // In real FunctionGemma, this might be a system message or specific turn structure.
            // For now, let's assume it's part of the user context or system prompt if we had one.
            // We will skip explicit complex schema formatting for this initial version.
        }

        for (const msg of messages) {
            prompt += `<start_of_turn>${msg.role}\n${msg.content}<end_of_turn>\n`;
        }

        prompt += '<start_of_turn>model\n'; // Prep for completion
        return prompt;
    }
}
