
import { describe, it, expect, vi } from 'vitest';
import { FunctionGemmaWeb } from '../src/index';
// Mock ModelManager to control generation output
import { ModelManager } from '../src/core/ModelManager';
import { Message } from '../src/types';

// We mock the entire module
vi.mock('../src/core/ModelManager');

describe('FunctionGemmaWeb Function Calling', () => {
    it('should parse function call, execute implementation, and return final response', async () => {
        const lib = new FunctionGemmaWeb();
        const mockGenerate = vi.fn();

        // Setup mock behavior for ModelManager
        // @ts-ignore - accessing private or mocked instance
        lib.modelManager.generate = mockGenerate;

        // 1. First call returns a function call request
        mockGenerate.mockResolvedValueOnce('<start_function_call>get_weather({"location": "Tokyo"})<end_function_call>');

        // 2. Second call (after tool execution) returns the final answer
        mockGenerate.mockResolvedValueOnce('The weather in Tokyo is sunny.');

        const messages: Message[] = [{ role: 'user', content: 'Weather in Tokyo?' }];
        const tools = [{
            name: 'get_weather',
            description: 'Get weather',
            parameters: { type: 'object', properties: { location: { type: 'string' } } },
            implementation: (args: any) => {
                return `Sunny in ${args.location}`;
            }
        }];

        const response = await lib.chat(messages, tools);

        // Verify the interactions
        expect(mockGenerate).toHaveBeenCalledTimes(2);

        // Verify tool execution logic
        expect(response).toBe('The weather in Tokyo is sunny.');

        // Verify messages history was updated during the loop (note: we passed by reference but typescript might behave differently if we replaced array, but here we push to it)
        // The implementation pushes to the 'messages' array.
        // Let's check if the messages array reflects the conversation turns
        expect(messages.length).toBe(3); // User + Model(Call) + Function(Result)
        // Note: The final response is returned but not pushed to messages array inside chat() loop logic unless we do it explicitly? 
        // In my implementation:
        // 1. messages.push(modelCall)
        // 2. messages.push(functionResult)
        // 3. generate called -> returns final response.
        // The final response is NOT pushed to messages inside the function, it's just returned.
        // So expected length is 3.

        expect(messages[1].role).toBe('model');
        expect(messages[1].content).toContain('get_weather');

        expect(messages[2].role).toBe('function');
        expect(messages[2].content).toContain('Sunny in Tokyo');
    });

    it('should handle multiple turns or stops if no function call', async () => {
        const lib = new FunctionGemmaWeb();
        const mockGenerate = vi.fn();
        // @ts-ignore
        lib.modelManager.generate = mockGenerate;

        mockGenerate.mockResolvedValueOnce('Just a normal response.');

        const messages: Message[] = [{ role: 'user', content: 'Hi' }];
        const response = await lib.chat(messages, []);

        expect(response).toBe('Just a normal response.');
        expect(mockGenerate).toHaveBeenCalledTimes(1);
    });
});
