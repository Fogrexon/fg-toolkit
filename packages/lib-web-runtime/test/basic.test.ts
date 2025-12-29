
import { describe, it, expect } from 'vitest';
import { TemplateEngine, Message } from '../src/prompts/TemplateEngine';
import { ModelManager } from '../src/core/ModelManager';

describe('TemplateEngine', () => {
    it('should format a simple user-model turn correctly', () => {
        const engine = new TemplateEngine();
        const messages: Message[] = [
            { role: 'user', content: 'Hello' },
            { role: 'model', content: 'Hi there' }
        ];

        // Test without tools first
        const prompt = engine.applyTemplate(messages);

        // Expected generic Gemma format (simplified in our implementation)
        // <start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>model\n

        expect(prompt).toContain('<start_of_turn>user\nHello<end_of_turn>');
        expect(prompt).toContain('<start_of_turn>model\nHi there<end_of_turn>');
        expect(prompt.endsWith('<start_of_turn>model\n')).toBe(true);
    });
});

describe('ModelManager', () => {
    it('should instantiate without error', () => {
        const manager = new ModelManager();
        expect(manager).toBeDefined();
    });
});
