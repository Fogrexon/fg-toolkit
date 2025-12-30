
import { describe, it, expect } from 'vitest';
import { ModelManager } from '../src/core/ModelManager';

describe('ModelManager', () => {
    it('should instantiate without error', () => {
        const manager = new ModelManager();
        expect(manager).toBeDefined();
    });
});
