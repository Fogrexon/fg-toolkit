import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    base: process.env.NODE_ENV === 'production' ? '/fg-toolkit/' : '/',
    resolve: {
        alias: {
            // Point to built ESM
            '@fg-toolkit/lib-web-runtime': resolve(__dirname, '../packages/lib-web-runtime/dist/index.mjs')
        }
    },
    build: {
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                basicChat: resolve(__dirname, 'basic-chat/index.html'),
                fineTunedChat: resolve(__dirname, 'fine-tuned-chat/index.html'),
                inventoryDemo: resolve(__dirname, 'inventory-demo/index.html')
            }
        }
    },
    server: {
        fs: {
            strict: false
        }
    },
    optimizeDeps: {
        exclude: ['@fg-toolkit/lib-web-runtime']
    }
});
