import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
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
                basicChat: resolve(__dirname, 'basic-chat/index.html')
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
