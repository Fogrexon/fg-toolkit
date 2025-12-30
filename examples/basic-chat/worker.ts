import { FunctionGemmaWeb, Message, FunctionTool } from '@fg-toolkit/lib-web-runtime';

// Define message types for type safety
type WorkerMessage =
    | { type: 'init', config: any }
    | { type: 'chat', messages: Message[] };

export type WorkerResponse =
    | { type: 'status', status: string, progress?: number }
    | { type: 'token', token: string }
    | { type: 'result', content: string }
    | { type: 'error', error: string };

let runtime: FunctionGemmaWeb | null = null;

// Demo tools
const tools: FunctionTool[] = [
    {
        name: 'get_current_time',
        description: 'Get the current time in a specific time zone',
        parameters: {
            type: 'object',
            properties: {
                timezone: {
                    type: 'string',
                    description: 'The time zone to get the time for (e.g., "UTC", "Asia/Tokyo", "America/New_York")',
                },
            },
            required: ['timezone'],
        },
        implementation: ({ timezone }: { timezone: string }) => {
            try {
                return new Date().toLocaleString('en-US', { timeZone: timezone });
            } catch (e) {
                return `Invalid timezone: ${timezone}`;
            }
        }
    }
];

self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const { type } = e.data;

    try {
        if (type === 'init') {
            const { config } = e.data as any;
            runtime = new FunctionGemmaWeb({
                ...config,
                progressCallback: (progress: number) => {
                    // Check if progress is a number (it might be an object from transformers.js in some versions/phases)
                    // But we typed it as number. If it's an object with status, handle it.
                    // Actually ModelManager casts to any, so we might receive objects.
                    const p = progress as any;
                    if (typeof p === 'number') {
                        self.postMessage({ type: 'status', status: 'progress', progress: p });
                    } else if (p.status) {
                        self.postMessage({ type: 'status', status: p.status, progress: p.progress });
                    }
                }
            });
            await runtime.init();
            self.postMessage({ type: 'status', status: 'ready' });
        }
        else if (type === 'chat') {
            if (!runtime) throw new Error('Runtime not initialized');
            const { messages } = e.data as any;

            const response = await runtime.chat(messages, tools, (token) => {
                self.postMessage({ type: 'token', token });
            });

            self.postMessage({ type: 'result', content: response });
        }
    } catch (err: any) {
        self.postMessage({ type: 'error', error: err.toString() });
    }
};
