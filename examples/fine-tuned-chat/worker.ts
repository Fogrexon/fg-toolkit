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

// Tool definitions (Matching the trainer's search_knowledge_base and search_google)
const tools: FunctionTool[] = [
  {
    name: 'search_knowledge_base',
    description: 'Search internal company documents, policies and project data.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The search query string.',
        },
      },
      required: ['query'],
    },
    implementation: async ({ query }: { query: string }) => {
      return `[Internal KB Result for: "${query}"] This is a mock response from the company knowledge base. Access to ADP portal is found on page 42 of the IT handbook.`;
    }
  },
  {
    name: 'search_google',
    description: 'Search public information.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The search query string.',
        },
      },
      required: ['query'],
    },
    implementation: async ({ query }: { query: string }) => {
      return `[Google Search Result for: "${query}"] According to Python documentation, list comprehensions provide a concise way to create lists. Simple syntax: [f(x) for x in sequence].`;
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
        progressCallback: (progress: any) => {
          if (typeof progress === 'number') {
            self.postMessage({ type: 'status', status: 'progress', progress: progress });
          } else if (progress.status) {
            self.postMessage({ type: 'status', status: progress.status, progress: progress.progress });
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
    console.error('[Worker Error]', err);
    let errorMessage = 'An unknown error occurred';

    if (typeof err === 'string') {
      errorMessage = err;
    } else if (err instanceof Error) {
      errorMessage = `${err.name}: ${err.message}\n${err.stack}`;
    } else if (typeof err === 'number') {
      errorMessage = `Numerical error: ${err}. Possible WASM memory/allocation failure or model path issue.`;
    } else {
      try {
        errorMessage = JSON.stringify(err);
      } catch {
        errorMessage = String(err);
      }
    }

    self.postMessage({ type: 'error', error: errorMessage });
  }
};
