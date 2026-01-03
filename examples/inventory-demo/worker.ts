import { FunctionGemmaWeb, Message, FunctionTool } from '@fg-toolkit/lib-web-runtime';

type WorkerMessage =
    | { type: 'init', config: any }
    | { type: 'chat', messages: Message[] };

export type WorkerResponse =
    | { type: 'status', status: string, progress?: number }
    | { type: 'token', token: string }
    | { type: 'result', content: string }
    | { type: 'error', error: string };

let runtime: FunctionGemmaWeb | null = null;

// Inventory state (in worker)
const inventory: Map<string, number> = new Map();

// Inventory management tools
const tools: FunctionTool[] = [
    {
        name: 'add_item',
        description: '在庫にアイテムを追加する',
        parameters: {
            type: 'object',
            properties: {
                name: {
                    type: 'string',
                    description: '追加するアイテムの名前',
                },
                quantity: {
                    type: 'number',
                    description: '追加する数量',
                },
            },
            required: ['name', 'quantity'],
        },
        implementation: ({ name, quantity }: { name: string; quantity: number }) => {
            const current = inventory.get(name) || 0;
            inventory.set(name, current + quantity);
            return `${name}を${quantity}個追加しました。現在の在庫: ${inventory.get(name)}個`;
        }
    },
    {
        name: 'remove_item',
        description: '在庫からアイテムを削除する',
        parameters: {
            type: 'object',
            properties: {
                name: {
                    type: 'string',
                    description: '削除するアイテムの名前',
                },
                quantity: {
                    type: 'number',
                    description: '削除する数量',
                },
            },
            required: ['name', 'quantity'],
        },
        implementation: ({ name, quantity }: { name: string; quantity: number }) => {
            const current = inventory.get(name) || 0;
            if (current < quantity) {
                return `エラー: ${name}の在庫が不足しています（現在: ${current}個）`;
            }
            const newQuantity = current - quantity;
            if (newQuantity === 0) {
                inventory.delete(name);
            } else {
                inventory.set(name, newQuantity);
            }
            return `${name}を${quantity}個削除しました。${newQuantity > 0 ? `残り: ${newQuantity}個` : '在庫がなくなりました'}`;
        }
    },
    {
        name: 'list_items',
        description: '現在の在庫のすべてのアイテムを一覧表示する',
        parameters: {
            type: 'object',
            properties: {},
            required: [],
        },
        implementation: () => {
            if (inventory.size === 0) {
                return '在庫にアイテムがありません。';
            }

            const items = Array.from(inventory.entries())
                .map(([name, quantity]) => `- ${name}: ${quantity}個`)
                .join('\n');
            return `現在の在庫:\n${items}`;
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
