import { Message } from '@fg-toolkit/lib-web-runtime';

const statusEl = document.getElementById('status')!;
const inventoryListEl = document.getElementById('inventory-list')!;
const chatHistoryEl = document.getElementById('chat-history')!;
const userInputEl = document.getElementById('user-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;

// Inventory state
const inventory: Map<string, number> = new Map();

// Create worker
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

let messages: Message[] = [];
let pendingResolver: ((value: string) => void) | null = null;
let currentModelMsgDiv: HTMLDivElement | null = null;
let currentFullResponse = "";

worker.onmessage = (e) => {
    const data = e.data;
    switch (data.type) {
        case 'status':
            if (data.status === 'ready') {
                statusEl.textContent = 'Status: Ready';
                userInputEl.disabled = false;
                sendBtn.disabled = false;
            } else if (data.status === 'progress') {
                statusEl.textContent = `Loading model: ${Math.round(data.progress)}%`;
            } else {
                statusEl.textContent = `Status: ${data.status}`;
            }
            break;
        case 'token':
            if (currentModelMsgDiv) {
                if (currentModelMsgDiv.textContent === '...') {
                    currentModelMsgDiv.textContent = "";
                }
                currentFullResponse += data.token;
                currentModelMsgDiv.textContent = currentFullResponse;
                chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
            }
            break;
        case 'result':
            statusEl.textContent = 'Status: Ready';
            if (pendingResolver) {
                pendingResolver(data.content);
                pendingResolver = null;
            }
            currentModelMsgDiv = null;

            // Restore UI
            userInputEl.disabled = false;
            sendBtn.disabled = false;
            userInputEl.focus();
            break;
        case 'error':
            statusEl.textContent = `Error: ${data.error}`;
            console.error(data.error);
            userInputEl.disabled = false;
            sendBtn.disabled = false;
            break;
    }
};

async function init() {
    statusEl.textContent = 'Status: Initializing worker...';
    worker.postMessage({
        type: 'init',
        config: {
            modelPath: '/models/inventory-management-gemma-it-onnx',
            quantized: true
        }
    });
}

function updateInventoryDisplay() {
    if (inventory.size === 0) {
        inventoryListEl.innerHTML = '<p class="empty-message">まだアイテムがありません</p>';
        return;
    }

    inventoryListEl.innerHTML = '';
    Array.from(inventory.entries())
        .sort(([a], [b]) => a.localeCompare(b, 'ja'))
        .forEach(([name, quantity]) => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'inventory-item';
            itemDiv.innerHTML = `
                <span class="item-name">${name}</span>
                <span class="item-quantity">${quantity}個</span>
            `;
            inventoryListEl.appendChild(itemDiv);
        });
}

function addItem(name: string, quantity: number): string {
    const current = inventory.get(name) || 0;
    inventory.set(name, current + quantity);
    updateInventoryDisplay();
    return `${name}を${quantity}個追加しました。現在の在庫: ${inventory.get(name)}個`;
}

function removeItem(name: string, quantity: number): string {
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
    updateInventoryDisplay();
    return `${name}を${quantity}個削除しました。${newQuantity > 0 ? `残り: ${newQuantity}個` : '在庫がなくなりました'}`;
}

function listItems(): string {
    if (inventory.size === 0) {
        return '在庫にアイテムがありません。';
    }

    const items = Array.from(inventory.entries())
        .map(([name, quantity]) => `- ${name}: ${quantity}個`)
        .join('\n');
    return `現在の在庫:\n${items}`;
}

function appendMessage(role: 'user' | 'model', text: string) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chatHistoryEl.appendChild(div);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    return div;
}

function chatWithWorker(msgs: Message[]): Promise<string> {
    return new Promise((resolve) => {
        pendingResolver = resolve;
        worker.postMessage({ type: 'chat', messages: msgs });
    });
}

async function handleSend() {
    const text = userInputEl.value.trim();
    if (!text) return;

    appendMessage('user', text);
    messages.push({ role: 'user', content: text });
    userInputEl.value = '';
    userInputEl.disabled = true;
    sendBtn.disabled = true;
    statusEl.textContent = 'Status: Generating...';

    // Create placeholder
    const modelMsgDiv = appendMessage('model', '...');
    currentModelMsgDiv = modelMsgDiv;
    currentFullResponse = "";

    try {
        const response = await chatWithWorker(messages);
        messages.push({ role: 'model', content: response });
    } catch (err) {
        console.error(err);
    }
}

sendBtn.addEventListener('click', handleSend);
userInputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});

init();
updateInventoryDisplay();
