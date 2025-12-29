import { Message } from '@fg-toolkit/lib-web-runtime';

const statusEl = document.getElementById('status')!;
const chatHistoryEl = document.getElementById('chat-history')!;
const userInputEl = document.getElementById('user-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;

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
            quantized: true
        }
    });
}

function appendMessage(role: 'user' | 'model', text: string) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chatHistoryEl.appendChild(div);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

// We wrap the worker response in a Promise to simulate async/await flow
function chatWithWorker(msgs: Message[]): Promise<string> {
    return new Promise((resolve, reject) => {
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
    const modelMsgDiv = document.createElement('div');
    modelMsgDiv.className = 'message model';
    modelMsgDiv.textContent = '...';
    chatHistoryEl.appendChild(modelMsgDiv);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

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
