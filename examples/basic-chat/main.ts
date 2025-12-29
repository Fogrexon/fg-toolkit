import { FunctionGemmaWeb } from '@fg-toolkit/lib-web-runtime';

const statusEl = document.getElementById('status')!;
const chatHistoryEl = document.getElementById('chat-history')!;
const userInputEl = document.getElementById('user-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;

// Allow local model loading if needed (though we use transformers.js default behavior usually)
// For this example, we assume internet access to download the model from HF
const runtime = new FunctionGemmaWeb({
    quantized: false,
    progressCallback: (progress: any) => {
        if (progress.status === 'progress') {
            statusEl.textContent = `Loading model: ${Math.round(progress.progress)}%`;
        } else {
            statusEl.textContent = `Status: ${progress.status}`;
        }
    }
});

let messages: { role: 'user' | 'model' | 'function'; content: string }[] = [];

async function init() {
    try {
        statusEl.textContent = 'Status: Loading model... (this may take a while first time)';
        await runtime.init();
        statusEl.textContent = 'Status: Ready';
        userInputEl.disabled = false;
        sendBtn.disabled = false;
    } catch (err) {
        statusEl.textContent = `Error: ${err}`;
        console.error(err);
    }
}

function appendMessage(role: 'user' | 'model', text: string) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chatHistoryEl.appendChild(div);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
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

    // Create a placeholder message for the model response
    const modelMsgDiv = document.createElement('div');
    modelMsgDiv.className = 'message model';
    modelMsgDiv.textContent = '...'; // Loading indicator
    chatHistoryEl.appendChild(modelMsgDiv);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

    let fullResponse = "";
    let isFirstToken = true;

    try {
        const response = await runtime.chat(messages, undefined, (token) => {
            if (isFirstToken) {
                modelMsgDiv.textContent = ""; // Clear loading indicator
                isFirstToken = false;
            }
            fullResponse += token;
            modelMsgDiv.textContent = fullResponse;
            chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
        });

        // Final update to ensure everything is consistent (though callback usually handles it)
        modelMsgDiv.textContent = response;
        messages.push({ role: 'model', content: response });
        statusEl.textContent = 'Status: Ready';
    } catch (err) {
        statusEl.textContent = `Error: ${err}`;
        console.error(err);
    } finally {
        userInputEl.disabled = false;
        sendBtn.disabled = false;
        userInputEl.focus();
    }
}

sendBtn.addEventListener('click', handleSend);
userInputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});

init();
