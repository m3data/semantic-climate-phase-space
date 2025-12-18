/**
 * Chat interface management
 */

// Set up chat input handlers
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chatInput');

    document.getElementById('sendBtn').addEventListener('click', sendMessage);

    // Handle Enter to send, Shift+Enter for new line
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-expand textarea as user types
    chatInput.addEventListener('input', autoExpandTextarea);
});

function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message || !window.ws || window.ws.readyState !== WebSocket.OPEN) {
        return;
    }

    // Warn about very large texts (only for local inference)
    const wordCount = message.split(/\s+/).length;
    const charCount = message.length;
    const isCloudProvider = config && ['together', 'anthropic', 'openai'].includes(config.provider);

    if (wordCount > 1000 && !isCloudProvider) {
        const proceed = confirm(
            `⚠️ Large text detected (${wordCount} words, ${charCount} characters)\n\n` +
            `With local inference, this may take 3-10 minutes to process.\n\n` +
            `Tip: For faster results, consider:\n` +
            `• Breaking into smaller sections\n` +
            `• Using a cloud provider (Together AI)\n\n` +
            `Continue anyway?`
        );
        if (!proceed) return;
    }

    // Add user message to chat
    addUserMessage(message);

    // Send to backend
    window.ws.send(JSON.stringify({
        type: 'message',
        text: message
    }));

    // Clear input and reset height
    input.value = '';
    input.style.height = 'auto';

    // Show loading indicator
    addLoadingMessage();

    // Scroll to bottom
    scrollToBottom();
}

function addUserMessage(text) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = 'message message-user';

    // Add icon
    const iconEl = document.createElement('i');
    iconEl.className = 'ph ph-user message-icon';

    const textEl = document.createElement('span');
    textEl.className = 'message-text';
    textEl.textContent = text;

    messageEl.appendChild(iconEl);
    messageEl.appendChild(textEl);
    messagesDiv.appendChild(messageEl);
    scrollToBottom();
}

function addAIMessage(text) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = 'message message-ai';

    // Add icon
    const iconEl = document.createElement('i');
    iconEl.className = 'ph ph-robot message-icon';

    const textEl = document.createElement('div');
    textEl.className = 'message-text';
    // Render markdown using marked.js
    textEl.innerHTML = marked.parse(text);

    messageEl.appendChild(iconEl);
    messageEl.appendChild(textEl);
    messagesDiv.appendChild(messageEl);
    scrollToBottom();
}

function addLoadingMessage() {
    const messagesDiv = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = 'message message-loading';
    messageEl.id = 'loadingMessage';
    messageEl.textContent = 'Thinking...';
    messagesDiv.appendChild(messageEl);
    scrollToBottom();

    // Provider-aware loading messages
    // Cloud providers (together, anthropic, openai) are fast (2-10s)
    // Local (ollama) can take 30s-10min depending on model/hardware
    const isCloudProvider = config && ['together', 'anthropic', 'openai'].includes(config.provider);

    // Update message for long waits
    let elapsed = 0;
    window.loadingInterval = setInterval(() => {
        if (document.getElementById('loadingMessage')) {
            elapsed += 5;
            if (isCloudProvider) {
                // Cloud providers - typically 2-10s, but can be longer under load
                if (elapsed < 10) {
                    messageEl.textContent = `Thinking... (${elapsed}s)`;
                } else if (elapsed < 30) {
                    messageEl.textContent = `Processing... (${elapsed}s)`;
                } else if (elapsed < 60) {
                    messageEl.textContent = `Still processing... (${elapsed}s)`;
                } else {
                    messageEl.textContent = `Waiting for response... (${Math.floor(elapsed/60)}m ${elapsed%60}s)`;
                }
            } else {
                // Ollama/local - can take minutes
                if (elapsed < 60) {
                    messageEl.textContent = `Thinking... (${elapsed}s)`;
                } else if (elapsed < 180) {
                    messageEl.textContent = `Processing... (${Math.floor(elapsed/60)}m ${elapsed%60}s - local inference can take 3-10 minutes)`;
                } else {
                    messageEl.textContent = `Still processing... (${Math.floor(elapsed/60)}m ${elapsed%60}s - please wait, max 10 minutes)`;
                }
            }
        } else {
            clearInterval(window.loadingInterval);
        }
    }, 5000); // Update every 5 seconds (faster for cloud feedback)
}

function removeLoadingMessage() {
    const loadingEl = document.getElementById('loadingMessage');
    if (loadingEl) {
        loadingEl.remove();
    }
    // Clear loading interval
    if (window.loadingInterval) {
        clearInterval(window.loadingInterval);
        window.loadingInterval = null;
    }
}

function addSystemMessage(text, isHtml = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageEl = document.createElement('div');
    messageEl.className = 'system-message';
    if (isHtml) {
        messageEl.innerHTML = text;
    } else {
        messageEl.textContent = text;
    }
    messagesDiv.appendChild(messageEl);
    scrollToBottom();
}

function clearChat() {
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.innerHTML = '';
    document.getElementById('turnCount').textContent = '0';
}

function autoExpandTextarea() {
    const textarea = document.getElementById('chatInput');

    // Reset height to recalculate
    textarea.style.height = '44px';

    // Calculate new height based on content
    const scrollHeight = textarea.scrollHeight;
    const maxHeight = 200;

    if (scrollHeight > 44) {
        const newHeight = Math.min(scrollHeight, maxHeight);
        textarea.style.height = newHeight + 'px';

        // Enable scrolling only when max height is reached
        if (scrollHeight > maxHeight) {
            textarea.style.overflowY = 'auto';
        } else {
            textarea.style.overflowY = 'hidden';
        }
    }
}

function scrollToBottom() {
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
