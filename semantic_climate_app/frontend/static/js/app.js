/**
 * Main application controller
 */

// Global state
let config = {
    provider: null,
    model: null,
    minTurns: 10
};

// Cache providers data
let providersCache = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Connect WebSocket immediately (for biosignal streaming before model selection)
    connectWebSocket(null, null);

    // Initialize EBS panel
    if (typeof initEBSPanel === 'function') {
        initEBSPanel();
    }

    // Show settings modal on start
    showSettings();

    // Set up event listeners
    document.getElementById('settingsBtn').addEventListener('click', showSettings);
    document.getElementById('cancelBtn').addEventListener('click', hideSettings);
    document.getElementById('saveBtn').addEventListener('click', saveSettings);
    document.getElementById('refreshModelsBtn').addEventListener('click', loadProviders);
    document.getElementById('retryProvidersBtn').addEventListener('click', loadProviders);
    document.getElementById('exportBtn').addEventListener('click', exportSession);
    document.getElementById('resetBtn').addEventListener('click', resetSession);

    // Details toggle (delayed revelation pattern)
    document.getElementById('toggleDetailsBtn').addEventListener('click', toggleDetails);

    // Load providers
    loadProviders();
}

// Delayed revelation: toggle metric details visibility
function toggleDetails() {
    const btn = document.getElementById('toggleDetailsBtn');
    const revealables = document.querySelectorAll('.revealable');
    const isRevealed = btn.classList.contains('active');

    if (isRevealed) {
        // Hide details
        btn.classList.remove('active');
        revealables.forEach(el => {
            el.classList.remove('revealed');
            el.classList.add('hidden');
        });
    } else {
        // Show details
        btn.classList.add('active');
        revealables.forEach(el => {
            el.classList.remove('hidden');
            el.classList.add('revealed');
        });
    }
}

async function loadProviders() {
    const statusEl = document.getElementById('providerStatus');
    statusEl.innerHTML = '<span class="spinner">⏳</span> Checking providers...';
    statusEl.className = 'status-indicator';

    try {
        const response = await fetch('/api/providers');
        const data = await response.json();
        providersCache = data.providers;

        // Count available providers
        const availableProviders = data.providers.filter(p => p.available);

        if (availableProviders.length > 0) {
            // Show provider list
            statusEl.innerHTML = `✅ ${availableProviders.length} provider(s) available`;
            statusEl.className = 'status-indicator running';

            document.getElementById('providerSelectGroup').style.display = 'block';
            document.getElementById('noProvidersMessage').classList.add('hidden');

            // Build provider radio buttons
            const providerList = document.getElementById('providerList');
            providerList.innerHTML = data.providers.map(provider => {
                const icon = getProviderIcon(provider.name);
                const statusClass = provider.available ? 'available' : 'unavailable';
                const disabled = provider.available ? '' : 'disabled';
                const checked = provider.available && !config.provider ? 'checked' : '';
                const note = provider.note || '';

                return `
                    <label class="provider-option ${statusClass}" ${disabled}>
                        <input type="radio" name="provider" value="${provider.name}" ${disabled} ${checked}>
                        <span class="provider-icon">${icon}</span>
                        <span class="provider-name">${provider.display_name}</span>
                        <span class="provider-status">${provider.available ? '✓' : '✗'}</span>
                        <span class="provider-note">${note}</span>
                    </label>
                `;
            }).join('');

            // Add event listeners to provider radios
            providerList.querySelectorAll('input[name="provider"]').forEach(radio => {
                radio.addEventListener('change', onProviderSelect);
            });

            // Select first available provider
            const firstAvailable = availableProviders[0];
            if (firstAvailable) {
                config.provider = firstAvailable.name;
                populateModels(firstAvailable);
            }

        } else {
            // No providers available
            statusEl.innerHTML = '❌ No providers available';
            statusEl.className = 'status-indicator error';

            document.getElementById('providerSelectGroup').style.display = 'none';
            document.getElementById('modelSelectGroup').style.display = 'none';
            document.getElementById('noProvidersMessage').classList.remove('hidden');
            document.getElementById('saveBtn').disabled = true;
        }

    } catch (error) {
        statusEl.innerHTML = '❌ Connection error';
        statusEl.className = 'status-indicator error';
        document.getElementById('saveBtn').disabled = true;
        console.error('Failed to load providers:', error);
    }
}

function getProviderIcon(providerName) {
    const icons = {
        'ollama': '🦙',
        'together': '⚡',
        'anthropic': '🤖',
        'openai': '🧠'
    };
    return icons[providerName] || '💬';
}

function onProviderSelect(event) {
    const providerName = event.target.value;
    config.provider = providerName;

    // Find provider data
    const provider = providersCache.find(p => p.name === providerName);
    if (provider) {
        populateModels(provider);
    }
}

function populateModels(provider) {
    const modelSelectGroup = document.getElementById('modelSelectGroup');
    const modelSelect = document.getElementById('modelSelect');
    const providerNote = document.getElementById('providerNote');

    if (provider.models && provider.models.length > 0) {
        modelSelectGroup.style.display = 'block';

        // Format model display based on provider
        modelSelect.innerHTML = provider.models.map(model => {
            const name = model.name || model;
            const size = model.size || '';
            const desc = model.description || '';
            const display = size ? `${name} (${size})` : name;
            const title = desc ? desc : '';
            return `<option value="${name}" title="${title}">${display}</option>`;
        }).join('');

        // Select first model
        if (modelSelect.options.length > 0) {
            modelSelect.selectedIndex = 0;
            config.model = modelSelect.value;
        }

        // Enable save button
        document.getElementById('saveBtn').disabled = false;

        // Show provider note
        if (provider.name === 'together') {
            providerNote.innerHTML = '<i class="ph ph-lightning"></i> Fast inference (2-5s) - reduces latency confound for biosignal coupling';
            providerNote.style.display = 'block';
        } else if (provider.name === 'ollama') {
            providerNote.innerHTML = '<i class="ph ph-lock"></i> Local inference - data stays on your machine';
            providerNote.style.display = 'block';
        } else {
            providerNote.style.display = 'none';
        }

    } else {
        modelSelectGroup.style.display = 'none';
        providerNote.style.display = 'none';
        document.getElementById('saveBtn').disabled = true;
    }
}

function showSettings() {
    document.getElementById('settingsModal').classList.remove('hidden');
}

function hideSettings() {
    document.getElementById('settingsModal').classList.add('hidden');
}

function saveSettings() {
    const select = document.getElementById('modelSelect');
    config.model = select.value;

    if (!config.provider || !config.model) {
        alert('Please select a provider and model');
        return;
    }

    // Connect WebSocket and configure with provider
    connectWebSocket(config.provider, config.model);

    // Enable chat
    document.getElementById('chatInput').disabled = false;
    document.getElementById('chatInput').placeholder = 'Type your message...';
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('exportBtn').disabled = false;
    document.getElementById('resetBtn').disabled = false;

    // Update UI
    addSystemMessage(`Connected to model: ${config.model}`);

    hideSettings();
}

function resetSession() {
    if (!confirm('Reset the current conversation?')) {
        return;
    }

    // Send reset message via WebSocket
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        window.ws.send(JSON.stringify({ type: 'reset' }));
    }

    // Clear UI
    clearChat();
    resetClimate();

    addSystemMessage('Session reset. Start a new conversation!');
}

function exportSession() {
    // Request session export via WebSocket
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        window.ws.send(JSON.stringify({ type: 'export' }));
    } else {
        alert('Not connected. Please start a conversation first.');
    }
}

function downloadSessionData(sessionData, savedPath, filename) {
    // Build session complete message
    // Only show field journal prompt if EBS was connected (e2e experiment)
    let message = `<i class="ph ph-check-circle"></i> Session exported — <code>${filename || 'sessions/'}</code>`;

    if (window.sessionStarted) {
        // EBS was connected - this is an e2e experiment
        message += `<br><span style="color: var(--text-muted); font-size: 11px;">Return to Field Journal to complete phenomenological capture.</span>`;
    }

    addSystemMessage(message, true);

    // Disable chat input to prevent further messages
    document.getElementById('chatInput').disabled = true;
    document.getElementById('sendBtn').disabled = true;
    document.getElementById('exportBtn').disabled = true;

}
