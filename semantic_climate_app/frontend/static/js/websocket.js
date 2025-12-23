/**
 * WebSocket connection management
 */

function connectWebSocket(provider, model, temperature = 0.7, systemPrompt = null) {
    // If already connected, just configure the model
    if (window.ws && window.ws.readyState === WebSocket.OPEN) {
        if (model) {
            const configMsg = {
                type: 'configure',
                provider: provider || 'ollama',
                model: model,
                temperature: temperature,
                system_prompt: systemPrompt || null
            };
            // Include ECP context if present (for session correlation)
            if (window.ecpContext) {
                configMsg.ecp_session_id = window.ecpContext.sessionId;
                configMsg.ecp_experiment_type = window.ecpContext.experimentType;
            }
            window.ws.send(JSON.stringify(configMsg));
        }
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    window.ws = new WebSocket(wsUrl);

    window.ws.onopen = () => {
        console.log('WebSocket connected');

        // Only configure if model is provided (not on initial page load)
        if (model) {
            const configMsg = {
                type: 'configure',
                provider: provider || 'ollama',
                model: model,
                temperature: temperature,
                system_prompt: systemPrompt || null
            };
            // Include ECP context if present (for session correlation)
            if (window.ecpContext) {
                configMsg.ecp_session_id = window.ecpContext.sessionId;
                configMsg.ecp_experiment_type = window.ecpContext.experimentType;
            }
            window.ws.send(JSON.stringify(configMsg));
        }
    };

    window.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    window.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addSystemMessage('Connection error. Please refresh the page.');
    };

    window.ws.onclose = () => {
        console.log('WebSocket disconnected');
        document.getElementById('chatInput').disabled = true;
        document.getElementById('sendBtn').disabled = true;
    };
}

function handleWebSocketMessage(data) {
    console.log('Received:', data);

    switch (data.type) {
        case 'config_success':
            console.log(data.message);
            break;

        case 'response':
            removeLoadingMessage();
            addAIMessage(data.text);

            // Update turn count
            document.getElementById('turnCount').textContent = data.turn_count;

            // Update metrics if available
            if (data.metrics) {
                document.getElementById('analysisStatus').textContent = 'Analysis active';
                document.getElementById('analysisStatus').className = 'status-active';

                updateClimate(data.metrics);
                updateChart(data.metrics);
            }
            break;

        case 'reset_success':
            console.log('Session reset');
            break;

        case 'export_data':
            downloadSessionData(data.data, data.saved_to, data.filename);
            break;

        case 'error':
            removeLoadingMessage();
            addSystemMessage('Error: ' + data.message);
            break;

        case 'biosignal':
            // Real-time biosignal update from EBS (1Hz)
            if (typeof updateBiosignal === 'function') {
                updateBiosignal(data.data);
            }
            break;

        default:
            console.log('Unknown message type:', data.type);
    }
}
