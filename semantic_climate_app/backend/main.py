"""
Semantic Climate Chat - FastAPI Backend

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Local web app for real-time semantic coupling analysis.
"""

# Disable tokenizers parallelism before any imports to avoid fork deadlocks
# This must be set before sentence-transformers/HuggingFace tokenizers are loaded
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import asyncio
import time

# Load environment variables from project .env file
# Use override=True to prefer project-specific keys over shell environment
from dotenv import load_dotenv
APP_DIR = Path(__file__).parent.parent  # semantic_climate_app/
load_dotenv(APP_DIR / ".env", override=True)

from llm_clients import (
    create_llm_client, get_available_providers, format_size,
    OllamaClient, TogetherClient
)
from embedding_service import EmbeddingService
from metrics_service import MetricsService
from session_manager import SessionManager
from ebs_client import EBSClient, get_ebs_client

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Climate Chat",
    description="Real-time semantic coupling analysis for AI dialogue",
    version="0.1.0"
)

# Add CORS middleware for ECP Field Journal integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3141", "http://127.0.0.1:3141"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global services (initialized on startup)
embedding_service = None
metrics_service = None
ebs_client = None
ebs_listen_task = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global embedding_service, metrics_service, ebs_client

    print("Semantic Climate Chat - Starting up...")
    print("=" * 50)

    # Initialize embedding service (nomic-embed-text via Ollama)
    print("\nLoading embedding model...")
    embedding_service = EmbeddingService()

    # Initialize metrics service
    print("\nInitializing metrics calculator...")
    metrics_service = MetricsService()

    # Check LLM providers status
    print("\nChecking LLM providers...")
    providers = get_available_providers()
    for provider in providers:
        status = "✓" if provider["available"] else "✗"
        print(f"   {status} {provider['display_name']}: {provider['note']}")
        if provider["available"] and provider["models"]:
            for model in provider["models"][:3]:
                name = model.get("name", str(model))
                print(f"      - {name}")

    # Initialize EBS client (optional - connects if EBS is running)
    print("\nChecking EarthianBioSense...")
    ebs_client = get_ebs_client()
    try:
        welcome = await ebs_client.connect()
        if "error" not in welcome:
            print(f"   Connected to EBS")
            # Start background listen task
            asyncio.create_task(ebs_client.start_listening())
        else:
            print(f"   EBS not available (start with: python src/app.py in Earthian-BioSense)")
    except Exception as e:
        print(f"   EBS connection failed: {e}")

    print("\n" + "=" * 50)
    print("Server ready at http://127.0.0.1:8000")
    print("=" * 50 + "\n")


@app.get("/")
async def get_home():
    """Serve main page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(content="<h1>Semantic Climate Chat</h1><p>Frontend not found</p>")


@app.get("/api/ollama/status")
async def get_ollama_status():
    """Legacy endpoint - Check if Ollama is running and get available models."""
    running = OllamaClient.is_running()

    if running:
        models = OllamaClient.get_available_models()
        # Format sizes
        for model in models:
            if isinstance(model['size'], int):
                model['size'] = format_size(model['size'])

        return {
            "running": True,
            "models": models,
            "endpoint": "http://localhost:11434"
        }
    else:
        return {
            "running": False,
            "models": [],
            "error": "Ollama server not accessible. Start with: ollama serve"
        }


@app.get("/api/providers")
async def get_providers():
    """
    Get all available LLM providers and their models.

    Returns list of providers with:
    - name: provider identifier ('ollama', 'together', 'anthropic', 'openai')
    - display_name: human-readable name
    - available: bool, whether it can be used
    - models: list of available models
    - note: status message or configuration hint
    """
    providers = get_available_providers()

    # Format Ollama model sizes for display
    for provider in providers:
        if provider["name"] == "ollama" and provider["models"]:
            for model in provider["models"]:
                if isinstance(model.get('size'), int):
                    model['size'] = format_size(model['size'])

    return {"providers": providers}


@app.get("/api/ebs/status")
async def get_ebs_status():
    """Check EBS connection status."""
    if ebs_client:
        return ebs_client.get_state()
    return {"connected": False, "error": "EBS client not initialized"}


# Track active session for external clients (e.g., ECP Field Journal)
active_session_info = {
    "active": False,
    "session_id": None,
    "model": None,
    "turn_count": 0,
    "started_at": None,
    "ebs_connected": False
}


def update_session_info(session, connected=True):
    """Update the global session info for external clients."""
    global active_session_info
    if session and connected:
        active_session_info = {
            "active": True,
            "session_id": session.session_id,
            "model": session.model_name,
            "turn_count": session.get_turn_count(),
            "started_at": session.created_at.isoformat(),
            "ebs_connected": ebs_client.is_connected() if ebs_client else False
        }
    else:
        active_session_info = {
            "active": False,
            "session_id": None,
            "model": None,
            "turn_count": 0,
            "started_at": None,
            "ebs_connected": ebs_client.is_connected() if ebs_client else False
        }


@app.get("/api/session/status")
async def get_session_status():
    """
    Get current session status for external clients (e.g., ECP Field Journal).

    Returns session state including model, turn count, and EBS connection.
    Used by EECP orchestration to pre-populate experiment metadata.
    """
    return active_session_info


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat and metrics.

    Protocol:
        Client sends: {"type": "configure", "model": "mistral"}
        Client sends: {"type": "message", "text": "Hello"}
        Server sends: {"type": "response", "text": "...", "metrics": {...}}
    """
    await websocket.accept()

    # Session state
    llm_client = None
    session = SessionManager(
        min_turns=10,
        max_turns=50,
        embedding_model=embedding_service.model_name
    )

    # EBS biosignal callback - store in session and forward to client
    sub_id = None

    async def on_phase_update(phase_data):
        session.add_biosignal_sample(phase_data.to_dict())
        # Forward biosignal to frontend for live psi_biosignal display
        try:
            await websocket.send_text(json.dumps({
                "type": "biosignal",
                "data": phase_data.to_dict()
            }))
        except Exception:
            pass  # Client may have disconnected

    # Subscribe to EBS if connected
    if ebs_client and ebs_client.is_connected():
        session.set_ebs_session(ebs_client.state.session_id)
        sub_id = ebs_client.subscribe(on_phase_update)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get("type")

            # Configure LLM model (supports multiple providers)
            if msg_type == "configure":
                model_name = message.get("model")
                provider = message.get("provider", "ollama")  # Default to ollama for backwards compat

                # Handle ECP context if present (when launched from Field Journal)
                ecp_session_id = message.get("ecp_session_id")
                ecp_experiment_type = message.get("ecp_experiment_type")
                if ecp_session_id:
                    session.set_ecp_context(ecp_session_id, ecp_experiment_type)
                    print(f"ECP context set: {ecp_session_id} ({ecp_experiment_type})")

                try:
                    llm_client = create_llm_client(provider, model_name)
                    session.set_model(f"{provider}:{model_name}")
                    update_session_info(session)  # Update for external clients
                    await websocket.send_text(json.dumps({
                        "type": "config_success",
                        "message": f"Connected to {llm_client.provider_name}: {model_name}",
                        "provider": provider,
                        "model": model_name
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Failed to connect to {provider}: {str(e)}"
                    }))

            # Handle user message
            elif msg_type == "message":
                if not llm_client:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Please select a model first"
                    }))
                    continue

                user_text = message.get("text", "").strip()
                if not user_text:
                    continue

                # Track when human message was received
                human_msg_received_at = time.time() * 1000  # ms

                try:
                    # Calculate human compose time (time since last AI response)
                    compose_time_ms = None
                    if session.turns:
                        last_turn = session.turns[-1]
                        if last_turn.speaker == 'ai':
                            last_ai_time = last_turn.timestamp.timestamp() * 1000
                            compose_time_ms = int(human_msg_received_at - last_ai_time)

                    # 1. Generate embedding for user message
                    t0 = time.time() * 1000
                    user_embedding = embedding_service.embed(user_text)
                    t1 = time.time() * 1000
                    print(f"[TIMING] User embed: {int(t1-t0)}ms", flush=True)
                    session.add_turn("human", user_text, user_embedding, compose_time_ms=compose_time_ms)

                    # 2. Get AI response from LLM (track latency)
                    llm_start = time.time() * 1000
                    ai_text = llm_client.chat(user_text)
                    llm_end = time.time() * 1000
                    api_latency_ms = int(llm_end - llm_start)
                    print(f"[TIMING] LLM call: {api_latency_ms}ms", flush=True)

                    # 3. Generate embedding for AI response
                    t2 = time.time() * 1000
                    ai_embedding = embedding_service.embed(ai_text)
                    t3 = time.time() * 1000
                    print(f"[TIMING] AI embed: {int(t3-t2)}ms", flush=True)
                    session.add_turn("ai", ai_text, ai_embedding, api_latency_ms=api_latency_ms)

                    # 4. Calculate metrics if enough turns
                    metrics_result = None
                    if session.can_analyze():
                        t4 = time.time() * 1000
                        embeddings = session.get_embeddings()
                        turn_texts = session.get_turn_texts()  # For affective substrate
                        previous_metrics = session.get_previous_metrics()  # For trajectory
                        semantic_shifts = session.get_semantic_shifts()  # For coherence

                        # Get latest biosignal for psi_biosignal computation
                        latest_biosignal = session.get_latest_biosignal()
                        biosignal_data = None
                        if latest_biosignal and 'hr' in latest_biosignal:
                            biosignal_data = {'heart_rate': latest_biosignal['hr']}

                        metrics_result = metrics_service.analyze(
                            embeddings=embeddings,
                            turn_texts=turn_texts,
                            previous_metrics=previous_metrics,
                            semantic_shifts=semantic_shifts,
                            biosignal_data=biosignal_data
                        )
                        t5 = time.time() * 1000
                        print(f"[TIMING] Metrics calc: {int(t5-t4)}ms", flush=True)
                        session.add_metrics_result(metrics_result)

                        # Send semiotic marker to EBS for JSONL coupling
                        if ebs_client and ebs_client.is_connected():
                            await ebs_client.send_semiotic_marker(metrics_result)

                    # 5. Include latest biosignal in response
                    biosignal = session.get_latest_biosignal()

                    # 6. Update session info for external clients
                    update_session_info(session)

                    # 7. Send response back to client
                    t_end = time.time() * 1000
                    print(f"[TIMING] TOTAL round-trip: {int(t_end - t0)}ms", flush=True)
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": ai_text,
                        "turn_count": session.get_turn_count(),
                        "metrics": metrics_result,
                        "biosignal": biosignal
                    }))

                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Error processing message: {str(e)}"
                    }))

            # Export session (and save to sessions/ directory)
            elif msg_type == "export":
                try:
                    session_data = session.export_session()

                    # Save to sessions/ directory for ECP integration
                    sessions_dir = BASE_DIR.parent / "sessions"
                    sessions_dir.mkdir(exist_ok=True)

                    # Generate filename
                    model_safe = session.model_name.replace("/", "-").replace(":", "-") if session.model_name else "unknown"
                    timestamp = session_data.get("metadata", {}).get("created_at", "").replace(":", "-").replace(".", "-")[:19]
                    session_id = session.session_id
                    filename = f"sc-{model_safe}-{timestamp}-{session_id[:8]}.json"

                    filepath = sessions_dir / filename
                    with open(filepath, 'w') as f:
                        json.dump(session_data, f, indent=2, default=str)

                    print(f"Session exported to: {filepath}")

                    await websocket.send_text(json.dumps({
                        "type": "export_data",
                        "data": session_data,
                        "saved_to": str(filepath),
                        "filename": filename
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Export failed: {str(e)}"
                    }))

            # Reset session
            elif msg_type == "reset":
                session.reset()
                if llm_client:
                    llm_client.reset_history()
                await websocket.send_text(json.dumps({
                    "type": "reset_success",
                    "message": "Session reset"
                }))

            # Field event (session markers)
            elif msg_type == "field_event":
                event = message.get("event", "unknown")
                note = message.get("note", "")
                # Forward to EBS for JSONL logging
                if ebs_client and ebs_client.is_connected():
                    await ebs_client.send_field_event(event, note)
                print(f"Field event: {event} - {note}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clear session info for external clients
        update_session_info(None, connected=False)
        # Unsubscribe from EBS when client disconnects
        if sub_id is not None and ebs_client:
            ebs_client.unsubscribe(sub_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
