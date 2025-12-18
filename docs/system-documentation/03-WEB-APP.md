# Web Application Architecture

**Directory:** `semantic_climate_app/`
**License:** ESL-A

## Overview

The Semantic Climate web application provides real-time analysis of human-AI dialogue through a local web interface. It connects to LLM providers (Ollama, Together AI) and optionally to Earthian BioSense for biosignal integration.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                    │
│                 frontend/index.html                      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Chat Panel  │  │ Metrics     │  │ Phase Space     │ │
│  │             │  │ Display     │  │ Visualization   │ │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
└─────────┼────────────────┼──────────────────┼──────────┘
          │                │                  │
          │    WebSocket   │                  │
          └────────────────┼──────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────┐
│                    FastAPI Backend                       │
│                    backend/main.py                       │
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ WebSocket       │  │ REST API        │              │
│  │ /ws/chat        │  │ /api/providers  │              │
│  │ /ws/ebs         │  │ /api/models     │              │
│  └────────┬────────┘  └─────────────────┘              │
│           │                                             │
│  ┌────────┴────────────────────────────────────┐       │
│  │              Service Layer                   │       │
│  │                                              │       │
│  │  ┌──────────────┐  ┌────────────────────┐   │       │
│  │  │ Embedding    │  │ Metrics            │   │       │
│  │  │ Service      │  │ Service            │   │       │
│  │  └──────┬───────┘  └─────────┬──────────┘   │       │
│  │         │                    │              │       │
│  │  ┌──────┴───────┐  ┌─────────┴──────────┐   │       │
│  │  │ LLM Clients  │  │ Session Manager    │   │       │
│  │  │ (Ollama,     │  │ (Conversation      │   │       │
│  │  │  Together)   │  │  State)            │   │       │
│  │  └──────────────┘  └────────────────────┘   │       │
│  │                                              │       │
│  │  ┌──────────────┐                           │       │
│  │  │ EBS Client   │  (Optional biosignal)     │       │
│  │  └──────────────┘                           │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Backend Services

### main.py — FastAPI Server

**Port:** 8000 (default)

**Endpoints:**

| Endpoint | Type | Purpose |
|----------|------|---------|
| `/` | GET | Serve frontend |
| `/ws/chat` | WebSocket | Real-time chat + metrics |
| `/ws/ebs` | WebSocket | Biosignal stream (EBS) |
| `/api/providers` | GET | Available LLM providers |
| `/api/models/{provider}` | GET | Models for provider |
| `/api/health` | GET | Server health check |

**Startup:**
```python
@app.on_event("startup")
async def startup_event():
    # Initialize embedding service (Ollama nomic-embed-text)
    embedding_service = EmbeddingService()

    # Initialize metrics service
    metrics_service = MetricsService()

    # Initialize EBS client (optional)
    ebs_client = get_ebs_client()
```

---

### embedding_service.py — Text Embeddings

**Purpose:** Convert turn text to embedding vectors.

**Backends:**

| Backend | Model | Dimensions |
|---------|-------|------------|
| Ollama | nomic-embed-text | 768 |
| sentence-transformers | all-mpnet-base-v2 | 768 |

**Usage:**
```python
from embedding_service import EmbeddingService

service = EmbeddingService(
    model_name='nomic-embed-text',  # or 'all-mpnet-base-v2'
    backend='ollama'                 # or 'sentence-transformers'
)

# Single embedding
embedding = service.embed("Hello, how are you?")

# Batch embedding
embeddings = service.embed_batch(["Turn 1", "Turn 2", "Turn 3"])
```

**Notes:**
- Ollama requires `ollama serve` running locally
- sentence-transformers downloads model on first use (~400MB)

---

### metrics_service.py — Coupling Analysis

**Purpose:** Orchestrate metric computation, mode detection, and risk assessment.

**Class: MetricsService**

```python
from metrics_service import MetricsService

service = MetricsService()

result = service.analyze(
    embeddings=embedding_list,
    turn_texts=text_list,
    turn_speakers=speaker_list,      # ['human', 'ai', ...]
    previous_metrics=prev_metrics,   # For trajectory calculation
    semantic_shifts=shift_list,      # For coherence analysis
    biosignal_data={'heart_rate': 72},
    trajectory_metrics=traj_list,    # For Δκ variance
    coherence_pattern='breathing'    # Pre-computed or None
)
```

**Returns:**
```python
{
    'metrics': {
        'delta_kappa': float,
        'alpha': float,
        'delta_h': float,
        'confidence_intervals': {...},
        'trends': {...}
    },
    'climate': {
        'temperature': float,   # Δκ
        'humidity': float,      # ΔH
        'pressure': float,      # α
        'temp_level': str,
        'humidity_level': str,
        'pressure_level': str
    },
    'mode': str,                # Base mode
    'coupling_mode': {
        'mode': str,
        'trajectory': str,
        'compound_label': str,
        'epistemic_risk': str,
        'risk_factors': [str],
        'confidence': float,
        'coherence': {...}
    },
    'psi_vector': {...},
    'psi_composite': float,
    'attractor_basin': {
        'name': str,
        'confidence': float
    },
    'affective_substrate': {...}
}
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `analyze()` | Full analysis pipeline |
| `compute_semantic_coherence()` | Autocorrelation-based coherence |
| `detect_coupling_mode()` | Trajectory-aware mode classification |
| `_detect_mode()` | Base mode (position only) |
| `_assess_epistemic_risk()` | Risk level + factors |

---

### llm_clients.py — LLM Providers

**Purpose:** Unified interface for conversation with AI models.

**Supported Providers:**

| Provider | Client Class | Requirements |
|----------|--------------|--------------|
| Ollama | `OllamaClient` | Local `ollama serve` |
| Together AI | `TogetherClient` | API key in `.env` |

**Usage:**
```python
from llm_clients import create_llm_client

# Auto-detect or specify provider
client = create_llm_client(provider='ollama', model='llama3.2')

# Streaming response
async for chunk in client.chat_stream(messages, model):
    print(chunk, end='')
```

**Discovery:**
```python
from llm_clients import get_available_providers

providers = get_available_providers()
# {'ollama': {'available': True, 'models': ['llama3.2', ...]}, ...}
```

---

### session_manager.py — Conversation State

**Purpose:** Track conversation history, embeddings, and metrics over session.

**Usage:**
```python
from session_manager import SessionManager

manager = SessionManager()

# Add turn
manager.add_turn(
    role='user',
    content='Hello!',
    embedding=embedding_vector,
    metrics=metrics_result
)

# Get history for LLM
messages = manager.get_messages()

# Get embeddings for analysis
embeddings = manager.get_embeddings()

# Export session
session_data = manager.export_session()
```

---

### ebs_client.py — Biosignal Integration

**Purpose:** Connect to Earthian BioSense for heart rate data.

**Protocol:** WebSocket to EBS backend (localhost:8765 default)

**Usage:**
```python
from ebs_client import EBSClient, get_ebs_client

# Get singleton client
client = get_ebs_client()

# Connect and listen
async def handle_biosignal(data):
    hr = data.get('heart_rate')
    # Update metrics with biosignal

await client.connect()
client.on_data = handle_biosignal
```

**Data Format:**
```python
{
    'heart_rate': float,      # BPM
    'rr_intervals': [float],  # ms between beats
    'timestamp': float
}
```

---

## Frontend

**Directory:** `frontend/`

**Structure:**
```
frontend/
├── index.html          # Main page
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   └── js/
│       └── app.js      # WebSocket client, UI logic
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| Chat Panel | User input, AI responses |
| Metrics Display | Δκ, α, ΔH values + gauges |
| Mode Badge | Current coupling mode |
| Risk Indicator | Epistemic risk level |
| Phase Space | 2D projection of Ψ trajectory |

---

## Configuration

### Environment Variables

Create `semantic_climate_app/.env`:

```env
# LLM Providers
TOGETHER_API_KEY=your_key_here

# Embedding
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BACKEND=ollama

# EBS Integration (optional)
EBS_HOST=localhost
EBS_PORT=8765
```

---

## Running the App

### Prerequisites

```bash
# Ollama (for embeddings and local LLM)
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2  # or preferred model

# Python dependencies
pip install fastapi uvicorn websockets python-dotenv
```

### Start Server

```bash
cd semantic-climate-phase-space/semantic_climate_app/backend
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access

Open browser to: `http://localhost:8000`

---

## WebSocket Protocol

### Chat WebSocket (`/ws/chat`)

**Client → Server:**
```json
{
    "type": "message",
    "content": "User's message text",
    "provider": "ollama",
    "model": "llama3.2"
}
```

**Server → Client (streaming):**
```json
{
    "type": "response_chunk",
    "content": "partial response text"
}
```

**Server → Client (metrics update):**
```json
{
    "type": "metrics_update",
    "metrics": {...},
    "climate": {...},
    "mode": "Resonant",
    "coupling_mode": {...},
    "psi_vector": {...},
    "attractor_basin": {...}
}
```

### EBS WebSocket (`/ws/ebs`)

**Server → Client (biosignal):**
```json
{
    "type": "biosignal",
    "heart_rate": 72,
    "timestamp": 1702345678.123
}
```
