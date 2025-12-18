# Semantic Climate Chat - Web Application

Real-time semantic coupling analysis for AI dialogue.

## Prerequisites

1. **Ollama** - Local LLM server
   ```bash
   # Install from https://ollama.ai
   ollama serve
   ollama pull llama3.2  # or any model you prefer
   ```

2. **Python 3.8+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **EarthianBioSense** (optional) - For biosignal integration

## Running

From the repository root:

```bash
cd semantic_climate_app
python backend/main.py
```

Or with auto-reload for development:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Access at: http://127.0.0.1:8000

## Architecture

```
semantic_climate_app/
├── backend/
│   ├── main.py              # FastAPI + WebSocket server
│   ├── metrics_service.py   # Wraps src/SemanticClimateAnalyzer
│   ├── session_manager.py   # Dialogue session state
│   ├── embedding_service.py # sentence-transformers integration
│   ├── ollama_client.py     # Ollama LLM client
│   └── ebs_client.py        # EarthianBioSense integration
├── frontend/
│   ├── index.html           # Main page
│   └── static/
│       ├── css/style.css    # Earth-warm dark theme
│       └── js/              # Chat, climate, WebSocket handlers
└── requirements.txt
```

## Features

- Real-time semantic coupling metrics (Δκ, ΔH, α)
- Trajectory-aware coupling mode detection
- Vector Ψ visualization (semantic, temporal, affective, biosignal)
- Attractor basin classification
- Session export to JSON
- Optional EBS biosignal integration
