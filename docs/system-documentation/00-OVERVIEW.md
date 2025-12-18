# Semantic Climate Phase Space — System Overview

## Purpose

This system provides tools for measuring and analyzing the **semantic climate** of human-AI dialogue — the movement of meaning through conversational phase space. It extends Morgoulis (2025) 4D Semantic Coupling Framework with dynamical systems concepts including attractor basins, trajectory dynamics, and multi-substrate coupling.

## Core Questions the System Addresses

1. **Is this dialogue exploring or converging?** (Semantic Curvature Δκ)
2. **How is meaning reorganizing?** (Entropy Shift ΔH)
3. **What patterns emerge over time?** (Fractal Similarity α)
4. **What coupling configuration are we in?** (Attractor Basin)
5. **What's the epistemic risk?** (Coupling Mode + Trajectory)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Web App (UI)   │  Batch Analyzer │  Direct Python Import       │
│  main.py        │  batch_analyze  │  from src import ...        │
│  WebSocket API  │  conversations  │                             │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       │
┌─────────────────────────────────────────────────────────────────┐
│                    MetricsService                               │
│    semantic_climate_app/backend/metrics_service.py              │
│    - Coupling mode detection                                    │
│    - Semantic coherence analysis                                │
│    - Risk assessment                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                SemanticClimateAnalyzer                          │
│                    src/extensions.py                            │
│    - Ψ vector computation (4 substrates)                        │
│    - Attractor basin detection (9 basins)                       │
│    - Dialogue context computation                               │
│    - Trajectory dynamics                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              SemanticComplexityAnalyzer                         │
│                   src/core_metrics.py                           │
│    - Δκ (Semantic Curvature) — local Frenet-Serret              │
│    - α (Fractal Similarity) — DFA on semantic velocity          │
│    - ΔH (Entropy Shift) — JS divergence on shared clustering    │
│                                                                 │
│    [Morgoulis 2025, MIT License, with fixes]                    │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
semantic-climate-phase-space/
├── src/                           # Core library
│   ├── __init__.py               # Public API exports
│   ├── core_metrics.py           # Morgoulis metrics (Δκ, α, ΔH)
│   ├── extensions.py             # Ψ vector, basins, dynamics
│   ├── api.py                    # Function-based API for testing
│   └── schema.py                 # Module versioning system
│
├── semantic_climate_app/          # Web application
│   ├── backend/
│   │   ├── main.py               # FastAPI server
│   │   ├── metrics_service.py    # Coupling mode detection
│   │   ├── embedding_service.py  # Text → embeddings
│   │   ├── llm_clients.py        # Ollama/Together AI/OpenAI/Anthropic clients
│   │   ├── session_manager.py    # Conversation state
│   │   └── ebs_client.py         # Earthian BioSense integration
│   └── frontend/                 # Web UI
│
├── scripts/
│   └── batch_analyze_conversations.py  # Archive analysis
│
├── tests/                         # Test suite (60+ tests)
│
└── docs/
    └── system-documentation/      # This documentation
```

## Data Flow

### Real-time Analysis (Web App)

```
User input
    │
    ▼
EmbeddingService.embed()          # Text → 768/384-dim vector
    │
    ▼
MetricsService.analyze()          # Full analysis pipeline
    │
    ├─► SemanticComplexityAnalyzer.calculate_all_metrics()
    │       └─► Δκ, α, ΔH + confidence intervals
    │
    ├─► MetricsService.compute_semantic_coherence()
    │       └─► autocorrelation, variance, pattern
    │
    ├─► MetricsService.detect_coupling_mode()
    │       └─► mode, trajectory, risk, compound label
    │
    └─► SemanticClimateAnalyzer.compute_coupling_coefficient()
            ├─► compute_semantic_substrate()
            ├─► compute_temporal_substrate()
            ├─► compute_affective_substrate()
            ├─► compute_dialogue_context()
            └─► detect_attractor_basin()
    │
    ▼
WebSocket → Frontend              # Real-time visualization
```

### Batch Analysis

```
JSON conversation files
    │
    ▼
batch_analyze_conversations.py
    │
    ├─► extract_turns()           # Parse conversation structure
    │
    ├─► EmbeddingService.embed_batch()
    │
    ├─► Windowed analysis loop:
    │       for each window:
    │           ├─► MetricsService.analyze(
    │           │       embeddings, turn_texts, turn_speakers,
    │           │       trajectory_metrics, coherence_pattern
    │           │   )
    │           └─► Accumulate trajectory_metrics
    │
    └─► Final full-conversation analysis
    │
    ▼
Analysis JSON + Summary Report
```

## Key Concepts

### The Three Core Metrics (Morgoulis, with Critical Fixes)

All three Morgoulis metrics required mathematical fixes (2025-12-08). The original implementations had gaps between what they claimed to measure and what they computed.

| Metric | Symbol | Measures | Original Problem | Fix |
|--------|--------|----------|------------------|-----|
| Semantic Curvature | Δκ | Trajectory bending | Chord deviation (circle→0) | Local Frenet-Serret |
| Fractal Similarity | α | Temporal self-organization | DFA on norms (~constant) | DFA on semantic velocity |
| Entropy Shift | ΔH | Semantic reorganization | Independent clustering | Shared clustering + JS |

**Ranges (after fix):**
- **Δκ:** [0, ~2] — Linear=0, high bending→1+
- **α:** [0.5, 1.5] — Noise=0.5, structured=0.7-0.9, drift=1.5
- **ΔH:** [0, 1] — Identical=0, reorganized=1

See [01-CORE-METRICS.md](01-CORE-METRICS.md) for full details on each fix.

### The Ψ Vector (Extensions)

Four-dimensional phase space position:

| Substrate | Symbol | Source | Range |
|-----------|--------|--------|-------|
| Semantic | Ψ_sem | PCA projection of Δκ, ΔH, α | [-1, 1] |
| Temporal | Ψ_temp | Metric stability over windows | [0, 1] |
| Affective | Ψ_aff | Sentiment + hedging + vulnerability | [-1, 1] |
| Biosignal | Ψ_bio | HR/HRV from EBS (optional) | [-1, 1] |

### Attractor Basins (9 canonical configurations)

| Basin | Characteristics |
|-------|-----------------|
| **Cognitive Mimicry** | High semantic, low affect, AI dominates, smooth Δκ |
| **Collaborative Inquiry** | High semantic, low affect, balanced turns, hedging present |
| **Reflexive Performance** | High semantic, performed uncertainty, scripted oscillation |
| **Sycophantic Convergence** | High alignment, low Δκ, low affect |
| **Creative Dilation** | High Δκ, high affect, divergent exploration |
| **Generative Conflict** | High divergent semantic, high affect, productive tension |
| **Deep Resonance** | All substrates high |
| **Embodied Coherence** | Low semantic, high biosignal |
| **Dissociation** | All substrates low |

### Coupling Modes (Trajectory-aware)

Modes combine position (where in metric space) with direction (trajectory):

- **Sycophantic** → Sycophantic-Progressive / Sycophantic-Regressive
- **Resonant**, **Generative**, **Contemplative**, **Emergent**, **Dialectical**
- **Chaotic**, **Dissociative**, **Liminal**, **Transitional**, **Exploratory**

## Epistemic Risk Assessment

Risk levels based on mode + trajectory + coherence:

| Risk Level | Indicators |
|------------|------------|
| **Critical** | Regressive sycophancy + minimal cognitive engagement |
| **High** | Complexity collapse, semantic fragmentation |
| **Moderate** | Progressive sycophancy, conceptual instability |
| **Low** | Healthy modes with stable/warming trajectories |

## Methodological Note

The mathematical fixes to the Morgoulis metrics (2025-12-08) changed how results should be interpreted:

1. **Past α values were noise** — any analyses using original implementation need revision
2. **Interpretive shift** — e.g., what appeared as "semantic fragmentation" (α~0.5) may actually be "semantic lock-in" (α~1.0)
3. **Validation** — all fixes validated against synthetic data with known properties

**Key insight:** The body-as-instrument (biosignal response) often detected patterns the broken metrics couldn't see. The fixes align the metrics with what the somatic system was already detecting.

## Version Tracking

Session exports include module-level versions for research reproducibility. This allows determining exactly which implementations produced results and whether sessions need re-analysis.

```python
from src import get_versions_dict, needs_reanalysis

# Current versions embedded in every export
versions = get_versions_dict()
# {'core_metrics': '1.1.0', 'extensions': '1.1.0', 'basin_detection': '2.0.0', ...}

# Check if old session needs re-analysis
if needs_reanalysis(session["metadata"].get("versions", {})):
    # Re-run with current implementations
```

See [06-VERSIONING.md](06-VERSIONING.md) for full details.

---

## License & Attribution

- **Core metrics (Δκ, α, ΔH)**: Morgoulis (2025), MIT License — with fixes
- **Extensions (Ψ, basins, trajectories)**: ESL-A (Earthian Stewardship License)
- **Web application**: ESL-A

Commercial use requires permission. Research/education use permitted.
