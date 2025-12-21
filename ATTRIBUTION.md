# Attribution

This document details the intellectual provenance of code and concepts in the Semantic Climate Phase Space project.

## Core Metrics - Daria Morgoulis (2025)

The foundational metrics implementation is preserved from the [4D Semantic Coupling Framework](https://github.com/daryamorgoulis/4d-semantic-coupling) by Daria Morgoulis.

### What Must Be Attributed to Morgoulis

**Algorithms:**
- Semantic Curvature (Δκ) - Geodesic deviation via linear interpolation
- Fractal Similarity (α) - Detrended Fluctuation Analysis with scaling exponent
- Entropy Shift (ΔH) - Cluster-level entropy via consensus clustering

**Empirical Thresholds:**
- Δκ ≥ 0.35 indicates cognitive complexity
- α ∈ [0.70, 0.90] indicates self-organizing patterns
- ΔH ≥ 0.12 indicates semantic reorganization

**Statistical Methods:**
- Bootstrap confidence intervals (1000 iterations, default)
- Permutation p-values for significance testing

**Location in codebase:**
- `src/core_metrics.py` - Preserved with minimal modification (import path only)

### Citation

```bibtex
@software{morgoulis2025semantic,
  author = {Morgoulis, Daria},
  title = {4D Semantic Coupling Framework},
  year = {2025},
  url = {https://github.com/daryamorgoulis/4d-semantic-coupling}
}
```

---

## Extensions - This Project

The following are original contributions by the Semantic Climate Phase Space project.

### Vector Ψ Representation

A 4-dimensional phase-space vector representing dialogue state:

```
Ψ = [ψ_semantic, ψ_temporal, ψ_affective, ψ_biosignal]
```

Each component is normalized to [-1, +1]:
- **ψ_semantic** - Derived from Δκ and α (using Morgoulis metrics)
- **ψ_temporal** - Response latency dynamics
- **ψ_affective** - Sentiment trajectory via VADER + GoEmotions hybrid (hedging/vulnerability patterns)
- **ψ_biosignal** - Heart rate variability coherence (via EarthianBioSense)

**Location:** `src/analyzer.py:SemanticClimateAnalyzer`, `src/substrates.py`

### Attractor Basin Framework

Ten canonical configurations in phase-space:

1. Deep Resonance - All substrates high, genuine mutual engagement
2. Collaborative Inquiry - Genuine co-exploration with mutual uncertainty
3. Cognitive Mimicry - Model performs engagement without genuine uncertainty (risk)
4. Reflexive Performance - Model appears to self-examine but pattern-matches
5. Sycophantic Convergence - High alignment, low curvature, low affect (risk)
6. Creative Dilation - Divergent exploration with high affect
7. Generative Conflict - Productive tension, high affect
8. Embodied Coherence - Low semantic, high biosignal (body leads)
9. Dissociation - All substrates low (risk)
10. Transitional - No clear basin, between states

**Location:** `src/basins.py:BasinDetector`

### Trajectory Dynamics

Windowed storage and derivative calculation:

- **TrajectoryBuffer** - Circular buffer with configurable window
- **Velocity** (dΨ/dt) - Central difference approximation
- **Acceleration** (d²Ψ/dt²) - Second derivative
- **Curvature** - Rate of directional change

**Location:** `src/trajectory.py:TrajectoryBuffer`

### Trajectory Integrity

Path-dependence analysis measuring coherent relationship between trajectory and its history:

- **IntegrityAnalyzer** - Computes autocorrelation, tortuosity, recurrence
- **TransformationDetector** - Detects constraint-altering events
- Score interpretation: fragmented (0.0-0.3), living (0.3-0.7), rigid (0.7-1.0)

**Location:** `src/integrity.py`

### Coupling Mode Classification

Trajectory-aware mode detection with epistemic risk assessment:

- 8 base modes (Sycophantic, Resonant, Generative, Contemplative, Emergent, Dialectical, Chaotic, Dissociative)
- 2 coherence-derived modes (Liminal, Transitional)
- Trajectory modifiers (Warming, Cooling, Oscillating, Stable)
- Epistemic risk levels (Low, Moderate, High, Critical)

**Location:** `semantic_climate_app/backend/metrics_service.py`

### Semantic Climate Model

Metaphorical interpretation of metrics:
- **Temperature** (Δκ) - Inferential complexity / cognitive energy
- **Humidity** (ΔH) - Conceptual diversity / exploration
- **Pressure** (α) - Fractal coherence / structure

### Web Application

Real-time semantic coupling analysis interface:
- FastAPI + WebSocket backend
- Live metric visualization with trajectory sparklines
- Multi-provider LLM support (Together AI, Anthropic, OpenAI, Ollama)
- GoEmotions-based hybrid affective analysis
- Epistemic/safety trajectory visualization
- Optional EarthianBioSense biosignal coupling

**Location:** `semantic_climate_app/`

### Function-Based API

Lightweight testing interface:
- `semantic_curvature()`, `dfa_alpha()`, `entropy_shift()`
- ICC and Bland-Altman utilities
- Bootstrap CI helpers

**Location:** `src/api.py`

---

## How to Cite

### Using Core Metrics Only

If you use only the Δκ, α, ΔH metrics, cite Morgoulis (2025).

### Using Extensions

If you use Vector Ψ, attractor basins, trajectory dynamics, or the web app, cite both:

```bibtex
@software{morgoulis2025semantic,
  author = {Morgoulis, Daria},
  title = {4D Semantic Coupling Framework},
  year = {2025},
  url = {https://github.com/daryamorgoulis/4d-semantic-coupling}
}

@software{semanticclimate2025,
  author = {Mytka, Mathew},
  title = {Semantic Climate Phase Space},
  year = {2025},
  url = {https://github.com/m3data/semantic-climate-phase-space}
}
```

---

## Licensing

**Core metrics (Morgoulis):** MIT License - preserved in `src/core_metrics.py`

**Extensions (this project):** Earthian Stewardship License (ESL-A)
- Non-commercial use permitted for research, education, community projects
- Commercial use requires explicit permission
- See LICENSE file for stewardship obligations and prohibited uses

Per ESL-A Section 7: "Portions of this repository contain code derived from Morgoulis (2025) under the MIT License. Those files retain their MIT headers; all additions surrounding or extending that code fall under this Earthian Stewardship License."

---

## Lineage Diagram

```
Morgoulis (2025) - 4D Semantic Coupling Framework [MIT License]
├── Semantic Curvature (Δκ)
├── Fractal Similarity (α)
├── Entropy Shift (ΔH)
└── Bootstrap CI / Permutation tests
        │
        ▼ PRESERVED WITH ATTRIBUTION
        │
This Project - Semantic Climate Phase Space [ESL-A License]
├── src/core_metrics.py (Morgoulis, MIT, minimal changes)
├── src/analyzer.py (NEW, ESL-A) - Main orchestrator
├── src/trajectory.py (NEW, ESL-A) - TrajectoryBuffer, geometry
├── src/substrates.py (NEW, ESL-A) - Ψ substrate computations
├── src/basins.py (NEW, ESL-A) - Basin detection, hysteresis
├── src/integrity.py (NEW, ESL-A) - Trajectory integrity
├── src/api.py (NEW, ESL-A) - Function-based testing API
├── src/extensions.py (DEPRECATED) - Legacy monolith
├── semantic_climate_app/ (NEW, ESL-A)
│   ├── backend/ - FastAPI + EmotionService (GoEmotions)
│   └── frontend/ - Trajectory viz, sparklines
└── tests/ (NEW, ESL-A)
    └── 251 validation tests
```
