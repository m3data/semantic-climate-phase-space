# Semantic Climate Phase Space

**It matters less about where we are, and more about how we move together.**

A dynamical-systems framework for studying how relational meaning unfolds in dialogue.  
Rather than treating mind as an internal system or reducing cognition to measurable substrates, this project analyses *semantic movement* and other observable traces as partial expressions of a wider ecology of sense-making. It incorporates the core semantic complexity metrics of Morgoulis (2025) but extends them into a relational phase‑space architecture that supports trajectory analysis, attractor dynamics, and optional semantic–autonomic integration across human–human and human–AI interaction.

## Epistemological Note

> **This system measures traces of experience.**
>
> Semantic metrics are not meaning in themselves.
> Physiological signals are not feeling.
> Dynamical signatures are not cognition.
>
> These measurements exist to *complement*, not replace, the lived, felt, relational nature of mind.
>
> First-person phenomenology and third-person quantification are different ways of knowing the same territory. They must mutually constrain one another. The moment we mistake the map for the territory—reducing experience to its measurable traces—we lose the thing we set out to understand.
>
> Use this framework with that awareness.

---

## Overview

This framework treats dialogue as a dynamical system, measuring how meaning moves through semantic space. Three core metrics capture different aspects of coupling quality:

| Metric | Symbol | What It Measures |
|--------|--------|------------------|
| **Semantic Curvature** | Δκ | Trajectory non-linearity (inferential complexity) |
| **Fractal Similarity** | α | Self-organising patterns via Detrended Fluctuation Analysis (coherence) |
| **Entropy Shift** | ΔH | Semantic reorganisation (conceptual diversity) |

Extensions in this project add:

- **Vector Ψ** - 4D phase-space representation (semantic, temporal, affective, biosignal)
- **Attractor Basin Detection** - 10 canonical dialogue configurations
- **Trajectory Dynamics** - Velocity, acceleration, curvature of meaning movement
- **Real-time Web Application** - Live semantic coupling analysis

## Research Novelty

Most prior work in semantic complexity and dialogue analysis treats conversational embeddings as **static sequences**, producing scalar descriptors rather than modelling the unfolding relational process. This project reframes those metrics within a **relational dynamical‑systems ontology**, treating semantic and autonomic trajectories as complementary, partial indicators of an ongoing ecology of sense‑making. The framework supports real‑time analysis of how meaning moves through multi‑dimensional phase‑space across human–human and human–AI interactions without reducing cognition to these signals.

Specifically, this framework introduces:

### 1. **Semantic Phase Space (Vector Ψ)**

The core innovation is the construction of a 4D semantic–temporal–affective–biosignal manifold. Scalar metrics from Morgoulis (2025) become **state variables**, allowing trajectory-based analysis rather than snapshot metrics.

### 2. **Trajectory Dynamics**

Meaning is treated as motion:

- velocity of semantic change  
- acceleration and curvature  
- stability and divergence  
- local derivatives via windowed buffers  

This shifts the epistemic framing from descriptive statistics to **differential geometry of meaning-making**.

### 3. **Attractor Basin Framework**

Dialogue is modelled as a dynamical system with identifiable attractors (e.g., generative flow, vigilant engagement, semantic drift). These are defined not only by metric values but by **movement patterns in Ψ-space**, enabling state-transition analysis.

### 4. **Semantic–Autonomic Coupling (Optional Integration)**

When integrated with EarthianBioSense, the system supports simultaneous modelling of:

- semantic trajectories in dialogue  
- autonomic trajectories (HRV-derived phase space)  

This enables the first computational framework capable of **joint modelling of semantic and somatic phase-space dynamics within human–AI interaction**.

### 5. **Real-Time Instrumentation**

Unlike prior post-hoc analytic approaches, this project supports:

- live state estimation  
- streaming trajectory visualization  
- attractor detection in real time  

This makes the system suitable for empirical research, experimental protocols, and embodied cognition studies involving human–AI coupling and embodied interaction human-to-human.

### Summary

This project builds beyond Morgoulis’s complexity metrics and constitutes a **new methodological frame**:

- embeddings become trajectories  
- metrics become dynamical substrates  
- conversation becomes a phase-space system  
- human–AI dialogue becomes a coupled dynamical ecology  

Researchers can now interrogate meaning-making, coherence formation, destabilization, and recovery with tools drawn from dynamical systems, cognitive science, and embodied interaction.

## Installation

```bash
git clone https://github.com/m3data/semantic-climate-phase-space.git
cd semantic-climate-phase-space
pip install -r semantic_climate_app/requirements.txt
```

### Computational Requirements

This framework uses `sentence-transformers` with `all-mpnet-base-v2` for embedding generation. Be aware:

- **PyTorch dependency** (~2GB download on first install)
- **Model download** (~400MB on first run)
- **GPU recommended** for reasonable embedding speed (CPU works but slower)
- **Memory:** 4GB+ RAM recommended for embedding operations

For local LLM inference via Ollama, additional resources required:
- **7B models:** ~8GB RAM/VRAM
- **70B models:** ~40GB+ RAM/VRAM (or quantized versions)

**Core dependencies:** numpy, scipy, scikit-learn, pandas, sentence-transformers, torch, fastapi

**Optional:** vaderSentiment (affective substrate), anthropic, openai (cloud providers)

## Quick Start

### Basic Usage

```python
from src import SemanticComplexityAnalyzer
import numpy as np

# Initialize analyzer (Morgoulis core metrics)
analyzer = SemanticComplexityAnalyzer()

# Generate embeddings for your dialogue turns
# (use sentence-transformers, OpenAI embeddings, etc.)
embeddings = [np.random.randn(384) for _ in range(20)]

# Calculate all metrics
results = analyzer.calculate_all_metrics(embeddings)

print(f"Semantic Curvature (Δκ): {results['delta_kappa']:.4f}")
print(f"Fractal Similarity (α):  {results['alpha']:.4f}")
print(f"Entropy Shift (ΔH):      {results['delta_h']:.4f}")
print(f"Complexity Detected:     {results['summary']['cognitive_complexity_detected']}")
```

### Extended Analysis with Vector Ψ

```python
from src import SemanticClimateAnalyzer
import numpy as np

# Extended analyzer with Vector Ψ and attractor detection
analyzer = SemanticClimateAnalyzer()

embeddings = np.array([np.random.randn(384) for _ in range(20)])
turn_texts = ["Hello, how are you?", "I'm doing well, thanks!"] * 10

result = analyzer.compute_coupling_coefficient(
    dialogue_embeddings=embeddings,
    turn_texts=turn_texts
)

print(f"Coupling Coefficient (Ψ): {result['psi']:.4f}")
print(f"Attractor Basin: {result['attractor_dynamics']['basin']}")
print(f"Confidence: {result['attractor_dynamics']['confidence']:.2%}")
```

### Function API (for testing/research)

```python
from src import semantic_curvature, dfa_alpha, entropy_shift

# Simple function calls without class instantiation
dk = semantic_curvature(embeddings)
alpha = dfa_alpha(similarity_series)
dh = entropy_shift(embeddings)
```

## Web Application

Real-time semantic coupling analysis with multi-provider LLM support.

### Supported LLM Providers

| Provider | Type | Latency | Setup |
|----------|------|---------|-------|
| **Together AI** | Cloud | 2-5s | Set `TOGETHER_API_KEY` env var |
| **Ollama** | Local | 30-120s | `ollama serve` + pull model |
| **Anthropic** | Cloud | 3-8s | Set `ANTHROPIC_API_KEY` env var |
| **OpenAI** | Cloud | 3-8s | Set `OPENAI_API_KEY` env var |

**Note:** Cloud providers (Together AI recommended) significantly reduce latency confounds in semantic coupling measurements. Local inference via Ollama provides privacy but introduces variable response times.

### Running the Web App

```bash
# Configure API keys (copy and edit)
cp semantic_climate_app/.env.example semantic_climate_app/.env

# Run web app
cd semantic_climate_app
python backend/main.py

# Access at http://127.0.0.1:8000
```

For local-only operation with Ollama:
```bash
ollama serve
ollama pull llama3.2
```

### Features

- Multi-provider LLM support with automatic detection
- Live metric gauges (Temperature/Humidity/Pressure metaphor)
- Trajectory-aware coupling mode detection
- Vector Ψ visualization
- Session export to JSON
- Optional EarthianBioSense biosignal integration

## Coupling Modes

The framework classifies dialogue into coupling modes based on metric position and trajectory:

| Mode | Δκ | ΔH | Character |
|------|-----|-----|-----------|
| **Sycophantic** | < 0.20 | < 0.08 | Minimal exploration, compliance |
| **Contemplative** | > 0.70 | < 0.05 | Dense sustained meaning-making |
| **Generative** | 0.50-0.75 | < 0.08 | Building new conceptual ground |
| **Resonant** | 0.35-0.65 | 0.08-0.18 | Co-emergent coupling |
| **Dialectical** | 0.35-0.70 | 0.12-0.25 | Productive tension |
| **Emergent** | > 0.65 | 0.05-0.12 | Complexity developing |
| **Chaotic** | high | > 0.25 | Fragmentation |

Trajectory modifiers: `-Warming`, `-Cooling`, `-Oscillating`, `-Stable`

Important: These are **proto-classifications** and should be interpreted contextually rather with mutual constraints of first person account and third-person observation.

## Attractor Basins

These proto-classifications represent relational configurations rather than mental contents.

Ten canonical configurations in phase-space:

1. **Deep Resonance** - All substrates high, genuine mutual engagement
2. **Collaborative Inquiry** - Genuine co-exploration with mutual uncertainty
3. **Cognitive Mimicry** - Model performs engagement without genuine uncertainty (risk)
4. **Reflexive Performance** - Model appears to self-examine but pattern-matches
5. **Sycophantic Convergence** - High alignment, low curvature, low affect (risk)
6. **Creative Dilation** - Divergent exploration with high affect
7. **Generative Conflict** - Productive tension, high affect
8. **Embodied Coherence** - Low semantic, high biosignal (body leads)
9. **Dissociation** - All substrates low (risk)
10. **Transitional** - No clear basin, between states

## Project Structure

```
semantic-climate-phase-space/
├── src/
│   ├── core_metrics.py     # Morgoulis (2025) - preserved with attribution
│   ├── extensions.py       # Vector Ψ, attractors, trajectory dynamics
│   └── api.py              # Function-based API for testing
├── tests/                  # 244 tests
├── semantic_climate_app/   # Real-time web application
│   ├── backend/            # FastAPI + WebSocket
│   └── frontend/           # Vanilla HTML/CSS/JS
└── docs/                   # Additional documentation
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Expected: 244 passed
```

## Attribution

### Core Metrics - Morgoulis (2025)

The foundational metrics implementation is by **Daria Morgoulis**:

- Semantic Curvature (Δκ) algorithm and empirical thresholds
- DFA-based Fractal Similarity (α) with scaling exponent
- Entropy Shift (ΔH) with consensus clustering
- Bootstrap CI methodology

**Source:** https://github.com/daryamorgoulis/4d-semantic-coupling

### Extensions - This Project

The following are original contributions:

- Vector Ψ representation (4D phase-space)
- Attractor basin framework (7 configurations)
- Trajectory dynamics (velocity, acceleration, curvature)
- Substrate computations (temporal, affective, biosignal)
- Semantic Climate web application
- Function-based testing API

## Citation

If using the core metrics, please cite Morgoulis (2025).

If using the extensions, please cite both:

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

## License

**Extensions (this project):** Earthian Stewardship License (ESL-A) - See LICENSE file.

**Core metrics (Morgoulis):** MIT License - preserved in `src/core_metrics.py` header.

The ESL-A license permits non-commercial use for research, education, and community projects. Commercial use requires explicit permission. See LICENSE for full terms including stewardship obligations and prohibited uses.

## Related Work

- [4D Semantic Coupling Framework](https://github.com/daryamorgoulis/4d-semantic-coupling) - Original metrics
- [EarthianBioSense](https://github.com/m3data/Earthian-BioSense) - Biosignal integration
