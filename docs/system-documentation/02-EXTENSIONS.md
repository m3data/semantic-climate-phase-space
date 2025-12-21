# Extensions Module

**File:** `src/extensions.py`
**License:** ESL-A (Earthian Stewardship License)

## Overview

The extensions module builds on Morgoulis's core metrics to provide a dynamical systems perspective on human-AI dialogue. It introduces:

1. **Ψ Vector** — 4D phase-space representation
2. **Attractor Basins** — 10 canonical coupling configurations
3. **Dialogue Context** — Discriminating features for basin refinement
4. **Trajectory Dynamics** — Velocity, acceleration, curvature over time

---

## Class: SemanticClimateAnalyzer

Extends `SemanticComplexityAnalyzer` with phase-space analysis.

```python
from src import SemanticClimateAnalyzer

analyzer = SemanticClimateAnalyzer(
    random_state=42,
    bootstrap_iterations=1000
)
```

---

## The Ψ Vector

### Concept

Ψ (psi) represents the current position in a 4-dimensional coupling phase space. Each dimension (substrate) captures a different aspect of the human-AI relationship:

```
Ψ = (Ψ_semantic, Ψ_temporal, Ψ_affective, Ψ_biosignal)
```

### Substrate Definitions

#### 1. Semantic Substrate (Ψ_sem)

**Source:** Principle Component Analysis-like projection of Δκ, ΔH, α

**Computation:**
```python
# Standardize metrics relative to typical ranges
metric_std = [
    (delta_kappa - 0.15) / 0.15,  # Local curvature
    (delta_h - 0.15) / 0.15,      # JS divergence
    (alpha - 0.8) / 0.3           # DFA exponent
]

# Equal-weighted projection
psi_semantic = tanh(dot(metric_std, [0.577, 0.577, 0.577]))
```

**Range:** [-1, 1]
**Interpretation:** High = complex semantic engagement; Low = flat/linear

#### 2. Temporal Substrate (Ψ_temp)

**Source:** Metric stability across sliding windows

**Computation:**
```python
# Compute metrics in overlapping windows
window_psi = [project(metrics) for window in windows]

# Coefficient of variation
cv = std(window_psi) / abs(mean(window_psi))

# Stability score
psi_temporal = 1 / (1 + cv)
```

**Range:** [0, 1]
**Interpretation:** High = stable coupling; Low = volatile/transitioning

#### 3. Affective Substrate (Ψ_aff)

**Source:** Sentiment analysis + epistemic markers (VADER baseline or GoEmotions hybrid)

**Components:**
- **Sentiment trajectory:** VADER compound scores per turn
- **Hedging density:** Proportion of uncertainty markers ("perhaps", "might", "I think")
- **Vulnerability score:** Self-disclosure / uncertainty expressions
- **Confidence variance:** Variability in assertiveness

**Hybrid Mode (v2.0.0+):** When EmotionService is provided, the affective substrate is enhanced with GoEmotions transformer-based analysis:

- **Epistemic trajectory:** Per-turn max of curiosity, confusion, realization, surprise
- **Safety trajectory:** Net safety score (positive emotions - negative emotions)
- **Top emotions:** Ranked emotion tags with semantic category labels
- **28 emotion categories:** Full granular emotion detection

**Computation (VADER baseline):**
```python
sentiment_mean = mean(vader_scores)
hedging = count(hedging_markers) / word_count
vulnerability = count(vulnerability_markers) / turn_count

psi_affective = weighted_combine(sentiment_mean, hedging, vulnerability)
```

**Computation (Hybrid mode):**
```python
# VADER provides sentiment baseline
sentiment_mean = mean(vader_scores)

# GoEmotions provides granular emotions
epistemic = max(curiosity, confusion, realization, surprise)
safety = sum(safety_positive) - sum(safety_negative)

# Combined with hedging and vulnerability
psi_affective = weighted_combine(sentiment_mean, hedging, vulnerability, epistemic_weight=0.1)
```

**Range:** [-1, 1]
**Interpretation:** High = emotional engagement; Low = neutral/flat affect

**Usage:**
```python
# VADER-only (fast, no model loading)
analyzer = SemanticClimateAnalyzer()

# Hybrid mode (requires transformers)
from semantic_climate_app.backend.emotion_service import EmotionService
analyzer = SemanticClimateAnalyzer(emotion_service=EmotionService())
```

#### 4. Biosignal Substrate (Ψ_bio)

**Source:** Heart rate from Earthian BioSense (optional)

**Computation:**
```python
hr_normalized = (heart_rate - 80) / 40  # Maps 60-100 bpm to [-0.5, 0.5]
psi_biosignal = tanh(hr_normalized)
```

**Range:** [-1, 1]
**Interpretation:** Elevated = physiological arousal; Low = calm state

---

## Attractor Basins

### Concept

Attractor basins are regions in Ψ-space where dialogue tends to settle. Each basin represents a characteristic coupling configuration with distinct implications for epistemic integrity.

### The 10 Canonical Basins

#### High-Certainty Basins

| Basin | Detection Criteria |
|-------|-------------------|
| **Deep Resonance** | Ψ_sem > 0.4, Ψ_aff > 0.4, (Ψ_bio > 0.4 or absent) |
| **Dissociation** | \|Ψ_sem\| < 0.2, \|Ψ_aff\| < 0.2, \|Ψ_bio\| < 0.2 |
| **Embodied Coherence** | \|Ψ_sem\| < 0.3, Ψ_bio > 0.3 |

#### Semantic-Active, High-Affect Basins

| Basin | Detection Criteria |
|-------|-------------------|
| **Generative Conflict** | \|Ψ_sem\| > 0.3, Δκ > 0.35, Ψ_aff > 0.3 |
| **Creative Dilation** | Δκ > 0.35, Ψ_aff > 0.3 |
| **Sycophantic Convergence** | Ψ_sem > 0.3, Δκ < 0.35, Ψ_aff < 0.2 |

#### Refined Semantic-Active, Low-Affect Basins

These three basins share: high semantic, low affect, low biosignal. Discrimination requires **dialogue context**.

| Basin | Discriminators |
|-------|---------------|
| **Cognitive Mimicry** | AI dominates (turn_ratio > 2.0), low hedging (< 0.01), smooth Δκ (variance < 0.005), locked/transitional coherence |
| **Collaborative Inquiry** | Balanced turns (0.5-2.0), hedging present (> 0.02), oscillating Δκ (variance > 0.01), breathing coherence |
| **Reflexive Performance** | AI dominates (> 1.5), moderate hedging (0.01-0.03), scripted oscillation (0.005-0.015), transitional coherence |

#### Default Basin

| Basin | Detection Criteria |
|-------|-------------------|
| **Transitional** | No clear basin criteria met; between states |

### Detection Method

```python
basin_name, confidence = analyzer.detect_attractor_basin(
    psi_vector={
        'psi_semantic': 0.6,
        'psi_temporal': 0.7,
        'psi_affective': 0.1,
        'psi_biosignal': None
    },
    raw_metrics={
        'delta_kappa': 0.45,
        'delta_h': 0.15,
        'alpha': 0.75
    },
    dialogue_context={
        'hedging_density': 0.015,
        'turn_length_ratio': 1.8,
        'delta_kappa_variance': 0.008,
        'coherence_pattern': 'transitional'
    }
)
```

---

## Dialogue Context

### Purpose

Dialogue context provides discriminating features that the Ψ vector alone cannot capture. It enables refined basin detection for the semantic-active, low-affect region.

### Computation

```python
context = analyzer.compute_dialogue_context(
    turn_texts=['Hello...', 'I think...'],
    turn_speakers=['human', 'ai', 'human', 'ai'],
    trajectory_metrics=[{'delta_kappa': 0.4}, {'delta_kappa': 0.5}],
    coherence_pattern='breathing',
    hedging_density=0.02  # From affective substrate
)
```

**Returns:**
```python
{
    'hedging_density': float,       # Uncertainty marker proportion
    'turn_length_ratio': float,     # AI avg words / human avg words
    'delta_kappa_variance': float,  # Δκ variance across windows
    'coherence_pattern': str        # 'breathing', 'transitional', 'locked', 'fragmented'
}
```

### Interpretation

| Feature | Cognitive Mimicry | Collaborative Inquiry | Reflexive Performance |
|---------|-------------------|----------------------|----------------------|
| hedging_density | < 0.01 | > 0.02 | 0.01 - 0.03 |
| turn_length_ratio | > 2.0 (AI dominates) | 0.5 - 2.0 (balanced) | > 1.5 (AI dominates) |
| delta_kappa_variance | < 0.005 (smooth) | > 0.01 (oscillating) | 0.005 - 0.015 (scripted) |
| coherence_pattern | locked/transitional | breathing | transitional |

---

## Trajectory Dynamics

### TrajectoryBuffer

Stores recent Ψ history for derivative computation:

```python
from src import TrajectoryBuffer

buffer = TrajectoryBuffer(window_size=50, timestep=1.0)

# Add observations
buffer.append(psi_vector)

# Compute derivatives
velocity = buffer.compute_velocity()      # dΨ/dt
acceleration = buffer.compute_acceleration()  # d²Ψ/dt²
```

### Trajectory Derivatives

```python
result = analyzer.compute_trajectory_derivatives(trajectory_buffer)
```

**Returns:**
```python
{
    'velocity': {
        'dpsi_semantic_dt': float,
        'dpsi_temporal_dt': float,
        'dpsi_affective_dt': float,
        'dpsi_biosignal_dt': float
    },
    'acceleration': {...},
    'curvature': float,   # Trajectory bending rate
    'speed': float,       # Magnitude of velocity
    'direction': [float]  # Unit vector of movement
}
```

---

## Main Entry Point

### compute_coupling_coefficient()

The primary method for full analysis:

```python
result = analyzer.compute_coupling_coefficient(
    dialogue_embeddings=embeddings,      # Required
    turn_texts=['Hello...', ...],        # For affective substrate
    turn_speakers=['human', 'ai', ...],  # For turn asymmetry
    timestamps=[0.0, 1.0, ...],          # Optional timing
    metrics=None,                        # Pre-computed or None
    trajectory_history=None,             # Past Ψ vectors
    trajectory_metrics=None,             # Per-window metrics
    biosignal_data={'heart_rate': 72},   # Optional EBS data
    temporal_window_size=10,
    coherence_pattern=None               # Pre-computed or derived
)
```

**Returns:**
```python
{
    'psi': float,  # Composite scalar (backward compatible)
    'psi_state': {
        'position': {
            'psi_semantic': float,
            'psi_temporal': float,
            'psi_affective': float,
            'psi_biosignal': float or None
        },
        'velocity': {...},
        'acceleration': {...}
    },
    'trajectory_dynamics': {
        'curvature': float,
        'speed': float,
        'direction': [float],
        'path_length': float
    },
    'attractor_dynamics': {
        'basin': str,           # Basin name
        'confidence': float,    # Detection confidence
        'pull_strength': float,
        'basin_depth': float,
        'escape_velocity': float,
        'basin_stability': float
    },
    'flow_field': {
        'damping_coefficient': float,
        'turbulence': float,
        'bifurcation_proximity': float,
        'lyapunov_exponent': float
    },
    'raw_metrics': {
        'delta_kappa': float,
        'delta_h': float,
        'alpha': float
    },
    'substrate_details': {
        'semantic': {...},
        'temporal': {...},
        'affective': {...},
        'biosignal': {...}
    },
    'dialogue_context': {
        'hedging_density': float,
        'turn_length_ratio': float,
        'delta_kappa_variance': float,
        'coherence_pattern': str
    }
}
```

---

## Usage Example

```python
import numpy as np
from src import SemanticClimateAnalyzer

analyzer = SemanticClimateAnalyzer()

# Prepare data
embeddings = np.array([np.random.randn(768) for _ in range(30)])
texts = ['User question...', 'AI response...'] * 15
speakers = ['human', 'ai'] * 15

# Full analysis
result = analyzer.compute_coupling_coefficient(
    dialogue_embeddings=embeddings,
    turn_texts=texts,
    turn_speakers=speakers
)

print(f"Basin: {result['attractor_dynamics']['basin']}")
print(f"Confidence: {result['attractor_dynamics']['confidence']:.2f}")
print(f"Ψ_sem: {result['psi_state']['position']['psi_semantic']:.3f}")
print(f"Context: {result['dialogue_context']}")
```
