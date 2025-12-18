# API Reference

## Quick Start

```python
from src import (
    SemanticComplexityAnalyzer,  # Core metrics (Morgoulis)
    SemanticClimateAnalyzer,     # Extended analysis (Ψ, basins)
    TrajectoryBuffer,            # Trajectory storage
    # Function API
    semantic_curvature,
    dfa_alpha,
    entropy_shift,
)
```

---

## Class API

### SemanticComplexityAnalyzer

Core metrics computation (Morgoulis 2025).

```python
class SemanticComplexityAnalyzer:
    def __init__(
        self,
        random_state: int = 42,
        bootstrap_iterations: int = 1000
    )
```

#### Methods

##### calculate_all_metrics

```python
def calculate_all_metrics(
    self,
    embedding_sequence: list
) -> dict
```

Compute all three core metrics with confidence intervals.

**Parameters:**
- `embedding_sequence`: List of embedding vectors (L2-normalized or not)

**Returns:**
```python
{
    'delta_kappa': float,
    'delta_kappa_ci': tuple[float, float],
    'alpha': float,
    'alpha_ci': tuple[float, float],
    'delta_h': float,
    'delta_h_ci': tuple[float, float],
    'summary': {
        'cognitive_complexity_detected': bool,
        'healthy_fractal_structure': bool,
        'significant_reorganization': bool
    }
}
```

##### semantic_curvature_enhanced

```python
def semantic_curvature_enhanced(
    self,
    embedding_sequence: list
) -> dict
```

Compute semantic curvature via local Frenet-Serret.

**Returns:**
```python
{
    'mean_curvature': float,
    'std_curvature': float,
    'local_curvatures': list[float],
    'total_arc_length': float
}
```

##### calculate_dfa_alpha

```python
def calculate_dfa_alpha(
    self,
    embedding_sequence: list
) -> dict
```

Compute DFA exponent on semantic velocity.

**Returns:**
```python
{
    'alpha': float,
    'fluctuations': list[float],
    'window_sizes': list[int],
    'r_squared': float
}
```

##### entropy_shift_comprehensive

```python
def entropy_shift_comprehensive(
    self,
    embedding_sequence: list
) -> dict
```

Compute entropy shift via JS divergence on shared clustering.

**Returns:**
```python
{
    'optimal_clusters': int,
    'first_half_entropy': float,
    'second_half_entropy': float,
    'entropy_shift': float,
    'cluster_distributions': {
        'first_half': list[float],
        'second_half': list[float]
    }
}
```

---

### SemanticClimateAnalyzer

Extended analysis with Ψ vector and attractor basins. Inherits from `SemanticComplexityAnalyzer`.

```python
class SemanticClimateAnalyzer(SemanticComplexityAnalyzer):
    def __init__(
        self,
        random_state: int = 42,
        bootstrap_iterations: int = 1000
    )
```

#### Methods

##### compute_coupling_coefficient

```python
def compute_coupling_coefficient(
    self,
    dialogue_embeddings: np.ndarray = None,
    turn_texts: list[str] = None,
    turn_speakers: list[str] = None,
    timestamps: list[float] = None,
    metrics: dict = None,
    trajectory_history: list[dict] = None,
    trajectory_metrics: list[dict] = None,
    biosignal_data: dict = None,
    temporal_window_size: int = 10,
    coherence_pattern: str = None
) -> dict
```

Full 4D coupling analysis.

**Parameters:**
- `dialogue_embeddings`: Embedding sequence (required if metrics not provided)
- `turn_texts`: Turn text strings for affective substrate
- `turn_speakers`: Speaker labels ('human', 'ai') for turn asymmetry
- `timestamps`: Turn timestamps for temporal synchrony
- `metrics`: Pre-computed metrics dict (optional)
- `trajectory_history`: Past Ψ vectors for dynamics
- `trajectory_metrics`: Per-window metrics for Δκ variance
- `biosignal_data`: Dict with 'heart_rate' etc.
- `temporal_window_size`: Window size for stability calculation
- `coherence_pattern`: Pre-computed coherence pattern

**Returns:** See [02-EXTENSIONS.md](02-EXTENSIONS.md#main-entry-point)

##### compute_dialogue_context

```python
def compute_dialogue_context(
    self,
    turn_texts: list[str] = None,
    turn_speakers: list[str] = None,
    trajectory_metrics: list[dict] = None,
    coherence_pattern: str = None,
    hedging_density: float = 0.0
) -> dict
```

Compute discriminating features for refined basin detection.

**Returns:**
```python
{
    'hedging_density': float,
    'turn_length_ratio': float,
    'delta_kappa_variance': float,
    'coherence_pattern': str
}
```

##### detect_attractor_basin

```python
def detect_attractor_basin(
    self,
    psi_vector: dict,
    raw_metrics: dict = None,
    dialogue_context: dict = None
) -> tuple[str, float]
```

Classify phase-space position into attractor basin.

**Parameters:**
- `psi_vector`: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
- `raw_metrics`: Dict with delta_kappa, delta_h, alpha
- `dialogue_context`: Dict from compute_dialogue_context()

**Returns:** `(basin_name, confidence)`

##### compute_semantic_substrate

```python
def compute_semantic_substrate(
    self,
    embeddings: np.ndarray,
    metrics: dict
) -> dict
```

**Returns:**
```python
{
    'psi_semantic': float,
    'alignment_score': float or None,
    'exploratory_depth': float,
    'raw_metrics': dict
}
```

##### compute_temporal_substrate

```python
def compute_temporal_substrate(
    self,
    embeddings: np.ndarray,
    timestamps: list[float] = None,
    metrics: dict = None,
    temporal_window_size: int = 10
) -> dict
```

**Returns:**
```python
{
    'psi_temporal': float,
    'metric_stability': float,
    'turn_synchrony': float or None,
    'rhythm_score': float or None
}
```

##### compute_affective_substrate

```python
def compute_affective_substrate(
    self,
    turn_texts: list[str],
    embeddings: np.ndarray = None
) -> dict
```

**Returns:**
```python
{
    'psi_affective': float,
    'sentiment_trajectory': list[float],
    'hedging_density': float,
    'vulnerability_score': float,
    'confidence_variance': float
}
```

---

### TrajectoryBuffer

Windowed storage for Ψ trajectory derivatives.

```python
class TrajectoryBuffer:
    def __init__(
        self,
        window_size: int = 50,
        timestep: float = 1.0
    )
```

#### Methods

##### append

```python
def append(
    self,
    psi_vector: dict,
    timestamp: float = None
) -> None
```

Add Ψ observation. Auto-manages window size.

##### compute_velocity

```python
def compute_velocity(self) -> dict or None
```

Central difference dΨ/dt. Returns None if < 3 observations.

##### compute_acceleration

```python
def compute_acceleration(self) -> dict or None
```

Second-order finite difference d²Ψ/dt². Returns None if < 4 observations.

---

## Function API

Lightweight functions for testing and research.

### semantic_curvature

```python
def semantic_curvature(
    embeddings: list,
    method: str = 'frenet'
) -> float
```

Returns mean curvature value only (no CI).

### semantic_curvature_ci

```python
def semantic_curvature_ci(
    embeddings: list,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float, float]
```

Returns `(mean, ci_low, ci_high)`.

### dfa_alpha

```python
def dfa_alpha(
    embeddings: list,
    min_window: int = 4,
    max_window: int = None
) -> float
```

Returns DFA exponent.

### entropy_shift

```python
def entropy_shift(
    embeddings: list,
    n_clusters: int = None
) -> float
```

Returns JS divergence value.

---

## Statistical Utilities

### icc_oneway_random

```python
def icc_oneway_random(
    ratings: np.ndarray
) -> float
```

Intraclass correlation coefficient (ICC 1,1).

### icc_bootstrap_ci

```python
def icc_bootstrap_ci(
    ratings: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float, float]
```

ICC with bootstrap confidence interval.

### bland_altman

```python
def bland_altman(
    method1: np.ndarray,
    method2: np.ndarray
) -> dict
```

Bland-Altman analysis for method comparison.

**Returns:**
```python
{
    'mean_diff': float,
    'std_diff': float,
    'upper_loa': float,
    'lower_loa': float
}
```

### cosine_sim

```python
def cosine_sim(
    a: np.ndarray,
    b: np.ndarray
) -> float
```

Cosine similarity between two vectors.

### bootstrap_ci

```python
def bootstrap_ci(
    data: np.ndarray,
    statistic: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple[float, float]
```

Generic bootstrap confidence interval.

---

## MetricsService (Web App)

Service wrapper for web application.

```python
from semantic_climate_app.backend.metrics_service import MetricsService

service = MetricsService()
```

### analyze

```python
def analyze(
    self,
    embeddings: list,
    turn_texts: list = None,
    turn_speakers: list = None,
    previous_metrics: dict = None,
    semantic_shifts: list = None,
    biosignal_data: dict = None,
    trajectory_metrics: list = None,
    coherence_pattern: str = None
) -> dict
```

Full analysis with coupling mode detection.

**Returns:** See [03-WEB-APP.md](03-WEB-APP.md#metricsservicepy--coupling-analysis)

### compute_semantic_coherence

```python
def compute_semantic_coherence(
    self,
    semantic_shifts: list
) -> SemanticCoherence
```

**Returns dataclass:**
```python
@dataclass
class SemanticCoherence:
    autocorrelation: float
    variance: float
    pattern: str  # 'breathing', 'transitional', 'locked', 'fragmented'
    coherence_score: float
```

### detect_coupling_mode

```python
def detect_coupling_mode(
    self,
    dk: float,
    dh: float,
    alpha: float,
    dk_trend: float = 0.0,
    dh_trend: float = 0.0,
    alpha_trend: float = 0.0,
    coherence: SemanticCoherence = None
) -> CouplingMode
```

**Returns dataclass:**
```python
@dataclass
class CouplingMode:
    mode: str
    trajectory: str
    compound_label: str
    epistemic_risk: str
    risk_factors: list
    confidence: float
    coherence: SemanticCoherence = None
```

---

## Type Hints Summary

```python
# Embeddings
Embedding = np.ndarray  # Shape: (dim,) typically 768 or 384
EmbeddingSequence = list[Embedding] | np.ndarray  # Shape: (n_turns, dim)

# Ψ Vector
PsiVector = dict[str, float | None]  # psi_semantic, psi_temporal, psi_affective, psi_biosignal

# Metrics
RawMetrics = dict[str, float]  # delta_kappa, delta_h, alpha

# Dialogue Context
DialogueContext = dict[str, float | str]  # hedging_density, turn_length_ratio, delta_kappa_variance, coherence_pattern

# Basin Detection
BasinResult = tuple[str, float]  # (basin_name, confidence)
```
