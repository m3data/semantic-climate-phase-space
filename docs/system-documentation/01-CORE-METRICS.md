# Core Metrics Module

**File:** `src/core_metrics.py`
**Original Author:** Daria Morgoulis (2025)
**License:** MIT
**Source:** https://github.com/daryamorgoulis/4d-semantic-coupling

## Overview

The core metrics module implements three operational metrics for measuring cognitive complexity in AI dialogue systems. These metrics form the foundation of the Semantic Climate model.

## Critical Fixes (2025-12-08)

**All three Morgoulis metrics required mathematical fixes.** The original implementations had significant gaps between what the metrics claimed to measure and what they actually computed:

| Metric | Original Problem | Fix Applied |
|--------|------------------|-------------|
| **Δκ** | Chord deviation from linear interpolation (not curvature) | Local Frenet-Serret curvature |
| **α** | Detrended Fluctuation Analysis (DFA) on embedding norms (~constant for normalized embeddings) | DFA on semantic velocity |
| **ΔH** | Independent clustering (cluster IDs don't correspond) | Shared clustering + JS divergence |

**Full Details:**
- [Appendix A1: Mathematical Review](A1-METRICS-MATHEMATICAL-REVIEW.md) — Complete analysis of original problems
- [Appendix A2: Fix Validation](A2-METRIC-FIX-VALIDATION.md) — Test cases and validation results

## Class: SemanticComplexityAnalyzer

### Initialization

```python
from src import SemanticComplexityAnalyzer

analyzer = SemanticComplexityAnalyzer(
    random_state=42,           # Reproducibility
    bootstrap_iterations=1000  # CI estimation
)
```

### Empirical Thresholds

```python
analyzer.thresholds = {
    'delta_kappa': 0.35,  # Cognitive complexity threshold
    'alpha_min': 0.70,    # Healthy complexity lower bound
    'alpha_max': 0.90,    # Healthy complexity upper bound
    'delta_h': 0.12       # Entropy reorganization threshold
}
```

---

## Metric 1: Semantic Curvature (Δκ)

### What It Measures

Semantic curvature quantifies how much the dialogue trajectory **bends** in embedding space. High curvature indicates non-linear exploration; low curvature indicates linear, predictable movement.

### The Original Problem

The Morgoulis implementation measured **chord deviation** — how far points deviate from a linear interpolation connecting start to end points:

```
Original: deviation from line(start → end)
```

**Critical issues:**
1. **Endpoint dependency:** A circular trajectory returning to start yields Δκ ≈ 0 despite maximum curvature
2. **Not geodesic:** Linear interpolation in ambient space ≠ shortest path on manifold
3. **No local information:** Measures global shape, not actual bending

### The Fix: Local Frenet-Serret Curvature

**Fixed implementation** uses discrete differential geometry:

```
Given embeddings e(t), e(t+1), e(t+2):

Velocity:     v(t) = e(t+1) - e(t)
Acceleration: a(t) = v(t+1) - v(t)

Local curvature: κ(t) = ||a_perp|| / ||v||²

where a_perp = a - (a·v̂)v̂  (perpendicular component of acceleration)
```

This computes **actual trajectory bending** independent of endpoints.

### Validation Results

| Trajectory Type | Original Δκ | Fixed Δκ |
|-----------------|-------------|----------|
| Linear (straight line) | ~0.0 | 0.0 ✓ |
| Random walk | variable | ~0.03 |
| Sine wave | variable | ~0.46 |
| Circle | **~0.0** ✗ | **~0.99** ✓ |

The circle case demonstrates the fix: original gave ~0 (wrong), fixed gives ~1 (correct).

### Method

```python
result = analyzer.semantic_curvature_enhanced(embedding_sequence)
```

**Returns:**
```python
{
    'mean_curvature': float,      # Average κ across trajectory
    'std_curvature': float,       # Curvature variability
    'local_curvatures': [float],  # Per-point κ values
    'total_arc_length': float     # Path length in embedding space
}
```

### Interpretation

| Δκ Value | Interpretation |
|----------|----------------|
| < 0.20 | Minimal exploration, linear dialogue |
| 0.20 - 0.35 | Moderate complexity |
| 0.35 - 0.65 | Healthy cognitive engagement |
| > 0.65 | High complexity / bending |

---

## Metric 2: Fractal Similarity Score (α)

### What It Measures

α quantifies the **temporal self-organization** of the dialogue using Detrended Fluctuation Analysis (DFA). It distinguishes:
- Random noise (α ≈ 0.5)
- Structured complexity (0.7 < α < 0.9)
- Brownian motion / drift (α ≈ 1.5)

### The Original Problem

The Morgoulis implementation computed DFA on **embedding norms** (`np.linalg.norm(emb)`):

```python
# Original (BROKEN):
signal = [np.linalg.norm(emb) for emb in embeddings]
```

**Critical issue:** For L2-normalized embeddings (standard practice), all norms ≈ 1.0. The DFA was analyzing an essentially constant signal — **the metric was meaningless**.

This was the most severe bug: any α values from the original implementation were noise.

### The Fix: DFA on Semantic Velocity

**Fixed implementation** applies DFA to **semantic velocity** — the cosine distance between consecutive turns:

```python
# Fixed:
signal = [1 - cosine_sim(emb[i], emb[i+1]) for i in range(len(emb)-1)]
```

This captures actual **movement through meaning space**.

### Mathematical Definition

```
1. Compute semantic velocity: v(t) = 1 - cos_sim(e(t), e(t+1))
2. Integrate: y(t) = Σ[v(i) - mean(v)] for i=1 to t
3. Divide into windows of size n
4. Fit trend in each window, compute fluctuation F(n)
5. α = slope of log(F) vs log(n)
```

### Validation Results

| Trajectory Type | Original α | Fixed α |
|-----------------|------------|---------|
| Smooth trajectory | ~0.5 (noise) | ~1.3 (long-range correlation) ✓ |
| Random trajectory | ~0.5 (noise) | ~0.67 (near white noise) ✓ |

### Interpretive Implications

The fix changed interpretation of past analyses. Example from Grunch denial session:

| Interpretation | Original | Corrected |
|----------------|----------|-----------|
| α value | ~0.5 | ~0.87-1.27 |
| Meaning | "Semantic fragmentation" | "Semantic lock-in" |
| Pattern | Random, chaotic | Trapped in predictable loop |

The body's biosignal response (HR rising, chest tightening) was detecting **attractor trap dynamics** that the broken metric couldn't see.

### Method

```python
result = analyzer.calculate_dfa_alpha(embedding_sequence)
```

**Returns:**
```python
{
    'alpha': float,              # DFA exponent
    'fluctuations': [float],     # F(n) values
    'window_sizes': [int],       # n values
    'r_squared': float           # Fit quality
}
```

### Interpretation

| α Value | Interpretation |
|---------|----------------|
| 0.5 | Uncorrelated noise (random) |
| 0.5 - 0.7 | Anti-persistent (choppy) |
| 0.7 - 0.9 | Healthy self-organization |
| 0.9 - 1.0 | Long-range correlations |
| > 1.0 | Non-stationary / drift |

**Boundary condition:** α = 0.5 returned when insufficient data for DFA (< 20 turns). This indicates "Exploratory" mode, not pathology.

---

## Metric 3: Entropy Shift (ΔH)

### What It Measures

ΔH quantifies **semantic reorganization** between dialogue halves. It measures how the distribution of meaning clusters changes from first half to second half.

### The Original Problem

The Morgoulis implementation clustered each half **independently**:

```python
# Original (BROKEN):
clusters_pre = KMeans(k).fit(first_half)
clusters_post = KMeans(k).fit(second_half)
# Compare distributions...
```

**Critical issue:** Cluster IDs are arbitrary. "Cluster 1" in the first half has no correspondence to "Cluster 1" in the second half. Comparing `P(cluster_1)` vs `Q(cluster_1)` was **apples to oranges**.

The metric measured diversity difference, not actual reorganization.

### The Fix: Shared Clustering + Jensen-Shannon Divergence

**Fixed implementation** uses a single clustering model across the full trajectory:

```python
# Fixed:
all_embeddings = first_half + second_half
clusters = KMeans(k).fit(all_embeddings)  # Shared cluster vocabulary

# Now cluster IDs correspond across halves
P = distribution(clusters[:len(first_half)])
Q = distribution(clusters[len(first_half):])
ΔH = jensen_shannon_divergence(P, Q)
```

### Mathematical Definition

```
1. Split trajectory into first/second halves
2. Cluster FULL trajectory (shared vocabulary)
3. Compute cluster distribution P (first half) and Q (second half)
4. ΔH = JS_divergence(P, Q) = 0.5 * [KL(P||M) + KL(Q||M)]
   where M = 0.5 * (P + Q)
```

**Why Jensen-Shannon:** JS divergence is symmetric (unlike KL) and bounded [0, 1], making it interpretable.

### Validation Results

| Scenario | Original ΔH | Fixed ΔH |
|----------|-------------|----------|
| Identical distributions | variable | 0.0 ✓ |
| Different random samples | variable | ~0.16 |
| Major semantic shift | variable | ~1.0 ✓ |

The fix ensures ΔH = 0 when halves have identical semantic structure, and ΔH → 1 for complete reorganization.

### Method

```python
result = analyzer.entropy_shift_comprehensive(embedding_sequence)
```

**Returns:**
```python
{
    'optimal_clusters': int,      # Data-driven k
    'first_half_entropy': float,  # H(P)
    'second_half_entropy': float, # H(Q)
    'entropy_shift': float,       # JS divergence
    'cluster_distributions': {    # Detailed breakdown
        'first_half': [float],
        'second_half': [float]
    }
}
```

### Interpretation

| ΔH Value | Interpretation |
|----------|----------------|
| < 0.08 | Stable semantic structure |
| 0.08 - 0.12 | Moderate reorganization |
| 0.12 - 0.20 | Significant restructuring |
| > 0.20 | Major semantic shift |

---

## Combined Analysis

### Main Method

```python
results = analyzer.calculate_all_metrics(embeddings)
```

**Returns:**
```python
{
    'delta_kappa': float,
    'delta_kappa_ci': (float, float),  # 95% CI
    'alpha': float,
    'alpha_ci': (float, float),
    'delta_h': float,
    'delta_h_ci': (float, float),
    'summary': {
        'cognitive_complexity_detected': bool,  # Δκ > threshold
        'healthy_fractal_structure': bool,      # α in range
        'significant_reorganization': bool      # ΔH > threshold
    }
}
```

### Bootstrap Confidence Intervals

All metrics include 95% bootstrap confidence intervals. The bootstrap resamples turns (with replacement) and recomputes metrics to estimate uncertainty.

```python
# Access CI
dk_low, dk_high = results['delta_kappa_ci']
```

---

## Usage Example

```python
import numpy as np
from src import SemanticComplexityAnalyzer

# Create analyzer
analyzer = SemanticComplexityAnalyzer()

# Generate/load embeddings (list of 768-dim vectors)
embeddings = [np.random.randn(768) for _ in range(30)]

# Calculate all metrics
results = analyzer.calculate_all_metrics(embeddings)

print(f"Δκ = {results['delta_kappa']:.3f} {results['delta_kappa_ci']}")
print(f"α  = {results['alpha']:.3f} {results['alpha_ci']}")
print(f"ΔH = {results['delta_h']:.3f} {results['delta_h_ci']}")
print(f"Complexity detected: {results['summary']['cognitive_complexity_detected']}")
```

---

## Implementation Notes

### Minimum Turn Requirements

- **Δκ:** Requires ≥3 embeddings (for velocity + acceleration)
- **α:** Requires ≥20 embeddings for meaningful DFA (returns 0.5 boundary otherwise)
- **ΔH:** Requires ≥6 embeddings (≥3 per half for clustering)

### Numerical Stability

- Division by zero protected with ε = 1e-10
- NaN values handled with fallbacks
- L2 normalization applied internally where needed

### Performance Considerations

- Bootstrap CI adds ~1000× computation per metric
- DFA window scaling is O(n log n)
- Clustering is O(n × k × iterations)

For real-time use, consider reducing `bootstrap_iterations` or caching results.
