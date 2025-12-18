# Appendix A1: Mathematical Review of Morgoulis Metrics

**Date:** 2025-12-08
**Reviewers:** Mat Mytka, with analysis support from Kairos
**Status:** Complete — fixes applied

## Executive Summary

This appendix documents the mathematical review of the three core metrics from Morgoulis (2025). While these metrics showed empirical utility, the underlying mathematics did not compute what the metric names claimed. All issues have been fixed.

| Metric | Claimed Measurement | Original Computation | Severity | Status |
|--------|--------------------|--------------------|----------|--------|
| **Δκ** | Geodesic deviation / trajectory curvature | Deviation from linear interpolation | High | **Fixed** |
| **α** | Fractal self-similarity via DFA | DFA on embedding norms (~constant) | Critical | **Fixed** |
| **ΔH** | Semantic reorganization | Independent clustering entropy difference | High | **Fixed** |

---

## 1. Semantic Curvature (Δκ) — Original Problems

### What Morgoulis Claimed

> "Measures the average angular deviation of dialogue's semantic trajectory from an idealized geodesic (linear interpolation) in embedding space."

### The Original Algorithm

```python
# Step 1: Construct "geodesic" reference
start_point = embeddings[0]
end_point = embeddings[-1]
linear_trajectory[i] = start + (end - start) * (i / (n-1))

# Step 2: Compute pointwise deviation via cosine distance
deviation[i] = 1 - cosine_similarity(actual[i], linear[i])

# Step 3: Gaussian-weighted average (emphasizes middle turns)
weights = exp(-0.5 * ((i - n/2) / (n/4))^2)
Δκ = weighted_average(deviations, weights)
```

### Problems Identified

#### Problem 1: The "geodesic" is not a geodesic

In differential geometry, a geodesic is the shortest path *on a manifold*. Sentence embeddings lie on (or near) a hypersphere for normalized embeddings.

- Linear interpolation creates a **chord through the interior**
- Intermediate points don't lie on the embedding manifold
- A true geodesic would follow a **great circle arc**

**Result:** Δκ measured "deviation from a chord," not actual curvature.

#### Problem 2: Start/end point dependency

The metric was entirely determined by the relationship to the line connecting endpoints:

| Trajectory Type | Expected Δκ | Original Δκ |
|-----------------|-------------|-------------|
| Circular (returns to start) | High (maximally curved) | **~0** (endpoints coincide) |
| Spiral outward | High | Depends on start/end only |
| Random walk ending near start | Variable | **~0** |

#### Problem 3: Interpolated vectors not renormalized

For L2-normalized embeddings, linear interpolation produces vectors with norm < 1. This introduced systematic bias in cosine similarity comparisons.

#### Problem 4: No directional information

Δκ measured magnitude of deviation but not direction. Opposite explorations scored identically.

### What Δκ Actually Measured (Originally)

> "How much does the middle portion of the dialogue deviate from a straight line connecting its start and end points?"

Valid geometric quantity, but:

- Not curvature in the differential-geometric sense
- Not geodesic deviation in the Riemannian sense
- Highly sensitive to endpoint selection

---

## 2. Fractal Similarity Score (α) — Original Problems

### What Morgoulis Claimed

> "Quantifies scale-invariant patterns indicating self-organization without collapse to periodicity or noise."

The α exponent from DFA is well-established:

- α ≈ 0.5: White noise (no correlations)
- α ≈ 1.0: 1/f noise (optimal complexity)
- α ≈ 1.5: Brownian motion

### The Original Algorithm

```python
# In calculate_all_metrics():
embedding_magnitudes = [np.linalg.norm(emb) for emb in embeddings]
fractal_result = fractal_similarity_robust(np.array(embedding_magnitudes))
```

### Problems Identified

#### Problem 1: Wrong input signal (CRITICAL)

DFA was computed on **embedding vector norms**.

For sentence embeddings:

- Most models produce **L2-normalized** output
- Normalized vectors have `||emb|| = 1.0` by definition
- The "signal" was essentially **constant**

**Consequence:** DFA on a constant signal produces meaningless α values. Any α values from the original implementation were noise.

#### Problem 2: No semantic content in signal

Even for non-normalized embeddings, vector magnitude has no clear semantic interpretation. A sentence isn't more meaningful because its embedding has larger norm.

Meaningful candidates for semantic DFA:

| Signal | Interpretation |
|--------|----------------|
| Inter-turn cosine distance | Semantic "velocity" |
| Projection onto PC1 | Position along dominant axis |
| Local curvature κ(t) | Instantaneous bending |

#### Problem 3: Short sequence instability

DFA returns fallback α = 0.5 for sequences < ~20 points. Many real dialogues are shorter.

### What α Actually Measured (Originally)

> Essentially nothing meaningful for normalized embeddings.

The DFA algorithm was correctly implemented. The problem was input signal choice.

---

## 3. Entropy Shift (ΔH) — Original Problems

### What Morgoulis Claimed

> "Measures information-theoretic reorganization across semantic clusters between pre- and post-intervention states."

### The Original Algorithm

```python
# Step 1: Split dialogue at midpoint
pre_embeddings = embeddings[:n//2]
post_embeddings = embeddings[n//2:]

# Step 2: Cluster each half INDEPENDENTLY
labels_pre = KMeans(n_clusters=8).fit_predict(pre_embeddings)
labels_post = KMeans(n_clusters=8).fit_predict(post_embeddings)

# Step 3: Compute Shannon entropy of cluster distributions
H_pre = -sum(p_i * log2(p_i))
H_post = -sum(p_j * log2(p_j))

# Step 4: Entropy shift
ΔH = |H_post - H_pre|
```

### Problems Identified

#### Problem 1: Independent clustering (CRITICAL)

Pre and post halves clustered **separately**:

- Cluster 1 in pre has no relationship to Cluster 1 in post
- Clusters don't track the same semantic regions
- **Comparing apples to oranges**

**Consequence:** Two identical halves could produce non-zero ΔH if K-means partitions them differently. A genuine semantic shift might produce ΔH ≈ 0 if both happen to cluster similarly.

#### Problem 2: Measures diversity, not reorganization

Shannon entropy of cluster distribution measures *within-half diversity*. It does not measure:

- Which semantic regions gained/lost mass
- Whether meaning moved between regions
- Directionality of change

#### Problem 3: Fixed split ratio

50% split assumes intervention at midpoint. Real perturbations occur at arbitrary points.

#### Problem 4: Cluster count sensitivity

Different n_clusters values produce different ΔH. Implementation used consensus across methods, which helped but didn't solve the fundamental issue.

### What ΔH Actually Measured (Originally)

> "How different are the cluster distribution entropies when K-means is run independently on each half?"

Valid statistical comparison, but:
- Conflated clustering artifacts with semantic change
- Couldn't track reorganization direction
- Measured diversity difference, not information flow

---

## 4. The Fixes Applied

### Δκ: Local Frenet-Serret Curvature

```python
# Fixed implementation:
velocity = embeddings[t+1] - embeddings[t]
acceleration = velocity[t+1] - velocity[t]
a_perp = acceleration - (acceleration · v̂) * v̂
κ(t) = ||a_perp|| / ||velocity||²
```

Measures actual trajectory bending, independent of endpoints.

### α: DFA on Semantic Velocity

```python
# Fixed implementation:
signal = [1 - cosine_sim(emb[i], emb[i+1]) for i in range(len-1)]
alpha = dfa(signal)
```

Captures movement through meaning space.

### ΔH: Shared Clustering + Jensen-Shannon

```python
# Fixed implementation:
all_embeddings = first_half + second_half
clusters = KMeans(k).fit(all_embeddings)  # Shared vocabulary
P = distribution(first_half_labels)
Q = distribution(second_half_labels)
ΔH = jensen_shannon_divergence(P, Q)
```

Cluster IDs now correspond. JS divergence is symmetric and bounded [0,1].

---

## 5. Validation Results

### Δκ Validation

| Trajectory | Original | Fixed |
|------------|----------|-------|
| Linear | ~0 | 0.0 ✓ |
| Random | variable | ~0.03 |
| Sine wave | variable | ~0.46 |
| Circle | **~0** ✗ | **~0.99** ✓ |

### α Validation

| Trajectory | Original | Fixed |
|------------|----------|-------|
| Smooth | ~0.5 (noise) | ~1.3 (correlation) ✓ |
| Random | ~0.5 (noise) | ~0.67 (near white) ✓ |

### ΔH Validation

| Scenario | Original | Fixed |
|----------|----------|-------|
| Identical halves | variable | 0.0 ✓ |
| Different samples | variable | ~0.16 |
| Major shift | variable | ~1.0 ✓ |

---

## 6. Implications

### Empirical Validity vs Mathematical Validity

The original metrics showed **empirical utility** despite mathematical issues:
- Correlated with perceived dialogue complexity
- Distinguished sycophantic from generative exchanges
- Coupling mode classifications felt intuitively correct

This suggests they captured *something real* as **proxy measures**, even while being mathematically mislabeled.

### Impact on Extensions

Work building on these metrics (Ψ vector, attractor basins) inherited the issues:

| Extension | Affected By |
|-----------|-------------|
| ψ_semantic (PC1 of Δκ, ΔH, α) | All three issues compounded |
| Attractor basins | Thresholds calibrated on flawed metrics |
| Coupling modes | Boundaries may not reflect true states |

All thresholds were re-validated after fixes.

### What Remained Valid

- **Trajectory-based approach** — dialogue traces paths through embedding space
- **Autocorrelation coherence** — mathematically correct, captures breathing patterns
- **Affective substrate** — independently grounded (VADER, hedging)
- **Empirical observations** — HR leading Δκ doesn't depend on metric interpretation

---

## 7. Key Insight

> **The body-as-instrument often detected patterns the broken metrics couldn't see.**

Example: In the Grunch denial session, biosignal response (HR rising, chest tightening) detected attractor trap dynamics. The broken α metric misclassified this as "fragmentation" (α~0.5). The corrected metric revealed "lock-in" (α~1.0) — matching what the body sensed.

The fixes align the metrics with what the somatic system was already detecting.

---

## References

- Morgoulis, D. (2025). 4D Semantic Coupling Framework. https://github.com/daryamorgoulis/4d-semantic-coupling
- Peng, C.K. et al. (1994). Mosaic organization of DNA nucleotides. Physical Review E.
- Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory.

---

*"The map is not the territory — but a useful map still helps you navigate."*
