# Finding A2 — Δκ Semantic Curvature

**Audit ID:** RAA-SC-001-A2
**Date:** 2026-02-25
**Auditor:** agent:kairos (claude-sonnet-4-6)

---

## Claim

Δκ (semantic curvature) correctly computes local trajectory curvature via discrete Frenet-Serret formula κ(t) = ||a_perp|| / ||v||², independent of start/end points.

---

## Files Examined

- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/core_metrics.py` lines 75–232 — `semantic_curvature_enhanced()`, `_compute_local_curvatures()`, `_calculate_curvature_single()`
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/tests/test_core_metrics.py` lines 25–116 — `TestSemanticCurvature`, `TestEdgeCases`, `TestMetricFixValidation`

---

## Evidence

### 1. Frenet-Serret formula implementation (core_metrics.py:197–222)

The implementation is:
- Velocity: `np.diff(embeddings, axis=0)` — first differences, shape `(n-1, d)` ✓
- Acceleration: `np.diff(velocities, axis=0)` — second differences, shape `(n-2, d)` ✓
- Unit velocity: `v_hat = v / v_norm` ✓
- Perpendicular acceleration: `a_parallel = dot(a, v_hat) * v_hat; a_perp = a - a_parallel` ✓
- Curvature: `kappa = ||a_perp|| / v_norm**2` ✓

This is mathematically correct. In any Euclidean space, κ = ||a_perp|| / ||v||² is the correct formula, and it is equivalent to the 3D cross-product formula κ = ||v × a|| / ||v||³ (provably equivalent because ||v × a||² = ||v||²||a_perp||² in 3D).

One documentation inconsistency: `semantic_curvature_enhanced()` docstring (line 87) shows `κ(t) = ||v × a|| / ||v||³` but the implementation uses the high-D projection form. These are equivalent in 3D and the projection form is the correct generalization — the issue is only in the docstring.

### 2. High-dimensional correctness (core_metrics.py:175–225)

The perpendicular projection approach correctly generalizes to arbitrary dimensions. Cross product is not defined for d > 3; the a_perp projection is the standard generalization used in differential geometry for arbitrary Euclidean spaces. The formula is well-founded.

### 3. Degenerate case handling (core_metrics.py:208–213)

Three cases are identified:

**Zero velocity** (`v_norm < 1e-10`): Returns `0.0` (core_metrics.py:209–212). This is conservative and stable, but mathematically debatable: zero velocity with non-zero acceleration is a cusp, formally undefined or infinite. In practice for sentence embeddings, exact zero velocity requires duplicate adjacent embeddings, which is unlikely in real data. The choice is reasonable for robustness.

**Collinear trajectory** (a parallel to v): `a_perp = 0`, κ = 0. Correctly identifies straight-line segments as zero curvature. ✓

**U-turn degenerate case**: If a trajectory exactly reverses direction (e.g., sequence [A, B, A]), v[0] = B-A, v[1] = -(B-A), a[0] = -2(B-A). Here a is exactly antiparallel to v, so a_parallel = -2(B-A) and a_perp = 0, producing κ = 0. A 180° reversal should yield high curvature, not zero. This is a known limitation of first-order discrete Frenet-Serret: exact direction reversals produce degenerate zero. In high-dimensional semantic embedding space (d=384), exact reversals are of measure zero for real data, but the edge case exists in principle. Tests do not cover this case.

**Duplicate embeddings** (all same point): velocity = 0, handled by zero-velocity check. Returns 0.0 without exception. ✓ (confirmed by `test_constant_embedding_handled`, core_metrics.py test line 327).

### 4. Bootstrap CI validity (core_metrics.py:139–151)

```python
boot_indices = np.sort(np.random.choice(n, size=n, replace=True))
boot_embeddings = embeddings[boot_indices]
```

Sorting preserves temporal ordering, which is correct for a trajectory bootstrap. However, sampling with replacement creates two systematic distortions:

1. **Duplicate indices** → adjacent identical embeddings → zero velocity → κ = 0 at those points. Expected duplicates ≈ 1 - (1-1/n)^n ≈ 63% of positions will be duplicated in a resample of size n, producing ~n/e ≈ 0.37n zero-velocity segments.

2. **Skipped indices** → larger gaps between successive embeddings → artificially elevated velocity and potentially altered curvature at gap boundaries.

These two biases partially cancel but not predictably. Block bootstrap (sampling contiguous blocks of fixed size) would preserve local temporal structure and is the standard approach for time series. The current method likely produces CIs with inflated lower bounds (from zero-velocity collapse) and variable upper bounds. The CI is returned but not used for classification (`threshold_met` depends only on `curvature >= 0.35`), so this is a reporting concern rather than a classification defect.

### 5. Permutation test (core_metrics.py:153–163)

```python
null_embeddings = np.random.permutation(embeddings)
p_value = np.mean(np.array(null_curvatures) >= curvature)
```

The null hypothesis is: "the trajectory's curvature is consistent with a random permutation of the same embeddings." `p_value = P(null >= observed)` — a small p_value means the observed curvature is unusually LOW compared to random permutations, indicating the temporal ordering imposes structure (smooth trajectory). A large p_value means the observed curvature is similar to or higher than random.

This is a valid and interpretable null, but note the direction: small p_value is associated with *lower* curvature than random, not higher. For the "complexity detected" use case (Δκ >= 0.35), a high-curvature trajectory would have a LARGE p_value (indistinguishable from random). This means the p_value as reported does not confirm that high curvature is statistically unusual — it only confirms that low curvature is. The p_value is not used in `threshold_met` (line 171), which depends solely on the 0.35 threshold, so there is no classification error. But the p_value interpretation in the broader framework needs clarification.

### 6. Mean curvature as summary statistic (core_metrics.py:135–136)

`curvature = np.mean(local_curvatures)` treats all n-2 interior points equally. Two concerns:

- **Slow-segment amplification**: κ = ||a_perp|| / ||v||² diverges as ||v|| → 0. Slow trajectory segments (small velocities) produce large κ values that can dominate the mean. An arc-length-weighted mean (weighted by ||v||) would be more geometrically natural (equivalent to total curvature / total arc length).

- **Outlier sensitivity**: A single sharp directional change at one interior point could produce a large local κ that elevates the mean. The `curvature_std` return value helps characterize this but is not used in threshold detection.

For typical sentence embedding trajectories, velocity magnitudes vary within an order of magnitude, so this concern is moderate rather than severe.

### 7. Threshold 0.35 provenance (core_metrics.py:69)

The code documents the threshold as "empirically-derived thresholds from validation study" with no citation. The Morgoulis (2025) reference is given for the original framework, but the threshold was modified in the 2025-12-08 fix. The 0.35 threshold is specific to the new local-curvature formula — it cannot be directly inherited from the chord-deviation formula. No calibration data, baseline distribution, or sensitivity analysis is present in the codebase or referenced tests. The threshold is asserted rather than derived.

### Test coverage

`TestSemanticCurvature` (lines 28–116) tests:
- Linear trajectory → κ < 1e-10 ✓
- Circular trajectory → κ > 0.5 ✓ (also validates fix over chord-deviation)
- Random trajectory → non-zero κ ✓
- Minimum length (n < 4) → graceful fallback ✓
- CI bounds, local_curvature count ✓

Missing tests:
- U-turn case ([A, B, A] pattern)
- Arc-length-weighted vs. unweighted curvature comparison
- Bootstrap CI validity under slow-segment conditions
- Threshold calibration justification

---

## Finding

**Verdict: PARTIAL**

The core Frenet-Serret formula is correctly implemented for high-dimensional Euclidean space. Velocity as first differences, acceleration as second differences, perpendicular projection for a_perp, and κ = ||a_perp|| / ||v||² are all mathematically sound. The fix from chord-deviation to local curvature is correctly made and the circular trajectory test validates independence from start/end points. Degenerate cases (zero velocity, collinear) are handled safely.

However, three concerns moderate confidence in the implementation:

1. **Bootstrap CI methodology**: Sorted replacement resampling distorts curvature by introducing zero-velocity segments (duplicates) and velocity spikes (gaps). The CI is computed but not used for classification, so this does not affect `threshold_met` — but it does affect reported uncertainty.

2. **Mean curvature sensitivity**: The unweighted mean of local κ(t) values can be dominated by slow trajectory segments where ||v|| is small. This is a systematic bias toward higher reported Δκ when the trajectory contains slow segments.

3. **Threshold 0.35 undocumented provenance**: The threshold was inherited or re-estimated for the local curvature formula without documented calibration. A threshold valid for chord-deviation is not directly transferable to local curvature without re-validation.

The formula is correct; the statistical infrastructure around it has addressable weaknesses.

---

## Notes

- The docstring inconsistency (line 87: `||v × a|| / ||v||³` vs. implementation `||a_perp|| / ||v||²`) should be corrected for clarity, even though the formulas are equivalent in 3D.
- The U-turn degenerate case (exact anti-parallel reversal → κ = 0) is low probability in high-dimensional embedding space but should be documented as a known limitation.
- The p_value direction (small p_value = trajectory is smoother than random) is counter-intuitive for readers expecting "small p_value = significant curvature." A note in the docstring would help.
- Block bootstrap with block size ≈ √n would be a straightforward improvement for CI validity.
- Arc-length-weighted mean (`np.average(local_curvatures, weights=||v[i]||)`) would reduce slow-segment bias.
