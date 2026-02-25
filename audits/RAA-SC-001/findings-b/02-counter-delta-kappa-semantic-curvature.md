# Counter-Finding B2 — Δκ Semantic Curvature

**Audit ID:** RAA-SC-001-B2
**Date:** 2026-02-25
**Auditor:** agent:kairos (claude-sonnet-4-6)
**Counter-auditing:** RAA-SC-001-A2

---

## Charge

COUNTER-AUDIT of Phase A finding on Δκ semantic curvature. Five specific questions were posed:

1. Bootstrap sorts resampled indices — does this bias curvature upward (sudden jumps) or downward (repeated points → zero velocity)?
2. κ(t) = ||a_perp|| / ||v||² — is this the correct discrete Frenet-Serret formula? Some references use ||v||³. Which is correct?
3. Zero velocity → 0.0: is this correct given zero velocity with non-zero acceleration implies infinite curvature?
4. Permutation null randomises entire trajectory — is this the right null hypothesis?
5. Mean curvature as summary statistic — does this hide distributional features?

---

## Files Examined (Independent Read)

- `src/core_metrics.py` lines 75–232: `semantic_curvature_enhanced()`, `_compute_local_curvatures()`, `_calculate_curvature_single()`

---

## Analysis

### Question 1: Bootstrap Bias Direction

**Phase A position:** "two biases partially cancel but not predictably."

**Counter-audit finding: Both biases are downward. They do not cancel.**

The bootstrap code (core_metrics.py:142–143):
```python
boot_indices = np.sort(np.random.choice(n, size=n, replace=True))
boot_embeddings = embeddings[boot_indices]
```

After sorting, two structural effects arise:

**Effect A — Duplicates (→ systematic κ = 0):** In a bootstrap resample of size n, the expected fraction of positions sampled at least once is 1 − (1 − 1/n)ⁿ → 1 − 1/e ≈ 0.632. The remaining ~36.8% of positions are *not* sampled. After sorting, the 63.2% that are sampled include many consecutive duplicates: whenever index i is sampled k ≥ 2 times, the sorted array contains k adjacent copies of `e[i]`. Between these, v[j] = e[i] − e[i] = 0, triggering the zero-velocity handler, which returns κ = 0.0. This produces a systematic floor of ~36.8% of curvature values clamped to zero.

**Effect B — Gaps (→ reduced κ from rescaling):** Unsampled indices create non-unit jumps in the resampled trajectory. For a smooth trajectory with step size Δ, a gap of size g produces velocity ≈ g·Δ and acceleration ≈ g·a. Then κ = ||a_perp|| / ||v||² ≈ g·||a_perp_orig|| / (g·||v_orig||)² = ||a_perp_orig|| / (g·||v_orig||²) = κ_orig / g. Since g ≥ 2 for any gap, **curvature at gap boundaries is halved or more from its true value.** This is a downward bias, not an upward one.

**Net effect:** Both effects reduce bootstrap curvature estimates below the original-trajectory estimate:
- Duplicates: ~36.8% of κ values floored at 0
- Gaps: remaining κ values reduced by a factor of 1/g

Phase A's characterisation that "sudden jumps" might bias upward is incorrect for the Frenet-Serret formula. Unlike chord-deviation (where gaps increase deviation), in the curvature formula the gap scales the denominator faster than the numerator, yielding *lower* κ, not higher.

**Consequence:** The bootstrap distribution is systematically below the observed curvature. The 95th percentile of bootstrap curvatures may fall *below* the point estimate, producing a CI that does not bracket the observed statistic. This is not merely "inflated lower bounds" — the CI may be entirely below the estimator it is supposed to characterise. The CI should not be interpreted as valid uncertainty bounds.

---

### Question 2: κ(t) = ||a_perp|| / ||v||² versus ||v × a|| / ||v||³

**Phase A position:** Both formulas are equivalent; no error.

**Counter-audit finding: Confirmed equivalent. Derivation given explicitly.**

For a curve r(t) parameterized by arbitrary t, curvature κ = ||dT/ds|| where T = unit tangent, s = arc length:

```
ds/dt = ||v||
T = v / ||v||
dT/dt = (a - (â·v)v̂) / ||v|| = a_perp / ||v||
κ = dT/ds = dT/dt · dt/ds = a_perp / ||v||²
→ κ = ||a_perp|| / ||v||²   ✓ (implemented)
```

In 3D, ||v × a||² = ||v||²||a||² − (v·a)² = ||v||² · ||a_perp||², so ||v × a|| = ||v|| · ||a_perp||. Therefore:

```
||v × a|| / ||v||³ = (||v|| · ||a_perp||) / ||v||³ = ||a_perp|| / ||v||²   ✓ (equivalent)
```

The ||v||³ formula is the 3D cross-product form. The ||v||² formula is the general d-dimensional form via perpendicular projection. They are algebraically identical wherever both are defined. The implementation is correct and the docstring inconsistency is documentation-only.

**One subtlety Phase A did not raise:** The implementation attributes curvature at "interior point i" using v[i] (left velocity: e[i+1] − e[i]) and a[i] = v[i+1] − v[i] (spanning e[i], e[i+1], e[i+2]). This is a *forward-biased* stencil — the curvature estimate at position i incorporates information from e[i+2] but not e[i−1]. The alternative central-difference stencil (v_central = (e[i+1] − e[i−1]) / 2, a_central = e[i+1] − 2e[i] + e[i−1]) has the same second-order accuracy but is symmetric around the evaluation point. Neither is "wrong," but the asymmetric stencil means the curvature at point i is effectively the curvature of the arc (e[i], e[i+1], e[i+2]), not the curvature *at* e[i+1]. The metric is labelled as curvature at interior points but is actually the curvature of the *following* arc. For independent verification of claims about where in the trajectory curvature is concentrated, this offset matters.

---

### Question 3: Zero Velocity → 0.0 (core_metrics.py:209–212)

**Phase A position:** "conservative and stable, but mathematically debatable… unlikely in real data. The choice is reasonable for robustness."

**Counter-audit finding: Phase A underweights the severity because it omits the bootstrap interaction.**

For real conversation data, Phase A is likely correct that exact zero velocity (||v|| < 1e-10) is rare. When e[i] and e[i+1] are distinct sentence embeddings from a transformer model, ||e[i+1] − e[i]|| is typically >> 1e-10. The mathematical error (0 rather than ∞) has negligible impact on observed curvature.

**However:** The zero-velocity handler is *systematically* invoked during bootstrap resampling, as shown in Question 1 above. Approximately 36.8% of bootstrap velocity vectors are zero by construction. Every one of these returns κ = 0.0. The zero-velocity handler was designed for a rare real-data edge case but is operating as a *de facto* downward-clamping function during bootstrap — a function that was never designed or calibrated for that role.

The combined failure mode is:
```
Bootstrap introduces ~36.8% zero-velocity steps (by design: sorted replacement)
  → Zero-velocity handler returns 0.0 (designed for rare real-data case)
  → Mean bootstrap curvature ≈ 0.632 × true_bootstrap_mean (at best)
  → CI lower bound collapses toward zero
```

Mathematically: if mean true curvature = μ, bootstrap mean ≈ 0.632μ (since ~36.8% of terms are forced to 0 and remaining terms are also suppressed by gap effects). This is a ~37–50% systematic underestimate of the CI centre, not a minor numerical artefact.

Phase A treated zero-velocity handling and bootstrap distortion as independent concerns. They are coupled, and the compound effect is more severe than either alone.

---

### Question 4: Permutation Null — Is This the Right Null Hypothesis?

**Phase A position:** "valid and interpretable null" with a note that direction is counter-intuitive.

**Counter-audit finding: The permutation null is structurally misaligned with the scientific claim, and likely uninformative for real data.**

The permutation null tests: "is the observed curvature consistent with a random ordering of the same embeddings?" Because curvature is computed from *consecutive differences*, it is sensitive to local correlations in the embedding sequence. Real semantic trajectories have strong local autocorrelation: consecutive turns are about the same topic and therefore more similar to each other than to distant turns. This is a structural property of dialogue, not a property specific to "complex" conversations.

**A random permutation destroys local autocorrelation.** The permuted trajectory has consecutive turns drawn from different parts of the conversation, with large semantic distances between adjacent steps. Curvature in the permuted trajectory is driven by large, jagged velocity changes — it will almost always be higher than in the original trajectory.

Consequence: for any typical conversation (complex or not), the original curvature will be *lower* than random permutation curvature, yielding p_value < 0.05 and the "significant" label meaning the trajectory is SMOOTHER than random. Nearly all conversations will receive small p_values, regardless of their actual complexity. The p_value is not discriminating between complex and non-complex trajectories — it is discriminating between trajectories (all conversations) and random noise (all permutations).

This makes the p_value reported by `semantic_curvature_enhanced()` **vacuously uninformative** for the stated use case. A test that is "significant" for essentially all inputs provides no information. The classification decision (`threshold_met`) correctly ignores p_value, but the p_value is still a reported output that could mislead users.

Phase A noted the direction issue but did not follow through to the conclusion that local autocorrelation makes the null distribution always higher than observed data. The counter-audit assesses this as a more serious flaw: the p_value is not a valid statistical test for the semantic complexity claim.

**Appropriate alternative null:** A *phase-randomised surrogate* (Fourier-based) preserves the autocorrelation structure while destroying specific temporal patterns. This would test whether the observed curvature pattern exceeds what autocorrelation alone would produce — a more meaningful test.

---

### Question 5: Mean Curvature — Distributional Hiding

**Phase A position:** "moderate rather than severe" concern for typical sentence embeddings.

**Counter-audit finding: Phase A correctly identifies the issue but underestimates it. The unweighted mean is measuring the wrong geometric quantity.**

The implemented statistic is (core_metrics.py:136):
```python
curvature = np.mean(local_curvatures)  # = Σ κ(t) / (n-2)
```

This measures **mean curvature per time step**. The geometrically correct summary for "how curved is this trajectory on average" is **mean curvature per unit arc length**:

```
κ_geom = Σ [κ(t) · ||v(t)||] / Σ ||v(t)||  (arc-length weighted)
```

The difference matters when ||v(t)|| varies along the trajectory. Since κ(t) = ||a_perp|| / ||v||², a time step with ||v|| = 0.1 contributes κ ∝ 100× more to the unweighted mean than a step with ||v|| = 1.0, even if both represent geometrically similar curvature. In dialogue, semantic step sizes vary substantially: turns introducing new topics cover more embedding space than turns that rephrase or acknowledge.

**Consequence for slow-segment amplification:** If a conversation contains short sequences of near-duplicate turns (very slow semantic motion, ||v|| small), those segments produce elevated κ values that dominate the unweighted mean. The classifier then triggers `threshold_met` not because the conversation traversed a curved trajectory, but because a few slow, similar turns created an artefactually large curvature contribution.

**Distributional concern:** The curvature distribution over trajectory points is right-skewed: most points have near-zero curvature (smooth local trajectories), and rare points have large curvature (sharp semantic pivots). The mean of a right-skewed distribution is sensitive to the high tail. Two trajectories with identical "cognitive complexity" but different outlier distributions could produce very different Δκ values. The `curvature_std` field is returned but not used in threshold detection — a hybrid statistic (e.g., trimmed mean, or mean + 0.5·std threshold) would be more robust.

Phase A's characterisation of this as "moderate" assumes velocity magnitudes vary within an order of magnitude. In practice, transformer sentence embeddings with diverse topics can produce velocity variations of 2–5x, creating a 4–25x difference in κ contribution between fastest and slowest segments. This is not negligible.

---

## Additional Issues Not Raised in Phase A

### Issue 1: Embedding Normalization Ambiguity

The docstring states "Sequence of embeddings (L2-normalized or not)" with no guidance on expected input. Many standard sentence encoders (SBERT, sentence-transformers) return L2-normalized embeddings living on the unit hypersphere S^(d-1).

For L2-normalized embeddings, the Euclidean first-difference v[i] = e[i+1] − e[i] is a *chord vector*, not a tangent vector to the hypersphere. The chord-based κ formula underestimates true geodesic curvature on the sphere, with error that grows with step size:

```
||v_chord|| = 2 sin(θ/2)  where θ = angle between e[i] and e[i+1]
||v_geodesic|| = θ
For θ = 0.2 rad: chord ≈ 0.1997, arc ≈ 0.200 (0.15% error — negligible)
For θ = 1.0 rad: chord ≈ 0.841, arc = 1.000 (16% error — relevant)
```

Large semantic shifts (low cosine similarity between adjacent turns, θ close to π/2) introduce systematic underestimation of curvature. For high-d unit sphere (d = 384), typical inter-sentence angles are moderate and this error is likely small but non-zero. The implementation should document whether unit-normalized inputs are expected, since the threshold 0.35 may have been calibrated on non-normalized or normalized data, and mixed usage would shift the distribution.

### Issue 2: The Threshold Transferability Problem Is Worse Than Noted

Phase A correctly noted the threshold 0.35 has undocumented provenance and cannot be directly inherited from the chord-deviation formula. The counter-audit extends this:

The four issues identified above (biased CI, unweighted mean, incorrect null, normalization ambiguity) all affect the *distribution* of Δκ values on real data. Any threshold calibrated on the Morgoulis 2025 dataset is calibrated on a specific joint configuration of:
- Embedding model (specific transformer)
- Normalization convention
- Dialogue length distribution
- Conversation domain

Changing any of these changes the Δκ distribution and renders the threshold invalid. There is no sensitivity analysis or held-out validation in the codebase confirming the threshold generalizes.

---

## Verdict

**PARTIAL — agreed with Phase A, with three corrections and two additional findings.**

**Agreements:**
- The Frenet-Serret formula κ = ||a_perp|| / ||v||² is correct and equivalent to ||v × a|| / ||v||³ (Question 2 confirmed)
- The docstring inconsistency is documentation-only, not an algorithmic error
- The zero-velocity handler is safe for real conversation data
- Mean curvature as a summary statistic has known weaknesses

**Corrections to Phase A:**

1. **Bootstrap bias direction (Question 1):** Phase A says biases "partially cancel." They do not cancel — both are downward. The CI can be non-bracketing (CI entirely below the point estimate). This is more severe than "inflated lower bounds."

2. **Zero-velocity interaction with bootstrap (Question 3):** Phase A treats these as independent concerns. They are coupled: bootstrap deliberately introduces ~36.8% zero-velocity steps, each returning κ = 0.0 via the handler. The compound effect is a ~37–50% underestimate of bootstrap CI centre.

3. **Permutation null (Question 4):** Phase A says "valid and interpretable." It is interpretable but structurally misaligned: local autocorrelation of real dialogue means all conversations will appear smoother than random, making the p_value vacuously near-zero for all inputs and therefore non-discriminating. This is not a minor direction-of-interpretation issue — the null is effectively broken for the intended use case.

**Additional findings not in Phase A:**

4. **Forward-biased stencil:** Curvature at "point i" is computed from the arc (e[i], e[i+1], e[i+2]), not centered at e[i+1]. The attribution of curvature to trajectory positions is off by one step.

5. **Embedding normalization ambiguity:** For L2-normalized inputs (common with SBERT), chord-based curvature underestimates geodesic curvature for large semantic steps. No documented expectation or correction for normalization convention.

---

## Summary Table

| Question | Phase A Verdict | Counter-Audit Verdict | Change |
|----------|----------------|----------------------|--------|
| Bootstrap bias direction | Partial cancel (unclear) | Both downward; CI non-bracketing | CORRECTED |
| Formula correctness | Equivalent, no error | Confirmed equivalent | CONFIRMED |
| Zero velocity handling | Reasonable, rare edge case | Coupled with bootstrap; ~37% CI underestimate | UPGRADED SEVERITY |
| Permutation null | Valid null, direction issue | Vacuously uninformative for all real dialogue | CORRECTED |
| Mean curvature statistics | Moderate concern | Wrong geometric quantity (time-step vs arc-length) | CONFIRMED, emphasis raised |
| Forward-biased stencil | Not raised | Off-by-one attribution | NEW |
| Normalization ambiguity | Not raised | Undocumented; affects threshold transferability | NEW |

---

## Notes

- The core mathematical claim — that Frenet-Serret curvature is correctly implemented and independent of start/end points — is correct and represents a genuine improvement over the original chord-deviation formula. The fix was well-motivated and well-executed.
- The statistical infrastructure (bootstrap CI, permutation test, threshold calibration) has addressable weaknesses that are more severe than Phase A assessed, particularly in combination.
- None of these issues invalidates the qualitative direction of the metric (more curvature = more directional change). They affect quantitative calibration, CI reliability, and the interpretation of the p_value.
- Priority remediation: (1) replace permutation null with phase-randomised surrogate or block-shuffle; (2) switch bootstrap to non-overlapping block bootstrap with block size ≈ √n; (3) implement arc-length-weighted mean as an alternative statistic; (4) document normalization expectation.
