# Finding A4: Ψ Vector Assembly

**Task:** a4-psi-vector-assembly
**Date:** 2026-02-25
**Auditor:** Claude Sonnet 4.6 (Phase A)

---

## Claim

Ψ vector assembly correctly composes four substrates (semantic, temporal, affective, biosignal) into a coherent phase-space representation with documented compression.

---

## Files Examined

- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/substrates.py` — full file (lines 1–595)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/analyzer.py` — lines 183–380 (assembly logic, `compute_coupling_coefficient`)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/extensions.py` — lines 263–435 (legacy Ψ assembly for comparison)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/tests/test_substrates.py` — full file
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/tests/fixtures/golden_outputs.json` — lines 1–52
- `/Users/m3untold/Code/EarthianLabs/publications/papers/cross-substrate-coupling-preprint.md` — lines 96–130 (Table 1, Section 3.2)

---

## Evidence

### Q1: z-score parameters — empirical calibration or arbitrary?

`compute_semantic_substrate` (`substrates.py:80–91`) standardises the three metrics before weighting:

```
metric_std = [
    (delta_kappa - 0.15) / 0.15,   # Δκ center=0.15, std=0.15
    (delta_h - 0.15) / 0.15,        # ΔH center=0.15, std=0.15
    (alpha - 0.8) / 0.3             # α  center=0.8, std=0.3
]
```

The comment reads "RECALIBRATED 2025-12-08 for fixed metric implementations" and documents the ranges as developer observations ("Range ~[0, 0.5]"), not computed statistics. No reference is made to a calibration dataset, systematic analysis, or cross-validation study. The identical center/std for Δκ and ΔH (both 0.15/0.15) is suspiciously symmetric. These parameters appear to be engineering estimates derived from inspecting a small number of test cases after the metric implementations were changed. No calibration artefacts exist in the repository.

**Verdict on Q1:** Parameters are undocumented engineering estimates, not empirically calibrated. The recalibration comment indicates they were updated once after a code fix but there is no study supporting the chosen values.

---

### Q2: Temporal substrate windowed recomputation — expensive or correct?

`compute_temporal_substrate` (`substrates.py:134–167`) creates a sliding window over all embeddings and calls `calculate_all_metrics()` on each window. With `window_size=10` (default), this produces O(N) calls to the full metric pipeline per analysis.

Two correctness concerns:

**a) DFA minimum sample violation.** The preprint (Section 3.2) states that DFA for α estimation "requires a minimum of 20 inter-turn velocity samples (~21 turns) for meaningful estimates." The sliding window uses `window_size=10`, with a guard of only `len(window_embs) >= 6` (`substrates.py:140`). Windows of 6–10 embeddings produce 5–9 velocity samples — far below the stated 20-sample minimum. The α values from these sub-minimal windows are unreliable by the preprint's own admission. Since α feeds the windowed CV calculation, `psi_temporal` is built on noisy α estimates.

**b) CV numerical instability.** The temporal stability formula (`substrates.py:159`) is:
```python
cv = np.std(window_psi) / (np.abs(np.mean(window_psi)) + 1e-10)
psi_temporal = 1 / (1 + cv)
```
When `mean(window_psi)` approaches zero (common with neutral or random-text embeddings), the `+ 1e-10` denominator guard causes CV to reach astronomically large values (>1e10), collapsing `psi_temporal` toward zero. The golden output fixture confirms this: `basic_embeddings_only` shows `psi_temporal = 1.22e-10` — effectively 0, an extreme artefact rather than a meaningful stability score. The fix would require a more robust CV denominator (e.g., the range or IQR of window_psi).

**Verdict on Q2:** Windowed recomputation is conceptually correct as a stability measure, but the implementation has two material flaws: the window size is below the DFA minimum required for reliable α estimates, and the CV denominator is numerically unstable near zero.

---

### Q3: Affective substrate — VADER for AI dialogue text?

VADER (`vaderSentiment`) was designed for short social media text and is not validated for AI-generated dialogue. However, the implementation uses VADER primarily to compute **variance** of compound scores across turns (`substrates.py:227`), not raw sentiment polarity. This is a stronger choice than using raw VADER for mood classification, since variance captures instability rather than valence — a property somewhat less dependent on VADER's social-media calibration.

The psi_affective composite (`substrates.py:298–305`) weights four components:
- 0.3 × sentiment_variance (VADER)
- 0.3 × hedging_density (regex patterns)
- 0.3 × vulnerability_score (regex patterns)
- 0.1 × confidence_variance (regex patterns)

Three of four components are regex-based, not VADER-dependent. The regime where VADER scoring matters most — distinguishing positive from negative affect in short AI turns — is handled only as one-third of one feature. The normalisation thresholds (`/0.5`, `/0.1`, `/0.05`) are undocumented engineering choices, not empirically derived.

**Range concern.** The formula `np.tanh(2 * (psi_affective_raw - 0.5))` maps `psi_affective_raw ∈ [0, 1]` to `tanh([-1, 1]) ≈ [-0.762, 0.762]`. The documentation claims the range is `[-1, 1]`; the practical range is approximately `±0.76`.

**Verdict on Q3:** The VADER limitation is partially mitigated by variance-based usage and the dominant weight of regex components. However, the normalisation thresholds are arbitrary, and the practical psi_affective range is ≈±0.76, narrower than the claimed ±1.

---

### Q4: Biosignal substrate — is `(HR−80)/40` with tanh appropriate?

`compute_biosignal_substrate` (`substrates.py:517–522`):
```python
hr_normalized = (hr - 80) / 40  # Maps ~60-100 to [-0.5, 0.5]
return float(np.tanh(hr_normalized))
```

Issues:

**a) Centre of 80 bpm is high.** Average resting HR in adults is approximately 70–75 bpm, not 80. Centering at 80 means a physiologically normal resting HR of 70 bpm gives: `(70−80)/40 = −0.25`, `tanh(−0.25) ≈ −0.24`. The biosignal substrate returns a negative value at normal rest — a representation bias the code does not acknowledge.

**b) Scale of 40 is coarse.** For a resting range of 60–100 bpm (±20 from 80), the linear component spans only [−0.5, +0.5] before tanh. This means the gradient through the physiologically active range is gentle: near-resting HR changes of 5–10 bpm produce psi_biosignal changes of approximately 0.1–0.2 — modest signals that the weighting formula (0.1×psi_biosignal) would further suppress.

**c) No calibration.** The preprint confirms these parameters without citing a source. The comment in code says "Normalize HR around resting (60-100 bpm typical)" which is circular justification.

The preprint Table 1 (`(HR − 80) / 40 | tanh to [−1, 1]`) accurately reflects the implementation. The practical resting-HR range of the output is approximately [−0.46, +0.46], not ±1.

**Verdict on Q4:** Implementation matches the preprint specification. The functional form (linear normalisation + tanh) is reasonable. The centre of 80 bpm is slightly high, and the coarse std of 40 limits sensitivity to physiologically relevant HR variation in resting dialogues.

---

### Q5: Comparable scales across substrates?

| Substrate | Theoretical range | Practical range (resting/typical input) |
|-----------|-------------------|-----------------------------------------|
| Ψ_semantic | [-1, 1] | Wide — depends on metric z-scores |
| Ψ_temporal | [0, 1] | Empirically near 0 (instability issue) or 0.5 (fallback) |
| Ψ_affective | [-1, 1] (claimed) | ≈[-0.76, 0.76] practical max |
| Ψ_biosignal | [-1, 1] (claimed) | ≈[-0.46, 0.46] for 60–100 bpm HR |

**Critical asymmetry.** Ψ_temporal is bounded `[0, 1]` (always non-negative), while the other substrates span negative values. The composite formula in `analyzer.py:369–380`:
```python
psi_composite = 0.5 * psi_semantic + 0.3 * psi_temporal + 0.2 * psi_affective
```
introduces a persistent positive bias: `psi_temporal` contributes `0.3 × [0, 1]` to the scalar, always pulling the composite toward positive values. A fully neutral dialogue (all substrates near zero) would yield `psi ≈ 0.3 × psi_temporal`, not zero.

This asymmetry is not disclosed in the preprint or documentation.

**Verdict on Q5:** Substrates are not on comparable scales. Ψ_temporal is non-negative while others can be negative. Ψ_affective and Ψ_biosignal have practical ranges narrower than the documented ±1. The composite psi scalar has a structural positive bias from the temporal term.

---

### Q6: Does the assembled Ψ vector match preprint Table 1?

Preprint Table 1 (Section 3.2) specifies:

| Ψ Component | Feature | Compression |
|-------------|---------|-------------|
| Ψ_semantic | Δκ, ΔH, α | z-scored, equal weight |
| Ψ_temporal | Metric stability | 1/(1 + CV) |
| Ψ_affective | Sentiment variance (0.3), hedging density (0.3), vulnerability (0.3), confidence variance (0.1) | tanh |
| Ψ_biosignal | (HR−80)/40 | tanh |

Each entry matches the implementation:
- Equal weights (1/√3) after z-scoring: `substrates.py:77, 84–88`
- 1/(1+CV): `substrates.py:160`
- Four affective components with stated weights: `substrates.py:298–303`
- (HR−80)/40 + tanh: `substrates.py:520–521`

**One discrepancy.** The preprint paragraph before Table 1 states substrates are "each compressed to [−1, 1]." Ψ_temporal is `[0, 1]`, not `[−1, 1]`. This is a factual error in the preprint.

**Verdict on Q6:** The implementation matches the Table 1 feature inventory. The preprint's prose claim that all substrates are compressed to "[-1, 1]" is inaccurate for Ψ_temporal, which is `[0, 1]`.

---

### Q7: Are the 1/√3 weights empirically supported?

The weights `[0.577, 0.577, 0.577]` = `1/√3` are described in the code as a "PC1 approximation with equal contributions" (`substrates.py:55`, `extensions.py:693`). The preprint Table 1 describes this more accurately as "z-scored, equal weight."

**Mathematical note.** Given `weights = [w, w, w]` where `w = 1/√3`:
- `np.linalg.norm(weights) = w × √3 = (1/√3) × √3 = 1`
- Therefore: `np.dot(metric_std, weights) / np.linalg.norm(weights)` = `np.dot(metric_std, weights)` = `(z_dk + z_dh + z_alpha) / √3`

This is a projection of the z-scored metric vector onto the equal-weight unit direction — equivalent to `mean(metric_std) × √3` or `sum(metric_std) / √3`. It is **not** PC1 in the data-analytic sense; it is a unit-vector projection assuming equal importance.

**What would be required for empirical justification:**
1. PCA on a corpus of dialogues to verify that the first principal component in the (Δκ, ΔH, α) space approximates (1/√3, 1/√3, 1/√3)
2. Sensitivity analysis showing that psi_semantic values and basin classifications are robust to variation in weights

Neither exists in the codebase or the preprint. The three metrics are likely correlated (they all measure aspects of semantic trajectory complexity), which means the equal-weight assumption is unlikely to be optimal; PCA would yield unequal weights.

**Verdict on Q7:** The 1/√3 weights are mathematically sound as a unit-vector projection but are not empirically supported as a PC1 approximation. The code comment overstates the theoretical justification; the preprint description ("equal weight") is more honest.

---

## Finding

**Verdict: PARTIAL**

The Ψ vector assembly correctly implements the four substrates as specified in preprint Table 1: the feature inventory, compression functions, and weights all match. However, three substantive concerns remain unaddressed:

1. **Temporal substrate reliability.** Windowed α estimates use window_size=10, which violates the preprint's own stated DFA minimum of ~20 samples. This makes Ψ_temporal unreliable in practice, confirmed by the golden-output artefact (psi_temporal ≈ 1e-10 for neutral embeddings due to CV denominator instability near zero).

2. **Non-comparable substrate scales.** Ψ_temporal is strictly `[0, 1]` while the other substrates span `[−1, 1]`, introducing a persistent positive bias in the composite scalar. The preprint describes all substrates as "compressed to [−1, 1]," which is incorrect for the temporal substrate.

3. **Calibration is undocumented.** The z-score parameters in the semantic substrate, the normalisation thresholds in the affective substrate, and the biosignal HR centre/scale are all engineering estimates without citation to calibration data or sensitivity analysis. The 1/√3 equal-weight assumption for semantic composition lacks empirical validation.

The assembly is internally consistent and matches the preprint Table 1 feature inventory. The concerns are about reliability, scale comparability, and calibration, not about the structural design of the Ψ vector.

---

## Notes

- **Golden output anomaly:** `psi_temporal = 1.22e-10` in `basic_embeddings_only` fixture confirms the CV denominator instability is not a theoretical concern but an observed artefact.
- **Scope of temporal substrate:** The `turn_synchrony` and `rhythm_score` fields are always `None` — two of the four temporal dimensions are unimplemented stubs. This is not a bug but it narrows what "temporal" means to metric stability only.
- **Legacy parity:** The `extensions.py` implementation (`compute_semantic_substrate`, `compute_temporal_substrate`, `compute_affective_substrate`) is byte-for-byte equivalent to `substrates.py` for the core logic. The deprecation is clean.
- **Affective substrate VADER applicability:** The VADER limitation is real but partially mitigated by using variance (not raw polarity) and by the regex-dominated composition. This is a known limitation in the preprint discussion.
- **Composite weighting:** The composite weights `(0.5, 0.3, 0.2)` / `(0.4, 0.3, 0.2, 0.1)` (`analyzer.py:369–380`) have no documented justification and are not mentioned in the preprint Table 1. These appear to be heuristic choices.
