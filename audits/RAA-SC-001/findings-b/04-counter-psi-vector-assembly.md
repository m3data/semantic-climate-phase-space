# Counter-Finding B4: Ψ Vector Assembly

**Task:** b4-counter-psi-vector-assembly
**Date:** 2026-02-25
**Auditor:** Claude Sonnet 4.6 (Phase B)
**Counter-auditing:** findings-a/04-psi-vector-assembly.md

---

## Scope

Independent re-examination of five questions against live source:
- `src/substrates.py` (full)
- `src/analyzer.py` (full)
- `src/extensions.py` lines 263–800
- `tests/fixtures/golden_outputs.json`

Phase A's verdict (PARTIAL) and seven sub-findings are evaluated below. New findings are appended.

---

## Q1: z-score parameters — calibrated or estimated?

**Phase A verdict: engineering estimates, not empirically calibrated.**

**Counter-verdict: CONFIRMED, with empirical demonstration of misalignment.**

Phase A identified the calibration parameters as developer observations without a corpus. This counter-audit adds concrete evidence from the golden fixture:

```
Actual golden values: Δκ=0.031, ΔH=0.356, α=0.5
Calibration centers:  Δκ=0.15,  ΔH=0.15,  α=0.8
```

The resulting z-scores:
- Δκ: (0.031 − 0.15) / 0.15 = **−0.79** (well below center)
- ΔH: (0.356 − 0.15) / 0.15 = **+1.37** (significantly above center)
- α:  (0.5 − 0.8) / 0.3   = **−1.0**  (one std below center)

The ΔH calibration center of 0.15 is a particularly poor fit: JSD values of 0.35+ appear even in a basic test case. The assumption that Δκ and ΔH share identical center/std (both 0.15/0.15) is falsified by this single example — their actual distributions differ substantially. This is not a calibration edge case; it is the test suite's own reference fixture.

**This means psi_semantic is systematically biased even for "normal" inputs.** A centre error of +1.37σ on ΔH pulls psi_semantic negative by ~0.8 for typical dialogues before tanh compression, regardless of whether the dialogue is genuinely in a low-engagement state.

Phase A's description ("suspiciously symmetric") was the right intuition; the data confirms the calibration is off in a directional, not just symmetric, sense.

---

## Q2: Temporal substrate — CV denominator instability

**Phase A verdict: two material flaws (window < DFA minimum; CV near zero).**

**Counter-verdict: PARTIALLY UPGRADED. The instability is structural and universal, not conditional.**

Phase A attributed the CV blowup to "neutral or random-text embeddings." The actual cause is more fundamental.

The temporal substrate pipeline:
```python
# 1. Collect window metrics across N-window_size+1 windows
window_array = np.array(window_metrics)          # shape: (W, 3)

# 2. Z-standardize ACROSS windows
window_array_std = (window_array - np.mean(window_array, axis=0)) \
                   / (np.std(window_array, axis=0) + 1e-10)  # substrates.py:153

# 3. Project onto equal-weight loadings
window_psi = np.dot(window_array_std, loadings)              # substrates.py:157

# 4. Compute CV of window_psi
cv = np.std(window_psi) / (np.abs(np.mean(window_psi)) + 1e-10)  # substrates.py:159
```

Step 2 **forces** `np.mean(window_array_std, axis=0) = [0, 0, 0]` by construction — that is the definition of z-standardization. Therefore `np.mean(window_psi) = np.dot([0,0,0], loadings) = 0` for any dialogue. The `+1e-10` guard is almost always the entire denominator. CV = std(window_psi) / 1e-10 ≈ 1e10 for any non-constant trajectory, collapsing psi_temporal → 0 universally.

The golden fixtures confirm this is not input-dependent:
- `basic_embeddings_only`: psi_temporal = **1.22e-10**
- `with_hedging_texts`: psi_temporal = **1.76e-10**

Both scenarios trigger the windowed path and produce the same near-zero output. The fallback of 0.5 is only reached when `len(embeddings) < window_size * 2` — meaning only very short dialogues avoid the instability.

**Practical consequence:** In any real-world dialogue with ≥20 turns, psi_temporal ≈ 0 unconditionally. The composite formula `0.3 × psi_temporal` contributes ~0 in all such cases, and the "temporal stability" dimension of Ψ is functionally absent. The positive-bias concern Phase A identified (psi_temporal always pulling composite positive) is actually inverted in practice: the windowed path collapses psi_temporal to 0, not 0.5 — but this still creates a floor artefact distinct from the intended signal.

**Phase A's DFA minimum violation concern is confirmed.** Window guard of 6 samples vs. stated minimum of ~20 compounds the instability.

---

## Q3: VADER for AI dialogue text — Phase A missed the GoEmotions hybrid path

**Phase A verdict: VADER limitation partially mitigated by variance-based usage.**

**Counter-verdict: PARTIALLY INCORRECT. Phase A described the wrong system.**

Phase A characterised the affective substrate as "VADER + regex" and assessed it as a stable function. This is accurate for the **fast path** (`compute_affective_substrate_fast`, substrates.py:177–314). However, as of the 2025-12-21 update, `substrates.py` implements a **two-tier hybrid system** that Phase A did not identify:

**Tier 1 (default): VADER-only fast path**
`compute_affective_substrate_fast` — lexicon + regex, returns `source='vader'`.

**Tier 2 (conditional): GoEmotions hybrid path**
When `emotion_service` is provided to the analyzer (`analyzer.py:90`), `compute_affective_substrate` calls `merge_affective_results` which produces a **structurally different** `psi_affective_refined`:

```python
# substrates.py:461–466
psi_affective_refined = np.tanh(
    0.3 * avg_epistemic +
    0.3 * avg_safety +
    0.2 * fast_result['hedging_density'] * 10 +   # ×10 undocumented
    0.2 * fast_result['vulnerability_score'] * 20  # ×20 undocumented
)
```

This formula:
1. Has scale factors (×10, ×20) that are undocumented and absent from the VADER path
2. Produces outputs in a different practical range than the VADER formula (VADER uses tanh of [0,1] composite; GoEmotions path directly tanhes raw signal sums)
3. Is not described in the preprint Table 1, which shows only one Ψ_affective formula
4. Replaces, rather than supplements, the VADER psi_affective value in the output

**Phase A's claim** that extensions.py and substrates.py are "byte-for-byte equivalent for the core logic" is **incorrect**. The extensions.py `compute_affective_substrate` method (lines 540–665) has no GoEmotions path — it is purely VADER + regex. The new substrates.py introduces a forked execution path invisible to extensions.py.

**Practical impact:** In the default configuration (`emotion_service=None`), Phase A's analysis holds. But any deployment with `emotion_service` configured receives a psi_affective computed via a different formula producing different values from the same input. The preprint's description of Ψ_affective is only accurate for the default path.

---

## Q4: Biosignal substrate — is HR normalization adequate?

**Phase A verdict: Implementation matches preprint. Centre of 80 bpm slightly high; coarse scale.**

**Counter-verdict: CONFIRMED. One additional observation.**

The implementation (substrates.py:504–522) is:
```python
hr = biosignal_data.get('heart_rate')
if hr is not None:
    hr_normalized = (hr - 80) / 40
    return float(np.tanh(hr_normalized))
return 0.0
```

Phase A's analysis is accurate. One additional observation: the function accepts a `biosignal_data` dict that nominally supports `hrv`, `gsr`, and "other biosignals" (per the docstring: "Future versions will incorporate HRV, GSR, and other biosignals"). Any dict keys beyond `heart_rate` are **silently ignored**. This means callers who believe they are providing multi-signal physiological data receive single-signal output with no warning.

The naming "biosignal substrate" implies multi-channel integration. The implementation is a scalar HR normalizer. The discrepancy between the name and the implementation creates a reliability risk if the system is extended by callers who pass HRV or GSR data expecting it to be used.

Phase A's assessment that the preprint accurately documents the formula is confirmed. The inadequacy is in the gap between the conceptual claim ("biosignal substrate") and the trivial implementation.

---

## Q5: Are the four substrates actually combined into a vector?

**Phase A verdict: Substrates computed independently, assembled as dict, collapsed to scalar. Ψ_temporal introduces persistent positive bias.**

**Counter-verdict: CONFIRMED AND EXTENDED.**

Phase A is correct that the "Ψ vector" is a Python dict, not a geometric vector:

```python
# analyzer.py:237–242
psi_vector = {
    'psi_semantic': semantic_result['psi_semantic'],
    'psi_temporal': temporal_result['psi_temporal'],
    'psi_affective': affective_result['psi_affective'],
    'psi_biosignal': psi_biosignal
}
```

**Extended analysis — geometric operations on an incoherent space:**

After assembly, `psi_vector` is appended to `TrajectoryBuffer` (`analyzer.py:246`), which computes geometric trajectory dynamics: velocity, acceleration, curvature, path length. These computations implicitly treat psi_vector as a point in ℝ⁴ and compute Euclidean distances. However:

1. psi_temporal ∈ [0, 1] while others span [−1, 1]: distances in the temporal dimension are compressed relative to other dimensions by a factor of 2
2. psi_biosignal is `None` when no biosignal data is provided (most cases): the geometric operations must handle a 3-dimensional (or partially filled) space
3. psi_temporal is empirically ≈0 for most real dialogues (see Q2): the temporal dimension is effectively collapsed to a point, not a range

These three facts compound: the "4D phase space" described by the preprint is in practice closer to a 3D space with one dimension compressed and another near its floor. Trajectory dynamics (velocity, curvature, path length) computed over this space are systematically distorted.

**Regarding the positive bias:** Phase A identified the theoretical bias from psi_temporal ∈ [0,1]. The empirical correction is that psi_temporal ≈ 0 in practice (not 0.5), so the actual composite bias from the temporal term is near zero for long dialogues — but this is worse: the temporal substrate contributes essentially nothing to the composite, making it a semantic+affective weighted sum in practice.

---

## New Findings

### B4-NEW-1: The GoEmotions formula is an undisclosed preprint deviation

The `merge_affective_results` hybrid path (substrates.py:391–501) produces a `psi_affective` using `avg_epistemic` and `avg_safety` signals from GoEmotions alongside `hedging_density × 10` and `vulnerability_score × 20`. This:

- Is not mentioned in the preprint
- Uses a different weighting structure than the VADER formula in Table 1
- Has no documented rationale for the ×10 and ×20 scale factors
- Produces output under `source='hybrid'` but the preprint does not disclose that two sources exist

This is a material deviation from the preprint's specification: the publication describes one Ψ_affective; the code has two, and they are not interchangeable.

### B4-NEW-2: CV instability is structural — all extended dialogues produce psi_temporal ≈ 0

As detailed in Q2, z-standardizing the window metrics before computing CV mathematically guarantees mean(window_psi) ≈ 0 for any input. The 1e-10 guard determines the CV denominator universally. This is not a bug in the numerical implementation so much as a logical flaw: the intent of CV is to measure relative variability (std/mean), but z-standardisation destroys the mean, making CV meaningless. A signal whose CV is computed after mean-removal is simply measuring std/0 — the guard prevents division by zero but doesn't restore interpretability.

**Fix direction:** The CV should be computed before z-standardisation, or the metric stability should use a scale-free measure (e.g., IQR/median, or the fraction of windows within one std of the trajectory median) that does not require a non-zero mean.

### B4-NEW-3: Single golden fixture data point falsifies Δκ/ΔH shared calibration assumption

From `tests/fixtures/golden_outputs.json`:
```
Δκ = 0.031   (calibration center 0.15, std 0.15)
ΔH = 0.356   (calibration center 0.15, std 0.15)
```

If these two metrics shared the same distributional parameters, we would expect similar z-scores for a typical sample. Instead, the same dialogue produces Δκ z-score ≈ −0.79 and ΔH z-score ≈ +1.37 — a difference of 2.2σ. The identical calibration center/std for Δκ and ΔH is empirically contradicted by the project's own golden fixture. The assumption should have been checked against actual metric outputs before the 2025-12-08 recalibration was committed.

---

## Summary Verdict

**Phase A verdict: PARTIAL — confirmed.**

Phase A's findings on the five specific questions are largely correct, with two corrections and two expansions:

| Phase A finding | Counter-verdict |
|----------------|----------------|
| Q1: z-score params are engineering estimates | **CONFIRMED + EXTENDED** — golden fixture provides empirical evidence of misalignment (ΔH off by 1.37σ; Δκ off by 0.79σ in the same test case) |
| Q2: CV instability near zero | **UPGRADED** — instability is structural (z-standardisation forces mean≈0) and universal for all extended dialogues, not conditional on "neutral" input |
| Q3: VADER limitation partially mitigated | **PARTIALLY INCORRECT** — Phase A missed the GoEmotions hybrid path, which uses a structurally different and undisclosed formula; Phase A's analysis holds only for the default VADER-only configuration |
| Q4: Biosignal is trivially simple HR normalizer | **CONFIRMED** — additional note: other biosignal keys (hrv, gsr) silently ignored |
| Q5: Substrates not combined as a geometric vector | **CONFIRMED + EXTENDED** — trajectory dynamics operate on a geometrically incoherent space (mixed ranges, missing dimensions, near-floor temporal); temporal positive bias is actually near-zero empirically, not 0.3 × 0.5 |

**Overall: Phase A's PARTIAL verdict is correct but understated. The temporal substrate is not merely "unreliable in practice" — it is functionally absent for most dialogues due to structural CV collapse. The affective substrate has an undisclosed dual-formula implementation that the preprint does not describe.**

---

## Notes

- The `psi_temporal ≈ 1e-10` pattern in both golden fixtures (not just `basic_embeddings_only` as Phase A noted, but also `with_hedging_texts`) confirms the universality of the CV collapse.
- The bare `except: pass` at substrates.py:149 and extensions.py:758 silently swallows all exceptions from windowed DFA computations, making debugging the temporal instability difficult without inspection of golden fixtures.
- Phase A correctly noted that turn_synchrony and rhythm_score are always `None` (two of four temporal dimensions are stubs); this is confirmed and unchanged.
- The composite weights (0.5/0.3/0.2 and 0.4/0.3/0.2/0.1) are confirmed to be undocumented heuristics not in the preprint.
