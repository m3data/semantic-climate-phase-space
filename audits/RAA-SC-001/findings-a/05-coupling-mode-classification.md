## Claim

Coupling mode classification (attractor basin detection) correctly maps Ψ position and raw metrics into 10 canonical dialogue configurations with discriminating features between performative and genuine engagement.

## Files Examined

- `/Users/m3untold/Code/EarthianLabs/worktrees/semantic-climate-phase-space/RAA-SC-001-a5-coupling-mode-classification-v1/src/basins.py` — lines 1–899 (full file)
- `/Users/m3untold/Code/EarthianLabs/worktrees/semantic-climate-phase-space/RAA-SC-001-a5-coupling-mode-classification-v1/tests/test_basins.py` — lines 1–522
- `/Users/m3untold/Code/EarthianLabs/publications/papers/cross-substrate-coupling-preprint.md` — lines 502–535 (Appendix B)

## Evidence

### 1. Basin Reachability

All 10 basins are reachable under some input combination.

**Reachability map** (in order of threshold checks in `_classify_basin()`, lines 592–715):

| Basin | Reaching Conditions |
|-------|---------------------|
| Deep Resonance | sem > 0.4, aff > 0.4, (bio > 0.4 or bio == 0.0) |
| Dissociation | abs(sem) < 0.2, abs(aff) < 0.2, abs(bio) < 0.2 |
| Embodied Coherence | abs(sem) < 0.3, bio > 0.3 |
| Generative Conflict | abs(sem) > 0.3, delta_kappa > 0.35, aff > 0.3 |
| Creative Dilation | delta_kappa > 0.35, aff > 0.3, sem ≤ 0.3 (after GC check fails) |
| Sycophantic Convergence | sem > 0.3, delta_kappa < 0.35, aff < 0.2 |
| Collaborative Inquiry | sem > 0.3, aff < 0.2, bio < 0.2, dk ≥ 0.35, inquiry_score highest |
| Cognitive Mimicry | sem > 0.3, aff < 0.2, bio < 0.2, dk ≥ 0.35, mimicry_score highest |
| Reflexive Performance | sem > 0.3, aff < 0.2, bio < 0.2, dk ≥ 0.35, reflexive_score highest |
| Transitional | No prior condition met, and no dominant dimension with qualifying dk |

**FINDING:** All 10 basins are reachable. However, the path to Collaborative Inquiry, Cognitive Mimicry, and Reflexive Performance requires `delta_kappa >= 0.35` (otherwise Sycophantic Convergence intercepts). Without `raw_metrics` input (delta_kappa defaults to 0.0), the three-way ambiguous zone is **never reached** — all high-semantic, low-affect, low-biosignal observations are classified as Sycophantic Convergence. This is a hidden input dependency not documented in the preprint.

### 2. Threshold Ordering

The ordering creates two significant shadowing effects:

**Effect A — Sycophantic Convergence shadows CI/CM/RP when delta_kappa = 0.**
Sycophantic Convergence check (line 622): `sem > 0.3 and delta_kappa < 0.35 and aff < 0.2`.
The three-way refined check (line 630): `abs(sem) > 0.3 and aff < 0.2 and bio < 0.2`.
When delta_kappa = 0 (no raw_metrics), the three-way zone is unreachable. This means callers that do not provide `raw_metrics` will never observe Collaborative Inquiry, Cognitive Mimicry, or Reflexive Performance — the taxonomy's most theoretically important basins for performative/genuine discrimination.

**Effect B — Undocumented secondary fallback (lines 697–715).**
After all primary checks fail, a secondary block can return Creative Dilation, Generative Conflict, Cognitive Mimicry, or Embodied Coherence before reaching Transitional. Notably, `Cognitive Mimicry` is returned when the dominant substrate is affective but delta_kappa ≤ 0.35 (line 711). This contradicts Cognitive Mimicry's defining features (high semantic activity, low affect) and is not documented in the preprint, which states the default when "no threshold met" is Transitional.

**Effect C — Deep Resonance bio == 0.0 condition (line 595).**
`(bio > 0.4 or bio == 0.0)` means all conversations without biosignal that have sem > 0.4 and aff > 0.4 are classified as Deep Resonance. Since most sessions in the corpus lack biosignal (24 of 34), this condition makes Deep Resonance the implicit classification for any emotionally engaged high-semantic exchange, without any biosignal evidence.

### 3. Performative/Genuine Distinction

**Code implements preprint Appendix B.1 table exactly.** The four-feature discrimination table (hedging_density, turn_length_ratio, delta_kappa_variance, coherence_pattern) matches the preprint precisely:

| Feature | Inquiry | Mimicry | Reflexive | Preprint Match |
|---------|---------|---------|-----------|----------------|
| Hedging > 0.02 | +0.3 | — | — | ✓ |
| Hedging < 0.01 | — | +0.3 | — | ✓ |
| Hedging 0.01–0.03 | — | — | +0.3 | ✓ |
| Turn ratio 0.5–2.0 | +0.3 | — | — | ✓ |
| Turn ratio > 2.0 | — | +0.3 | — | ✓ |
| Turn ratio > 1.5 | — | — | +0.2 | ✓ |
| dk_variance > 0.01 | +0.2 | — | — | ✓ |
| dk_variance < 0.005 | — | +0.2 | — | ✓ |
| dk_variance 0.005–0.015 | — | — | +0.3 | ✓ |
| Coherence breathing | +0.2 | — | — | ✓ |
| Coherence locked/transitional | — | +0.2 | — | ✓ |
| Coherence transitional | — | — | +0.2 | ✓ |

**Concern: thresholds are asserted, not empirically derived.** The preprint describes these as "derived through iterative threshold analysis during pilot sessions." No quantitative derivation, validation corpus, or inter-rater reliability is reported. The hedging range overlap (0.02–0.03 contributes to both Inquiry and Reflexive Performance simultaneously) is a design choice that may increase noise at the boundary.

**Concern: missing dialogue_context defaults to Cognitive Mimicry (hidden bias).** When called without `dialogue_context`:
- inquiry_score = 0.3 (turn_ratio 1.0 is within 0.5–2.0)
- mimicry_score = 0.7 (hedging 0.0 < 0.01, dk_variance 0.0 < 0.005, coherence 'transitional')
- reflexive_score = 0.2 (coherence 'transitional')

Cognitive Mimicry wins with margin 0.4 > 0.1 threshold, giving it high confidence via `abs(sem) * (0.5 + 0.7 * 0.5)`. This default behaviour is not disclosed in the preprint and means that any system component that omits `dialogue_context` will systematically classify the ambiguous zone as performative rather than genuine engagement.

### 4. Hysteresis

`HysteresisConfig` correctly implements entry < exit threshold asymmetry for all 10 basins:
- Entry thresholds: 0.20–0.35 across basins
- Exit thresholds: 0.30–0.45 across basins (consistently higher)

The state machine in `detect_with_hysteresis()` (lines 423–556) implements UNKNOWN → PROVISIONAL → ESTABLISHED progression, and prevents exit from ESTABLISHED until exit threshold is crossed.

**Concern: conceptual confusion in exit threshold comparison (lines 525–538).**
When `state_status == 'established'` and a new basin is proposed:
```
if raw_confidence < current_config.exit_threshold:
    final_basin = current_basin  # stay
```
Here, `raw_confidence` is the classifier's confidence in the **proposed new basin**, while `current_config.exit_threshold` is the exit threshold for the **current basin**. These measure different things. The logic should compare the residual confidence in the current basin, not the confidence in the proposed replacement. In practice this works as a proxy (high confidence in a new basin implies departure from the old one) but it is not a rigorous implementation of threshold-based hysteresis. Under edge cases — e.g., where two basins simultaneously have high confidence — the comparison could prevent valid exits.

**Confidence scale issue.** The exit thresholds (0.30–0.45) are fixed values, but raw confidence from `_classify_basin()` is computed heterogeneously:
- Deep Resonance: `min(sem, aff)` — can approach 1.0
- Sycophantic Convergence: `sem * (1.0 - dk/0.35) * (1.0 - abs(aff))` — product of three terms, generally low
- Three-way basins: `abs(sem) * (0.5 + best_score * 0.5)` — bounded by sem magnitude

There is no calibration of raw_confidence against a common scale, so exit thresholds of 0.35–0.45 may be too tight for Deep Resonance (where confidence approaches 1.0 easily) and too loose for Sycophantic Convergence (where confidence is typically 0.1–0.3 due to the product formula).

### 5. Soft Membership

`compute_soft_membership()` (lines 717–803) correctly implements softmax on negative squared distances to basin centroids. The implementation is numerically stable (max subtraction before exponentiation). KL divergence for distribution shift is included with epsilon guard.

**Concern: ambiguous-zone centroids are nearly identical in psi space.**

Pairwise squared distances among the three ambiguous basins in (psi_sem, psi_aff, dk, psi_bio) space:

| Pair | Squared Distance |
|------|-----------------|
| Collaborative Inquiry vs Reflexive Performance | 0.0025 |
| Collaborative Inquiry vs Cognitive Mimicry | 0.02 |
| Reflexive Performance vs Cognitive Mimicry | 0.0125 |

CI and RP differ by only 0.05 in delta_kappa and nothing else (distance² = 0.0025). At temperature=1.0, softmax will assign nearly identical weights to these two basins for any input in the ambiguous zone, accurately reflecting ambiguity. However, the actual discrimination between CI, CM, and RP depends on dialogue_context features (hedging, turn_ratio, dk_variance, coherence_pattern) — dimensions **not represented in the centroids**.

The centroids do not include the dialogue context dimensions. Soft membership therefore cannot distinguish between the three ambiguous basins regardless of the dialogue context provided. The preprint claims soft membership "provides continuous confidence estimates," but for the most theoretically important basins, these estimates are uninformative. A position solidly in the Collaborative Inquiry zone (based on dialogue context) will still show near-equal weights for CI, CM, and RP in soft membership.

### 6. Over-specification of the Taxonomy

The 4D input (psi_semantic, psi_affective, psi_biosignal, delta_kappa) supports clear partitioning for 7 basins (Deep Resonance, Dissociation, Embodied Coherence, Generative Conflict, Creative Dilation, Sycophantic Convergence, Transitional). The remaining 3 basins (Collaborative Inquiry, Cognitive Mimicry, Reflexive Performance) require 4 additional dialogue-context features. The taxonomy therefore implicitly requires 8 input dimensions to be fully specified.

This is an honest architectural choice (the preprint acknowledges the ambiguous zone requires dialogue context), but it means:
1. The claim "maps Ψ position and raw metrics into 10 canonical configurations" is incomplete — the 3 theoretically critical basins require dialogue context that is separate from both Ψ and raw metrics.
2. Any pipeline component that provides psi_vector and raw_metrics but not dialogue_context will silently default to Cognitive Mimicry for the ambiguous zone.

### 7. Preprint Taxonomy Match

The code substantially matches Appendix B. The 10 basins, their characterisations, and the Appendix B.1 feature table are faithfully implemented. The "highest score wins" rule with 0.1 ambiguity margin (line 689, preprint §B.1) is correctly implemented. Confidence halving on ambiguous scores (`abs(sem) * 0.5`, line 692) matches the preprint's stated behaviour.

Minor discrepancy: basin numbering in code comments (e.g., "Basin 7: Deep Resonance") does not match position in the `BASINS` list (Deep Resonance is position 1). This is cosmetic but adds confusion.

## Finding

**Verdict: PARTIAL**

The coupling mode taxonomy is well-designed and the code faithfully implements the preprint's Appendix B specification. All 10 basins are reachable, the feature-based discrimination table matches the preprint exactly, hysteresis implements asymmetric thresholds, and the soft membership computation is mathematically correct.

However, four substantive concerns prevent a CONFIRMED verdict:

1. **Default input bias**: When `raw_metrics` is absent (delta_kappa = 0), Sycophantic Convergence intercepts all high-semantic, low-affect observations, making the three theoretically critical basins unreachable. When `dialogue_context` is absent, the ambiguous zone defaults to Cognitive Mimicry with high confidence. Neither behaviour is disclosed in the preprint.

2. **Secondary fallback returns Cognitive Mimicry for affect-dominant states** (line 711), contradicting Cognitive Mimicry's defining features. This undocumented path can produce incorrect labels.

3. **Hysteresis exit comparison is conceptually confused**: `raw_confidence` of the proposed new basin is compared against the exit threshold of the current basin. These are not equivalent quantities; the comparison works heuristically but is not a principled implementation of hysteresis.

4. **Soft membership does not discriminate the ambiguous trio**: Centroids for Collaborative Inquiry, Reflexive Performance, and Cognitive Mimicry differ by only 0.005–0.02 in squared distance, and the dialogue context dimensions that actually discriminate these basins are not included in the centroid representation. The soft membership distributions for these basins will be near-uniform regardless of dialogue context.

The taxonomy's theoretical distinction between performative and genuine engagement is properly operationalised in the explicit classification path, but the default-input behaviour, secondary fallback, and soft membership limitations mean that the claim holds only for well-formed inputs with all context features provided.

## Notes

- Test coverage for Reflexive Performance is absent — there is no test asserting that a specific input produces Reflexive Performance as the primary label (only implicit coverage through the three-way test at line 397).
- The `coherence_pattern` value is never computed anywhere in the visible pipeline; it defaults to `'transitional'`, which systematically biases against Collaborative Inquiry (which requires `'breathing'`). If `coherence_pattern` is always `'transitional'` in production, Collaborative Inquiry's maximum achievable score is 0.6 (hedging + turn_ratio + dk_variance), while Cognitive Mimicry can reach 0.7 and Reflexive Performance 0.8. This structural advantage for Reflexive Performance over Collaborative Inquiry is unexamined.
- The `bio == 0.0` condition in Deep Resonance (line 595) was likely intended to handle missing biosignal, but treating absence as vacuous satisfaction inflates Deep Resonance detection in the dominant (semantic-only) use case.
- Confidence values are not calibrated across basins; the exit threshold comparisons in hysteresis assume a comparable scale that the heterogeneous confidence formulas do not provide.
