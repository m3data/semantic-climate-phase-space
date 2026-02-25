# RAA-SC-001 Phase A Synthesis

**Audit:** RAA-SC-001 — Semantic Climate Phase Space
**Phase:** A (5 findings)
**Synthesised:** 2026-02-25
**Synthesiser:** Claude Sonnet 4.6 (a-synthesis task)

---

## 1. One-Line Verdicts

| ID | Metric | Verdict | One-line summary |
|----|--------|---------|------------------|
| A1 | DFA/α fractal similarity | **LOW confidence** | Algorithm is structurally correct, but scale range is critically insufficient for typical dialogue lengths (0.0–0.54 log-decades vs. ≥1 required), and the IID bootstrap CI is statistically invalid — α values for the Grunch conversation cannot be distinguished from noise. |
| A2 | Δκ semantic curvature | **PARTIAL** | Core Frenet-Serret formula correctly generalises to high-dimensional space; statistical infrastructure (bootstrap CI, slow-segment mean bias, uncalibrated 0.35 threshold) has addressable weaknesses that affect reported uncertainty but not the classification boolean. |
| A3 | ΔH entropy shift via JSD | **PARTIAL** | JSD formula and shared clustering are mathematically correct and properly implemented; confidence intervals are invalid (CI uses k=8 KMeans only, point estimate averages 6 combinations), and consensus averaging conflates topological granularities. |
| A4 | Ψ vector assembly | **PARTIAL** | Implementation matches preprint Table 1 feature inventory, but Ψ_temporal is unreliable (DFA window < stated minimum, CV denominator instability confirmed by golden-output artefact), substrates have incomparable scales (Ψ_temporal is [0,1] not [−1,1]), and all calibration parameters are undocumented engineering estimates. |
| A5 | Coupling mode classification | **PARTIAL** | Taxonomy faithfully implements preprint Appendix B; however, hidden defaults (Cognitive Mimicry when context absent, Sycophantic Convergence when delta_kappa absent), an incorrect secondary fallback, and soft membership blind to the ambiguous trio mean correct classification is input-conditional in undisclosed ways. |

---

## 2. Concern Clusters

### Cluster 1 — Bootstrap invalidity (A1, A2, A3)

All three lower-level metrics use statistically invalid confidence interval methods:

- **A1/α**: IID resampling destroys temporal autocorrelation structure. Every bootstrap sample produces α ≈ 0.5 by construction; CI reflects white-noise variability, not uncertainty about the observed series.
- **A2/Δκ**: Sorted replacement resampling creates ≈63% duplicate-index segments → zero velocity → κ = 0 artificially, while skipped indices create velocity spikes. CI has inflated lower bound.
- **A3/ΔH**: Bootstrap runs k=8 KMeans only; point estimate averages 6 combinations (kmeans×{6,8,10} + gmm×{6,8,10}). CI is not formally valid for the consensus metric.

The bootstrap problem is systemic, not isolated. Every reported confidence interval in the pipeline is estimating a different quantity than the corresponding point estimate, or is estimating the wrong distribution entirely.

### Cluster 2 — Undocumented / unempirical calibration (A2, A4, A5)

Threshold and parameter values across the pipeline are asserted rather than derived:

- **A2**: The 0.35 curvature threshold was "empirically-derived" per the docstring but has no documented calibration artefact. The threshold was reset after the chord-deviation→local-curvature fix without re-validation.
- **A4**: z-score parameters (Δκ center=0.15/std=0.15, ΔH center=0.15/std=0.15, α center=0.8/std=0.3) are developer observations from a small number of test cases. The identical Δκ/ΔH parameters are suspiciously symmetric. The 1/√3 equal-weight assumption for semantic substrate composition is labelled "PC1 approximation" but PCA was not run. Affective normalisation thresholds (/0.5, /0.1, /0.05) and biosignal HR centre (80 bpm; normal resting is 70–75) are uncited.
- **A5**: All basin entry/exit thresholds, and the performative-vs-genuine feature scoring weights, are described as "derived through iterative threshold analysis during pilot sessions" with no quantitative derivation or inter-rater reliability reported.

### Cluster 3 — Short-sequence / small-sample degradation (A1, A4)

The pipeline assumes longer sequences than typical use produces:

- **A1**: DFA requires ≥1 log-decade of scale range for reliable α estimation, which requires N ≈ 500 velocity samples (501 turns). Typical dialogues (20–60 turns) yield 0.0–0.54 log-decades and 2–11 usable integer scales. At <25 turns, DFA silently falls back to α = 0.5 without notification.
- **A4**: Ψ_temporal is computed via a sliding window of size 10. Ten-embedding windows produce 9 velocity values — below the preprint's own stated DFA minimum of ~20 samples. The golden-output fixture confirms the consequence: psi_temporal ≈ 1.22×10⁻¹⁰ for neutral embeddings due to CV denominator instability near zero.

These are compounding problems: A4 relies on A1's DFA algorithm but at window sizes that A1 already flags as sub-minimal.

### Cluster 4 — Hidden input dependencies and silent defaults (A5)

The classification layer has undisclosed branching on input presence:

- When `raw_metrics` is absent, `delta_kappa` defaults to 0.0, and Sycophantic Convergence intercepts all high-semantic, low-affect observations — making Collaborative Inquiry, Cognitive Mimicry, and Reflexive Performance unreachable.
- When `dialogue_context` is absent, the three-way ambiguous zone defaults to Cognitive Mimicry (mimicry_score 0.7 vs inquiry_score 0.3 vs reflexive_score 0.2 from neutral defaults).
- A secondary fallback path returns Cognitive Mimicry for affect-dominant states (line 711), contradicting Cognitive Mimicry's defining features (high semantic, low affect). This path is undocumented in the preprint.

### Cluster 5 — Scale incompatibility in Ψ assembly (A4)

Ψ_temporal is bounded [0,1] while Ψ_semantic and Ψ_affective can be negative. The composite formula `0.5×Ψ_semantic + 0.3×Ψ_temporal + 0.2×Ψ_affective` therefore has a structural positive bias: a fully neutral dialogue (Ψ_semantic = Ψ_affective ≈ 0) yields psi ≈ 0.3×Ψ_temporal ≠ 0. The preprint's prose describes all substrates as "compressed to [−1,1]," which is factually incorrect for Ψ_temporal.

---

## 3. Findings Most Worth Adversarial Review

### Priority 1 — A1: DFA/α Scale Range (P1)

This is the highest-priority finding for adversarial review because:

1. α is the *specific metric* in the preprint's load-bearing Grunch claim (α = 0.77–1.27 during denial phase → lock-in).
2. The scale-range failure is quantitatively documented and mechanistically clear — it is not an interpretation dispute.
3. A1 is the only finding with verdict **LOW confidence** (not PARTIAL); the others acknowledge correct core computation.
4. The preprint's stated range (0.77–1.27) spans the 0.77–1.2 "healthy vs. lock-in" boundary by exactly the margin that the noise floor makes unresolvable at typical dialogue lengths.
5. The fallback (α = 0.5 at <25 turns) is silent — any analysis reporting α for short conversations may be reporting a constant.

**Adversarial question:** For the Grunch denial-phase conversation, how many turns does it contain? If <100 turns, the α estimates are in the unreliable regime. Is there any segment-level analysis, or is this a single α for the full conversation?

### Priority 2 — A4: Ψ_temporal CV Denominator Instability (confirmed artefact)

This deserves adversarial review because it is not theoretical — it is observed in the golden-output fixture:

- `psi_temporal = 1.22e-10` for neutral-embeddings input is a confirmed artefact, not a concern about edge cases.
- Ψ_temporal contributes 30% of the composite psi scalar.
- A conversation that should read as "neutral / undetermined" has Ψ_temporal ≈ 0, which then shifts the composite via the `[0,1]` vs `[−1,1]` asymmetry.

**Adversarial question:** Does the Grunch analysis report Ψ_temporal close to 0? If so, the "lock-in" composite may be partly artefactual (psi_temporal collapsing + positive structural bias).

### Priority 3 — A5: Cognitive Mimicry as Default (hidden classification bias)

This deserves adversarial review because the three theoretically critical basins (Collaborative Inquiry, Cognitive Mimicry, Reflexive Performance) are the ones used to distinguish performative from genuine engagement — the core scientific claim of the cross-substrate coupling preprint. The default behaviour systematically biases this distinction:

- Without full dialogue_context, the ambiguous zone always returns Cognitive Mimicry (performative label) with high confidence.
- `coherence_pattern` appears to never be computed in the visible pipeline (always defaults to 'transitional'), which gives Cognitive Mimicry a 0.7 vs Collaborative Inquiry's maximum 0.6 structural advantage.

**Adversarial question:** What dialogue_context features were provided for the Grunch conversation? Was coherence_pattern computed, or was it left at the default 'transitional'? If the latter, Collaborative Inquiry was structurally disadvantaged by the implementation.

---

## 4. Contradictions Between Findings

No direct logical contradictions were identified between findings. However, two compounding tensions exist:

**Tension A: A1 and A4 share a DFA minimum, but A4 violates it more severely.**
A1 finds that a minimum of ~25 turns is required for any DFA output, and ~40–100 turns for marginal reliability. A4 then finds that the temporal substrate calls DFA with window_size=10 (9 velocity values). This is not a contradiction — it is A4 inheriting A1's limitation at an even smaller scale, compounding the reliability problem. A4's temporal substrate is doubly unreliable: DFA is called below its own minimum, and the CV aggregation of those unreliable values is then numerically unstable.

**Tension B: A3 says "no preprint-blocking issue" for ΔH, but A4 includes ΔH as an input to Ψ_semantic with uncalibrated z-scores.**
A3's preprint-impact note rates ΔH's CI mismatch as non-blocking (the point estimate is likely reasonable, just the CI is approximate). A4 then shows that ΔH enters Ψ_semantic via an uncalibrated z-score (center=0.15, std=0.15 — same values as Δκ, suggesting copy-paste rather than derivation). The propagation of ΔH's granularity-averaging through an uncalibrated z-score means the non-blocking concern at A3 compounds with the uncalibrated compression at A4. Neither finding flags this downstream consequence individually.

---

## 5. Grunch Lock-In Claim Assessment

**Claim:** α = 0.77–1.27 during the Grunch denial phase indicates semantic lock-in.

This claim is affected at three points in the pipeline:

### Direct threat — A1 (scale range)

The finding explicitly notes: *"The preprint's reported range α = 0.77–1.27 for the Grunch conversation may not be distinguishable from noise given these scale constraints."*

- If the Grunch denial phase is a typical-length dialogue segment (20–60 turns), the scale range is 0.0–0.54 log-decades.
- The difference between α = 0.77 and α = 1.27 — the span of the reported range — is larger than the uncertainty floor, but the difference between α = 1.0 ("healthy") and α = 1.2 ("lock-in threshold") may not be.
- The bootstrap CI is invalid (IID resampling), so the reported interval cannot be taken as a credible uncertainty bound.
- **No test exists** for a synthetic locked-in signal producing α > 1.2 at realistic dialogue lengths. The threshold is asserted, not validated.

### Propagation threat — A4 (Ψ vector)

α enters Ψ_semantic via z-scoring with `center=0.8, std=0.3`. The reported α range of 0.77–1.27 maps to z-scores of approximately −0.1 to +1.57, giving Ψ_semantic contributions of ≈ −0.06 to +0.91 from α alone. If the center/std are off (which A4 indicates is likely given the uncalibrated status), the contribution of α to Ψ_semantic is correspondingly biased.

The Ψ_temporal artefact (psi_temporal ≈ 0 for neutral embeddings, positive structural bias) means the composite psi scalar for the Grunch conversation may show elevation for reasons unrelated to semantic lock-in.

### Classification threat — A5 (basin detection)

Basin classification for the Grunch conversation depends on inputs that A5 identifies as potentially incomplete:
- If `dialogue_context` was not fully populated (specifically `coherence_pattern`), the three-way discrimination defaults in ways that advantage Cognitive Mimicry over Collaborative Inquiry.
- The soft membership computation does not discriminate between Collaborative Inquiry, Cognitive Mimicry, and Reflexive Performance — these basins have centroids separated by only 0.005–0.02 squared distance.

### Summary impact on claim

The lock-in claim rests primarily on α, which has the most serious reliability problem (A1, verdict: LOW). The claim's specific form — a *range* of α values (0.77–1.27) rather than a single value — partially hedges against point-estimate noise, but the hedge is insufficient: the range spans the 0.77–1.2 "healthy" zone and the >1.2 "lock-in" zone simultaneously, meaning the claimed evidence for lock-in is the upper portion of the range, which is precisely where the noise floor is most consequential.

**The claim requires Phase B adversarial review focused on:**
1. Conversation length for the Grunch denial phase (is α reliable at that length?)
2. Whether the bootstrapped CI was reported alongside α = 0.77–1.27 (and if so, whether it was the invalid IID CI)
3. Whether any segment-level analysis (not just conversation-level α) was used to ground the lock-in interpretation

---

## Overall Phase A Assessment

| Concern | Findings | Severity | Preprint Impact |
|---------|----------|----------|-----------------|
| DFA scale range → unreliable α | A1 | **Critical** | Directly threatens Grunch lock-in claim |
| IID bootstrap CIs (invalid across pipeline) | A1, A2, A3 | **High** | All reported CIs are untrustworthy |
| Ψ_temporal CV instability (confirmed artefact) | A4 | **High** | Composite psi bias; 30% weight unreliable |
| Undocumented calibration throughout | A2, A4, A5 | **Medium** | Thresholds and weights are assertions, not evidence |
| Ψ_temporal scale incompatibility | A4 | **Medium** | Structural positive bias; preprint prose incorrect |
| Hidden classification defaults | A5 | **Medium** | Performative/genuine distinction unreliable without full context |
| Curvature mean bias (slow segments) | A2 | **Low** | Δκ point estimate elevated in slow-trajectory conversations |
| Consensus granularity averaging | A3 | **Low** | ΔH consensus metric is a mixed estimator |
| U-turn degenerate case (κ = 0 on reversal) | A2 | **Low** | Negligible probability in high-dimensional embedding space |
| Cognitive Mimicry secondary fallback error | A5 | **Medium** | Incorrect labels for affect-dominant states |

The pipeline's core mathematical components (DFA algorithm, Frenet-Serret curvature, JSD, shared clustering) are structurally correct. The weaknesses accumulate in the statistical infrastructure (CIs, calibration, scale comparability) and the classification layer (hidden defaults). The Grunch lock-in finding is the most vulnerable because it depends on the metric with the most serious reliability problem (α / DFA) and because the reported α range straddles the threshold used to interpret lock-in.

Phase B should prioritise: (1) bootstrap remediation for α CIs; (2) verification of the Grunch conversation length against the DFA scale-range table in A1; (3) dialogue_context feature completeness for the basin classifications reported in the preprint.
