# RAA-SC-001 Convergence Report

**Audit:** RAA-SC-001 — Semantic Climate Phase Space
**Evaluator:** Claude Sonnet 4.6 (convergence-eval task)
**Date:** 2026-02-25
**Phase A findings:** `findings-a/01–05` + `SYNTHESIS-A.md`
**Phase B findings:** `findings-b/01–05` (counter-audits)

---

## 1. Executive Summary

Five claims across the Semantic Climate Phase Space pipeline were audited in Phase A and adversarially counter-audited in Phase B. The two phases converge well on the existence and direction of every major problem but diverge on severity in two cases and on coverage — Phase B consistently found additional issues Phase A missed. No Phase A finding was overturned. Overall convergence score: **0.73 / 1.0**.

The core claim that α = 0.77–1.27 during the Grunch denial phase indicates semantic lock-in is **not definitively invalidated** but is **not fully supportable** at current evidence quality. The claim requires either (a) demonstration that the Grunch conversation is long enough for reliable DFA estimation, (b) a stationarity test ruling out the conversational-arc confound, and (c) out-of-sample validation of the α > 1.2 threshold. Without these, the preprint's lock-in interpretation must be qualified to a directional tendency, not a threshold crossing.

---

## 2. Per-Finding Convergence Scores

### Finding 1 — DFA/α Fractal Similarity

| | Phase A | Phase B |
|---|---|---|
| **Verdict** | LOW confidence | DOWNGRADE (Phase A too conservative on stationarity; all A findings confirmed) |
| **Algorithm correctness** | PASS | PASS (confirmed) |
| **Scale range** | FAIL P1 | CONFIRM P1 + extended with stationarity concern |
| **Bootstrap CI** | FAIL P2 | CONFIRM P2 + two additional failure modes |
| **Lock-in threshold** | P3 (not validated) | P2 elevated to circularly derived — same data |

**Convergence score: 0.82**

Both phases agree on every Phase A finding. Phase B adds a harder-to-fix, conceptual-level concern: semantic velocity has structural non-stationarity (bounded distribution, systematically declining variance over the conversational arc) that DFA-1 cannot detrend. A conversation exhibiting lock-in would show declining semantic velocity — precisely the pattern DFA-1 would estimate as high α regardless of true fractal structure. Phase A frames the problems as "fixable"; Phase B argues one layer is not fixable without either a different algorithm or a stationarity test. This is an extension, not a contradiction.

**Phase B net additions:**
- B1-F1 (P1): Semantic velocity has structural non-stationarity; DFA-1 conflates conversational arc with power-law correlation
- B1-F2 (P2): Bootstrap helper uses 10 vs 20 scales; scale collapse (not just IID resampling) drives bootstrap to 0.5 for short sequences
- B1-F3 (P2): α > 1.2 threshold and [0.70, 0.90] healthy range are circularly derived from the Grunch dataset itself — cannot provide independent evidence

---

### Finding 2 — Δκ Semantic Curvature

| | Phase A | Phase B |
|---|---|---|
| **Verdict** | PARTIAL | PARTIAL (corrections to three points + two new findings) |
| **Core formula** | PASS | PASS (confirmed; full algebraic derivation given) |
| **Bootstrap bias direction** | "Partial cancellation" | Both effects are downward; CI can be entirely below the point estimate |
| **Zero-velocity + bootstrap interaction** | Independent concerns | Coupled: ~36.8% of bootstrap κ values are forced to 0 by construction |
| **Permutation null** | "Valid and interpretable" | Structurally misaligned; near-universal small p-values for all real dialogue |

**Convergence score: 0.63**

The core finding (formula correct, statistical infrastructure has addressable weaknesses) converges. Two meaningful disagreements on severity:

1. **Bootstrap bias direction:** Phase A says the two biases "partially cancel." Phase B's analysis shows both effects (duplicates → κ=0; gaps → κ/g for g≥2) are downward, compounding rather than cancelling. The CI can be positioned entirely below the point estimate, not merely with an inflated lower bound.

2. **Permutation null validity:** Phase A calls it "valid and interpretable." Phase B argues it is vacuously uninformative because all real conversations have strong local autocorrelation — they will always appear smoother than a random permutation, making p-value < 0.05 for essentially all inputs regardless of complexity.

**Phase B net additions:**
- B2-N4: Forward-biased stencil — curvature attributed to "point i" is the curvature of the arc (e[i], e[i+1], e[i+2]), not centred at e[i+1]
- B2-N5: Embedding normalisation ambiguity — chord-based κ underestimates geodesic curvature for L2-normalised inputs with large semantic steps; threshold 0.35 may not transfer across normalisation conventions

---

### Finding 3 — ΔH Entropy Shift via Jensen-Shannon Divergence

| | Phase A | Phase B |
|---|---|---|
| **Verdict** | PARTIAL | Phase A verdict upheld; Phase A was too lenient |
| **JSD formula** | PASS | PASS (confirmed; detailed) |
| **Shared clustering** | PASS | PASS (confirmed) |
| **Bootstrap-estimator mismatch** | Primary concern (P1) | CONFIRM + extended with sparse estimation and clustering variance |
| **Minimum sample threshold** | Flagged n_pre < k as edge case | Promotes to normal operating condition: n_pre ≥ 5k required (n_pre=10, k=10 is maximum-noise case) |
| **stability_score bounds** | Implicitly accepted `min(1,...)` as real guard | `min(1,...)` is dead code — score cannot exceed 1 by construction |

**Convergence score: 0.75**

Good agreement on the core (JSD correct, shared clustering correct, bootstrap mismatch is primary flaw). Two material Phase B contributions:

1. **Statistical degeneracy at k=n/2:** Phase A flagged n_pre < k as a degenerate edge case. Phase B shows the standard 20-turn conversation with k=10 (k = n_pre) is equally degenerate — SE/mean ≈ 100% at k=10, 71% at k=6. This is not an edge case; it is the default operating condition for any conversation under ~60 turns.

2. **GMM with n≈k in 768 dimensions:** Phase A did not examine this. A full-covariance GMM fitting ~2 observations per component in 768 dimensions is numerically degenerate. The GMM column of the consensus average is artefact of regularisation, not data geometry.

**Phase B factual correction:** `stability_score > 1` is mathematically impossible (JSD ≥ 0 ⟹ ratio ≥ 0 ⟹ score ≤ 1). The `min(1, ...)` guard is dead code, suggesting imprecise range reasoning in the original implementation.

**Phase B net additions:**
- B3-N5: `transition_summary` reports the best-silhouette single combination's JS value, not the consensus mean — potentially contradicting the numeric metric in the same output

---

### Finding 4 — Ψ Vector Assembly

| | Phase A | Phase B |
|---|---|---|
| **Verdict** | PARTIAL | CONFIRMED AND EXTENDED |
| **z-score calibration** | Engineering estimates, suspicious symmetry | CONFIRM + empirical evidence: ΔH z-score off by 1.37σ in golden fixture |
| **CV instability** | "Conditional on neutral/random embeddings" | Structural and universal — z-standardisation mathematically forces mean≈0 for all extended dialogues |
| **VADER limitation** | VADER+regex, limitation partially mitigated | MISS: did not identify the GoEmotions hybrid path (different formula, undisclosed in preprint) |
| **Biosignal substrate** | Correctly assessed | CONFIRM + HRV/GSR keys silently ignored |
| **Substrate scale incompatibility** | [0,1] vs [−1,1], positive bias | CONFIRM + empirical correction: psi_temporal ≈ 0 in practice (not 0.5), so temporal dimension is functionally absent |

**Convergence score: 0.65**

Two material disagreements:

1. **CV instability scope:** Phase A frames this as conditional on "neutral or random-text embeddings." Phase B shows it is structural: z-standardisation forces `mean(window_array_std) = 0` by construction, making `mean(window_psi) = np.dot([0,0,0], loadings) = 0` for any dialogue. The 1e-10 guard is universally the denominator. Both golden fixtures (basic_embeddings_only AND with_hedging_texts) confirm near-zero psi_temporal.

2. **GoEmotions hybrid path:** Phase A missed that `substrates.py` has a two-tier affective implementation. When `emotion_service` is configured, `merge_affective_results` produces psi_affective via a structurally different formula (using avg_epistemic, avg_safety, with undocumented ×10 and ×20 scale factors) that is not described in the preprint. The preprint's Ψ_affective description is only accurate for the default VADER-only configuration.

**Phase B net additions:**
- B4-NEW-1 (P2): GoEmotions path is an undisclosed preprint deviation
- B4-NEW-2 (P2): CV instability is structural — all extended dialogues produce psi_temporal ≈ 0 due to mathematical necessity of z-standardisation before CV
- B4-NEW-3 (P2): Golden fixture empirically contradicts the assumption that Δκ and ΔH share calibration parameters (2.2σ difference from same input)

---

### Finding 5 — Coupling Mode Classification

| | Phase A | Phase B |
|---|---|---|
| **Verdict** | PARTIAL | PARTIAL (all Phase A concerns confirmed + 4 new findings) |
| **Absent raw_metrics → Sycophantic Convergence** | IDENTIFIED | CONFIRM |
| **Absent dialogue_context → Cognitive Mimicry** | IDENTIFIED | CONFIRM + quantified (mimicry_score 0.7 vs inquiry_score 0.3) |
| **Secondary fallback contradiction** | IDENTIFIED (line 711) | CONFIRM |
| **Hysteresis conceptually confused** | IDENTIFIED | CONFIRM |
| **Soft membership for ambiguous trio** | Near-uniform due to centroid proximity | CONFIRM + extended: T=1.0 causes near-uniform across all 10 basins (CI=11.3% at exact CI centroid) |

**Convergence score: 0.78**

All Phase A concerns confirmed. Phase B contributions are primarily extensions and quantifications, not corrections.

**Phase B net additions:**
- B5-N1 (P2): Deep Resonance check fires before GC/CD and contains no delta_kappa guard — for all bio=0 sessions (majority of corpus), the taxonomy cannot distinguish convergent resonance from creative tension
- B5-N2 (P2): BASIN_CENTROIDS placed in empirically unreachable regions — psi_affective and psi_biosignal centroids exceed observed maxima in golden fixtures; centroid geometry derived from threshold boundary midpoints, not corpus observations
- B5-N3 (P2): Negative psi_semantic states (common in practice) are uniformly distant from all centroids, making soft membership uninformative in the negative-semantic half of the space
- B5-N4 (P2): Shannon capacity analysis confirms structural 1.0–1.3 bit deficit in 4D centroid space — 10-basin taxonomy requires 8D space for full discrimination; 4D provides only ~2.0–2.3 bits for the 6-basin semantic-active cluster

---

## 3. Overall Convergence Score

| Finding | Score | Classification |
|---------|-------|----------------|
| F1: DFA/α | 0.82 | High |
| F2: Δκ curvature | 0.63 | Moderate |
| F3: ΔH JSD | 0.75 | Good |
| F4: Ψ assembly | 0.65 | Moderate |
| F5: Basin classification | 0.78 | High |
| **Overall** | **0.73** | **Good** |

**Interpretation:** The two phases are broadly convergent — no Phase A finding was overturned, and the overall direction of every verdict was confirmed. The lower convergence in F2 and F4 reflects genuine analytical gaps in Phase A (permutation null severity, GoEmotions path omission, CV instability universality), not contradictions.

---

## 4. Structurally Unresolvable Findings

These findings cannot be resolved by static code analysis alone and require empirical investigation:

**U1 — Grunch conversation length**
The DFA scale-range table in A1 shows reliable α estimation requires ~100 turns. The Grunch denial phase length is not determinable from code. If <100 turns, the scale range is ≤0.78 decades and the α estimates are unreliable by the preprint's own implicit standards. This is the single highest-priority unresolvable issue.

**U2 — Semantic velocity stationarity**
Whether semantic velocity from real dialogue satisfies DFA-1's homogeneous-variance assumption cannot be determined without running a stationarity test (e.g., KPSS on velocity subwindows) on actual Grunch data. Phase B's non-stationarity concern is structurally compelling but empirically unconfirmed.

**U3 — Lock-in threshold external validity**
Whether α > 1.2 reliably distinguishes semantic lock-in from healthy persistence requires an independent validation corpus. The threshold is currently set on the same data being evaluated — a circularity that cannot be resolved by code inspection.

**U4 — Production affective path**
Whether the Grunch preprint results used the VADER-only or GoEmotions hybrid path for psi_affective cannot be determined from code alone. The two paths produce different numerical outputs.

**U5 — Calibration parameter validity**
Whether the z-score parameters (center=0.15/std=0.15 for Δκ and ΔH, center=0.8/std=0.3 for α) are approximately correct for the intended use case requires an empirical corpus comparison. The golden fixture already shows the ΔH calibration is off by 1.37σ for at least one test case.

---

## 5. Does Anything Invalidate the Core Lock-In Claim?

**Claim under assessment:** α = 0.77–1.27 during the Grunch denial phase indicates semantic lock-in.

**Short answer:** The claim is not invalidated in the sense of being demonstrated wrong. It is not supportable as stated because its primary metric is unreliable at typical dialogue lengths and its interpretation threshold is circularly derived.

### What the evidence establishes

The DFA algorithm is structurally correct (both phases agree). Computing α on semantic velocity is conceptually plausible (a locked-in conversation should produce persistent, low-variance velocity → high Hurst exponent). The α range 0.77–1.27 is above the 0.5 white-noise baseline, suggesting *some* persistence structure.

### What threatens the threshold interpretation

**Threat 1 — Scale range (P1):** At typical dialogue lengths (20–60 turns), the DFA regression uses 2–11 integer scales spanning 0.0–0.54 log-decades. Standard guidance requires ≥1 decade. At these scales, the difference between α = 1.0 ("healthy complexity") and α = 1.2 ("lock-in threshold") — a 0.2-unit difference — is within the noise floor. The claimed range (0.77–1.27) straddles the threshold by exactly the margin that the noise makes unresolvable.

**Threat 2 — Conversational arc confound (B1-F1, P1):** A conversation exhibiting lock-in would show declining semantic velocity over time (settling into narrow topic). Declining velocity = non-stationary variance. DFA-1 cannot distinguish this structural pattern from power-law correlation — both produce elevated α. The metric may be measuring conversational arc, not fractal persistence.

**Threat 3 — Circular threshold (B1-F3, P2):** The α > 1.2 threshold was derived from Morgoulis (2025), which includes the Grunch conversation. Applied to evaluate the Grunch conversation, the threshold provides no independent evidence.

**Threat 4 — Invalid CI (A1-F2, P2):** The confidence intervals for α are based on IID bootstrap resampling. For typical dialogue lengths, almost all bootstrap iterations return α = 0.5 (via scale collapse and temporal destruction), making the CI systematically compressed near 0.5. Any reported CI cannot be interpreted as bounding the true α.

### Net assessment

The directional claim ("elevated α in the Grunch denial phase") is plausible but unconfirmed. The threshold claim ("this indicates lock-in") requires:

1. Verification that the Grunch denial phase is long enough for reliable DFA estimation (≥100 turns)
2. A stationarity test on the velocity signal to rule out conversational arc confounding
3. Out-of-sample validation of the α > 1.2 threshold

Without these three, the claim should be reframed in the preprint from "indicates semantic lock-in" to "is consistent with elevated semantic persistence, pending stationarity and length validation."

**Preprint status of core claim: REQUIRES QUALIFICATION (not retraction)**

---

## 6. Remediation Actions (Severity-Ordered)

### P0 — Preprint-Blocking (must address before submission or add explicit limitation)

| ID | Action | Finding |
|----|--------|---------|
| P0-1 | Add preprint text disclosing the DFA scale limitation for the Grunch conversation. Report conversation turn count for the denial phase. Rephrase "indicates lock-in" to "is consistent with elevated semantic persistence" unless length ≥100 turns can be confirmed. | A1/B1 |
| P0-2 | Add methods limitation stating all reported α CIs are based on IID bootstrap and are not valid uncertainty bounds. Either replace with block bootstrap (methodological fix) or label CIs as "approximate." | A1-F2 |
| P0-3 | Add methods limitation noting that semantic velocity may exhibit non-stationary variance (decreasing over conversational arc) that DFA-1 cannot distinguish from power-law correlation. Provide or cite a stationarity test. | B1-F1 |

### P1 — Should Fix Before Submission

| ID | Action | Finding |
|----|--------|---------|
| P1-1 | **DFA scale parameters:** Lower `min_scale` to 2 and raise `max_scale_factor` to 0.4. Add R² threshold filter (discard α if R² < 0.9). Document minimum reliable dialogue length (~100 turns for R² > 0.95). | A1-F1 |
| P1-2 | **Ψ_temporal CV instability (structural fix):** Move CV computation before z-standardisation, or replace with a scale-free stability measure (e.g., IQR/median of window metrics) that does not require a non-zero mean. | B4-NEW-2 |
| P1-3 | **Δκ threshold recalibration:** The 0.35 curvature threshold was reset after the chord-deviation→local-curvature formula change without documented recalibration. Add calibration basis or explicit uncertainty caveat. | A2, B2 |

### P2 — Fix After Submission (Next Release)

| ID | Action | Finding |
|----|--------|---------|
| P2-1 | **Block bootstrap for DFA α:** Replace IID resampling with circular block bootstrap (block size = correlation length from ACF). Standard method for correlated time series CIs. | A1-F2, B1-F2 |
| P2-2 | **Block bootstrap for Δκ:** Replace sorted-replacement bootstrap with non-overlapping block bootstrap (block size ≈ √n) to avoid systematic zero-velocity introduction. | A2, B2 |
| P2-3 | **Matching bootstrap for ΔH:** Run all 6 combinations (kmeans×{6,8,10} + gmm×{6,8,10}) per bootstrap iteration and take their mean — matching the consensus point estimate. Currently only k=8 KMeans is used. | A3, B3 |
| P2-4 | **ΔH minimum sample guard:** Add validation requiring `n_pre ≥ 5 × max(n_clusters_range)` (or dynamically filter k > n_pre // 5). Standard 20-turn conversation fails this criterion at k=10. | B3-Q1 |
| P2-5 | **GoEmotions path disclosure:** The preprint describes one Ψ_affective formula; the code has two. Add footnote or methods section noting that when `emotion_service` is configured, a different formula applies. | B4-NEW-1 |
| P2-6 | **Substrate range alignment:** Ψ_temporal is [0,1] while others span [−1,1]. Rescale Ψ_temporal to [−1,1] or update the preprint to accurately describe the range asymmetry and the resulting structural positive bias in the composite. | A4, B4 |
| P2-7 | **Cognitive Mimicry secondary fallback correction:** Line 711 returns Cognitive Mimicry for affect-dominant states, contradicting CM's defined signature (high-semantic, low-affect). Change to Generative Conflict or Transitional. | A5, B5 |
| P2-8 | **Deep Resonance delta_kappa blindness:** Add a delta_kappa condition to the Deep Resonance check (e.g., `dk < 0.35`) so that high-dk states with `sem > 0.4, aff > 0.4, bio == 0.0` correctly route to Generative Conflict or Creative Dilation. | B5-N1 |
| P2-9 | **BASIN_CENTROIDS empirical recalibration:** Current centroids were derived from threshold boundary midpoints. Recalibrate from actual corpus observations; many centroids (DR: psi_aff=0.5, EC: psi_bio=0.5) are in empirically unreachable regions. | B5-N2 |
| P2-10 | **Lock-in threshold out-of-sample validation:** Design a synthetic locked-in signal test (slow random walk → α > 1.2 at realistic lengths) and/or validate on an independent conversation corpus. | A1-F4, B1-F3 |

### P3 — Future Work

| ID | Action | Finding |
|----|--------|---------|
| P3-1 | **Stationarity test integration:** Add KPSS test on velocity subwindows to detect non-stationary variance before DFA; flag results where velocity is non-stationary and suppress or caveat α interpretation. | B1-F1 |
| P3-2 | **DFA-2 / adaptive DFA:** Quadratic detrending (DFA-2) partially addresses non-stationary linear trends. Document when DFA-1 vs DFA-2 is appropriate for semantic velocity signals. | B1-F1 |
| P3-3 | **Arc-length-weighted curvature mean:** Replace unweighted mean of κ(t) with `Σ[κ(t) · ‖v(t)‖] / Σ‖v(t)‖` to reduce slow-segment amplification. | A2, B2 |
| P3-4 | **Phase-randomised surrogate for Δκ significance:** Replace permutation null (always significant for real dialogue due to local autocorrelation) with Fourier-based phase-randomised surrogate that preserves autocorrelation structure while destroying specific temporal patterns. | B2-Q4 |
| P3-5 | **Soft membership temperature calibration:** Calibrate temperature T to actual inter-centroid distances in the corpus. T=1.0 produces near-uniform membership across all 10 basins from any starting position (CI = 11.3% at the exact CI centroid). | B5-N4 |
| P3-6 | **Expand centroid space to include dialogue context:** Add hedging_density, turn_ratio, dk_variance, coherence_pattern dimensions to BASIN_CENTROIDS and `compute_soft_membership` for the ambiguous trio (CI, RP, CM). | A5, B5 |
| P3-7 | **DFA minimum length documentation:** Update documentation to state: DFA fallback activates at <25 turns; marginal reliability at ~40 turns; reasonable reliability requires ~100+ turns. Remove "20-turn minimum viable" framing. | A1-F3 |
| P3-8 | **Test coverage for realistic dialogue lengths:** Add DFA tests with N=25–60 velocity samples; add synthetic lock-in test; add Reflexive Performance as explicit primary classification test case. | A1, A5 |

---

## 7. Convergence Topology Summary

```
Finding         PhA   PhB   Score   Pattern
DFA/α           LOW   CONFIRM↓  0.82   Extension (non-stationarity, circular threshold)
Δκ curvature    PART  CORRECT   0.63   Correction (bootstrap direction, permutation null)
ΔH JSD          PART  EXTEND    0.75   Extension (k=n/2 degeneracy, dead code)
Ψ assembly      PART  EXTEND+   0.65   Extension + Gap (GoEmotions path, CV universality)
Basin class.    PART  CONFIRM+  0.78   Extension (centroid geometry, DK-blindness, capacity)
Overall                         0.73   Good convergence; Phase B consistently sharpens
```

No finding exhibits the CIRCLING pattern (irreducible disagreement on verdict direction). Phase B's primary contribution is sharpening severity assessments and identifying coverage gaps, not contesting core conclusions.

---

---

## 8. Resolution Status (Updated 2026-03-11)

The following table tracks which remediation actions have been addressed. The preprint is at `publications/papers/cross-substrate-coupling-preprint.md`.

### P0 — Preprint-Blocking

| ID | Action | Status | Resolution |
|----|--------|--------|------------|
| P0-1 | DFA scale limitation disclosure; reframe lock-in language | **RESOLVED** | §5.5 explicitly states 40 turns, <1 log-decade scale range. All lock-in claims throughout the paper reframed to "consistent with elevated semantic persistence" or "possible lock-in dynamics." Turn count reported. |
| P0-2 | Bootstrap CI caveat | **RESOLVED** | §3.3 states block bootstrap planned but not conducted. §5.5 notes cross-correlation p-values violate independence assumptions. CIs not presented as valid uncertainty bounds. |
| P0-3 | Stationarity caveat | **RESOLVED** | §5.5 describes the conversational-arc confound explicitly: declining variance conflated with power-law correlation by DFA-1. Notes KPSS test not conducted. Lock-in interpretation labelled as directional. |

### P1 — Should Fix Before Submission

| ID | Action | Status | Resolution |
|----|--------|--------|------------|
| P1-1 | DFA scale parameter tuning + R² filter | **DEFERRED** | Code-level fix for next release. Paper discloses limitation (§5.5). |
| P1-2 | Ψ_temporal CV structural fix | **DEFERRED** | Code-level fix for next release. Ψ_temporal is not load-bearing for any preprint claim. |
| P1-3 | Δκ threshold recalibration documentation | **PARTIALLY RESOLVED** | §3.2 notes implementations differ from Morgoulis. Threshold not presented as validated. |

### P2 — Fix After Submission

| ID | Action | Status | Notes |
|----|--------|--------|-------|
| P2-1 through P2-10 | Code-level fixes | **DEFERRED** | All P2 items are implementation improvements for the next codebase release, not preprint content issues. |

### P3 — Future Work

| ID | Action | Status | Notes |
|----|--------|--------|-------|
| P3-1 through P3-8 | Research extensions | **OPEN** | Mapped to future work in §5.5 and §6 of the preprint. |

### Structurally Unresolvable Findings

| ID | Finding | Status | Notes |
|----|---------|--------|-------|
| U1 | Grunch conversation length | **RESOLVED** | 40 turns, disclosed in §5.5. Below 100-turn reliability threshold — paper qualifies all α interpretations accordingly. |
| U2 | Semantic velocity stationarity | **ACKNOWLEDGED** | Not tested; disclosed as limitation in §5.5. |
| U3 | Lock-in threshold external validity | **ACKNOWLEDGED** | Not validated out-of-sample; paper reframes claims as directional. |
| U4 | Production affective path | **RESOLVED** | VADER-only path used for all reported results. GoEmotions path is a code-level alternative not used in the preprint. |
| U5 | Calibration parameter validity | **ACKNOWLEDGED** | Not validated against corpus. Paper presents metrics as exploratory instrumentation, not calibrated scales. |

*RAA-SC-001 convergence evaluation complete. Resolution status added 2026-03-11.*
