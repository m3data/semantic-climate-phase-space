# Counter-Finding B3 — ΔH Entropy Shift via Jensen-Shannon Divergence

**Audit:** RAA-SC-001
**Task:** b3-counter-delta-h-entropy-shift-jsd
**Date:** 2026-02-25
**Phase A finding:** `audits/RAA-SC-001/findings-a/03-delta-h-entropy-shift-jsd.md`
**Phase A verdict:** PARTIAL

---

## Purpose

This document independently re-examines Phase A's analysis of `entropy_shift_comprehensive()`
against the five specific questions posed in the task brief. Phase A's code was re-read
from scratch before consulting Phase A's conclusions.

---

## Independent Code Review

`entropy_shift_comprehensive()` (lines 414–607) and its closures were re-examined
in full. The five questions below are answered from first principles.

---

## Question 1 — Is KMeans with n_clusters=10 on 20 points statistically meaningful?

**Setup:** `calculate_all_metrics()` with 20 turns and `split_ratio=0.5` produces:
- `split_point = int(20 * 0.5) = 10`
- `n_pre = 10`, `n_post = 10`, `all_embeddings.shape = (20, d)`
- KMeans with `n_clusters=10` fits 10 clusters on 20 total points: k/n = 0.5

**Finding: No — statistically degenerate.**

With k=10 on n=20, the expected points per cluster is 2. For the pre-distribution
estimation: 10 pre-points are assigned over 10 cluster bins, yielding an expected
count of 1 per bin. The variance of each frequency estimate is:

```
Var(p̂_c) = p_c(1-p_c) / n_pre ≈ 0.1 * 0.9 / 10 = 0.009
SE(p̂_c) ≈ 0.095
```

For a uniform distribution with true p_c = 0.1, the standard error is 95% of the
true value — the noise is comparable to the signal. Equivalently: with 10 i.i.d.
observations falling into 10 bins, each bin has an expected count of 1, and roughly
35% of bins will have count 0 (by (9/10)^10 ≈ 0.35). The pre-distribution is
effectively a sparse spike pattern determined more by sampling noise than by the
true underlying distribution.

This is not an edge case. It is the **normal operating condition** for a 20-turn
conversation with the default `n_clusters_range=[6, 8, 10]`. The k=10 configuration
will always produce distribution estimates with SE/mean ≈ 1 at n_pre=10.

Even the k=6 configuration is marginal at n_pre=10:

```
SE(p̂_c) = sqrt(0.167 * 0.833 / 10) ≈ 0.118
SE / true_mean = 0.118 / 0.167 ≈ 71%
```

The conventional rule-of-thumb for reliable frequency estimation from a categorical
distribution is n ≥ 5k per half. At k=6 this requires n_pre ≥ 30, at k=10 it
requires n_pre ≥ 50. A conversation would need 60–100 turns before the k=6–10 range
becomes statistically meaningful.

**Phase A gap:** Phase A flagged the n_pre < k case (e.g., n_pre=3 with k=6) as
degenerate. But the statistically meaningful bound is n_pre >> k, not n_pre ≥ k.
Phase A set the bar at a point that excludes only the most extreme failures. The
standard 20-turn conversation at k=10 (where k = n_pre) is equally degenerate by
the frequency-estimation criterion, and Phase A did not flag this.

**Consequence for the reported ΔH=0.228 fragmentation spike:** If this value was
computed from a 20-turn session, the k=10 configurations are producing distributions
with ~60–100% relative standard error per bin. The consensus mean will absorb noise
as signal.

---

## Question 2 — Does averaging JSD across different granularities produce a meaningful metric?

**Finding: No — the average has a systematic upward bias, not just interpretive
ambiguity.**

Phase A correctly identified this as a tension ("different levels of topological
resolution"). My independent analysis adds a directional consequence: JSD is
monotone-non-decreasing with partition refinement.

By the **data processing inequality**, for a deterministic coarsening f that maps
k fine-grained clusters to fewer coarse clusters:

```
JSD(p_coarse, q_coarse) ≤ JSD(p_fine, q_fine)
```

Equivalently, JSD(k=6) ≤ JSD(k=8) ≤ JSD(k=10) in expectation for a given dataset.
(With finite samples this is not strict, but the directional bias holds in expectation.)

The consequence: `mean(JSD_6, JSD_8, JSD_10)` is systematically higher than JSD_6
(the most conservative, lowest-variance estimate) and lower than JSD_10 (the
highest-variance, most noise-inflated estimate). The ensemble average is not a
neutral combination — it is biased toward the high-k extreme.

If the ΔH=0.12 threshold was calibrated with any particular k in mind, the consensus
mean will cross it at a lower true reorganization level than that k alone would require.
The calibration is implicitly against a mixture of granularities rather than a well-
defined scale.

**Additional issue:** GMM with default `covariance_type='full'` at n=20, k=10 fits
a 768-dimensional Gaussian covariance matrix (~295,000 parameters) from ~2 points
per component. The result is a numerically degenerate fit; sklearn adds `reg_covar=1e-6`
internally but this cannot rescue estimation from 2 points in 768 dimensions. The GMM
column of the consensus average is fitting noise, and the JSD values it contributes
are artefacts of regularization, not data geometry. Phase A did not examine this.

---

## Question 3 — Does the epsilon affect the JSD [0,1] bound?

**Finding: No. Phase A is correct. The [0,1] bound holds.**

After adding 1e-12 to all entries and renormalising, both vectors sum to 1 and are
positive. JSD under log2 between any two valid probability distributions is bounded
[0, 1]. The epsilon shifts each entry by at most ~1e-12 per bin (for k=10), which
is ≤ 1e-11 of any typical non-zero entry value.

For the maximally disjoint case p = [1, 0, ..., 0] and q = [0, ..., 0, 1]:
- After epsilon: p ≈ [1, ε, ..., ε], q ≈ [ε, ..., ε, 1] (normalised)
- m ≈ [0.5, ε, ..., ε, 0.5]
- KL(p||m) ≈ log2(1/0.5) = 1 bit
- JSD ≈ 0.5 + 0.5 = 1.0 ✓

For the identical case p = q: both get the same epsilon perturbation, m = p,
KL(p||m) = 0, JSD = 0. Epsilon introduces no positive bias for zero-reorganization. ✓

**No counter-finding. Phase A correct on this point.**

One observation Phase A did not make: the epsilon is added to all bins including
those genuinely zero because n_clusters > n_pre. This does not break the bound but
means the effective support is always the full k-dimensional simplex. Two distributions
that are maximally disjoint (each concentrated on different clusters) will always
compute JSD < 1 by a negligible margin (not a concern in practice).

---

## Question 4 — Does the bootstrap produce a valid CI?

**Finding: No — and for three compounding reasons.**

**Confirmed from Phase A (primary):** The CI (lines 575–591) uses k=8 KMeans only,
while `consensus_delta_h` averages 6 combinations (kmeans×{6,8,10} + gmm×{6,8,10}).
These are different estimators. The CI is for a sub-estimator of the reported point
estimate, not for the point estimate itself.

**New from this review (secondary):** The k=8 bootstrap operates on n=20 points
(n_pre=10 per resample) with k=8. Expected points per cluster = 2.5 in the full
stacked sample; n_pre points per cluster for the distribution estimate ≈ 1.25. This
is not statistically different from the degenerate k=10 case described in Q1. The
CI inherits the full noise of sparse distribution estimation.

**New from this review (tertiary):** Each bootstrap iteration re-clusters the
resampled data with a fixed `random_state=42`. KMeans with the same seed but different
data produces different cluster geometries. The CI conflates:
- Sampling variance: how JSD varies across samples from the same populations
- Clustering variance: how JSD varies because KMeans converges differently on each
  bootstrap sample

These are distinct sources of variability. The second is a form of model instability,
not sampling uncertainty. The CI width includes both, making it wider than a pure
sampling CI — but without any label indicating this is a total-uncertainty interval
rather than a sampling CI.

**Compound consequence:** For the 20-turn case, the CI reports the combined variability
of: (a) the wrong estimator, (b) sparse distribution estimation, and (c) clustering
instability. The interval is not interpretable as a standard 95% CI.

---

## Question 5 — Does the clamp (max 0, min 1) hide instability?

**Finding: Phase A contains a factual error. `stability_score` CANNOT exceed 1.**

Phase A (task brief framing) states: "this can be negative or > 1". The Phase A
document describes only the negative case. My independent analysis shows the upper
bound is unreachable:

```
stability_score = 1 - std(js_values) / (mean_js + 1e-12)
```

Since JSD ≥ 0 always, `js_values` is a list of non-negative reals.
Therefore `mean_js ≥ 0` and `std(js_values) ≥ 0`.
The ratio `std / (mean + 1e-12) ≥ 0`.
Therefore `stability_score = 1 - [non-negative] ≤ 1`.

**The `min(1, stability_score)` in `max(0, min(1, stability_score))` (line 603) is
dead code.** It can never be triggered by real inputs. Phase A's statement that the
score "can be negative or > 1" is partially incorrect — it can be negative but
not > 1.

The `max(0, ...)` clamp is real and does discard information (as Phase A correctly
notes). CV=5 and CV=50 both produce stability_score=0. This is a genuine expressivity
limitation.

The dead `min(1, ...)` guard suggests the original author misidentified the formula's
range, possibly confusing it with a different metric or deriving the bound incorrectly.
It is not harmful but is evidence of imprecise reasoning about the formula's properties.

---

## Additional Finding — Transition Summary / Point Estimate Inconsistency

Not in the task brief but observed in the independent code review:

`_generate_transition_summary()` (line 626) formats `js_div` from `best_result`,
which is the JS value from the **best-silhouette single combination** — not the
consensus mean. The human-readable output (e.g., "Substantial reorganization
(JS=0.231)") can show a materially different JS value than the reported
`consensus_delta_h` / `js_divergence` (the mean of all 6). A caller comparing
the numeric metric to the summary text may encounter apparent contradictions.

---

## Counter-Assessment of Phase A

### What Phase A got right

- JSD formula correctness (Q1 in Phase A) — confirmed
- Shared clustering correctness (Q2 in Phase A) — confirmed
- Bootstrap-estimator mismatch as the primary bootstrap concern — confirmed
- Epsilon [0,1] bound — confirmed
- Stability score clamp hiding severity for large negative CV — confirmed

### Where Phase A was insufficient

| Issue | Phase A | This counter-audit |
|-------|---------|-------------------|
| k=n/2 at 20 turns | Only flagged n_pre < k as edge case | k = n_pre is normal operation; SE/mean ≈ 1 at k=10, 71% at k=6 — statistically degenerate for any conversation under ~60 turns |
| Consensus averaging | "Tension" — interpretive | Systematic upward bias via data processing inequality |
| GMM at n=20, k=10 | Noted hard assignments are appropriate | Full-covariance GMM fitting ~2 pts in 768 dims is numerically degenerate; GMM column of consensus is noise |
| Stability > 1 | Implicitly accepted as possible (dead code left unexplained) | `min(1, ...)` is provably dead code; score is bounded above by construction |
| Summary inconsistency | Not noted | `transition_summary` uses best-k JS, not consensus mean |

### Revised severity of weaknesses

**Phase A weakness 1 (Bootstrap-estimator mismatch):** Confirmed. Priority: HIGH.
Compounded by sparse distribution estimation in the bootstrap iterations.

**Phase A weakness 2 (No minimum size validation):** Promoted from "no minimum size
check" to a more precise statement: the threshold should be n_pre ≥ 5k, not n_pre ≥ k.
For k=10, this requires n_pre ≥ 50 (100-turn conversation). The standard 20-turn
case is statistically degenerate at k=10. Priority: HIGH (not the MEDIUM implied
by Phase A's framing).

**Phase A weakness 3 (Consensus conflates granularities):** Confirmed and extended
with the directional bias from the data processing inequality. Priority: MODERATE.

---

## Counter-Verdict

**Phase A verdict (PARTIAL) is upheld but Phase A was too lenient.**

The core algorithm — JSD formula, shared clustering, epsilon bound — is correctly
implemented. The claim "correctly computes semantic reorganization" is partially true
for the core method. The [0,1] bound holds.

However, for the primary use case of a 20-turn conversation with `n_clusters_range=[6, 8, 10]`:

1. The k=10 configuration is distributing 10 observations over 10 bins — maximum
   variance, minimum signal. This is not an edge case; it is the default operation.

2. The bootstrap CI is estimated from the wrong estimator, at equally sparse sample
   sizes, conflating sampling and clustering variance.

3. The GMM component of the consensus is numerically degenerate at n≈k with full
   covariance in high dimensions.

A ΔH value from a 20-turn conversation should be treated as a rough directional
indicator, not a calibrated metric. The reported CI should not be taken at face value.
The claim of bounded [0,1] output is correct; the claim that the value represents
"true reorganization" at small n requires a much stronger caveat than Phase A provided.

**One factual correction to Phase A:** `stability_score > 1` is impossible given
the formula. The `min(1, ...)` clamp is dead code.

---

## Notes

- **Preprint impact:** The ΔH=0.228 fragmentation spike would require knowing the
  exact turn count of the session to assess severity. If from a 20-turn session, the
  k=10 configuration contributes two of the six ensemble members under maximum-variance
  conditions. The value is not necessarily wrong but confidence in precision (vs. order
  of magnitude) is low.

- **Actionable fix for k-range:** A minimum sample size guard of
  `n_pre >= 5 * max(n_clusters_range)` would catch the standard 20-turn case.
  Alternatively, filter `n_clusters_range` to exclude k > n_pre // 5.

- **Actionable fix for bootstrap:** Run all 6 combinations per bootstrap iteration
  and take their mean before appending to `bootstrap_js`. Three lines of change.

- **Actionable fix for stability:** Remove `min(1, ...)` from the clamp or document
  why it was included. Current presence implies the original author expected values > 1
  which cannot occur.
