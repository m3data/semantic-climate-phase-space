# Finding A3 — ΔH Entropy Shift via Jensen-Shannon Divergence

**Audit:** RAA-SC-001
**Task:** a3-delta-h-entropy-shift-jsd
**Date:** 2026-02-25
**Claim:** ΔH (entropy shift) correctly computes semantic reorganization via Jensen-Shannon
divergence on shared clustering of pre/post embeddings, bounded [0,1].

---

## Claim

`entropy_shift_comprehensive()` measures semantic reorganization by:
1. Fitting a shared cluster model on the full embedding trajectory (pre + post combined)
2. Computing probability distributions over cluster IDs for each half
3. Computing Jensen-Shannon divergence between those distributions
4. Using consensus across clustering methods (kmeans, gmm) and cluster counts (6, 8, 10)

The metric is claimed to be bounded [0,1] and to capture *true reorganization* — where
probability mass moved across the shared semantic space — rather than entropy difference.

---

## Files Examined

- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/core_metrics.py`
  - `entropy_shift_comprehensive()`: lines 414–607
  - `compute_js_divergence()`: lines 461–485 (closure)
  - `calculate_distributions_shared()`: lines 487–527 (closure)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/tests/test_core_metrics.py`
  - `TestEntropyShift` class: lines 172–258
  - `TestMetricFixValidation.test_entropy_shift_uses_shared_clustering`: lines 425–447

---

## Evidence

### Q1 — JSD formula correctness

`compute_js_divergence(p, q)` (lines 461–485):

- Pads both vectors to the same length with zeros, then adds epsilon (1e-12) to all
  entries before renormalising. After renormalisation, both vectors are valid probability
  distributions.
- Computes `m = 0.5 * (p_padded + q_padded)` — correct mixture.
- Computes `kl_pm = sum(p * log2(p/m))` and `kl_qm = sum(q * log2(q/m))`.
- Returns `0.5 * kl_pm + 0.5 * kl_qm` — correct JSD formula.
- Uses `np.log2` throughout, giving JSD in bits and guaranteeing the [0,1] bound
  (max 1 bit when distributions are completely disjoint).

The formula is mathematically correct. The epsilon-before-renormalise pattern avoids
`log(0)` while introducing only negligible distortion (each zero entry becomes
~1e-12/sum, contributing ≈ 1e-12 * log2(2e-11) ≈ -4e-11 to KL — negligible). The [0,1]
bound is maintained after epsilon+renormalisation because the result remains a divergence
between two valid probability distributions under log2.

Padding (`max_len = max(len(p), len(q))`; shorter vector zero-padded) is irrelevant in
practice: shared clustering always produces n_clusters-length distributions for both pre
and post. No length mismatch can occur in the main path.

### Q2 — Shared clustering correctness

`calculate_distributions_shared(all_emb, n_pre, method, n_clusters)` (lines 487–527):

- KMeans/GMM is fitted on `all_emb` (the full stacked trajectory, shape `[n_pre+n_post, d]`).
- `all_labels = clusterer.fit_predict(all_emb)` — both KMeans and GaussianMixture return
  hard cluster assignments. Correct.
- Labels are split: `pre_labels = all_labels[:n_pre]`, `post_labels = all_labels[n_pre:]`.
  The boundary index `n_pre` is correctly maintained.
- Distributions are computed over `np.arange(n_clusters)` — all cluster IDs, not just
  observed ones. A cluster absent in pre gets probability 0 (before epsilon). This is
  the semantically correct shared cluster space.
- Pre denominator: `n_pre`; post denominator: `len(all_emb) - n_pre`. These are
  equivalent to explicit `n_pre` / `n_post` — correct but slightly inconsistent.

The shared clustering design correctly ensures cluster IDs have the same semantics in
pre and post. The fix from the original Morgoulis implementation (independent clustering)
is sound and properly implemented.

### Q3 — Consensus across methods and cluster counts

Six JS values are computed: kmeans×{6,8,10} and gmm×{6,8,10}. The mean is reported as
`consensus_delta_h` / `js_divergence`.

The rationale (robustness via ensemble) is reasonable in principle, but the design has
a tension: different k-values measure reorganization at different *granularities*. With
k=6, two embeddings in the same broad region are in the same cluster; with k=10, they
may split into different clusters, inflating JS. The mean of JS(k=6), JS(k=8), JS(k=10)
is not an estimate of a single well-defined quantity — it is an average across different
levels of topological resolution.

The silhouette score selects the best clustering for the `pre_distribution` /
`post_distribution` output, but this best-clustering is NOT the same as the consensus
mean. The human-readable summary describes the best clustering, while the metric reports
the average of all six.

There is no weighting by silhouette quality — a poor-fit k=6 model contributes equally
to the average as a well-fit k=8 model.

### Q4 — stability_score formula

```python
stability_score = 1 - np.std(js_values) / (mean_js + 1e-12)
```
Clipped via `max(0, min(1, stability_score))`.

This is the inverted coefficient of variation (CV). Properties:
- `std=0` → score=1.0 (all methods agree). Correct.
- `std=mean` → score=0 (CV=1). Reasonable floor signal.
- `mean≈0, std≈0` → score≈1.0 (stable consensus: no reorganization). Correct.
- `mean=0.01, std=0.05` → score = 1-5 = -4 → clipped to 0. The clip hides severity
  (CV=5 and CV=50 both produce score=0).

The clamp at [0,1] prevents negative values but discards information about *how*
unstable the consensus is. A score of 0 is ambiguous — it could mean mild or extreme
dispersion. This is a minor expressivity limitation, not a bug.

Edge case: if `mean_js` is exactly 0 but `std` is not zero (theoretically possible if
some JS values are tiny positive and the mean rounds to 0 after floating point), the
formula divides by 1e-12 and gives a large negative number, clipped to 0. In practice
this is unlikely given the epsilon smoothing.

### Q5 — Bootstrap CI validity

Bootstrap (lines 575–591) resamples pre and post independently (with replacement), then
re-clusters with **k=8 KMeans only** (not the full 6-combination consensus).

**Core mismatch:** The point estimate (`consensus_delta_h`) is the mean of 6 JS values
(2 methods × 3 cluster counts). The bootstrap CI estimates the sampling distribution of
a single k=8 KMeans JS value. These are different estimators. The CI is not
formally valid for the reported `consensus_delta_h`.

A valid CI for the consensus would require, for each bootstrap iteration, running all 6
combinations and computing their mean. The current implementation runs only one of the
six. This produces a CI that is narrower and offset relative to the true CI of the
consensus metric.

Additional note: each bootstrap iteration re-clusters the resampled data. Clustering
randomness (even with fixed `random_state`, KMeans with different data produces
different labels) adds variance to the bootstrap distribution beyond pure sampling
variance. This inflates CI width slightly. This is not incorrect (it captures total
estimation uncertainty), but it conflates sampling variance with algorithmic variance.

Independent resampling of pre and post (vs. joint resampling) is appropriate: it
estimates how JSD would vary if we observed different samples from the same two
populations, which is the standard two-sample bootstrap.

### Q6 — Epsilon effect on [0,1] bound

Adding 1e-12 before renormalisation does not break the [0,1] bound. After renormalisation,
the distributions are valid (positive, sum to 1). JSD under log2 between any two valid
distributions is bounded [0,1]. The epsilon shifts each entry by at most ~1e-12 (for
distributions over 10 clusters), which is negligible relative to any typical entry value.

For a completely sparse distribution (one-hot), the non-peak entries become ~1e-12/
(1+9e-12) ≈ 1e-12. Their KL contribution is ~-3.6e-10 bits (negligible). The bound
holds. ✓

### Q7 — Size asymmetry and near-identical embeddings

**Very different pre/post sizes:**
- At split_ratio=0.5, sizes are equal. But `entropy_shift_comprehensive()` is called
  directly with arbitrary arrays; `calculate_all_metrics()` sets split_point = int(n *
  0.5), so the split is exactly equal (with floor rounding, one half may be 1 shorter).
- If pre has very few embeddings (e.g., n_pre=3), the cluster distribution is estimated
  from 3 points. With n_clusters=6, some clusters will have 0 points in pre by necessity
  (6 clusters, 3 points → at least 3 empty). This produces a degenerate distribution.
  No minimum size check exists. The function returns a plausible-looking JS value with
  no warning.
- Bootstrap degrades further: with n_pre=3, `np.random.choice(3, 3, replace=True)`
  produces one of 10 possible bootstrap samples — CI is meaningless.

**Near-identical embeddings:**
- KMeans with n_clusters=6 on near-identical points converges with overlapping
  centroids; cluster assignments are semi-arbitrary (depend on initialisation).
- Pre and post will receive similar assignments (since they span the same region), so
  pre_dist ≈ post_dist and JS ≈ 0. This is the correct semantic interpretation (no
  reorganization). ✓
- If ALL embeddings are exactly identical: all KMeans labels will be 0 (single cluster
  "wins" by convention). `len(np.unique(all_labels)) == 1`, so silhouette_score returns
  -1 (protected by the conditional). The distribution is [1, 0, 0, ...] for both pre
  and post → JS = 0. ✓

**Empty or single-point input:**
- `np.vstack` with empty arrays raises ValueError. Caught by `except Exception: return
  None`. With all 6 combinations returning None, `js_values = []`, `mean_js = 0`, CI =
  (0, 0), stability = 0. Silent failure — returns a valid-looking result (JS=0) that is
  indistinguishable from genuine zero-reorganization.

---

## Finding

**Verdict: PARTIAL**

The core algorithm — JSD formula and shared clustering — is mathematically correct and
properly implemented. The formula uses log2, guaranteeing the [0,1] bound. The shared
clustering correctly fits on the full trajectory and splits labels to pre/post, enabling
genuine reorganization measurement. The test suite covers the critical fix (shared vs.
independent clustering) and the zero-divergence and high-divergence boundary conditions.

Three methodological weaknesses are present:

1. **Bootstrap–estimator mismatch (primary concern).** The reported `confidence_interval`
   is estimated using k=8 KMeans only, while `consensus_delta_h` averages 6 combinations
   (2 methods × 3 cluster counts). The CI is not valid for the reported point estimate.
   To fix: run all 6 combinations per bootstrap iteration and take the mean, matching the
   main estimator.

2. **No minimum size validation.** With very small pre or post arrays (e.g., n_pre < k),
   the function produces unreliable distributions and degenerate bootstrap CIs without
   warning. Silent failure (empty input → JS=0) is indistinguishable from genuine
   zero-reorganization.

3. **Consensus averaging conflates granularities.** JSD(k=6) and JSD(k=10) measure
   reorganization at different topological resolutions and are not estimates of the same
   quantity. Averaging them produces a number without a clean mathematical interpretation.
   The stability_score partially compensates but loses scale below CV=1.

None of these concerns invalidate the metric's conceptual validity, and the core JSD
and shared clustering are implemented correctly. The claim "correctly computes" is
partially true: the computation is correct, but the CI and consensus mean exceed what
the implementation strictly delivers.

---

## Notes

- **Preprint impact:** ΔH is used as one component of the Ψ vector (semantic substrate)
  and contributes to basin classification. A reported ΔH=0.228 "fragmentation spike"
  would be affected by all three concerns above. The point estimate itself is likely
  reasonable (the mean of 6 related but not identical estimators), but the CI should be
  treated as approximate. No preprint-blocking issue.

- **GMM fit_predict:** GaussianMixture.fit_predict() returns MAP hard assignments,
  consistent with KMeans label semantics. Soft membership (predict_proba) is not used.
  This is a deliberate choice; hard clustering is appropriate for distribution computation.

- **Denominator spelling:** Post denominator uses `len(all_emb) - n_pre` (line 513)
  rather than the `n_post` variable. Functionally equivalent; minor inconsistency.

- **Stability score clamp:** `max(0, min(1, stability_score))` hides negative CV scores.
  A stability of -4 and -40 both become 0. Callers have no way to detect severity.

- **Test gap:** No test covers the bootstrap CI estimator mismatch. The test suite
  checks that CI is present and lower ≤ upper, but not that it corresponds to the same
  estimator as the point estimate.
