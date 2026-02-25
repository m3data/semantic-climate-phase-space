# Finding B5: Counter-Audit — Coupling Mode Classification

**Task:** b5-counter-coupling-mode-classification
**Date:** 2026-02-25
**Auditor:** Claude Sonnet 4.6 (Phase B)
**Counter-auditing:** `findings-a/05-coupling-mode-classification.md`

---

## Claim (from Phase A)

Coupling mode classification (attractor basin detection) correctly maps Ψ position and raw metrics into 10 canonical dialogue configurations with discriminating features between performative and genuine engagement.

## Phase A Verdict

PARTIAL — four concerns: default input bias, secondary fallback returning CM for affect-dominant states, hysteresis exit comparison conceptually confused, soft membership not discriminating the ambiguous trio.

---

## Independent Analysis

Files examined independently:
- `src/basins.py` lines 1–899 (full)
- `src/substrates.py` (via Phase A findings on psi ranges)
- `tests/fixtures/golden_outputs.json` (via prior exploration)

---

## Interrogation of Five Counter-Audit Targets

### Target 1: Basin detection ordering — does Deep Resonance first create priority bias?

**Phase A (Effect C):** Noted that `bio == 0.0` in the Deep Resonance condition (line 595) makes DR the implicit classification for any high-semantic, high-affective exchange without biosignal. Framed as inflation of Deep Resonance across the majority (24/34) no-biosignal corpus.

**Counter-audit finding: Phase A undersells the severity; delta_kappa blindness is the deeper problem.**

The Deep Resonance check (lines 594–597):
```python
if sem > 0.4 and aff > 0.4 and (bio > 0.4 or bio == 0.0):
```
fires before the Generative Conflict (line 612) and Creative Dilation (line 617) checks. Both GC and CD require `delta_kappa > 0.35`, which is the defining discriminator of productive tension vs convergent resonance. A state with `sem=0.7, aff=0.5, dk=0.8, bio=0.0` — empirically a high-divergence, high-energy exchange — would be classified as Deep Resonance without ever consulting delta_kappa.

The taxonomy's theoretical purpose distinguishes Deep Resonance (synchronized, convergent, all-substrates-high) from Generative Conflict (productive semantic tension, high affect, high delta_kappa). For bio=0 sessions, this distinction is **permanently collapsed**. Delta_kappa is irrelevant to Deep Resonance classification.

**Phase A is correct about the empirical consequence (DR inflation) but does not name the theoretical cost: the taxonomy cannot distinguish convergence from creative tension in the majority use case.**

Verify: Does ordering bias Transitional? Yes — Transitional is the final fallback and therefore receives only what no other basin claims. But the more important priority structure is Deep Resonance shadowing GC/CD for no-biosignal sessions.

---

### Target 2: What happens when dialogue_context is absent?

**Task description claim:** "All three scores default to 0, so classification falls through to Transitional."

**This claim is factually wrong.**

Code trace (lines 586–695) with `dialogue_context = None`:

```python
hedging = 0.0         # ctx.get('hedging_density', 0.0)
turn_ratio = 1.0      # ctx.get('turn_length_ratio', 1.0)  ← NOT 0.0
dk_variance = 0.0     # ctx.get('delta_kappa_variance', 0.0)
coherence_pattern = 'transitional'  # ctx.get('coherence_pattern', 'transitional')
```

`turn_length_ratio` defaults to **1.0, not 0.0**. `coherence_pattern` defaults to `'transitional'`, not neutral.

Computing the three scores with these defaults:

| Score | Rule triggered | Value |
|-------|---------------|-------|
| `inquiry_score` | `0.5 < turn_ratio < 2.0` (+0.3) — turn_ratio=1.0 qualifies | **0.3** |
| `mimicry_score` | hedging < 0.01 (+0.3), dk_variance < 0.005 (+0.2), coherence in ('locked','transitional') (+0.2) | **0.7** |
| `reflexive_score` | coherence == 'transitional' (+0.2) only — others fail | **0.2** |

Best: Cognitive Mimicry (0.7). Margin over second: 0.7 − 0.3 = 0.4 ≥ 0.1 threshold. Not ambiguous.

Classification returns: `("Cognitive Mimicry", abs(sem) * (0.5 + 0.7 * 0.5))` = `abs(sem) * 0.85`.

**Conclusion: the absent-context default is Cognitive Mimicry with confidence ~0.85·|sem|, not Transitional. Phase A already identified this correctly. The task description's "fallthrough to Transitional" claim is incorrect.**

Why does the task description get this wrong? Likely because it reasoned that default values (hedging=0, dk_variance=0) would produce zero scores, forgetting that (a) `turn_ratio` defaults to 1.0 (which qualifies for inquiry's balanced-turn criterion) and (b) `coherence_pattern='transitional'` scores mimicry and reflexive. The non-zero inquiry_score from `turn_ratio=1.0` is the counterintuitive element.

---

### Target 3: BASIN_CENTROIDS vs `_classify_basin` — are psi_semantic values the same? Scale mismatch?

**Same value:** Yes. Both `_classify_basin` (line 578) and `compute_soft_membership` (line 741) extract `psi_vector.get('psi_semantic', 0.0)`. There is no semantic mismatch between the two methods.

**Scale mismatch: CONFIRMED and more severe than Phase A implies.**

`BASIN_CENTROIDS` (lines 321–332) place all 10 basins at coordinates in [0, 0.5]:

| Basin | psi_sem | psi_aff | dk | psi_bio |
|-------|---------|---------|-----|---------|
| Deep Resonance | 0.5 | 0.5 | 0.5 | 0.5 |
| Collaborative Inquiry | 0.4 | 0.1 | 0.4 | 0.0 |
| Cognitive Mimicry | 0.5 | 0.1 | 0.3 | 0.0 |
| Reflexive Performance | 0.4 | 0.1 | 0.35 | 0.0 |
| Sycophantic Convergence | 0.5 | 0.1 | 0.2 | 0.0 |
| Creative Dilation | 0.3 | 0.4 | 0.5 | 0.0 |
| Generative Conflict | 0.4 | 0.4 | 0.5 | 0.0 |
| Embodied Coherence | 0.1 | 0.2 | 0.3 | 0.5 |
| Dissociation | 0.1 | 0.1 | 0.1 | 0.1 |
| Transitional | 0.25 | 0.25 | 0.3 | 0.0 |

All psi_affective centroid values: {0.1, 0.2, 0.25, 0.4, 0.5}.
All psi_biosignal centroid values: {0.0, 0.1, 0.5}.

Actual psi ranges (from golden fixtures and Phase A substrate analysis):

| Component | Documented range | Practical range (golden fixtures) |
|-----------|-----------------|-----------------------------------|
| psi_semantic | [-1, 1] | [-0.923, 0.849] |
| psi_affective | ≈[-0.76, 0.76] | [-0.628, **0.248**] |
| psi_biosignal | ≈[-0.46, 0.46] | [-0.124, **0.124**] (60–100 bpm HR) |
| delta_kappa | [0, ∞) | [0, ~0.5] typical |

**Critical observations:**

1. **psi_affective max observed = 0.248**. The DR centroid requires psi_aff = 0.5. Four centroids (DR, GC, CD at 0.4, and GC/CD at 0.5) are placed above the empirically observed maximum of psi_affective. Deep Resonance centroid (psi_aff=0.5) is in a region that typical affective states cannot reach, as the formula `tanh(2*(raw - 0.5))` saturates well below 0.5 for typical dialogue sentiment variance.

2. **psi_biosignal max observed = 0.124**. Embodied Coherence centroid (psi_bio=0.5) and DR centroid (psi_bio=0.5) require biosignal values only achievable at extreme HR deviations (~100 bpm baseline). The typical resting-dialogue HR of 70–85 bpm produces psi_biosignal ≈ [-0.24, +0.12].

3. **psi_semantic is frequently negative** (4 of 5 non-precomputed golden scenarios). All centroids are positive. Negative-semantic states are uniformly distant from all centroids (distances ≥ 0.1 in the semantic dimension alone), making soft membership nearly uniform for any state where semantic substrate is negative.

**Consequence for soft membership:** A large fraction of real-world inputs fall outside the positive half of the centroid space. For such inputs, the soft membership distribution is not informative about which basin the state "prefers" — all basins appear equidistant because the centroid placement assumes positive psi_semantic and aff values that typical dialogues do not produce.

The `_classify_basin` classifier is less affected because it uses threshold comparisons (`sem > 0.4`, `abs(sem) < 0.2`, etc.) that work symmetrically with abs() guards. The soft membership calculator has no such compensation.

**Phase A identified the centroid proximity concern; it did not identify that the centroids are placed in empirically unreachable regions of the actual psi distribution.**

---

### Target 4: Soft membership temperature = 1.0

**Phase A:** Noted T=1.0 "may be too high for 4D distances." Attributed the soft membership failure primarily to centroid proximity, not temperature.

**Counter-audit: T=1.0 is independently responsible for near-uniform distributions. The temperature and proximity problems compound.**

At the exact Collaborative Inquiry centroid `(sem=0.4, aff=0.1, dk=0.4, bio=0.0)`, squared distances to all 10 centroids:

| Basin | d² from CI centroid |
|-------|---------------------|
| **CI** | **0.0000** |
| RP | 0.0025 |
| CM | 0.0200 |
| SC | 0.0500 |
| Trans | 0.0550 |
| GC | 0.1000 |
| CD | 0.1100 |
| Dis | 0.1900 |
| EC | 0.3600 |
| DR | 0.4300 |

Softmax at T=1.0: `exp(-d²/1.0)` for each basin, normalized:

| Basin | exp(-d²) | membership |
|-------|----------|------------|
| CI | 1.0000 | **11.3%** |
| RP | 0.9975 | **11.3%** |
| CM | 0.9802 | **11.1%** |
| SC | 0.9512 | 10.7% |
| Trans | 0.9465 | 10.7% |
| GC | 0.9048 | 10.2% |
| CD | 0.8958 | 10.1% |
| Dis | 0.8270 | 9.3% |
| EC | 0.6977 | 7.9% |
| DR | 0.6505 | 7.3% |
| *Sum* | *8.851* | *100%* |

At the exact CI centroid, CI receives only 11.3% membership — barely above the uniform prior of 10%. The ambiguity metric (1 − (max − second)) = 1 − (0.113 − 0.113) ≈ 1.0, which would correctly signal maximum ambiguity but is uninformative about which basin is more likely.

**Does a lower temperature solve the CI/RP discrimination?** At T=0.01:
- CI: exp(0/0.01) = 1.0
- RP: exp(-0.0025/0.01) = exp(-0.25) ≈ 0.779

CI and RP memberships converge to approximately 56% vs. 44% — still poor discrimination, regardless of temperature. The d² = 0.0025 between CI and RP centroids (differing only by 0.05 in delta_kappa) is structurally too small for any temperature to overcome.

**Root cause:** At T=1.0, the softmax collapses to near-uniform across all 10 basins. Reducing T sharpens the distribution but cannot resolve the CI/RP degeneracy because their centroids are separated by only 0.05 in a single dimension. This is an architectural problem (centroid placement), not a tuning problem (temperature selection).

**Phase A is correct that soft membership fails for the ambiguous trio. The counter-audit adds: T=1.0 causes soft membership to be uninformative for the entire taxonomy — not just the ambiguous trio — and the CI/RP degeneracy persists even at low temperatures.**

---

### Target 5: Information-theoretic capacity of 10 basins in 4D space

**Phase A:** Noted that centroids for CI, RP, CM differ by only 0.005–0.02 in squared distance. Described the 10-basin taxonomy as requiring 8 input dimensions for full specification.

**Counter-audit: Shannon capacity analysis reveals the deficit is structural, not addressable by parameter tuning.**

The effective number of discriminable basins in the 4D centroid space is determined by the ratio of centroid separation to measurement noise. With typical psi variation of σ ≈ 0.05–0.1 across turns, the probability of correctly classifying a state into its nearest centroid basin (nearest-centroid classifier) depends on the minimum pairwise distances.

**Minimum pairwise squared distances among all 10 centroids:**

| Pair | d² |
|------|----|
| CI vs RP | 0.0025 |
| GC vs CD | 0.0100 |
| CM vs SC | 0.0100 |
| CM vs RP | 0.0125 |
| CI vs CM | 0.0200 |
| CI vs SC | 0.0500 |

For σ = 0.05 noise: P(correct: CI vs RP) ≈ Φ(d_min / (2σ)) = Φ(0.05/0.10) = Φ(0.5) ≈ 0.69 — substantially error-prone. For σ = 0.1: Φ(0.25) ≈ 0.60 — barely above chance.

**Structural capacity estimate:**

The 4D centroid space can reliably discriminate approximately 4–5 basins given empirical psi noise:

| Group | Basins | Why discriminable |
|-------|--------|-------------------|
| Isolated | Deep Resonance | Far from all others (d² ≥ 0.43 from CI) |
| Isolated | Dissociation | Far from all others (d² ≥ 0.19 from CI) |
| Isolated | Embodied Coherence | High bio dimension separates it |
| Moderate | Transitional | Moderate separation |
| Cluster | CI, RP, CM, SC, GC, CD | All within d² ≤ 0.11 of each other |

The 10-basin taxonomy encodes ~log₂(10) ≈ 3.32 bits of basin identity. The 4D centroid representation, given the clustering of 6 basins in the semantic-active region, provides approximately log₂(4–5) ≈ 2.0–2.3 bits of discriminable information. **The structural deficit is ~1.0–1.3 bits.**

This deficit corresponds precisely to the 4 additional dialogue context dimensions (hedging, turn_ratio, dk_variance, coherence_pattern) that the 3-way CI/RP/CM classification requires. Including those dimensions in the centroid representation (an 8D centroid space) would restore the missing bits — but `compute_soft_membership` does not accept `dialogue_context`.

**Additional centroid geometry concern: the 6-basin semantic-active cluster (CI, RP, CM, SC, GC, CD) all share psi_affective ∈ {0.1, 0.4}. The primary 4D discriminators within this cluster are psi_semantic (range: 0.3–0.5) and delta_kappa (range: 0.2–0.5). These two dimensions must carry all the discrimination for 6 basins — approximately log₂(6) ≈ 2.6 bits — over ranges of only 0.2–0.3 in each dimension.**

---

## Phase A Confirmation and Corrections

| Phase A claim | Counter-audit verdict |
|---------------|----------------------|
| Default absent `raw_metrics` routes all high-sem, low-aff states to Sycophantic Convergence | **CONFIRMED** |
| Default absent `dialogue_context` routes ambiguous zone to Cognitive Mimicry (not Transitional) | **CONFIRMED** (task description's claim of Transitional is wrong) |
| Secondary fallback (line 711) returns Cognitive Mimicry for affect-dominant states | **CONFIRMED**: `elif dominant == 'affective': return ("Generative Conflict" if delta_kappa > 0.35 else "Cognitive Mimicry", ...)` — contradicts CM's defined high-semantic, low-affect signature |
| Hysteresis exit comparison is conceptually confused | **CONFIRMED**: `raw_confidence` of proposed new basin is compared to `current_config.exit_threshold` — these are not equivalent quantities |
| Soft membership doesn't discriminate the ambiguous trio | **CONFIRMED AND EXTENDED**: T=1.0 causes near-uniform membership across all 10 basins (CI=11.3% at exact CI centroid); reducing temperature to T=0.01 still cannot resolve CI/RP degeneracy (centroid distance 0.05 in dk only) |

---

## New Findings Not in Phase A

### B5-N1: Deep Resonance ordering is delta_kappa-blind, not just bio-blind

Phase A correctly notes the `bio == 0.0` condition inflates Deep Resonance. The deeper problem: Deep Resonance fires before Generative Conflict and Creative Dilation, and contains **no delta_kappa check**. For all bio=0 sessions with `sem > 0.4` and `aff > 0.4`, the value of delta_kappa is irrelevant — the state is Deep Resonance regardless of whether the semantic activity is convergent (low dk) or expansively divergent (high dk). This collapses the most theoretically significant dimension of the taxonomy for the dominant use case.

### B5-N2: BASIN_CENTROIDS placed in empirically unreachable regions

Centroids assume psi_affective up to 0.5 and psi_biosignal up to 0.5. Observed psi_affective max from golden fixtures is 0.248; psi_biosignal max is 0.124 for typical resting HR. Deep Resonance centroid `(0.5, 0.5, 0.5, 0.5)` requires affective engagement approximately double the maximum observed in test cases. **The centroid geometry was derived from threshold values in `_classify_basin`, not from empirical basin observations.** A centroid at 0.5 placed above a realistic maximum of 0.25 is not representative of the basin's actual psi distribution.

Consequence: soft membership assigns systematically low weights to Deep Resonance even for inputs that the hard classifier correctly identifies as Deep Resonance, because the soft membership's centroid is further from real observations than from the Transitional or Sycophantic centroids.

### B5-N3: Negative psi_semantic states are invisible to soft membership

psi_semantic is frequently negative in real sessions (4 of 5 non-precomputed golden fixture scenarios). All 10 centroids have positive psi_semantic (range: 0.1–0.5). A session with psi_semantic = -0.5 (legitimate, observed for hedging-heavy text) is uniformly far from all centroids: minimum d² ≈ 0.36 (from Dissociation centroid at 0.1). The soft membership distribution for such sessions will be dominated by the proximity hierarchy of non-semantic dimensions — affective and biosignal — rather than by any meaningful basin signal. The entire semantic substrate becomes a distortion source for soft membership in the negative-semantic half of the space.

### B5-N4: Temperature 1.0 causes macro-uniform membership, not just ambiguous-trio failure

Phase A frames T=1.0 as a problem specific to the ambiguous trio. The calculated membership at the exact CI centroid (11.3% CI, 10.7% SC, 10.2% GC, 9.3% Dis) shows T=1.0 produces near-uniform membership across all 10 basins from any starting position. This is not a basin-specific failure — it means the soft membership is effectively uninformative for any input, not just for CI/RP/CM. The API caller receives a distribution that is always near-uniform, making `primary_basin` and `secondary_basin` output nearly arbitrary.

---

## Revised Assessment

**Verdict: PARTIAL** (maintaining Phase A verdict; adding two concerns to the four identified)

Phase A's four concerns are confirmed. Two additional concerns:

5. **Centroids placed in empirically unreachable regions**: BASIN_CENTROIDS were derived from threshold boundary midpoints, not from empirical basin observations. Several centroids require psi values (particularly psi_affective ≥ 0.4, psi_biosignal = 0.5) that typical dialogue sessions do not produce. Soft membership is therefore systematically miscalibrated for real inputs, independent of the temperature problem.

6. **Deep Resonance is delta_kappa-blind**: The ordering check classifies all high-sem, high-aff, bio=0 states as Deep Resonance without consulting delta_kappa. High-dk states (divergent, expansive) that should route to Generative Conflict or Creative Dilation are captured by Deep Resonance instead. For the dominant bio=0 use case, the taxonomy cannot distinguish convergent resonance from productive creative tension.

**The underlying architectural tension: the classification logic and the centroid geometry were built from different design assumptions. The logic was designed around threshold regions for the hard classifier; the centroids were placed at round-number proxies for those regions. A rigorous soft membership implementation would require (a) empirically calibrated centroids from actual corpus observations, (b) centroid expansion to include dialogue context dimensions for the ambiguous trio, and (c) temperature calibrated to the actual inter-basin distances in the empirical corpus.**

---

## Notes

- The task description's counter-claim that "all three scores default to 0, so classification falls through to Transitional" is **factually incorrect**. The correct default behavior (Cognitive Mimicry with 0.7 score) was accurately identified by Phase A. The turn_length_ratio default of 1.0 (not 0.0) and the coherence_pattern default of 'transitional' (not neutral) are the critical elements the counter-claim misses.
- Phase A's note about absent Reflexive Performance tests stands: no test directly asserts RP as a primary classification output.
- Phase A's note about `coherence_pattern` always being 'transitional' in production (Collaborative Inquiry can never achieve its maximum score) remains unaddressed. With coherence locked at 'transitional', CI maximum achievable score = 0.6, vs Reflexive Performance maximum = 0.8, creating a permanent structural advantage for RP over CI in any real session.
- The `bio == 0.0` condition in Deep Resonance (line 595) was likely intended to express "missing biosignal should not disqualify resonance." But logically, absence of evidence is not evidence of the condition — especially when `bio > 0.4` is the intended discriminator. The condition effectively makes biosignal presence penalizing (requiring bio > 0.4) while biosignal absence is permissive (always satisfies the check), inverting the evidential logic.
