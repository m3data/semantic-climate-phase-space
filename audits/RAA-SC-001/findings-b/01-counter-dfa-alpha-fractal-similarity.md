# Counter-Audit B1: DFA/α Fractal Similarity

**Task:** `b1-counter-dfa-alpha-fractal-similarity`
**Date:** 2026-02-25
**Auditor:** Claude Sonnet 4.6 (RAA-SC-001 Phase B — adversarial counter-audit)
**Phase A finding under review:** `findings-a/01-dfa-alpha-fractal-similarity.md`
**Claim under audit:** DFA/α implementation correctly computes fractal similarity via detrended fluctuation analysis on semantic velocity (inter-turn cosine distances), producing meaningful α exponents where α≈0.5=white noise, α≈1.0=pink noise, α>1.2=lock-in.

---

## Counter-Audit Methodology

This counter-audit proceeds in two passes:

1. **Independent first-pass** — I read `src/core_metrics.py` and form my own assessment of the five targeted questions before consulting Phase A's conclusions.
2. **Comparative second-pass** — I compare my findings to Phase A and explicitly assess: Confirm / Extend / Challenge.

The goal is adversarial: find findings Phase A missed, find places Phase A overclaimed severity, and identify the single most load-bearing weakness for the Grunch lock-in claim.

---

## Independent Analysis

### Question 1: Does IID bootstrap destroy temporal structure?

**Code under review:** `fractal_similarity_robust`, lines 335–344:

```python
bootstrap_alphas = []
for _ in range(min(self.bootstrap_iterations, 300)):
    boot_indices = np.random.choice(
        len(token_sequence), size=len(token_sequence), replace=True
    )
    boot_sequence = token_sequence[boot_indices]
    boot_alpha = self._calculate_alpha_single(boot_sequence, ...)
    bootstrap_alphas.append(boot_alpha)
```

**Finding: Yes, IID resampling invalidates the CI — and there are two additional failure modes Phase A did not identify.**

**Primary failure (confirmed from Phase A):** The bootstrap resamples velocity values with replacement and random ordering. DFA measures the autocorrelation structure of the ordered sequence. IID resampling produces a sequence with the same marginal distribution but α ≈ 0.5 (white noise) by construction — the bootstrap CI estimates the variability of a white-noise DFA, not the uncertainty in the observed α.

**Secondary failure (Phase A missed):** `_calculate_alpha_single` wraps its entire body in `except: return 0.5`. For short bootstrap sequences (especially when resampling a 20-value sequence), many bootstrap iterations will fail silently and return exactly 0.5. This creates a **spike at 0.5** in `bootstrap_alphas`. The resulting percentile CI will be artificially compressed around 0.5 — a narrow CI that looks precise but is doubly invalid: wrong center (biased toward 0.5) and wrong width (computed from a spike distribution).

**Tertiary failure (Phase A missed):** `_calculate_alpha_single` uses **10** log-spaced scales (line 369), not 20 as in the main function (line 273). For the same sequence length, 10 candidate scale points produces fewer or equal unique integer scales than 20 points. For a 20-value bootstrap sample with max_scale=5, 10 log-spaced points from log10(4) to log10(5) still collapses to [4, 5] — 2 unique scales — triggering the `len(fluctuations) < 3` fallback return 0.5. The bootstrap function therefore always returns 0.5 for bootstrap samples of length 20–24, regardless of their content.

**Combined effect:** For typical dialogue lengths (21–40 turns), nearly all 300 bootstrap iterations return 0.5 — either from the scale fallback (silently from the except) or from the temporal destruction. The reported CI `(0.5, 0.5)` or a very narrow band near 0.5 is an artifact, not an estimate.

---

### Question 2: With 19 velocity samples, is α meaningful?

**Code path for 19 velocity samples (20-turn dialogue):**

- `calculate_all_metrics`: `len(semantic_velocities) < 20` → `19 < 20` = True → **DFA never called**
- Returns hard-coded `alpha=0.5, confidence_interval=(0.5, 0.5), scales_used=0`

**Code path for 20 velocity samples (21-turn dialogue):**

- `fractal_similarity_robust` called with 20-element array
- `len(token_sequence) < 20` → `20 < 20` = False → proceeds
- `max_scale = max(int(20 * 0.25), 5) = max(5, 5) = 5`
- `np.logspace(log10(4), log10(5), 20)` spans [4.0, 5.0], all values in [4.0, 5.0)
- `.astype(int)` maps values < 5.0 to 4, value = 5.0 to 5
- `np.unique(...)` = **[4, 5]** — 2 unique scales
- Inner check: `len(log_scales) < 3` → `2 < 3` = True → **fallback, DFA not executed**

**Verdict: At 20 velocities (21 turns), DFA falls back identically to the 19-velocity case.** Phase A's table is accurate.

**An additional degradation Phase A underweights:** Even when the `< 3` scale check passes (at 25+ turns giving 3 scales), a linear regression through **exactly 3 points** is numerically degenerate in an important sense: it has 1 residual degree of freedom. R² can appear high from one noisy point falling near the regression line, while the slope estimate is wildly unstable. With 3 points, removing any single one changes the slope by a potentially arbitrary amount. The R² reported by the code at 25 turns will appear plausible (0.8–0.99) while the slope (α) is unreliable.

**Quantitative verification of task description claim:** The task description states "max_scale = max(int(19*0.25), 5) = 5." This is correct: `int(19*0.25) = int(4.75) = 4`, and `max(4, 5) = 5`. However, this path is never reached for 19 velocities because the outer guard in `calculate_all_metrics` returns the fallback first. The analysis is correct in substance but slightly misattributes which fallback fires.

---

### Question 3: What errors does the bare except clause mask?

**Code under review:** `_calculate_alpha_single`, lines 360–412:

```python
try:
    ...
    alpha, _ = np.polyfit(log_scales, log_fluctuations, 1)
    return alpha
except:
    return 0.5
```

**Errors that could legitimately occur and be silently masked:**

| Error | Condition | Probability | Impact |
|-------|-----------|-------------|--------|
| `numpy.linalg.LinAlgError` | `polyfit` receives a rank-deficient system; occurs when all x-values are identical (e.g., all scales collapse to same integer) | Low but non-zero for degenerate sequences | Signals genuinely pathological input; silently returns 0.5 misrepresents as "white noise" |
| `ValueError: zero-size array` | `log_scales` or `log_fluctuations` empty after `np.isfinite` filtering; can occur when all fluctuations are zero (constant velocity segment) | Low | Same as above |
| `ValueError: array must not contain infs or NaNs` | `polyfit` from NumPy raises this when passed inf values; could happen before the valid_idx filter runs if implementation changes | Low | Masked |
| `MemoryError` | Bootstrap on very large sequences; 300 × large array operations | Negligible in practice | Would be catastrophic if masked |
| `SystemExit` / `KeyboardInterrupt` | These are **not** caught by bare `except` since they don't inherit from `BaseException`... wait, actually **bare `except:` catches SystemExit and KeyboardInterrupt** in Python 2 but NOT in Python 3 (they inherit from `BaseException` not `Exception`, and bare `except:` catches `BaseException` in Python 3 too) | Negligible | Would silently prevent interruption |

**The key concern is not any single error — it's the combination:** When `_calculate_alpha_single` returns 0.5 silently in bootstrap iterations due to scale collapse (the common case for short sequences), the except clause is actually *not* the primary cause. The inner `return 0.5` on line 396 (`if len(fluctuations) < 3`) is the primary cause, and this is a legitimate fallback. The bare except is redundant for the common paths but masks the uncommon pathological cases.

**Specific error Phase A understated:** `np.polyfit` with `deg=1` requires at least 2 distinct x-values. For a sequence of exactly 20 values where all scales collapse to identical integers, the `np.unique` step produces a 1-element array. The code then computes `log_scales[:len(fluctuations)]` where `fluctuations` also has 1 element. The `len(log_scales) < 3` check fires and returns 0.5 normally — so the except clause is actually *not needed* for this path. The bare except is primarily a code quality issue, not a functional error in the common paths.

---

### Question 4: Does semantic velocity have the stationarity properties DFA assumes?

This question receives the most superficial treatment in Phase A ("reasonable", "plausible") and deserves independent deeper analysis.

**What DFA actually requires:**

Standard DFA (Peng et al. 1994, Physica A) was designed to detect long-range power-law correlations while being robust to *polynomial non-stationarity* in the mean. It detrends each window by a polynomial of degree `polynomial_order` (here, 1). This handles linear drifts in the signal.

However, DFA assumes:
1. The scaling law F(s) ~ s^α holds consistently across the full scale range
2. The signal's statistical properties are **homogeneous across time** (second-order stationarity, except for the polynomial mean trend)
3. The signal is long enough to sample multiple scales reliably

**Problems with semantic velocity as a DFA input:**

**(A) Hard lower bound — structural non-Gaussianity**

Cosine distance ∈ [0, 1] for typical sentence embeddings (after L2 normalization, embeddings tend to be positive-valued, so cosine distance is rarely > 1). The hard lower bound at 0 means the marginal distribution is right-skewed and bounded. This causes the cumulative sum Y(k) = Σ[velocity(i) - mean_velocity] to be bounded from below when mean_velocity > 0, preventing the unbounded random walk behavior that produces clean power-law scaling in F(s).

**(B) Non-stationary variance across conversational arc**

Conversations have internal structure: opening sequences, topic introduction, development, convergence. Early turns typically show higher semantic velocity (introducing topics, shifting frames). Later turns in a focused dialogue show lower velocity (topic maintenance, elaboration). This constitutes **variance non-stationarity** — the variance of velocity changes systematically over time — which DFA-1 (linear detrending) cannot remove. Only MFDFA or higher-order DFA would partially address this.

For the lock-in claim specifically: a conversation exhibiting "lock-in" would show a characteristic *decrease* in velocity over time (settling into a narrow topic). This temporal trend in variance is exactly what DFA-1 cannot distinguish from genuine power-law correlation. A declining variance sequence will produce α estimates that drift toward higher values — mimicking lock-in even in a non-fractal process.

**(C) Conversational structure as pseudo-correlation**

Dialogue follows social/rhetorical norms: topic introduction, elaboration, transition, closure. These produce temporal patterns in semantic velocity that are structural rather than stochastic. DFA interprets any systematic temporal pattern as correlation (higher α). A question-answer-follow-up structure produces alternating low-high velocity that DFA would characterize as anti-correlated (α < 0.5), while a sustained monologue produces persistent low velocity that DFA would characterize as positively correlated (α > 1.0). These are not "fractal" properties — they're discourse structure.

**(D) The signal is already a first-difference**

The velocity v(i) = 1 - cos_sim(e(i), e(i+1)) is a first-difference of the embedding trajectory. DFA then computes the cumulative sum Y(k) = Σv(i) — effectively the second-order integration of the trajectory. The relationship between α on Y(k) and any meaningful property of the original conversation is indirect and not theoretically motivated. The standard DFA application to physiological data (heartbeats, neural oscillations) applies DFA to the raw signal, not to a derived difference signal. The double-integration makes the scaling interpretation non-standard.

**Summary on stationarity:** Phase A's "PASS" for the input signal choice is too lenient. The semantic velocity signal likely violates the homogeneous variance assumption, and the conversational arc structure creates pseudo-correlations that DFA cannot distinguish from genuine power-law correlations. These are not "minor notes" — they call into question whether α from this implementation is interpretable as a Hurst exponent for *any* dialogue length.

---

### Question 5: Are interpretation thresholds from DFA literature or arbitrary?

**Standard DFA thresholds — literature provenance:**

| Threshold | Value | From DFA literature? | Reference |
|-----------|-------|---------------------|-----------|
| White noise | α ≈ 0.5 | **YES** | Peng et al. (1994) |
| Pink noise / 1/f | α ≈ 1.0 | **YES** | Peng et al. (1994), Buldyrev et al. |
| Brownian motion | α ≈ 1.5 | **YES** | Peng et al. (1994) |
| Lock-in threshold | α > 1.2 | **NOT from DFA literature** | Morgoulis (2025) — no peer review |
| "Healthy complexity" | 0.70–0.90 | **NOT from DFA literature** | Morgoulis (2025) — no peer review |

**Analysis of the 1.2 threshold:**

In standard DFA nomenclature:
- α ∈ (0.5, 1.0): persistent long-range correlations (sub-diffusive)
- α ∈ (1.0, 1.5): stronger persistence (between 1/f and Brownian)
- α > 1.5: super-diffusive, associated with heavy autocorrelation

α = 1.2 is in the range (1.0, 1.5) — stronger-than-pink-noise persistence. The choice of 1.2 vs 1.0 vs 1.5 as the "lock-in boundary" has no theoretical basis in the DFA literature. The code comment calls these "empirically-derived thresholds from validation study" but:
1. No citation is provided beyond the Morgoulis GitHub
2. The validation study is the same preprint under audit
3. Circular: thresholds derived from observations on the Grunch conversation, applied to validate claims about the Grunch conversation

**Analysis of the 0.70–0.90 "healthy complexity" range:**

This range is stated in the code as `alpha_min=0.70, alpha_max=0.90`. This zone is above white noise (0.5) and below the lock-in threshold (1.2). The "healthy" label is interpretive, not validated. In cardiac DFA literature, α ≈ 1.0 ± 0.2 is associated with healthy heart rate variability — this may be where the analogical reasoning originates — but:
1. Cardiac DFA uses a very different signal (RR intervals, 10,000+ data points)
2. The mapping from cardiac "healthy complexity" to semantic "healthy complexity" is by analogy, not evidence
3. The range is not tested against ground-truth labeled dialogues

**Critical additional finding:** The thresholds are codified constants in `__init__` (lines 68–73) attributed to an "empirically-derived validation study" that is *itself the preprint being audited*. This is circular validation: the thresholds define what counts as "healthy" or "locked-in," and the study reports finding α values that cross these thresholds, but the thresholds were set with this data in mind. No out-of-sample validation exists.

---

## Comparative Assessment: Phase A vs Independent Findings

### Where Phase A is correct

| Phase A Finding | Assessment | Verdict |
|----------------|------------|---------|
| Scale range critically undersized (P1) | Math checks out; 20 velocities → 2 scales → fallback. Table of turns vs. scales is accurate | **CONFIRM** |
| Bootstrap CI uses IID resampling (P2) | Fundamental methodological error correctly identified | **CONFIRM** |
| 20-turn "minimum viable" misleading (P3) | Correct — DFA fallback activates for ≤20 velocities; actual minimum is 24+ velocities | **CONFIRM** |
| Lock-in threshold not validated (P4) | Correct — no test exists, no out-of-sample validation | **CONFIRM** |
| Bare except masks errors (P4) | Correct identification, appropriate severity | **CONFIRM** |

### Where Phase A is too lenient

| Area | Phase A Assessment | Counter-Assessment |
|------|-------------------|-------------------|
| Input signal validity | "PASS — reasonable and correct" | **Overclaimed**: semantic velocity has structural stationarity violations (bounded, non-stationary variance, conversational arc effects) that DFA-1 cannot handle. This is a **conceptual** not just parametric problem. |
| Algorithm structural correctness | "Structurally correct" repeated prominently | This framing creates false confidence. Mathematical correctness of DFA steps is irrelevant if the input signal violates DFA's assumptions. A correctly implemented algorithm applied to inappropriate data is not "correct." |
| Threshold provenance | "empirically derived from Morgoulis' study" | This understates the circularity — the thresholds were set from the same preprint being validated. Phase A frames this as "unverified" rather than "circularly defined." |
| Stationarity | Not addressed as a finding | **Omitted**: this deserves its own finding. Non-stationary variance in semantic velocity creates systematic bias in α that mimics lock-in for purely structural reasons. |

### Where Phase A may have overclaimed severity

Phase A's finding table assigns P1 to scale range and P2 to bootstrap CI. These severity labels are reasonable. However, Phase A does not note that fixing P1 and P2 (wider scales, block bootstrap) would still leave the stationarity problem unresolved. The framing suggests these are "fixable" issues — a block bootstrap and wider scales are implementable changes. The stationarity concern is harder to fix: it questions whether DFA is the right tool at all for semantic velocity on dialogue data.

### Net additional findings

| ID | Severity | Finding (not in Phase A) |
|----|----------|--------------------------|
| B1-F1 | **P1** | Semantic velocity has structural non-stationarity (bounded distribution, variance changing over conversational arc) that DFA-1 cannot detrend. Even with adequate scale range, α estimates will be biased toward higher values by declining variance — confounding lock-in detection. |
| B1-F2 | **P2** | Bootstrap function uses 10 scales (vs. 20 in main), further collapsing to [4, 5] for bootstrap samples of length 20–24. Near-universal fallback to 0.5 in bootstrap iterations is not solely due to IID resampling — it's also an artifact of the reduced scale count in the helper function. |
| B1-F3 | **P2** | Threshold range [0.70, 0.90] and lock-in value 1.2 were set on the same dataset being evaluated in the preprint (circular validation). Phase A calls this "unverified" — more precisely, it is **circularly defined** and cannot provide independent evidence. |
| B1-F4 | **P3** | With exactly 3 scale points (25–27 turn dialogues), the log-log regression has 1 residual degree of freedom. R² can appear high (>0.8) while the slope is unstable. The R² quality filter would not catch this case — a reported R²=0.95 at 3 scale points has near-zero diagnostic value. |

---

## Overall Verdict

**Phase A's confidence assessment: LOW — CONFIRMED**

However, the reasons are stronger than Phase A states. Phase A's "LOW confidence" is primarily grounded in scale range and bootstrap issues — both parametric/methodological failures that are potentially fixable. My independent analysis adds a third category of concern: **conceptual invalidity** of DFA for semantic velocity at dialogue timescales.

The Grunch lock-in claim (α = 0.77–1.27) is doubly threatened:

1. **At the implementation level** (Phase A's focus): Scale range is too narrow for dialogue lengths; CI is invalid due to IID bootstrap. These are fixable.

2. **At the conceptual level** (this counter-audit's addition): Declining semantic velocity over a "locked-in" conversation creates a non-stationary variance pattern that DFA-1 would estimate as high α regardless of true fractal structure. The threshold α > 1.2 is derived circularly from the same data. Even a fixed implementation could produce systematically inflated α values by measuring the conversational arc, not power-law correlations.

**The most load-bearing weakness for the lock-in claim** is not bootstrap resampling (fixable) or scale range (fixable) but the confounding between **structural variance decline** and **high DFA α**. Both produce α > 1.0 in DFA, but only one is "lock-in." This distinction is not made in the preprint or the code, and cannot be resolved without either a stationarity test (e.g., KPSS test on velocity subwindows) or a non-stationary DFA variant (e.g., DFA-2 with quadratic detrending, or adaptive DFA).

---

## Revised Findings Summary

| ID | Source | Severity | Finding |
|----|--------|----------|---------|
| A1-F1 | Phase A | **P1** | Scale range undersized: <0.55 log-decades for typical dialogues. Confirmed. |
| A1-F2 | Phase A | **P2** | Bootstrap CI uses IID resampling. Confirmed with extensions: helper uses only 10 scales and bare except creates spike at 0.5, further compressing CI toward 0.5. |
| A1-F3 | Phase A | **P3** | 20-turn minimum misleading. Confirmed. |
| A1-F4 | Phase A | **P3** | Lock-in threshold not validated. Confirmed and elevated: threshold is circularly derived. |
| A1-F5 | Phase A | **P4** | Bare except masks errors. Confirmed. Specific error types enumerated above. |
| **B1-F1** | **Counter** | **P1** | Semantic velocity has structural non-stationarity; DFA-1 cannot distinguish declining variance (conversational arc) from power-law correlation. High α is confounded with structural deceleration. |
| **B1-F2** | **Counter** | **P2** | Bootstrap helper uses 10 vs 20 scales; near-universal 0.5 return for short-sequence bootstrap iterations is also scale-collapse driven, not just IID-driven. |
| **B1-F3** | **Counter** | **P2** | Lock-in threshold 1.2 and healthy range 0.70–0.90 are circularly validated from the Grunch dataset itself. Cannot provide independent evidence for the lock-in claim. |
| **B1-F4** | **Counter** | **P3** | 3-point log-log regression (at 25–27 turns) has 1 residual dof; R² > 0.8 is uninformative at this scale count. |

**Phase A overall confidence: LOW — UPHELD**
**Counter-audit revision:** The "LOW" label is warranted but understates the conceptual problem. A phrase like "structurally correct algorithm applied to an inappropriate signal" better captures the situation than "reliable algorithm with fixable parameter issues."
