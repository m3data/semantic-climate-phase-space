# Finding A1: DFA/α Fractal Similarity — Audit Report

**Task:** `a1-dfa-alpha-fractal-similarity`
**Date:** 2026-02-25
**Auditor:** Claude Sonnet 4.6 (RAA-SC-001 Phase A)
**Claim under audit:** DFA/α implementation correctly computes fractal similarity via detrended fluctuation analysis on semantic velocity (inter-turn cosine distances), producing meaningful α exponents where α≈0.5=white noise, α≈1.0=pink noise, α>1.2=lock-in.

---

## Summary

The DFA algorithm is **structurally correct** — the core steps (mean-centering, cumulative sum, non-overlapping segmentation, polynomial detrending, RMS fluctuation, log-log regression) follow the standard DFA-1 procedure. The input signal (semantic velocity via cosine distance) is the **right choice** and represents a genuine improvement over the original Morgoulis implementation (which used embedding norms, a near-constant signal for L2-normalized embeddings).

However, **two critical issues** compromise the reliability of α estimates for typical dialogue lengths:

1. **Scale range critically undersized for short conversations** (P1) — dialogues of 20–60 turns produce scale ranges spanning 0.0–0.54 log-decades, far below the ~1 decade minimum for reliable DFA slope estimation.
2. **Bootstrap CI uses IID resampling** (P2) — destroys temporal structure; confidence intervals are statistically invalid.

Additionally, the 20-turn "minimum viable" threshold is misleading: DFA is not actually computed at 20 turns (fallback path activates). The practical minimum for any DFA result is ~25 turns, and for a marginally reliable result is ~30–40 turns.

---

## Check 1: DFA Algorithm Correctness

**Files examined:** `src/core_metrics.py:234–355` (`fractal_similarity_robust`) and `:357–412` (`_calculate_alpha_single`)

### Assessment: PASS

The implementation correctly follows DFA-1:

| Step | Code | Correct? |
|------|------|----------|
| Mean-centre input | `mean_centered = token_sequence - np.mean(token_sequence)` | ✓ |
| Cumulative sum (profile) | `cumulative_sum = np.cumsum(mean_centered)` | ✓ |
| Non-overlapping segments | `range(0, len(cumulative_sum) - scale + 1, scale)` stride=scale | ✓ |
| Polynomial detrending | `np.polyfit(x, segment, polynomial_order=1)` (linear, DFA-1) | ✓ |
| RMS fluctuation per segment | `np.sqrt(np.mean(detrended**2))` | ✓ |
| Mean fluctuation per scale | `np.mean(segment_fluctuations)` | ✓ |
| Log-log regression | `np.polyfit(log10(scales), log10(fluctuations), 1)` | ✓ |
| Goodness-of-fit | R² computed correctly from residuals | ✓ |
| Inf/NaN filtering | `valid_idx = np.isfinite(...) & np.isfinite(...)` | ✓ |
| Minimum scales guard | `if len(log_scales) < 3: return fallback` | ✓ |

The log-log regression uses `np.polyfit` (OLS). This is standard for DFA. No edge-case failures were found in the regression path.

**Note on `_calculate_alpha_single`:** This helper uses 10 log-spaced scales (vs. 20 in the main function). Acceptable for bootstrap resampling where computation cost matters, but produces slightly less resolution.

---

## Check 2: Input Signal — Semantic Velocity

**File:** `src/core_metrics.py:662–682` (`calculate_all_metrics`)

### Assessment: PASS

The signal passed to DFA is inter-turn cosine distance:

```python
cos_sim = cosine_similarity(
    embeddings_array[i].reshape(1, -1),
    embeddings_array[i + 1].reshape(1, -1)
)[0, 0]
semantic_velocities.append(1.0 - cos_sim)
```

This is correct. For a dialogue of N turns, it produces N-1 scalar velocity values measuring how fast semantic content changes between consecutive turns.

**The fix from the original Morgoulis implementation is valid.** Computing DFA on `np.linalg.norm(embedding)` for L2-normalized embeddings yields a near-constant signal (≈1.0 throughout), producing a degenerate DFA result. Semantic velocity is a meaningful and varying signal.

**Range of values:** Cosine distance ∈ [0, 2] in principle, but for typical sentence embeddings cosine distance is ∈ [0, 1] (since embeddings are non-negative after normalisation). The DFA applies regardless of the input range since it mean-centres before integration. ✓

---

## Check 3: Scale Parameters — Adequacy for Typical Dialogue Lengths

**Parameters:** `min_scale=4`, `max_scale_factor=0.25`, 20 log-spaced scale candidates

### Assessment: FAIL (P1)

This is the most significant finding.

#### Scale range computation

For a velocity sequence of length N:
- `max_scale = max(int(N * 0.25), min_scale + 1) = max(int(N/4), 5)`
- Scales are 20 log-spaced integers from 4 to max_scale, then `np.unique()`

#### Concrete analysis for typical conversation lengths

| Turns | Velocities (N) | max_scale | Unique scales | log-decade range | Regression viable? |
|-------|---------------|-----------|---------------|------------------|--------------------|
| 20 | 19 | _(fallback before DFA)_ | — | — | **No (fallback)** |
| 21 | 20 | 5 | [4, 5] = 2 | 0.097 | **No (<3)** |
| 22–24 | 21–23 | 5–5 | [4, 5] = 2 | 0.097 | **No (<3)** |
| 25 | 24 | 6 | [4, 5, 6] = 3 | 0.176 | Barely (3 pts) |
| 30 | 29 | 7 | [4, 5, 6, 7] = 4 | 0.243 | Marginal |
| 40 | 39 | 9 | [4, 5, 6, 7, 8, 9] = 6 | 0.352 | Poor |
| 60 | 59 | 14 | [4..14] ≈ 11 | 0.544 | Inadequate |
| 100 | 99 | 24 | [4..24] ≈ 16 | 0.778 | Borderline |
| 500 | 499 | 124 | [4..124] ≈ 20 | 1.49 | Acceptable |

**Standard DFA guidance requires at least 1 log-decade of scale range for reliable slope estimation.** At 1 decade, you need N ≈ 500 velocity samples (501 turns). This is far beyond typical dialogue lengths.

For the system's stated application domain (20–60 turn dialogues), the achievable scale range is 0.0–0.54 decades. Within this range:
- Only 2–11 distinct integer scales are available
- The slope estimate (α) is highly sensitive to noise at any single scale
- Small perturbations in a single fluctuation value can shift α by ±0.3 or more

#### Consequence for the lock-in claim

The claim states α > 1.2 indicates lock-in. With only 4–11 scales spanning <0.55 decades, the regression produces α values that are **unreliable estimates** of the true Hurst exponent. A difference of α = 0.8 vs. α = 1.2 (the "healthy" vs. "lock-in" boundary) could easily be artefact.

The preprint's reported range α = 0.77–1.27 for the Grunch conversation (presumably a typical-length dialogue) may not be distinguishable from noise given these scale constraints.

---

## Check 4: Bootstrap CI — Resampling Method

**File:** `src/core_metrics.py:335–344`

### Assessment: FAIL (P2) — Critical methodological error

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

**The bootstrap resamples velocity values IID (with replacement, random order).** This is fundamentally incompatible with DFA.

DFA measures *temporal autocorrelation structure*. The Hurst exponent α characterises how correlations decay across scales. When you resample IID:
1. The temporal order is destroyed — consecutive pairs are randomised
2. The resampled sequence has the same marginal distribution but α ≈ 0.5 (white noise) by construction
3. Every bootstrap sample therefore estimates α ≈ 0.5 regardless of the true α
4. The resulting CI reflects the variability of white-noise DFA estimates, not uncertainty about the observed series

**What you get:** CI for a white-noise process
**What you need:** CI reflecting estimation uncertainty for the *observed* temporal structure

The correct approach is a **block bootstrap** (moving blocks or circular blocks) which preserves local autocorrelation. Block size should be ≥ the correlation length, typically estimated from the ACF.

**Consequence:** All reported confidence intervals for α are statistically invalid. The CI does not bound the true α with any meaningful coverage probability. In the Grunch preprint, if α CIs were used to support the lock-in claim, those CIs cannot be trusted.

**Note:** A secondary consequence is that short bootstrap sequences (which arise when the resampled indices happen to have short effective lengths for DFA scales) also produce the fallback α = 0.5 more often, further biasing CI estimates toward 0.5.

---

## Check 5: Log-Log Regression Edge Cases

**File:** `src/core_metrics.py:307–322`

### Assessment: PASS (with minor notes)

Edge cases handled:

| Scenario | Handling | Correct? |
|----------|----------|----------|
| Zero fluctuation (constant segment) | `log10(0) = -inf`, removed by `valid_idx` | ✓ |
| Insufficient valid scales (<3) | Returns fallback `alpha=0.5, r_squared=0.0` | ✓ |
| Zero variance in log-fluctuations | `ss_tot == 0 → r_squared = 0` | ✓ |
| Empty segments list | `if len(segments) == 0: continue` | ✓ |

**Minor note:** When `ss_tot = 0` (all fluctuations identical across scales — a pathological but possible case), R² = 0 is returned but α is still computed from `np.polyfit`. The fit result would be `slope = 0, intercept = log10(constant)`, which is mathematically defined. The returned α = 0 in this case is not a fallback value and could be misinterpreted. Low confidence should be signalled via R².

**Minor note:** The `_calculate_alpha_single` helper wraps its entire body in `try/except: return 0.5`. This is a broad exception suppressor — errors in polyfit or DFA are silently swallowed. This makes debugging difficult. The broad catch is acceptable in bootstrap context but hides potential issues.

---

## Check 6: Interpretation Correctness

### Assessment: PARTIAL PASS

The standard DFA interpretation:

| α value | Interpretation | Code alignment |
|---------|----------------|----------------|
| α ≈ 0.5 | White noise (uncorrelated) | ✓ (test exists) |
| 0.5 < α < 1.0 | Long-range correlations (persistent) | ✓ (implied) |
| α ≈ 1.0 | 1/f noise (pink noise) | ✓ (stated in claim) |
| α ≈ 1.5 | Brownian motion (random walk) | ✓ (test: `1.2 < α < 1.8`) |
| α > 1.5 | Super-diffusive / unbounded correlations | Not discussed |

**Target range [0.70, 0.90]** (code threshold `alpha_min/alpha_max`): This represents the "healthy complexity" zone. It's reasonable — above white noise (0.5) but below lock-in territory.

**Lock-in threshold α > 1.2:** Stated in the claim and conceptually sound (persistent single-topic behaviour → high α on semantic velocity). However:
- This threshold is **not validated in tests** — no test checks that a "locked-in" synthetic signal produces α > 1.2
- The threshold appears to be empirically derived from Morgoulis' original study; whether it transfers to semantic velocity on shorter time series is not verified
- Given the scale-range constraints identified in Check 3, α > 1.2 vs. α < 1.2 may not be distinguishable at typical dialogue lengths

**Conceptual validity of applying DFA to semantic velocity:** Reasonable. Semantic velocity captures turn-to-turn variation; DFA on this signal can detect whether the variation is random (α ≈ 0.5), shows scaling structure (α ≈ 1.0), or is heavily autocorrelated / slowly varying (α > 1.5). The lock-in intuition (conversation stuck on one topic → small, persistent velocity values → high Hurst exponent) is plausible but not validated.

---

## Check 7: Behaviour at Exactly 20 Turns

### Assessment: 20 turns produce fallback — DFA not computed

Execution path for 20-turn dialogue:

1. `calculate_all_metrics(dialogue_embeddings)` receives 20 embeddings
2. Loop `for i in range(len(embeddings_array) - 1)` = `range(19)` → 19 velocity values
3. Check: `if len(semantic_velocities) < 20` → `19 < 20` = **True**
4. Returns hard-coded fallback: `alpha=0.5, r_squared=0.0, confidence_interval=(0.5, 0.5), scales_used=0, target_range_met=False`
5. DFA is **never called**

For 21 turns (20 velocities):
1. Outer check: `20 < 20` = False → calls `fractal_similarity_robust` with 20 values
2. Inner check: `20 < 20` = False → proceeds
3. `max_scale = max(int(20 * 0.25), 5) = 5`
4. Scales: `np.unique(logspace(log10(4), log10(5), 20).astype(int))` = `[4, 5]` (2 scales)
5. After removing invalid: `len(log_scales) = 2 < 3` → **fallback**

**Practical minimum for any DFA result: 25 turns** (24 velocities → 3 scales [4,5,6])
**Practical minimum for marginally reliable result: ~40 turns** (39 velocities → 6 scales, 0.35 decades)

The documented "minimum viable" of 20 turns is misleading — DFA is silently replaced with a fallback value. Any analysis reporting α for <25 turn dialogues is reporting the constant 0.5, not a computed value.

---

## Test Coverage Assessment

| Test | Coverage | Issue |
|------|----------|-------|
| `test_white_noise_alpha_near_half` (N=200) | White noise α ≈ 0.5 | Uses 200 samples — not representative of dialogue lengths |
| `test_brownian_motion_alpha_near_1_5` (N=200) | Brownian α ≈ 1.5 | Uses 200 samples — not representative |
| `test_minimum_length` (N<20) | Fallback handling | Correct ✓ |
| `test_target_range_detection` (N=100) | Range [0.70, 0.90] | Uses 100 samples — still above typical dialogue length |
| `test_alpha_on_semantic_velocity_not_norms` | Semantic velocity vs norms | Present ✓ |
| **Missing:** Bootstrap validity | IID vs. block bootstrap | **Not tested** |
| **Missing:** Short-dialogue scale range | N=20–60 | **Not tested** |
| **Missing:** Lock-in threshold validation | α > 1.2 for stuck dialogue | **Not tested** |

All DFA tests use N=100–200, well above typical dialogue lengths. No test validates behaviour at the realistic 20–60 turn range.

---

## Findings Summary

| ID | Severity | Finding |
|----|----------|---------|
| A1-F1 | **P1** | Scale range undersized: typical dialogues (20–60 turns) produce 0.0–0.54 log-decades of scale range, below the ~1 decade minimum for reliable DFA. α estimates in this regime are unreliable. |
| A1-F2 | **P2** | Bootstrap CI uses IID resampling, destroying temporal structure. All reported α confidence intervals are statistically invalid. |
| A1-F3 | **P3** | 20-turn "minimum viable" is misleading: DFA fallback activates at <21 turns; practical minimum for any DFA output is 25 turns; for marginal reliability, ~40 turns. |
| A1-F4 | **P3** | Lock-in threshold α > 1.2 is stated but not validated by any test; no synthetic "locked-in" signal test exists. |
| A1-F5 | **P4** | `_calculate_alpha_single` uses broad `except: return 0.5` silently suppressing all errors; complicates debugging. |

**Confidence in claim:** LOW
The DFA algorithm is structurally correct. The input signal choice (semantic velocity) is correct. However, the scale constraints mean α values for typical dialogue lengths are **unreliable slope estimates** computed from too few points over too narrow a range. The IID bootstrap makes confidence intervals meaningless. The Grunch lock-in finding (α = 0.77–1.27) cannot be validated against noise with the current implementation.

---

## Recommendations

1. **P1 — Scale range:** Lower `min_scale` to 2 and/or raise `max_scale_factor` to 0.4 to widen the effective range; add R² threshold filtering (e.g., discard α if R² < 0.9); document minimum reliable dialogue length (~100 turns for R² > 0.95).
2. **P2 — Bootstrap:** Replace IID bootstrap with block bootstrap (e.g., circular block bootstrap, block size 3–5). This is standard for correlated time series.
3. **P3 — Documentation:** Update minimum viable length to 25 turns (any DFA output) and add caveat that results below 40 turns are low-confidence.
4. **P4 — Test coverage:** Add tests with N=25–60 velocity samples; add synthetic lock-in test (slow random walk → α > 1.2).
