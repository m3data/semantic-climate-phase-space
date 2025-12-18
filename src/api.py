"""
Semantic Climate Phase Space - Function API

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

Lightweight, stateless functions for metric calculation and statistical analysis.
Designed for testing, scripting, and research workflows where class instantiation
is unnecessary overhead.

These functions provide simplified interfaces to the core metrics from
Morgoulis (2025) and statistical utilities for validation studies.

Usage:
    from src.api import semantic_curvature, dfa_alpha, entropy_shift

    # Calculate metrics directly
    kappa = semantic_curvature(embeddings)
    alpha = dfa_alpha(magnitude_signal)
    delta_h = entropy_shift(pre_embeddings, post_embeddings)

Core metric algorithms based on Morgoulis (2025, MIT License):
    https://github.com/daryamorgoulis/4d-semantic-coupling
"""

from typing import List, Tuple, Dict
import itertools
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

__all__ = [
    # Core metrics (simplified)
    'semantic_curvature',
    'semantic_curvature_ci',
    'dfa_alpha',
    'entropy_shift',
    # Statistical utilities
    'icc_oneway_random',
    'icc_bootstrap_ci',
    'bland_altman',
    'all_pairs_bland_altman',
    # Helpers
    'cosine_sim',
    'bootstrap_ci',
]


# =============================================================================
# Helper Functions
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1D vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        float: Cosine similarity in range [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return 1.0 - cosine(a, b)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval for the mean.

    Args:
        values: Array of values to bootstrap
        n_boot: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.mean(values[idx]))
    alpha = (1 - ci) / 2
    return (float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha)))


# =============================================================================
# Core Metrics - Based on Morgoulis (2025)
# =============================================================================

def semantic_curvature(embeddings: np.ndarray) -> float:
    """
    Calculate Semantic Curvature (Δκ) for a dialogue trajectory.

    CRITICAL FIX (2025-12-08): Now uses discrete local curvature via Frenet-Serret
    instead of chord deviation. This measures TRUE curvature (how fast direction
    changes) rather than deviation from start-to-end line.

    Algorithm: Discrete Frenet-Serret curvature, based on Morgoulis (2025) concept

    Args:
        embeddings: Array of shape (n_turns, embedding_dim), ordered by turn

    Returns:
        float: Mean local curvature (κ = ||a_perp|| / ||v||²)
    """
    n = embeddings.shape[0]
    if n < 4:  # Need at least 4 points for 2 curvature measurements
        return 0.0

    # Compute velocities and accelerations
    velocities = np.diff(embeddings, axis=0)  # n-1 velocities
    accelerations = np.diff(velocities, axis=0)  # n-2 accelerations

    # Compute local curvature at each interior point
    local_curvatures = []
    for i in range(len(accelerations)):
        v = velocities[i]
        a = accelerations[i]

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            local_curvatures.append(0.0)
            continue

        # Curvature: κ = ||a_perp|| / ||v||²
        # a_perp = a - (a·v̂)v̂
        v_hat = v / v_norm
        a_parallel = np.dot(a, v_hat) * v_hat
        a_perp = a - a_parallel
        kappa = np.linalg.norm(a_perp) / (v_norm ** 2)
        local_curvatures.append(kappa)

    if not local_curvatures:
        return 0.0

    return float(np.mean(local_curvatures))


def semantic_curvature_ci(
    embeddings: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate Semantic Curvature with bootstrap confidence interval.

    Args:
        embeddings: Array of shape (n_turns, embedding_dim)
        n_boot: Number of bootstrap iterations
        ci: Confidence level
        random_state: Random seed

    Returns:
        tuple: (point_estimate, (ci_lower, ci_upper))
    """
    n = embeddings.shape[0]
    if n < 4:  # Need at least 4 points for local curvature
        return 0.0, (0.0, 0.0)

    rng = np.random.default_rng(random_state)

    def resample_once():
        idx = np.sort(rng.integers(0, n, n))
        return semantic_curvature(embeddings[idx])

    boots = np.array([resample_once() for _ in range(n_boot)])
    alpha = (1 - ci) / 2

    return float(boots.mean()), (
        float(np.quantile(boots, alpha)),
        float(np.quantile(boots, 1 - alpha))
    )


def dfa_alpha(
    signal: np.ndarray,
    min_scale: int = 4,
    max_scale_fraction: float = 0.25
) -> float:
    """
    Calculate Fractal Similarity Score (α) via Detrended Fluctuation Analysis.

    Quantifies scale-invariant patterns in a time series. Values in the
    range [0.70, 0.90] indicate healthy self-organization without collapse
    to periodicity (α → 1.5) or white noise (α → 0.5).

    Algorithm: Morgoulis (2025), based on Peng et al. (1994)

    Args:
        signal: 1D array (e.g., embedding magnitudes over turns)
        min_scale: Minimum window size for DFA
        max_scale_fraction: Maximum window as fraction of signal length

    Returns:
        float: DFA scaling exponent α
    """
    x = signal - np.mean(signal)
    y = np.cumsum(x)
    N = len(y)

    max_scale = max(min(int(N * max_scale_fraction), N // 2), min_scale + 1)
    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        16
    ).astype(int))

    F = []
    valid_scales = []

    for s in scales:
        nseg = N // s
        if nseg < 2:
            continue

        segs = y[:nseg * s].reshape(nseg, s)
        rms = []

        for seg in segs:
            t_idx = np.arange(s)
            coeff = np.polyfit(t_idx, seg, 1)
            trend = np.polyval(coeff, t_idx)
            detr = seg - trend
            rms.append(np.sqrt(np.mean(detr**2)))

        F.append(np.mean(rms))
        valid_scales.append(s)

    if len(F) < 2:
        return 0.5

    log_s = np.log10(np.array(valid_scales))
    log_F = np.log10(np.array(F))
    alpha, _ = np.polyfit(log_s, log_F, 1)

    return float(alpha)


def entropy_shift(
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42
) -> float:
    """
    Calculate Entropy Shift (ΔH) between pre and post dialogue segments.

    CRITICAL FIX (2025-12-08): Now uses SHARED CLUSTERING so cluster IDs
    correspond between pre and post. Measures true reorganization via
    Jensen-Shannon divergence instead of entropy difference.

    Algorithm: Shared clustering with JS divergence, based on Morgoulis (2025) concept

    Args:
        embeddings_pre: Pre-intervention embeddings, shape (n, d)
        embeddings_post: Post-intervention embeddings, shape (m, d)
        n_clusters: Number of clusters for KMeans
        random_state: Random seed for reproducibility

    Returns:
        float: Jensen-Shannon divergence [0, 1]
    """
    n_pre = len(embeddings_pre)
    n_post = len(embeddings_post)

    if n_pre < 2 or n_post < 2:
        return 0.0

    # Reduce clusters if insufficient data
    n_clusters = min(n_pre + n_post, n_clusters)
    if n_clusters < 2:
        return 0.0

    # SHARED CLUSTERING: cluster the full trajectory
    all_embeddings = np.vstack([embeddings_pre, embeddings_post])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(all_embeddings)

    # Split labels back to pre and post
    labels_pre = labels[:n_pre]
    labels_post = labels[n_pre:]

    # Compute distributions over shared cluster space
    def _distribution(labels_subset, n_k):
        counts = np.zeros(n_k)
        for label in labels_subset:
            counts[label] += 1
        return counts / counts.sum()

    p = _distribution(labels_pre, n_clusters)
    q = _distribution(labels_post, n_clusters)

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)

    def _kl_divergence(pk, qk):
        # KL(p || q) with smoothing
        pk = np.clip(pk, 1e-12, 1)
        qk = np.clip(qk, 1e-12, 1)
        return float(np.sum(pk * np.log2(pk / qk)))

    js = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    return float(js)


# =============================================================================
# Statistical Utilities - For Validation Studies
# =============================================================================

def icc_oneway_random(data: np.ndarray) -> float:
    """
    Calculate ICC(1,1) - Intraclass Correlation Coefficient (one-way random).

    Used for assessing agreement between multiple raters/instruments.

    Args:
        data: Array of shape (n_targets, n_raters)

    Returns:
        float: ICC value in range [-1, 1], where 1 = perfect agreement
    """
    n, k = data.shape
    grand_mean = np.mean(data)
    mpt = np.mean(data, axis=1)  # Mean per target

    SSB = k * np.sum((mpt - grand_mean)**2)  # Between-target sum of squares
    SST = np.sum((data - grand_mean)**2)     # Total sum of squares
    SSW = SST - SSB                           # Within-target sum of squares

    MSB = SSB / (n - 1)
    MSW = SSW / (n * (k - 1))

    ICC = (MSB - MSW) / (MSB + (k - 1) * MSW + 1e-12)

    return float(ICC)


def icc_bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate ICC(1,1) with bootstrap confidence interval.

    Args:
        data: Array of shape (n_targets, n_raters)
        n_boot: Number of bootstrap iterations
        ci: Confidence level
        random_state: Random seed

    Returns:
        tuple: (point_estimate, (ci_lower, ci_upper))
    """
    rng = np.random.default_rng(random_state)
    n = data.shape[0]

    ics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ics.append(icc_oneway_random(data[idx]))

    ics = np.array(ics)
    point = float(np.mean(ics))
    alpha = (1 - ci) / 2

    return point, (float(np.quantile(ics, alpha)), float(np.quantile(ics, 1 - alpha)))


def bland_altman(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Calculate Bland-Altman statistics for method comparison.

    Measures agreement between two measurement methods by analyzing
    the differences against the mean of both methods.

    Args:
        a: Measurements from method A
        b: Measurements from method B

    Returns:
        dict: {
            'mean_diff': Mean difference (bias),
            'sd_diff': Standard deviation of differences,
            'loa_low': Lower limit of agreement (mean - 1.96*SD),
            'loa_high': Upper limit of agreement (mean + 1.96*SD)
        }
    """
    diff = a - b
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd

    return {
        "mean_diff": md,
        "sd_diff": sd,
        "loa_low": loa_low,
        "loa_high": loa_high
    }


def all_pairs_bland_altman(
    matrix: np.ndarray,
    labels: List[str]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate Bland-Altman statistics for all pairs of raters/methods.

    Args:
        matrix: Array of shape (n_targets, n_raters)
        labels: List of rater/method names

    Returns:
        dict: {(label_i, label_j): bland_altman_stats} for all pairs
    """
    out = {}
    k = len(labels)

    for i, j in itertools.combinations(range(k), 2):
        a = matrix[:, i]
        b = matrix[:, j]
        out[(labels[i], labels[j])] = bland_altman(a, b)

    return out
