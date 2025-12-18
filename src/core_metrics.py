"""
4D Semantic Coupling Framework - Core Metrics Implementation

This module implements the three key metrics for measuring cognitive complexity
in AI dialogue systems:
- Semantic Curvature (Δκ)
- Fractal Similarity Score (α)
- Entropy Shift (ΔH)

Original Author: Daria Morgoulis (2025)
License: MIT

Source: https://github.com/daryamorgoulis/4d-semantic-coupling

This file preserves Daria Morgoulis's original implementation with minimal
modifications for modular import structure. All core algorithms, thresholds,
and statistical methods are unchanged from the original.

Citation:
    Morgoulis, D. (2025). 4D Semantic Coupling Framework for Measuring
    Cognitive Complexity in AI Dialogue Systems.
    https://github.com/daryamorgoulis/4d-semantic-coupling

Modifications from original:
    - Adjusted import for sklearn.metrics.pairwise.cosine_similarity
      (functional equivalent, required for some sklearn versions)
    - Removed example_analysis() from module level (moved to examples/)
    - Added __all__ export list for clean namespace
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

__all__ = ['SemanticComplexityAnalyzer']


class SemanticComplexityAnalyzer:
    """
    Main class for calculating cognitive complexity metrics in AI dialogue.

    This class implements the 4D Semantic Coupling Framework with three
    operational metrics validated across multiple AI platforms.

    Original implementation by Daria Morgoulis (2025).
    """

    def __init__(self, random_state=42, bootstrap_iterations=1000):
        """
        Initialize the analyzer with default parameters.

        Args:
            random_state (int): Random seed for reproducibility
            bootstrap_iterations (int): Number of bootstrap samples for CI
        """
        self.random_state = random_state
        self.bootstrap_iterations = bootstrap_iterations
        np.random.seed(random_state)

        # Empirically-derived thresholds from validation study
        self.thresholds = {
            'delta_kappa': 0.35,
            'alpha_min': 0.70,
            'alpha_max': 0.90,
            'delta_h': 0.12
        }

    def semantic_curvature_enhanced(self, embedding_sequence):
        """
        Calculate Semantic Curvature (Δκ) via discrete local curvature.

        CRITICAL FIX (2025-12-08): The original Morgoulis implementation measured
        deviation from a linear chord connecting start to end points. This is NOT
        geodesic curvature — it's chord deviation, highly sensitive to endpoints.
        A circular trajectory returning to start would yield Δκ ≈ 0.

        This fix computes TRUE LOCAL CURVATURE using discrete Frenet-Serret:
        - Velocity: v(t) = e(t+1) - e(t)
        - Acceleration: a(t) = v(t+1) - v(t)
        - Local curvature: κ(t) = ||v × a|| / ||v||³

        For high-dimensional spaces where cross product isn't defined, we use:
        κ(t) = ||a_perp|| / ||v||² where a_perp is acceleration perpendicular to v

        This gives actual trajectory curvature independent of start/end points.

        See: context-building/REVIEW_morgoulis-4d-metrics-mathematical-analysis.md

        Args:
            embedding_sequence (list): Sequence of embeddings (L2-normalized or not)

        Returns:
            dict: {
                'curvature': float,           # Mean local curvature
                'curvature_std': float,       # Std of local curvatures (trajectory variability)
                'confidence_interval': tuple,
                'p_value': float,
                'threshold_met': bool,
                'local_curvatures': list      # κ(t) at each interior point
            }
        """
        n = len(embedding_sequence)
        if n < 4:  # Need at least 4 points for 2 curvature measurements
            return {
                'curvature': 0.0,
                'curvature_std': 0.0,
                'confidence_interval': (0.0, 0.0),
                'p_value': 1.0,
                'threshold_met': False,
                'local_curvatures': []
            }

        embeddings = np.array(embedding_sequence)

        # Compute local curvatures using discrete Frenet-Serret
        local_curvatures = self._compute_local_curvatures(embeddings)

        if len(local_curvatures) == 0:
            return {
                'curvature': 0.0,
                'curvature_std': 0.0,
                'confidence_interval': (0.0, 0.0),
                'p_value': 1.0,
                'threshold_met': False,
                'local_curvatures': []
            }

        # Mean curvature (the primary metric)
        curvature = np.mean(local_curvatures)
        curvature_std = np.std(local_curvatures)

        # Bootstrap confidence interval
        bootstrap_curvatures = []
        for _ in range(min(self.bootstrap_iterations, 500)):
            boot_indices = np.sort(np.random.choice(n, size=n, replace=True))
            boot_embeddings = embeddings[boot_indices]
            boot_local = self._compute_local_curvatures(boot_embeddings)
            if len(boot_local) > 0:
                bootstrap_curvatures.append(np.mean(boot_local))

        if len(bootstrap_curvatures) > 0:
            ci_lower, ci_upper = np.percentile(bootstrap_curvatures, [2.5, 97.5])
        else:
            ci_lower, ci_upper = curvature, curvature

        # Statistical significance: compare to shuffled trajectory
        null_curvatures = []
        for _ in range(200):
            null_embeddings = np.random.permutation(embeddings)
            null_local = self._compute_local_curvatures(null_embeddings)
            if len(null_local) > 0:
                null_curvatures.append(np.mean(null_local))

        if len(null_curvatures) > 0:
            p_value = np.mean(np.array(null_curvatures) >= curvature)
        else:
            p_value = 1.0

        return {
            'curvature': curvature,
            'curvature_std': curvature_std,
            'confidence_interval': (ci_lower, ci_upper),
            'p_value': p_value,
            'threshold_met': curvature >= self.thresholds['delta_kappa'],
            'local_curvatures': local_curvatures.tolist()
        }

    def _compute_local_curvatures(self, embeddings):
        """
        Compute local curvature at each interior point using discrete Frenet-Serret.

        For a trajectory e(t), curvature at point t is:
        κ(t) = ||a_perp|| / ||v||²

        where:
        - v = e(t+1) - e(t) is velocity
        - a = v(t+1) - v(t) is acceleration
        - a_perp = a - (a·v̂)v̂ is acceleration perpendicular to velocity

        Args:
            embeddings: np.array of shape (n, d)

        Returns:
            np.array of local curvatures at points 1 to n-2
        """
        n = len(embeddings)
        if n < 4:
            return np.array([])

        # Compute velocities: v[i] = e[i+1] - e[i]
        velocities = np.diff(embeddings, axis=0)  # Shape: (n-1, d)

        # Compute accelerations: a[i] = v[i+1] - v[i]
        accelerations = np.diff(velocities, axis=0)  # Shape: (n-2, d)

        local_curvatures = []
        for i in range(len(accelerations)):
            v = velocities[i]  # Velocity at point i
            a = accelerations[i]  # Acceleration at point i

            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                # Zero velocity — trajectory is stationary, curvature undefined
                local_curvatures.append(0.0)
                continue

            # Unit velocity vector
            v_hat = v / v_norm

            # Perpendicular component of acceleration
            a_parallel = np.dot(a, v_hat) * v_hat
            a_perp = a - a_parallel

            # Curvature = ||a_perp|| / ||v||²
            kappa = np.linalg.norm(a_perp) / (v_norm ** 2)
            local_curvatures.append(kappa)

        return np.array(local_curvatures)

    def _calculate_curvature_single(self, embeddings):
        """Helper method for single curvature calculation using local curvature."""
        local_curvatures = self._compute_local_curvatures(embeddings)
        if len(local_curvatures) == 0:
            return 0.0
        return np.mean(local_curvatures)

    def fractal_similarity_robust(self, token_sequence, min_scale=4,
                                max_scale_factor=0.25, polynomial_order=1):
        """
        Calculate Fractal Similarity Score (α) via Detrended Fluctuation Analysis.

        Quantifies scale-invariant patterns indicating self-organization without
        collapse to periodicity or noise.

        Args:
            token_sequence (array): Numerical representation of semantic tokens
            min_scale (int): Minimum window size for DFA
            max_scale_factor (float): Maximum window as fraction of sequence length
            polynomial_order (int): Polynomial degree for detrending

        Returns:
            dict: {
                'alpha': float,
                'r_squared': float,
                'confidence_interval': tuple,
                'scales_used': int,
                'target_range_met': bool
            }
        """
        if len(token_sequence) < 20:
            return {
                'alpha': 0.5,
                'r_squared': 0.0,
                'confidence_interval': (0.5, 0.5),
                'scales_used': 0,
                'target_range_met': False
            }

        # Convert to fluctuation profile
        mean_centered = token_sequence - np.mean(token_sequence)
        cumulative_sum = np.cumsum(mean_centered)

        # Generate scales with higher resolution
        max_scale = max(int(len(cumulative_sum) * max_scale_factor), min_scale + 1)
        scales = np.unique(np.logspace(
            np.log10(min_scale), np.log10(max_scale), 20
        ).astype(int))

        fluctuations = []

        for scale in scales:
            segments = []

            # Extract non-overlapping segments
            for start in range(0, len(cumulative_sum) - scale + 1, scale):
                segment = cumulative_sum[start:start + scale]
                if len(segment) == scale:
                    segments.append(segment)

            if len(segments) == 0:
                continue

            # Detrend each segment
            segment_fluctuations = []
            for segment in segments:
                x = np.arange(len(segment))

                # Polynomial detrending
                coefficients = np.polyfit(x, segment, polynomial_order)
                trend = np.polyval(coefficients, x)
                detrended = segment - trend

                # Root mean square fluctuation
                rms_fluctuation = np.sqrt(np.mean(detrended**2))
                segment_fluctuations.append(rms_fluctuation)

            mean_fluctuation = np.mean(segment_fluctuations)
            fluctuations.append(mean_fluctuation)

        # Robust regression in log-log space
        log_scales = np.log10(scales[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)

        # Remove any inf/-inf values
        valid_idx = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
        log_scales = log_scales[valid_idx]
        log_fluctuations = log_fluctuations[valid_idx]

        if len(log_scales) < 3:
            return {
                'alpha': 0.5,
                'r_squared': 0.0,
                'confidence_interval': (0.5, 0.5),
                'scales_used': 0,
                'target_range_met': False
            }

        # Linear regression
        alpha, intercept = np.polyfit(log_scales, log_fluctuations, 1)

        # Goodness of fit
        predicted = alpha * log_scales + intercept
        ss_res = np.sum((log_fluctuations - predicted)**2)
        ss_tot = np.sum((log_fluctuations - np.mean(log_fluctuations))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Bootstrap confidence interval for alpha
        bootstrap_alphas = []
        for _ in range(min(self.bootstrap_iterations, 300)):
            boot_indices = np.random.choice(
                len(token_sequence), size=len(token_sequence), replace=True
            )
            boot_sequence = token_sequence[boot_indices]
            boot_alpha = self._calculate_alpha_single(
                boot_sequence, min_scale, max_scale_factor, polynomial_order
            )
            bootstrap_alphas.append(boot_alpha)

        ci_lower, ci_upper = np.percentile(bootstrap_alphas, [2.5, 97.5])

        return {
            'alpha': alpha,
            'r_squared': r_squared,
            'confidence_interval': (ci_lower, ci_upper),
            'scales_used': len(scales),
            'target_range_met': (self.thresholds['alpha_min'] <= alpha <=
                               self.thresholds['alpha_max'])
        }

    def _calculate_alpha_single(self, token_sequence, min_scale,
                              max_scale_factor, polynomial_order):
        """Helper method for single alpha calculation."""
        try:
            if len(token_sequence) < 20:
                return 0.5

            mean_centered = token_sequence - np.mean(token_sequence)
            cumulative_sum = np.cumsum(mean_centered)

            max_scale = max(int(len(cumulative_sum) * max_scale_factor), min_scale + 1)
            scales = np.unique(np.logspace(
                np.log10(min_scale), np.log10(max_scale), 10
            ).astype(int))

            fluctuations = []

            for scale in scales:
                segments = []
                for start in range(0, len(cumulative_sum) - scale + 1, scale):
                    segment = cumulative_sum[start:start + scale]
                    if len(segment) == scale:
                        segments.append(segment)

                if len(segments) == 0:
                    continue

                segment_fluctuations = []
                for segment in segments:
                    x = np.arange(len(segment))
                    coefficients = np.polyfit(x, segment, polynomial_order)
                    trend = np.polyval(coefficients, x)
                    detrended = segment - trend
                    rms_fluctuation = np.sqrt(np.mean(detrended**2))
                    segment_fluctuations.append(rms_fluctuation)

                fluctuations.append(np.mean(segment_fluctuations))

            if len(fluctuations) < 3:
                return 0.5

            log_scales = np.log10(scales[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)

            valid_idx = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
            log_scales = log_scales[valid_idx]
            log_fluctuations = log_fluctuations[valid_idx]

            if len(log_scales) < 3:
                return 0.5

            alpha, _ = np.polyfit(log_scales, log_fluctuations, 1)
            return alpha

        except:
            return 0.5

    def entropy_shift_comprehensive(self, pre_embeddings, post_embeddings,
                                  clustering_methods=['kmeans', 'gmm'],
                                  n_clusters_range=[6, 8, 10]):
        """
        Calculate Entropy Shift (ΔH) via Jensen-Shannon divergence on shared clustering.

        CRITICAL FIX (2025-12-08): The original Morgoulis implementation clustered
        pre and post halves INDEPENDENTLY. This means cluster 1 in pre has NO
        correspondence to cluster 1 in post — we're comparing apples to oranges.
        The metric measured diversity difference, not actual reorganization.

        This fix uses SHARED CLUSTERING:
        1. Cluster the ENTIRE trajectory to establish consistent cluster IDs
        2. Compute cluster distribution for pre and post separately
        3. Measure divergence using Jensen-Shannon divergence (symmetric, bounded [0,1])

        This captures TRUE REORGANIZATION — where probability mass actually moved.

        See: context-building/REVIEW_morgoulis-4d-metrics-mathematical-analysis.md

        Args:
            pre_embeddings (array): Initial conversation embeddings
            post_embeddings (array): Final conversation embeddings
            clustering_methods (list): Clustering algorithms for consensus
            n_clusters_range (list): Numbers of clusters to evaluate

        Returns:
            dict: {
                'consensus_delta_h': float,      # Mean JS divergence across methods
                'js_divergence': float,          # Primary metric (JS divergence)
                'confidence_interval': tuple,
                'method_results': dict,
                'threshold_met': bool,
                'stability_score': float,
                'pre_distribution': list,        # Cluster distribution in pre
                'post_distribution': list,       # Cluster distribution in post
                'transition_summary': str        # Human-readable summary
            }
        """
        pre_embeddings = np.array(pre_embeddings)
        post_embeddings = np.array(post_embeddings)

        # Combine for shared clustering
        all_embeddings = np.vstack([pre_embeddings, post_embeddings])
        n_pre = len(pre_embeddings)
        n_post = len(post_embeddings)

        def compute_js_divergence(p, q):
            """Jensen-Shannon divergence (symmetric, bounded [0, 1])."""
            # Ensure same length (pad with zeros if needed)
            max_len = max(len(p), len(q))
            p_padded = np.zeros(max_len)
            q_padded = np.zeros(max_len)
            p_padded[:len(p)] = p
            q_padded[:len(q)] = q

            # Add small epsilon for numerical stability
            p_padded = p_padded + 1e-12
            q_padded = q_padded + 1e-12

            # Renormalize
            p_padded = p_padded / np.sum(p_padded)
            q_padded = q_padded / np.sum(q_padded)

            # Mixture distribution
            m = 0.5 * (p_padded + q_padded)

            # JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
            kl_pm = np.sum(p_padded * np.log2(p_padded / m))
            kl_qm = np.sum(q_padded * np.log2(q_padded / m))

            return 0.5 * kl_pm + 0.5 * kl_qm

        def calculate_distributions_shared(all_emb, n_pre, method, n_clusters):
            """Cluster full trajectory, compute distributions for each half."""
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters,
                                     random_state=self.random_state, n_init=10)
                elif method == 'gmm':
                    clusterer = GaussianMixture(n_components=n_clusters,
                                              random_state=self.random_state)
                else:
                    return None

                # Fit on FULL trajectory
                all_labels = clusterer.fit_predict(all_emb)

                # Split labels back to pre/post
                pre_labels = all_labels[:n_pre]
                post_labels = all_labels[n_pre:]

                # Compute distributions over shared cluster space
                all_clusters = np.arange(n_clusters)

                pre_dist = np.array([
                    np.sum(pre_labels == c) / n_pre for c in all_clusters
                ])
                post_dist = np.array([
                    np.sum(post_labels == c) / (len(all_emb) - n_pre) for c in all_clusters
                ])

                # Quality metric
                silhouette = silhouette_score(all_emb, all_labels) if len(np.unique(all_labels)) > 1 else -1

                return {
                    'pre_dist': pre_dist,
                    'post_dist': post_dist,
                    'silhouette': silhouette,
                    'n_clusters_used': len(np.unique(all_labels))
                }

            except Exception:
                return None

        # Comprehensive analysis across methods and cluster numbers
        results = {}
        js_values = []
        best_result = None
        best_silhouette = -1

        for method in clustering_methods:
            method_results = {}

            for n_clusters in n_clusters_range:
                dist_result = calculate_distributions_shared(
                    all_embeddings, n_pre, method, n_clusters
                )

                if dist_result is None:
                    continue

                # Jensen-Shannon divergence
                js_div = compute_js_divergence(dist_result['pre_dist'], dist_result['post_dist'])

                method_results[f'n_clusters_{n_clusters}'] = {
                    'js_divergence': js_div,
                    'pre_distribution': dist_result['pre_dist'].tolist(),
                    'post_distribution': dist_result['post_dist'].tolist(),
                    'silhouette': dist_result['silhouette'],
                    'n_clusters_used': dist_result['n_clusters_used']
                }

                js_values.append(js_div)

                # Track best clustering for detailed output
                if dist_result['silhouette'] > best_silhouette:
                    best_silhouette = dist_result['silhouette']
                    best_result = {
                        'pre_dist': dist_result['pre_dist'],
                        'post_dist': dist_result['post_dist'],
                        'js_div': js_div
                    }

            results[method] = method_results

        # Consensus metrics
        mean_js = np.mean(js_values) if js_values else 0
        stability_score = 1 - np.std(js_values) / (mean_js + 1e-12) if js_values else 0

        # Bootstrap confidence interval
        bootstrap_js = []
        for _ in range(min(self.bootstrap_iterations, 200)):
            try:
                boot_pre_idx = np.random.choice(n_pre, n_pre, replace=True)
                boot_post_idx = np.random.choice(n_post, n_post, replace=True)
                boot_all = np.vstack([pre_embeddings[boot_pre_idx], post_embeddings[boot_post_idx]])

                dist_result = calculate_distributions_shared(boot_all, n_pre, 'kmeans', 8)
                if dist_result is not None:
                    boot_js = compute_js_divergence(dist_result['pre_dist'], dist_result['post_dist'])
                    bootstrap_js.append(boot_js)
            except Exception:
                continue

        if bootstrap_js:
            ci_lower, ci_upper = np.percentile(bootstrap_js, [2.5, 97.5])
        else:
            ci_lower, ci_upper = mean_js, mean_js

        # Generate human-readable summary
        transition_summary = self._generate_transition_summary(best_result) if best_result else "Insufficient data"

        return {
            'consensus_delta_h': mean_js,  # Keep old name for API compatibility
            'js_divergence': mean_js,      # More accurate name
            'confidence_interval': (ci_lower, ci_upper),
            'method_results': results,
            'threshold_met': mean_js >= self.thresholds['delta_h'],
            'stability_score': max(0, min(1, stability_score)),
            'pre_distribution': best_result['pre_dist'].tolist() if best_result else [],
            'post_distribution': best_result['post_dist'].tolist() if best_result else [],
            'transition_summary': transition_summary
        }

    def _generate_transition_summary(self, result):
        """Generate human-readable summary of semantic reorganization."""
        if result is None:
            return "No transition data"

        pre_dist = result['pre_dist']
        post_dist = result['post_dist']
        js_div = result['js_div']

        # Find biggest changes
        changes = post_dist - pre_dist
        max_gain_idx = np.argmax(changes)
        max_loss_idx = np.argmin(changes)

        if js_div < 0.05:
            return "Minimal reorganization — semantic distribution stable"
        elif js_div < 0.15:
            return f"Moderate reorganization — cluster {max_gain_idx} gained (+{changes[max_gain_idx]:.2f}), cluster {max_loss_idx} lost ({changes[max_loss_idx]:.2f})"
        else:
            return f"Substantial reorganization (JS={js_div:.3f}) — significant probability mass shift from cluster {max_loss_idx} to {max_gain_idx}"

    def calculate_all_metrics(self, dialogue_embeddings, split_ratio=0.5):
        """
        Calculate all three complexity metrics for a dialogue session.

        Args:
            dialogue_embeddings (list): Sequence of sentence embeddings
            split_ratio (float): Split point for pre/post comparison

        Returns:
            dict: Complete results for all metrics
        """
        if len(dialogue_embeddings) < 6:
            return {
                'delta_kappa': 0.0,
                'alpha': 0.5,
                'delta_h': 0.0,
                'error': 'Insufficient data points'
            }

        # Calculate Semantic Curvature
        curvature_result = self.semantic_curvature_enhanced(dialogue_embeddings)

        # Calculate Fractal Similarity using semantic velocity (inter-turn cosine distances)
        #
        # CRITICAL FIX (2025-12-08): The original Morgoulis implementation computed DFA
        # on embedding magnitudes (np.linalg.norm), which is ~constant for L2-normalized
        # embeddings (i.e., the signal was essentially noise). This rendered α meaningless.
        #
        # Semantic velocity (cosine distance between consecutive turns) captures how fast
        # meaning changes turn-to-turn — a meaningful signal for DFA analysis.
        #
        # See: context-building/REVIEW_morgoulis-4d-metrics-mathematical-analysis.md
        embeddings_array = np.array(dialogue_embeddings)
        semantic_velocities = []
        for i in range(len(embeddings_array) - 1):
            # Cosine distance = 1 - cosine_similarity
            cos_sim = cosine_similarity(
                embeddings_array[i].reshape(1, -1),
                embeddings_array[i + 1].reshape(1, -1)
            )[0, 0]
            semantic_velocities.append(1.0 - cos_sim)

        # Handle edge case: need at least 20 points for meaningful DFA
        if len(semantic_velocities) < 20:
            fractal_result = {
                'alpha': 0.5,
                'r_squared': 0.0,
                'confidence_interval': (0.5, 0.5),
                'scales_used': 0,
                'target_range_met': False
            }
        else:
            fractal_result = self.fractal_similarity_robust(np.array(semantic_velocities))

        # Calculate Entropy Shift
        split_point = int(len(dialogue_embeddings) * split_ratio)
        pre_embeddings = np.array(dialogue_embeddings[:split_point])
        post_embeddings = np.array(dialogue_embeddings[split_point:])

        entropy_result = self.entropy_shift_comprehensive(pre_embeddings, post_embeddings)

        return {
            'delta_kappa': curvature_result['curvature'],
            'delta_kappa_ci': curvature_result['confidence_interval'],
            'delta_kappa_significant': curvature_result['threshold_met'],

            'alpha': fractal_result['alpha'],
            'alpha_ci': fractal_result['confidence_interval'],
            'alpha_in_range': fractal_result['target_range_met'],

            'delta_h': entropy_result['consensus_delta_h'],
            'delta_h_ci': entropy_result['confidence_interval'],
            'delta_h_significant': entropy_result['threshold_met'],

            'summary': {
                'cognitive_complexity_detected': all([
                    curvature_result['threshold_met'],
                    fractal_result['target_range_met'],
                    entropy_result['threshold_met']
                ])
            }
        }
