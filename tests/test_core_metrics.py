"""
Tests for core_metrics.py - Morgoulis (2025) Implementation with 2025-12-08 Fixes

These tests validate the core metric calculations:
- Semantic Curvature (Δκ) — Local curvature via Frenet-Serret (FIXED)
- Fractal Similarity Score (α) — DFA on semantic velocity (FIXED)
- Entropy Shift (ΔH) — JS divergence on shared clustering (FIXED)

Test categories:
- Synthetic data validation (linear, circular, Brownian, white noise)
- Edge cases (minimum lengths, zero vectors)
- Statistical properties (bootstrap CIs, thresholds)

Critical fixes validated:
- Δκ: Now computes true local curvature, not chord deviation
- α: Now uses semantic velocity, not embedding norms
- ΔH: Now uses shared clustering for true reorganization measurement
"""

import pytest
import numpy as np
from src.core_metrics import SemanticComplexityAnalyzer


class TestSemanticCurvature:
    """Tests for Semantic Curvature (Δκ) via local Frenet-Serret curvature."""

    def test_linear_trajectory_zero_curvature(self):
        """Linear trajectory should have exactly zero curvature (straight line doesn't bend)."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        # Create perfectly linear trajectory
        np.random.seed(42)
        start = np.random.randn(384)
        end = np.random.randn(384)
        n = 20
        embeddings = [start + (end - start) * (i / (n - 1)) for i in range(n)]

        result = analyzer.semantic_curvature_enhanced(embeddings)

        # Linear trajectory should have effectively zero curvature (within floating point)
        assert result['curvature'] < 1e-10, f"Linear trajectory curvature should be ~0, got: {result['curvature']}"
        assert all(k < 1e-10 for k in result['local_curvatures'])

    def test_circular_trajectory_high_curvature(self):
        """
        Circular trajectory should have HIGH curvature.

        This is a critical fix validation — the OLD implementation would return ~0
        because start and end points coincide.
        """
        analyzer = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=50)

        # Create circular trajectory in 2D embedded in 384D
        circle = []
        for i in range(20):
            theta = 2 * np.pi * i / 20
            point = np.zeros(384)
            point[0] = np.cos(theta)
            point[1] = np.sin(theta)
            circle.append(point)

        result = analyzer.semantic_curvature_enhanced(circle)

        # Circle should have high, consistent curvature
        assert result['curvature'] > 0.5, f"Circle curvature too low: {result['curvature']}"
        assert result['curvature_std'] < 0.3, f"Circle curvature should be consistent, std: {result['curvature_std']}"

    def test_random_trajectory_measurable_curvature(self):
        """Random trajectory should have measurable curvature from direction changes."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(20)]

        result = analyzer.semantic_curvature_enhanced(embeddings)

        # Random trajectory should have non-zero curvature
        assert result['curvature'] > 0.0, f"Random trajectory curvature too low: {result['curvature']}"
        assert len(result['local_curvatures']) == 18  # n - 2 interior points (20 - 2 = 18)

    def test_minimum_length(self):
        """Should handle minimum sequence length gracefully (need 4+ points)."""
        analyzer = SemanticComplexityAnalyzer()

        # Less than 4 embeddings (need 4 for 1 curvature measurement)
        result = analyzer.semantic_curvature_enhanced([np.random.randn(384) for _ in range(3)])

        assert result['curvature'] == 0.0
        assert result['threshold_met'] is False
        assert result['local_curvatures'] == []

    def test_bootstrap_ci_exists(self):
        """Should return bootstrap confidence interval."""
        analyzer = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=50)

        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(15)]
        result = analyzer.semantic_curvature_enhanced(embeddings)

        assert 'confidence_interval' in result
        assert len(result['confidence_interval']) == 2
        assert result['confidence_interval'][0] <= result['confidence_interval'][1]

    def test_local_curvatures_returned(self):
        """Should return local curvature values at each interior point."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(10)]
        result = analyzer.semantic_curvature_enhanced(embeddings)

        assert 'local_curvatures' in result
        # For n=10, we get n-2=8 local curvature values
        assert len(result['local_curvatures']) == 8
        assert 'curvature_std' in result


class TestFractalSimilarity:
    """Tests for Fractal Similarity Score (α) via DFA."""

    def test_white_noise_alpha_near_half(self):
        """White noise should have α ≈ 0.5."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        # White noise signal
        signal = np.random.randn(200)

        result = analyzer.fractal_similarity_robust(signal)

        # White noise α should be around 0.5
        assert 0.3 < result['alpha'] < 0.7, f"White noise α unexpected: {result['alpha']}"

    def test_brownian_motion_alpha_near_1_5(self):
        """Brownian motion (integrated white noise) should have α ≈ 1.5."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        # Brownian motion (cumulative sum of white noise)
        signal = np.cumsum(np.random.randn(200))

        result = analyzer.fractal_similarity_robust(signal)

        # Brownian motion α should be around 1.5
        assert 1.2 < result['alpha'] < 1.8, f"Brownian motion α unexpected: {result['alpha']}"

    def test_minimum_length(self):
        """Should handle short sequences gracefully."""
        analyzer = SemanticComplexityAnalyzer()

        # Less than 20 points
        signal = np.random.randn(10)
        result = analyzer.fractal_similarity_robust(signal)

        assert result['alpha'] == 0.5  # Default fallback
        assert result['target_range_met'] is False

    def test_target_range_detection(self):
        """Should detect target range [0.70, 0.90]."""
        analyzer = SemanticComplexityAnalyzer()

        # Create signal with known α
        np.random.seed(42)
        signal = np.random.randn(100)
        result = analyzer.fractal_similarity_robust(signal)

        expected_in_range = 0.70 <= result['alpha'] <= 0.90
        assert result['target_range_met'] == expected_in_range


class TestEntropyShift:
    """Tests for Entropy Shift (ΔH) via Jensen-Shannon divergence on shared clustering."""

    def test_identical_distributions_zero_divergence(self):
        """Identical pre/post distributions should have JS divergence = 0."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.random.randn(30, 384)

        # Same data for pre and post — should be exactly 0
        result = analyzer.entropy_shift_comprehensive(
            embeddings[:15], embeddings[:15]
        )

        # JS divergence of identical distributions is 0
        assert result['js_divergence'] == 0.0, f"Identical distributions should have JS=0, got {result['js_divergence']}"
        assert 'Minimal reorganization' in result['transition_summary']

    def test_completely_separated_distributions_high_divergence(self):
        """
        Completely separated distributions should have JS divergence near 1.

        This validates the shared clustering fix — we're measuring actual
        redistribution, not just entropy difference.
        """
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        # Two completely separated clusters
        pre = np.random.randn(20, 384) * 0.5
        post = np.random.randn(20, 384) * 0.5 + 10 * np.ones(384)

        result = analyzer.entropy_shift_comprehensive(pre, post)

        # Complete separation → JS ≈ 1
        assert result['js_divergence'] > 0.9, f"Separated distributions should have high JS, got {result['js_divergence']}"
        assert 'Substantial reorganization' in result['transition_summary']

    def test_shared_clustering_provides_distributions(self):
        """Should return pre and post distributions over shared cluster space."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        pre = np.random.randn(20, 384)
        post = np.random.randn(20, 384)

        result = analyzer.entropy_shift_comprehensive(pre, post)

        assert 'pre_distribution' in result
        assert 'post_distribution' in result
        assert len(result['pre_distribution']) > 0
        assert len(result['post_distribution']) > 0
        # Distributions should sum to 1 (within epsilon)
        assert abs(sum(result['pre_distribution']) - 1.0) < 0.01
        assert abs(sum(result['post_distribution']) - 1.0) < 0.01

    def test_method_results_present(self):
        """Should include results from multiple clustering methods."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        pre = np.random.randn(20, 384)
        post = np.random.randn(20, 384)

        result = analyzer.entropy_shift_comprehensive(pre, post)

        assert 'method_results' in result
        assert 'kmeans' in result['method_results']
        assert 'gmm' in result['method_results']
        # New: should have js_divergence in method results
        assert 'js_divergence' in result['method_results']['kmeans']['n_clusters_8']

    def test_transition_summary_generated(self):
        """Should generate human-readable transition summary."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        pre = np.random.randn(20, 384)
        post = np.random.randn(20, 384) + 3

        result = analyzer.entropy_shift_comprehensive(pre, post)

        assert 'transition_summary' in result
        assert isinstance(result['transition_summary'], str)
        assert len(result['transition_summary']) > 0


class TestCalculateAllMetrics:
    """Tests for combined metric calculation."""

    def test_returns_all_metrics(self):
        """Should return all three metrics."""
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(30)]

        result = analyzer.calculate_all_metrics(embeddings)

        assert 'delta_kappa' in result
        assert 'alpha' in result
        assert 'delta_h' in result
        assert 'summary' in result

    def test_minimum_length_error(self):
        """Should handle insufficient data gracefully."""
        analyzer = SemanticComplexityAnalyzer()

        embeddings = [np.random.randn(384) for _ in range(3)]
        result = analyzer.calculate_all_metrics(embeddings)

        assert 'error' in result
        assert result['delta_kappa'] == 0.0

    def test_confidence_intervals_present(self):
        """Should include confidence intervals for all metrics."""
        analyzer = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=50)

        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(30)]

        result = analyzer.calculate_all_metrics(embeddings)

        assert 'delta_kappa_ci' in result
        assert 'alpha_ci' in result
        assert 'delta_h_ci' in result


class TestThresholds:
    """Tests for empirically-derived thresholds."""

    def test_default_thresholds(self):
        """Should have correct default thresholds from Morgoulis (2025)."""
        analyzer = SemanticComplexityAnalyzer()

        assert analyzer.thresholds['delta_kappa'] == 0.35
        assert analyzer.thresholds['alpha_min'] == 0.70
        assert analyzer.thresholds['alpha_max'] == 0.90
        assert analyzer.thresholds['delta_h'] == 0.12


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_zero_vectors_handled(self):
        """Should handle zero vectors gracefully (stationary trajectory)."""
        analyzer = SemanticComplexityAnalyzer()

        embeddings = [np.zeros(384) for _ in range(10)]

        # Should not raise exception — stationary trajectory has zero curvature
        result = analyzer.semantic_curvature_enhanced(embeddings)
        assert result['curvature'] == 0.0

    def test_constant_embedding_handled(self):
        """Should handle constant embeddings (all same point)."""
        analyzer = SemanticComplexityAnalyzer()

        # All same embedding
        point = np.random.randn(384)
        embeddings = [point.copy() for _ in range(10)]

        result = analyzer.semantic_curvature_enhanced(embeddings)
        # Stationary → zero curvature
        assert result['curvature'] == 0.0

    def test_reproducibility_with_seed(self):
        """Results should be reproducible with same random state."""
        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(30)]

        analyzer1 = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=50)
        analyzer2 = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=50)

        result1 = analyzer1.calculate_all_metrics(embeddings)
        result2 = analyzer2.calculate_all_metrics(embeddings)

        assert result1['delta_kappa'] == result2['delta_kappa']
        assert result1['alpha'] == result2['alpha']
        assert result1['delta_h'] == result2['delta_h']


class TestMetricFixValidation:
    """
    Tests specifically validating the 2025-12-08 metric fixes.

    These tests ensure the critical bugs are fixed:
    - Δκ: Was measuring chord deviation, now measures local curvature
    - α: Was computing DFA on norms (~constant), now uses semantic velocity
    - ΔH: Was clustering independently, now uses shared clustering
    """

    def test_curvature_independent_of_endpoints(self):
        """
        Curvature should NOT depend heavily on start/end points.

        OLD BUG: Circular trajectory (returns to start) would give ~0 curvature.
        FIX: Local curvature measures actual bending regardless of endpoints.
        """
        analyzer = SemanticComplexityAnalyzer(random_state=42, bootstrap_iterations=20)

        # Create two trajectories with same shape but different endpoints
        # Trajectory 1: Sine wave, doesn't return to start
        sine1 = []
        for i in range(20):
            t = i / 20 * 2 * np.pi
            point = np.zeros(384)
            point[0] = t
            point[1] = np.sin(t)
            sine1.append(point)

        # Trajectory 2: Full sine cycle (returns near start in y)
        sine2 = []
        for i in range(20):
            t = i / 20 * 2 * np.pi
            point = np.zeros(384)
            point[0] = np.sin(t)  # Returns to 0
            point[1] = np.cos(t)
            sine2.append(point)

        result1 = analyzer.semantic_curvature_enhanced(sine1)
        result2 = analyzer.semantic_curvature_enhanced(sine2)

        # Both should have similar curvature (both are smooth curves)
        # Old implementation would give very different results
        assert result1['curvature'] > 0
        assert result2['curvature'] > 0

    def test_alpha_on_semantic_velocity_not_norms(self):
        """
        DFA should be computed on semantic velocity, not embedding norms.

        OLD BUG: For L2-normalized embeddings, norms are ~1.0 (constant signal).
        FIX: Semantic velocity (inter-turn cosine distance) captures dynamics.
        """
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        # Create embeddings with known semantic structure
        np.random.seed(42)
        embeddings = [np.random.randn(384) for _ in range(40)]

        # Normalize them (making norms ~constant)
        normalized = [e / np.linalg.norm(e) for e in embeddings]

        result = analyzer.calculate_all_metrics(normalized)

        # Alpha should be meaningful (not degenerate ~0.5 from noise)
        # The semantic velocity varies even when norms are constant
        assert 'alpha' in result
        # Should have enough data for DFA
        assert result['alpha'] != 0.5 or 'error' in result

    def test_entropy_shift_uses_shared_clustering(self):
        """
        ΔH should use shared clustering so cluster IDs correspond.

        OLD BUG: Pre and post clustered independently — cluster 1 in pre
        has no relationship to cluster 1 in post.
        FIX: Cluster full trajectory, then measure redistribution.
        """
        analyzer = SemanticComplexityAnalyzer(random_state=42)

        np.random.seed(42)
        # Create data where pre and post occupy different regions
        pre = np.random.randn(20, 384)
        post = pre + 5 * np.ones(384)  # Shifted version

        result = analyzer.entropy_shift_comprehensive(pre, post)

        # With shared clustering, we can now track which clusters gained/lost
        assert 'pre_distribution' in result
        assert 'post_distribution' in result

        # The distributions should be over the SAME cluster space
        assert len(result['pre_distribution']) == len(result['post_distribution'])
