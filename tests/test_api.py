"""
Tests for api.py - Function-based API

These tests validate the lightweight function interface:
- Core metric functions
- Statistical utilities
- Helper functions
"""

import pytest
import numpy as np
from src.api import (
    semantic_curvature,
    semantic_curvature_ci,
    dfa_alpha,
    entropy_shift,
    icc_oneway_random,
    icc_bootstrap_ci,
    bland_altman,
    all_pairs_bland_altman,
    cosine_sim,
    bootstrap_ci,
)


class TestSemanticCurvatureFunction:
    """Tests for semantic_curvature() function."""

    def test_basic_calculation(self):
        """Should calculate curvature for valid embeddings."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 384)

        result = semantic_curvature(embeddings)

        assert isinstance(result, float)
        assert 0 <= result <= 2  # Reasonable range

    def test_linear_trajectory(self):
        """Linear trajectory should have low curvature."""
        start = np.random.randn(384)
        end = np.random.randn(384)
        n = 20
        embeddings = np.array([start + (end - start) * (i / (n - 1)) for i in range(n)])

        result = semantic_curvature(embeddings)

        assert result < 0.1

    def test_minimum_length(self):
        """Should return 0 for sequences < 3."""
        embeddings = np.random.randn(2, 384)

        result = semantic_curvature(embeddings)

        assert result == 0.0

    def test_with_ci(self):
        """Should return point estimate and CI."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 384)

        mean, ci = semantic_curvature_ci(embeddings, n_boot=50)

        assert isinstance(mean, float)
        assert len(ci) == 2
        assert ci[0] <= ci[1]


class TestDfaAlphaFunction:
    """Tests for dfa_alpha() function."""

    def test_white_noise(self):
        """White noise should have α ≈ 0.5."""
        np.random.seed(42)
        signal = np.random.randn(200)

        result = dfa_alpha(signal)

        assert 0.3 < result < 0.7

    def test_brownian_motion(self):
        """Brownian motion should have α ≈ 1.5."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(200))

        result = dfa_alpha(signal)

        assert 1.2 < result < 1.8

    def test_short_signal(self):
        """Should handle short signals gracefully."""
        signal = np.random.randn(10)

        result = dfa_alpha(signal)

        # Should return default or calculated value without error
        assert isinstance(result, float)


class TestEntropyShiftFunction:
    """Tests for entropy_shift() function."""

    def test_basic_calculation(self):
        """Should calculate entropy shift."""
        np.random.seed(42)
        pre = np.random.randn(20, 384)
        post = np.random.randn(20, 384)

        result = entropy_shift(pre, post)

        assert isinstance(result, float)
        assert result >= 0

    def test_identical_distributions(self):
        """Identical distributions should have low ΔH."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 384)

        result = entropy_shift(embeddings, embeddings)

        assert result < 0.1

    def test_small_sample_handling(self):
        """Should handle small samples by reducing clusters."""
        np.random.seed(42)
        pre = np.random.randn(5, 384)
        post = np.random.randn(5, 384)

        result = entropy_shift(pre, post, n_clusters=8)

        # Should not raise, should return valid result
        assert isinstance(result, float)


class TestIccFunctions:
    """Tests for ICC functions."""

    def test_perfect_agreement(self):
        """Perfect agreement should have ICC ≈ 1."""
        # All raters give same scores
        data = np.tile(np.array([[1], [2], [3], [4], [5]]), (1, 3))

        result = icc_oneway_random(data)

        assert result > 0.99

    def test_no_agreement(self):
        """Random data should have low ICC."""
        np.random.seed(42)
        data = np.random.randn(10, 3)

        result = icc_oneway_random(data)

        # Random data ICC should be near 0 (can be negative)
        assert -0.5 < result < 0.5

    def test_bootstrap_ci(self):
        """Should return point estimate and CI."""
        np.random.seed(42)
        data = np.random.randn(10, 3)

        point, ci = icc_bootstrap_ci(data, n_boot=50)

        assert isinstance(point, float)
        assert len(ci) == 2


class TestBlandAltmanFunctions:
    """Tests for Bland-Altman functions."""

    def test_basic_calculation(self):
        """Should calculate Bland-Altman statistics."""
        np.random.seed(42)
        a = np.random.randn(20)
        b = a + np.random.randn(20) * 0.1  # b similar to a

        result = bland_altman(a, b)

        assert 'mean_diff' in result
        assert 'loa_low' in result
        assert 'loa_high' in result
        assert result['loa_low'] < result['loa_high']

    def test_identical_arrays(self):
        """Identical arrays should have zero mean difference."""
        a = np.array([1, 2, 3, 4, 5])

        result = bland_altman(a, a)

        assert result['mean_diff'] == 0.0

    def test_all_pairs(self):
        """Should compute BA for all pairs."""
        np.random.seed(42)
        matrix = np.random.randn(10, 4)
        labels = ['A', 'B', 'C', 'D']

        result = all_pairs_bland_altman(matrix, labels)

        # Should have 6 pairs (4 choose 2)
        assert len(result) == 6
        assert ('A', 'B') in result


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_cosine_sim_identical(self):
        """Identical vectors should have similarity 1."""
        v = np.array([1, 2, 3])

        result = cosine_sim(v, v)

        assert abs(result - 1.0) < 0.001

    def test_cosine_sim_orthogonal(self):
        """Orthogonal vectors should have similarity 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])

        result = cosine_sim(v1, v2)

        assert abs(result) < 0.001

    def test_cosine_sim_zero_vector(self):
        """Zero vector should return 0."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])

        result = cosine_sim(v1, v2)

        assert result == 0.0

    def test_bootstrap_ci_basic(self):
        """Should return valid CI."""
        np.random.seed(42)
        values = np.random.randn(50)

        ci = bootstrap_ci(values, n_boot=100)

        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_bootstrap_ci_coverage(self):
        """CI should contain population mean for normal data."""
        np.random.seed(42)
        # Sample from N(0, 1)
        values = np.random.randn(100)

        ci = bootstrap_ci(values, n_boot=500, ci=0.95)

        # 0 should be in or near the CI
        assert ci[0] < 0.5 and ci[1] > -0.5


class TestConsistencyWithClass:
    """Tests that function API matches class API."""

    def test_curvature_consistency(self):
        """Function and class should give same curvature."""
        from src.core_metrics import SemanticComplexityAnalyzer

        np.random.seed(42)
        embeddings = np.random.randn(20, 384)

        # Function API
        func_result = semantic_curvature(embeddings)

        # Class API
        analyzer = SemanticComplexityAnalyzer(random_state=42)
        class_result = analyzer.semantic_curvature_enhanced(list(embeddings))

        # Should be very close (algorithm is same, minor float differences possible)
        assert abs(func_result - class_result['curvature']) < 0.01
