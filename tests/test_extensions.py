"""
Tests for extensions.py - Vector Ψ and Attractor Extensions

These tests validate the extensions to Morgoulis (2025):
- TrajectoryBuffer (temporal dynamics)
- SemanticClimateAnalyzer (Vector Ψ, substrates, attractors)
"""

import pytest
import numpy as np
from src.extensions import TrajectoryBuffer, SemanticClimateAnalyzer


class TestTrajectoryBuffer:
    """Tests for TrajectoryBuffer class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        buffer = TrajectoryBuffer(window_size=20, timestep=0.5)

        assert buffer.window_size == 20
        assert buffer.timestep == 0.5
        assert len(buffer.history) == 0

    def test_append_basic(self):
        """Should append Ψ vectors to history."""
        buffer = TrajectoryBuffer()

        psi = {'psi_semantic': 0.5, 'psi_temporal': 0.3, 'psi_affective': 0.1, 'psi_biosignal': None}
        buffer.append(psi)

        assert len(buffer.history) == 1
        assert buffer.history[0][1] == psi

    def test_append_with_timestamp(self):
        """Should use provided timestamp."""
        buffer = TrajectoryBuffer()

        psi = {'psi_semantic': 0.5, 'psi_temporal': 0.3, 'psi_affective': 0.1, 'psi_biosignal': None}
        buffer.append(psi, timestamp=10.5)

        assert buffer.history[0][0] == 10.5

    def test_window_size_maintained(self):
        """Should maintain window size limit."""
        buffer = TrajectoryBuffer(window_size=5)

        for i in range(10):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        assert len(buffer.history) == 5

    def test_velocity_calculation(self):
        """Should compute velocity via central difference."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Linear trajectory: psi_semantic increases by 0.1 per step
        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        velocity = buffer.compute_velocity()

        assert velocity is not None
        # dpsi_semantic/dt should be ~0.1 (central diff over 2 steps = 0.2/2)
        assert abs(velocity['dpsi_semantic_dt'] - 0.1) < 0.01

    def test_velocity_insufficient_history(self):
        """Should return None with insufficient history."""
        buffer = TrajectoryBuffer()

        buffer.append({'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 0.2, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        velocity = buffer.compute_velocity()
        assert velocity is None

    def test_acceleration_calculation(self):
        """Should compute acceleration via second-order difference."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Quadratic trajectory: psi_semantic = 0.1 * t^2
        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * (i ** 2),
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        acceleration = buffer.compute_acceleration()

        assert acceleration is not None
        # d2psi/dt2 for quadratic should be constant (0.2)
        assert acceleration['d2psi_semantic_dt2'] is not None

    def test_curvature_calculation(self):
        """Should compute trajectory curvature."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Curved trajectory
        for i in range(5):
            buffer.append({
                'psi_semantic': np.sin(i * 0.5),
                'psi_temporal': np.cos(i * 0.5),
                'psi_affective': 0.1 * i,
                'psi_biosignal': None
            })

        curvature = buffer.compute_curvature()

        # Should return a numeric value
        assert curvature is None or isinstance(curvature, float)

    def test_get_trajectory_segment(self):
        """Should retrieve trajectory segment."""
        buffer = TrajectoryBuffer()

        for i in range(5):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        # Get last 3 points
        segment = buffer.get_trajectory_segment(n_points=3)

        assert len(segment) == 3
        assert segment[-1][1]['psi_semantic'] == 0.4


class TestSemanticClimateAnalyzer:
    """Tests for SemanticClimateAnalyzer class."""

    def test_inherits_from_core(self):
        """Should inherit from SemanticComplexityAnalyzer."""
        from src.core_metrics import SemanticComplexityAnalyzer
        analyzer = SemanticClimateAnalyzer()

        assert isinstance(analyzer, SemanticComplexityAnalyzer)
        assert hasattr(analyzer, 'calculate_all_metrics')
        assert hasattr(analyzer, 'thresholds')

    def test_compute_coupling_coefficient_basic(self):
        """Should compute coupling coefficient with embeddings only."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.array([np.random.randn(384) for _ in range(20)])

        result = analyzer.compute_coupling_coefficient(dialogue_embeddings=embeddings)

        assert 'psi' in result
        assert 'psi_state' in result
        assert 'raw_metrics' in result
        assert 'attractor_dynamics' in result

    def test_psi_state_structure(self):
        """Should return correct Ψ state structure."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.array([np.random.randn(384) for _ in range(20)])

        result = analyzer.compute_coupling_coefficient(dialogue_embeddings=embeddings)

        assert 'position' in result['psi_state']
        assert 'psi_semantic' in result['psi_state']['position']
        assert 'psi_temporal' in result['psi_state']['position']
        assert 'psi_affective' in result['psi_state']['position']
        assert 'psi_biosignal' in result['psi_state']['position']

    def test_attractor_basin_detection(self):
        """Should detect attractor basin."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.array([np.random.randn(384) for _ in range(20)])

        result = analyzer.compute_coupling_coefficient(dialogue_embeddings=embeddings)

        assert result['attractor_dynamics']['basin'] is not None
        assert result['attractor_dynamics']['confidence'] >= 0

    def test_with_turn_texts(self):
        """Should compute affective substrate when turn texts provided."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.array([np.random.randn(384) for _ in range(10)])
        texts = [
            "I think this is interesting.",
            "I'm not sure about that.",
            "Maybe we could explore this further?",
            "I feel uncertain about the approach.",
            "This seems promising to me.",
        ] * 2

        result = analyzer.compute_coupling_coefficient(
            dialogue_embeddings=embeddings,
            turn_texts=texts
        )

        # Affective substrate should be computed
        assert result['substrate_details']['affective']['psi_affective'] != 0.0

    def test_raw_metrics_preserved(self):
        """Should preserve raw metrics from core analyzer."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        np.random.seed(42)
        embeddings = np.array([np.random.randn(384) for _ in range(20)])

        result = analyzer.compute_coupling_coefficient(dialogue_embeddings=embeddings)

        assert 'delta_kappa' in result['raw_metrics']
        assert 'delta_h' in result['raw_metrics']
        assert 'alpha' in result['raw_metrics']


class TestSubstrateComputation:
    """Tests for individual substrate computations."""

    def test_semantic_substrate(self):
        """Should compute semantic substrate from metrics."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        metrics = {'delta_kappa': 0.5, 'delta_h': 0.15, 'alpha': 0.8}
        result = analyzer.compute_semantic_substrate(embeddings=np.array([]), metrics=metrics)

        assert 'psi_semantic' in result
        assert -1 <= result['psi_semantic'] <= 1
        assert result['exploratory_depth'] == 0.5

    def test_temporal_substrate_neutral_default(self):
        """Should return neutral value when insufficient data."""
        analyzer = SemanticClimateAnalyzer(random_state=42)

        embeddings = np.random.randn(5, 384)  # Too short
        result = analyzer.compute_temporal_substrate(embeddings=embeddings)

        assert result['psi_temporal'] == 0.5  # Neutral

    def test_affective_substrate_hedging(self):
        """Should detect hedging patterns in affective substrate."""
        analyzer = SemanticClimateAnalyzer()

        texts = [
            "I think maybe this could be right.",
            "Perhaps we should consider other options.",
            "I'm not sure, but it seems like...",
        ]

        result = analyzer.compute_affective_substrate(turn_texts=texts)

        assert result['hedging_density'] > 0


class TestAttractorBasins:
    """Tests for attractor basin detection."""

    def test_all_basin_types_exist(self):
        """Should recognize all 7 canonical basin types."""
        analyzer = SemanticClimateAnalyzer()

        basin_types = [
            "Cognitive Mimicry",
            "Embodied Coherence",
            "Generative Conflict",
            "Sycophantic Convergence",
            "Creative Dilation",
            "Deep Resonance",
            "Dissociation",
            "Transitional"
        ]

        # Test specific configurations
        test_cases = [
            ({'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5},
             {'delta_kappa': 0.5}),  # Deep Resonance candidate
            ({'psi_semantic': 0.0, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': 0.0},
             {'delta_kappa': 0.2}),  # Dissociation candidate
        ]

        for psi_vector, raw_metrics in test_cases:
            basin, confidence = analyzer.detect_attractor_basin(psi_vector, raw_metrics)
            assert basin in basin_types
            assert 0 <= confidence <= 1

    def test_sycophantic_detection(self):
        """Should detect sycophantic convergence pattern."""
        analyzer = SemanticClimateAnalyzer()

        # High semantic alignment + low Δκ + low affect
        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.1,
            'psi_biosignal': 0.0
        }
        raw_metrics = {'delta_kappa': 0.2}  # Below threshold

        basin, confidence = analyzer.detect_attractor_basin(psi_vector, raw_metrics)

        assert basin == "Sycophantic Convergence"
