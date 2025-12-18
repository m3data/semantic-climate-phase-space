"""
Tests for the SemanticClimateAnalyzer orchestrator.

Tests cover:
- Basic initialization and configuration
- Compute coupling coefficient functionality
- Backward compatibility with extensions.py output structure
- track_history flag behavior
- compute_integrity flag behavior
- reset_history method
- trajectory_integrity output
- Enhanced attractor_dynamics with residence_time, previous_basin
- Multiple calls accumulating history
"""

import pytest
import numpy as np

from src.analyzer import SemanticClimateAnalyzer
from src.extensions import SemanticClimateAnalyzer as LegacyAnalyzer


class TestAnalyzerInitialization:
    """Tests for analyzer initialization."""

    def test_default_initialization(self):
        """Analyzer initializes with default settings."""
        analyzer = SemanticClimateAnalyzer()
        assert analyzer.track_history is True
        assert analyzer.compute_integrity is True
        assert analyzer.trajectory is not None
        assert analyzer.basin_history is not None
        assert analyzer.basin_detector is not None
        assert analyzer.integrity_analyzer is not None

    def test_disable_history(self):
        """Analyzer can disable history tracking."""
        analyzer = SemanticClimateAnalyzer(track_history=False)
        assert analyzer.track_history is False
        assert analyzer.trajectory is None
        assert analyzer.basin_history is None

    def test_disable_integrity(self):
        """Analyzer can disable integrity computation."""
        analyzer = SemanticClimateAnalyzer(compute_integrity=False)
        assert analyzer.compute_integrity is False
        assert analyzer.integrity_analyzer is None

    def test_inherits_from_core(self):
        """Analyzer inherits from SemanticComplexityAnalyzer."""
        from src.core_metrics import SemanticComplexityAnalyzer
        analyzer = SemanticClimateAnalyzer()
        assert isinstance(analyzer, SemanticComplexityAnalyzer)

    def test_random_state_inherited(self):
        """Random state is passed to parent class."""
        analyzer = SemanticClimateAnalyzer(random_state=123)
        assert analyzer.random_state == 123


class TestComputeCouplingCoefficient:
    """Tests for compute_coupling_coefficient method."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    @pytest.fixture
    def sample_texts(self):
        """Generate sample turn texts."""
        return [
            "I think this is interesting",
            "Yes, I agree with that perspective",
            "Perhaps we could explore this further",
            "I'm not sure about that approach",
        ] * 5  # 20 turns

    @pytest.fixture
    def sample_speakers(self):
        """Generate sample speaker labels."""
        return ['human', 'ai'] * 10  # 20 turns

    def test_basic_call(self, sample_embeddings):
        """Basic call returns expected structure."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        assert 'psi' in result
        assert 'psi_state' in result
        assert 'trajectory_dynamics' in result
        assert 'attractor_dynamics' in result
        assert 'flow_field' in result
        assert 'raw_metrics' in result
        assert 'substrate_details' in result
        assert 'dialogue_context' in result

    def test_psi_is_float(self, sample_embeddings):
        """psi value is a float."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)
        assert isinstance(result['psi'], float)

    def test_psi_state_structure(self, sample_embeddings):
        """psi_state has correct structure."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        assert 'position' in result['psi_state']
        assert 'velocity' in result['psi_state']
        assert 'acceleration' in result['psi_state']

        position = result['psi_state']['position']
        assert 'psi_semantic' in position
        assert 'psi_temporal' in position
        assert 'psi_affective' in position
        assert 'psi_biosignal' in position

    def test_raw_metrics_structure(self, sample_embeddings):
        """raw_metrics has correct structure."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        assert 'delta_kappa' in result['raw_metrics']
        assert 'delta_h' in result['raw_metrics']
        assert 'alpha' in result['raw_metrics']

    def test_attractor_dynamics_structure(self, sample_embeddings):
        """attractor_dynamics has correct structure."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        ad = result['attractor_dynamics']
        assert 'basin' in ad
        assert 'confidence' in ad
        assert 'pull_strength' in ad
        assert 'basin_depth' in ad
        assert 'escape_velocity' in ad
        assert 'basin_stability' in ad
        # New fields
        assert 'residence_time' in ad
        assert 'previous_basin' in ad
        assert 'raw_confidence' in ad

    def test_with_turn_texts(self, sample_embeddings, sample_texts):
        """Call with turn texts computes affective substrate."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(
            sample_embeddings,
            turn_texts=sample_texts
        )

        affective = result['substrate_details']['affective']
        assert affective['psi_affective'] != 0.0  # Should have some value
        assert 'hedging_density' in affective
        assert affective['hedging_density'] > 0.0  # "I think", "perhaps" in texts

    def test_with_biosignal_data(self, sample_embeddings):
        """Call with biosignal data computes biosignal substrate."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(
            sample_embeddings,
            biosignal_data={'heart_rate': 75}
        )

        biosignal = result['substrate_details']['biosignal']
        assert biosignal['psi_biosignal'] is not None

    def test_requires_embeddings_or_metrics(self):
        """Raises error if neither embeddings nor metrics provided."""
        analyzer = SemanticClimateAnalyzer()
        with pytest.raises(ValueError):
            analyzer.compute_coupling_coefficient()

    def test_accepts_precomputed_metrics(self, sample_embeddings):
        """Can accept pre-computed metrics."""
        analyzer = SemanticClimateAnalyzer()
        # First compute metrics
        metrics = analyzer.calculate_all_metrics(sample_embeddings)

        # Then use them
        result = analyzer.compute_coupling_coefficient(metrics=metrics)
        assert result['raw_metrics']['delta_kappa'] == metrics['delta_kappa']


class TestTrajectoryIntegrity:
    """Tests for trajectory integrity computation."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    def test_trajectory_integrity_in_output(self, sample_embeddings):
        """trajectory_integrity field is present in output."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)
        assert 'trajectory_integrity' in result

    def test_integrity_none_on_first_call(self, sample_embeddings):
        """trajectory_integrity is not sufficient on first call (insufficient data)."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        # First call - only 1 point in trajectory, needs at least 5
        if result['trajectory_integrity'] is not None:
            assert result['trajectory_integrity']['sufficient_data'] is False

    def test_integrity_available_after_multiple_calls(self, sample_embeddings):
        """trajectory_integrity becomes available after sufficient calls."""
        analyzer = SemanticClimateAnalyzer()

        # Make multiple calls to build up trajectory
        for i in range(10):
            result = analyzer.compute_coupling_coefficient(sample_embeddings)

        ti = result['trajectory_integrity']
        assert ti is not None
        assert ti['sufficient_data'] is True
        assert 'autocorrelation' in ti
        assert 'tortuosity' in ti
        assert 'recurrence_rate' in ti
        assert 'integrity_score' in ti
        assert 'integrity_label' in ti

    def test_integrity_none_when_disabled(self, sample_embeddings):
        """trajectory_integrity is None when compute_integrity=False."""
        analyzer = SemanticClimateAnalyzer(compute_integrity=False)

        for i in range(10):
            result = analyzer.compute_coupling_coefficient(sample_embeddings)

        assert result['trajectory_integrity'] is None

    def test_integrity_label_values(self, sample_embeddings):
        """integrity_label is one of expected values."""
        analyzer = SemanticClimateAnalyzer()

        for i in range(10):
            result = analyzer.compute_coupling_coefficient(sample_embeddings)

        ti = result['trajectory_integrity']
        if ti is not None and ti['sufficient_data']:
            assert ti['integrity_label'] in ['fragmented', 'living', 'rigid']


class TestHistoryTracking:
    """Tests for trajectory and basin history tracking."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    def test_trajectory_accumulates(self, sample_embeddings):
        """Trajectory buffer accumulates entries."""
        analyzer = SemanticClimateAnalyzer()

        for i in range(5):
            analyzer.compute_coupling_coefficient(sample_embeddings)

        assert len(analyzer.trajectory.history) == 5

    def test_basin_history_accumulates(self, sample_embeddings):
        """Basin history accumulates entries."""
        analyzer = SemanticClimateAnalyzer()

        for i in range(5):
            analyzer.compute_coupling_coefficient(sample_embeddings)

        assert len(analyzer.basin_history.history) == 5

    def test_residence_time_increases(self, sample_embeddings):
        """Residence time increases when staying in same basin."""
        analyzer = SemanticClimateAnalyzer()

        # Multiple calls should accumulate residence time
        residence_times = []
        for i in range(10):
            result = analyzer.compute_coupling_coefficient(sample_embeddings)
            residence_times.append(result['attractor_dynamics']['residence_time'])

        # Should generally increase (or reset on basin change)
        # At minimum, later values should be >= earlier values if same basin
        assert max(residence_times) > 0

    def test_reset_history_clears_all(self, sample_embeddings):
        """reset_history clears trajectory and basin history."""
        analyzer = SemanticClimateAnalyzer()

        # Build up some history
        for i in range(5):
            analyzer.compute_coupling_coefficient(sample_embeddings)

        assert len(analyzer.trajectory.history) == 5
        assert len(analyzer.basin_history.history) == 5

        # Reset
        analyzer.reset_history()

        assert len(analyzer.trajectory.history) == 0
        assert len(analyzer.basin_history.history) == 0

    def test_no_history_when_disabled(self, sample_embeddings):
        """No history accumulated when track_history=False."""
        analyzer = SemanticClimateAnalyzer(track_history=False)

        for i in range(5):
            analyzer.compute_coupling_coefficient(sample_embeddings)

        assert analyzer.trajectory is None
        assert analyzer.basin_history is None


class TestEnhancedAttractorDynamics:
    """Tests for enhanced attractor_dynamics output."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    def test_residence_time_field(self, sample_embeddings):
        """residence_time is present in attractor_dynamics."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)
        assert 'residence_time' in result['attractor_dynamics']
        assert isinstance(result['attractor_dynamics']['residence_time'], int)

    def test_previous_basin_field(self, sample_embeddings):
        """previous_basin is present in attractor_dynamics."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)
        assert 'previous_basin' in result['attractor_dynamics']

    def test_raw_confidence_field(self, sample_embeddings):
        """raw_confidence is present in attractor_dynamics."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)
        assert 'raw_confidence' in result['attractor_dynamics']
        assert isinstance(result['attractor_dynamics']['raw_confidence'], float)


class TestBackwardCompatibility:
    """Tests for backward compatibility with extensions.py."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    @pytest.fixture
    def sample_texts(self):
        """Generate sample turn texts."""
        return ["Hello world"] * 20

    def test_same_output_keys(self, sample_embeddings, sample_texts):
        """New analyzer has same output keys as legacy."""
        new_analyzer = SemanticClimateAnalyzer(track_history=False, compute_integrity=False)
        legacy_analyzer = LegacyAnalyzer()

        new_result = new_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )
        legacy_result = legacy_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )

        # Check all legacy keys are present in new output
        for key in legacy_result.keys():
            assert key in new_result, f"Missing key: {key}"

    def test_same_psi_value(self, sample_embeddings, sample_texts):
        """New and legacy analyzers produce same psi value."""
        new_analyzer = SemanticClimateAnalyzer(track_history=False, compute_integrity=False)
        legacy_analyzer = LegacyAnalyzer()

        new_result = new_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )
        legacy_result = legacy_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )

        assert abs(new_result['psi'] - legacy_result['psi']) < 1e-10

    def test_same_raw_metrics(self, sample_embeddings, sample_texts):
        """New and legacy analyzers produce same raw metrics."""
        new_analyzer = SemanticClimateAnalyzer(track_history=False, compute_integrity=False)
        legacy_analyzer = LegacyAnalyzer()

        new_result = new_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )
        legacy_result = legacy_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )

        for metric in ['delta_kappa', 'delta_h', 'alpha']:
            assert abs(new_result['raw_metrics'][metric] - legacy_result['raw_metrics'][metric]) < 1e-10

    def test_same_basin_detection(self, sample_embeddings, sample_texts):
        """New and legacy analyzers detect same basin."""
        new_analyzer = SemanticClimateAnalyzer(track_history=False, compute_integrity=False)
        legacy_analyzer = LegacyAnalyzer()

        new_result = new_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )
        legacy_result = legacy_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )

        # Basin names should match
        assert new_result['attractor_dynamics']['basin'] == legacy_result['attractor_dynamics']['basin']

    def test_same_substrate_values(self, sample_embeddings, sample_texts):
        """New and legacy analyzers compute same substrate values."""
        new_analyzer = SemanticClimateAnalyzer(track_history=False, compute_integrity=False)
        legacy_analyzer = LegacyAnalyzer()

        new_result = new_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )
        legacy_result = legacy_analyzer.compute_coupling_coefficient(
            sample_embeddings, turn_texts=sample_texts
        )

        # psi_semantic
        new_sem = new_result['substrate_details']['semantic']['psi_semantic']
        legacy_sem = legacy_result['substrate_details']['semantic']['psi_semantic']
        assert abs(new_sem - legacy_sem) < 1e-10

        # psi_temporal
        new_temp = new_result['substrate_details']['temporal']['psi_temporal']
        legacy_temp = legacy_result['substrate_details']['temporal']['psi_temporal']
        assert abs(new_temp - legacy_temp) < 1e-10

        # psi_affective
        new_aff = new_result['substrate_details']['affective']['psi_affective']
        legacy_aff = legacy_result['substrate_details']['affective']['psi_affective']
        assert abs(new_aff - legacy_aff) < 1e-10


class TestDialogueContext:
    """Tests for dialogue context computation."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    def test_dialogue_context_structure(self, sample_embeddings):
        """dialogue_context has expected structure."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(sample_embeddings)

        ctx = result['dialogue_context']
        assert 'hedging_density' in ctx
        assert 'turn_length_ratio' in ctx
        assert 'delta_kappa_variance' in ctx
        assert 'coherence_pattern' in ctx

    def test_coherence_pattern_passthrough(self, sample_embeddings):
        """coherence_pattern is passed through correctly."""
        analyzer = SemanticClimateAnalyzer()
        result = analyzer.compute_coupling_coefficient(
            sample_embeddings,
            coherence_pattern='breathing'
        )

        assert result['dialogue_context']['coherence_pattern'] == 'breathing'


class TestLegacyTrajectoryHistory:
    """Tests for legacy trajectory_history parameter."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(20, 384)

    def test_accepts_trajectory_history(self, sample_embeddings):
        """Accepts legacy trajectory_history parameter."""
        analyzer = SemanticClimateAnalyzer(track_history=False)

        trajectory_history = [
            {'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None},
            {'psi_semantic': 0.2, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': None},
            {'psi_semantic': 0.3, 'psi_temporal': 0.5, 'psi_affective': 0.2, 'psi_biosignal': None},
        ]

        result = analyzer.compute_coupling_coefficient(
            sample_embeddings,
            trajectory_history=trajectory_history
        )

        # Should have computed derivatives from the provided history
        assert result['psi_state']['velocity'] is not None
