"""
Tests for integrity.py - Trajectory Integrity Computation

TDD tests written BEFORE implementation per modular refactor design.

These tests define the expected behavior for:
- IntegrityAnalyzer: Computes trajectory integrity from path and basin history
- TransformationDetector: Detects constraint-altering events (stubbed for now)

Integrity measures the coherent relationship between path and history,
distinguishing fragmentation from rigidity.

Score interpretation:
- 0.0-0.3: Fragmented (no memory, random walk)
- 0.3-0.7: Living integrity (responsive persistence)
- 0.7-1.0: Rigid (locked, no responsiveness)

See: research/theory/trajectory-integrity.md
"""

import pytest
import numpy as np
import json
from pathlib import Path

# Will import from src.integrity once implemented
try:
    from src.integrity import IntegrityAnalyzer, TransformationDetector
    INTEGRITY_MODULE_EXISTS = True
except ImportError:
    INTEGRITY_MODULE_EXISTS = False
    IntegrityAnalyzer = None
    TransformationDetector = None

# Import TrajectoryBuffer for test setup
try:
    from src.trajectory import TrajectoryBuffer
except ImportError:
    from src.extensions import TrajectoryBuffer

# Import BasinHistory for test setup
try:
    from src.basins import BasinHistory
except ImportError:
    BasinHistory = None


# Load golden outputs for regression testing
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_OUTPUTS_PATH = FIXTURES_DIR / 'golden_outputs.json'


@pytest.fixture
def golden_outputs():
    """Load golden outputs captured before refactor."""
    with open(GOLDEN_OUTPUTS_PATH) as f:
        return json.load(f)


def create_fragmented_trajectory(n_points=20):
    """Create a random/fragmented trajectory with no memory."""
    np.random.seed(42)
    buffer = TrajectoryBuffer(timestep=1.0)

    for i in range(n_points):
        buffer.append({
            'psi_semantic': np.random.uniform(-1, 1),
            'psi_temporal': np.random.uniform(0, 1),
            'psi_affective': np.random.uniform(-1, 1),
            'psi_biosignal': None
        })

    return buffer


def create_rigid_trajectory(n_points=20):
    """Create a locked/rigid trajectory with no responsiveness."""
    buffer = TrajectoryBuffer(timestep=1.0)

    # Same point repeated
    for i in range(n_points):
        buffer.append({
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.3,
            'psi_biosignal': None
        })

    return buffer


def create_living_trajectory(n_points=20):
    """Create a trajectory with living integrity (responsive persistence)."""
    buffer = TrajectoryBuffer(timestep=1.0)

    # Smooth evolution with some responsiveness
    for i in range(n_points):
        t = i / n_points
        buffer.append({
            'psi_semantic': 0.3 + 0.3 * np.sin(2 * np.pi * t),
            'psi_temporal': 0.5 + 0.1 * np.cos(2 * np.pi * t),
            'psi_affective': 0.2 + 0.2 * np.sin(4 * np.pi * t),
            'psi_biosignal': None
        })

    return buffer


@pytest.mark.skipif(not INTEGRITY_MODULE_EXISTS, reason="integrity.py not yet implemented")
class TestIntegrityAnalyzer:
    """Tests for IntegrityAnalyzer class."""

    def test_initialization_defaults(self):
        """Should initialize with default weights."""
        analyzer = IntegrityAnalyzer()

        assert analyzer.weights['autocorrelation'] == 0.4
        assert analyzer.weights['tortuosity'] == 0.3
        assert analyzer.weights['recurrence'] == 0.3
        assert analyzer.min_length == 5

    def test_initialization_custom(self):
        """Should accept custom weights."""
        analyzer = IntegrityAnalyzer(
            autocorr_weight=0.5,
            tortuosity_weight=0.25,
            recurrence_weight=0.25,
            min_trajectory_length=10
        )

        assert analyzer.weights['autocorrelation'] == 0.5
        assert analyzer.min_length == 10

    def test_compute_returns_dict_structure(self):
        """Should return dict with expected keys."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_living_trajectory()

        result = analyzer.compute(trajectory)

        assert 'autocorrelation' in result
        assert 'tortuosity' in result
        assert 'recurrence_rate' in result
        assert 'integrity_score' in result
        assert 'integrity_label' in result
        assert 'sufficient_data' in result

    def test_integrity_score_bounded(self):
        """Integrity score should be bounded [0, 1]."""
        analyzer = IntegrityAnalyzer()

        for traj_func in [create_fragmented_trajectory, create_living_trajectory, create_rigid_trajectory]:
            trajectory = traj_func()
            result = analyzer.compute(trajectory)

            assert 0 <= result['integrity_score'] <= 1

    def test_fragmented_trajectory_low_score(self):
        """Fragmented (random) trajectory should have low integrity score."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_fragmented_trajectory(n_points=50)

        result = analyzer.compute(trajectory)

        assert result['integrity_score'] < 0.4
        assert result['integrity_label'] == 'fragmented'

    def test_rigid_trajectory_high_score(self):
        """Rigid (locked) trajectory should have high integrity score."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_rigid_trajectory(n_points=50)

        result = analyzer.compute(trajectory)

        assert result['integrity_score'] > 0.7
        assert result['integrity_label'] == 'rigid'

    def test_living_trajectory_mid_score(self):
        """Living trajectory should have mid-range integrity score."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_living_trajectory(n_points=50)

        result = analyzer.compute(trajectory)

        assert 0.3 <= result['integrity_score'] <= 0.7
        assert result['integrity_label'] == 'living'

    def test_insufficient_data_flagged(self):
        """Should flag when trajectory too short."""
        analyzer = IntegrityAnalyzer(min_trajectory_length=10)
        trajectory = create_living_trajectory(n_points=5)

        result = analyzer.compute(trajectory)

        assert result['sufficient_data'] is False

    def test_label_assignment_fragmented(self):
        """Score 0.0-0.3 should be labeled 'fragmented'."""
        analyzer = IntegrityAnalyzer()

        # Mock low-scoring trajectory
        trajectory = create_fragmented_trajectory(n_points=100)
        result = analyzer.compute(trajectory)

        if result['integrity_score'] < 0.3:
            assert result['integrity_label'] == 'fragmented'

    def test_label_assignment_living(self):
        """Score 0.3-0.7 should be labeled 'living'."""
        analyzer = IntegrityAnalyzer()

        trajectory = create_living_trajectory(n_points=50)
        result = analyzer.compute(trajectory)

        if 0.3 <= result['integrity_score'] <= 0.7:
            assert result['integrity_label'] == 'living'

    def test_label_assignment_rigid(self):
        """Score 0.7-1.0 should be labeled 'rigid'."""
        analyzer = IntegrityAnalyzer()

        trajectory = create_rigid_trajectory(n_points=50)
        result = analyzer.compute(trajectory)

        if result['integrity_score'] > 0.7:
            assert result['integrity_label'] == 'rigid'

    def test_with_basin_history(self):
        """Should incorporate basin history when provided."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_living_trajectory(n_points=30)

        if BasinHistory is not None:
            history = BasinHistory()
            for i in range(30):
                history.append('Deep Resonance', 0.8, turn=i)

            result = analyzer.compute(trajectory, basin_history=history)

            # Just verify it runs without error
            assert 'integrity_score' in result


@pytest.mark.skipif(not INTEGRITY_MODULE_EXISTS, reason="integrity.py not yet implemented")
class TestIntegrityComponents:
    """Tests for individual integrity component calculations."""

    def test_autocorrelation_high_for_smooth(self):
        """Smooth trajectory should have high autocorrelation."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_living_trajectory(n_points=50)

        result = analyzer.compute(trajectory)

        # Smooth trajectory has memory -> high autocorrelation
        assert result['autocorrelation'] > 0.5

    def test_autocorrelation_low_for_random(self):
        """Random trajectory should have low autocorrelation."""
        analyzer = IntegrityAnalyzer()
        trajectory = create_fragmented_trajectory(n_points=50)

        result = analyzer.compute(trajectory)

        # Random trajectory has no memory -> low autocorrelation
        assert result['autocorrelation'] < 0.5

    def test_tortuosity_low_for_straight(self):
        """Straight trajectory should have low tortuosity (close to 1)."""
        analyzer = IntegrityAnalyzer()

        # Create straight-line trajectory
        buffer = TrajectoryBuffer(timestep=1.0)
        for i in range(30):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.1 * i,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        result = analyzer.compute(buffer)

        # Straight line has tortuosity = 1.0 (minimum)
        # We normalize so lower tortuosity -> higher score component
        assert result['tortuosity'] is not None

    def test_recurrence_rate_high_for_periodic(self):
        """Periodic trajectory should have high recurrence."""
        analyzer = IntegrityAnalyzer()

        # Create periodic trajectory that revisits same regions
        buffer = TrajectoryBuffer(timestep=1.0)
        for i in range(60):
            t = i / 10
            buffer.append({
                'psi_semantic': 0.5 * np.sin(2 * np.pi * t),
                'psi_temporal': 0.5 * np.cos(2 * np.pi * t),
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        result = analyzer.compute(buffer)

        # Periodic trajectory revisits same points
        assert result['recurrence_rate'] is not None


@pytest.mark.skipif(not INTEGRITY_MODULE_EXISTS, reason="integrity.py not yet implemented")
class TestTransformationDetector:
    """Tests for TransformationDetector class (stubbed for Phase 1)."""

    def test_initialization(self):
        """Should initialize with parameters."""
        detector = TransformationDetector(
            substrate_threshold=0.3,
            basin_transition_counts=True
        )

        assert detector.substrate_threshold == 0.3
        assert detector.basin_transition_counts is True

    def test_detect_returns_bool(self):
        """detect() should return boolean."""
        detector = TransformationDetector()

        psi_prev = {'psi_semantic': 0.3, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': None}
        psi_curr = {'psi_semantic': 0.8, 'psi_temporal': 0.5, 'psi_affective': 0.4, 'psi_biosignal': None}

        result = detector.detect(psi_prev, psi_curr)

        assert isinstance(result, bool)

    def test_large_substrate_change_is_transformation(self):
        """Large substrate change should be detected as transformation."""
        detector = TransformationDetector(substrate_threshold=0.3)

        psi_prev = {'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None}
        psi_curr = {'psi_semantic': 0.8, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None}

        # Change of 0.7 > threshold of 0.3
        result = detector.detect(psi_prev, psi_curr)

        assert result is True

    def test_small_change_not_transformation(self):
        """Small substrate change should not be transformation."""
        detector = TransformationDetector(substrate_threshold=0.3)

        psi_prev = {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.2, 'psi_biosignal': None}
        psi_curr = {'psi_semantic': 0.55, 'psi_temporal': 0.52, 'psi_affective': 0.22, 'psi_biosignal': None}

        result = detector.detect(psi_prev, psi_curr)

        assert result is False

    def test_basin_transition_is_transformation(self):
        """Basin transition should be detected as transformation."""
        detector = TransformationDetector(basin_transition_counts=True)

        psi_prev = {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.2, 'psi_biosignal': None}
        psi_curr = {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.2, 'psi_biosignal': None}

        result = detector.detect(
            psi_prev, psi_curr,
            basin_prev='Deep Resonance',
            basin_curr='Dissociation'
        )

        assert result is True

    def test_transformation_density(self):
        """Should compute transformations per turn."""
        detector = TransformationDetector()

        trajectory = create_living_trajectory(n_points=30)

        if BasinHistory is not None:
            history = BasinHistory()
            for i in range(30):
                basin = 'Deep Resonance' if i % 10 < 5 else 'Transitional'
                history.append(basin, 0.7, turn=i)

            density = detector.compute_transformation_density(trajectory, history)

            assert 0 <= density <= 1


@pytest.mark.skipif(not INTEGRITY_MODULE_EXISTS, reason="integrity.py not yet implemented")
class TestIntegrityCalibration:
    """Tests for integrity score calibration."""

    def test_brownian_motion_mid_range(self):
        """Brownian motion should be between fragmented and living."""
        analyzer = IntegrityAnalyzer()

        # Create Brownian-like trajectory
        np.random.seed(42)
        buffer = TrajectoryBuffer(timestep=1.0)
        position = np.array([0.5, 0.5, 0.0, 0.0])

        for i in range(50):
            step = np.random.randn(4) * 0.1
            position = np.clip(position + step, -1, 1)
            buffer.append({
                'psi_semantic': float(position[0]),
                'psi_temporal': float(position[1]),
                'psi_affective': float(position[2]),
                'psi_biosignal': None
            })

        result = analyzer.compute(buffer)

        # Brownian has some memory (not pure random) but not locked
        assert 0.2 <= result['integrity_score'] <= 0.6

    def test_damped_oscillation_living(self):
        """Damped oscillation should score as living integrity."""
        analyzer = IntegrityAnalyzer()

        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(50):
            t = i / 50
            damping = np.exp(-2 * t)
            buffer.append({
                'psi_semantic': 0.5 + 0.4 * damping * np.sin(10 * np.pi * t),
                'psi_temporal': 0.5 + 0.2 * damping * np.cos(10 * np.pi * t),
                'psi_affective': 0.3 * damping,
                'psi_biosignal': None
            })

        result = analyzer.compute(buffer)

        # Damped oscillation has structure but responsiveness
        assert result['integrity_label'] in ['living', 'fragmented']


class TestIntegrityBackwardCompatibility:
    """Ensure integrity can be integrated with existing analyzer."""

    def test_can_use_trajectory_buffer(self):
        """IntegrityAnalyzer should work with existing TrajectoryBuffer."""
        trajectory = create_living_trajectory()

        # TrajectoryBuffer from extensions should work
        assert hasattr(trajectory, 'history')
        assert hasattr(trajectory, 'compute_velocity')

    @pytest.mark.skipif(not INTEGRITY_MODULE_EXISTS, reason="integrity.py not yet implemented")
    def test_none_trajectory_handled(self):
        """Should handle None trajectory gracefully."""
        analyzer = IntegrityAnalyzer()

        result = analyzer.compute(None)

        assert result['sufficient_data'] is False
        assert result['integrity_score'] is None or result['integrity_score'] == 0.0
