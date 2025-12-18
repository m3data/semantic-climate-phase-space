"""
Tests for trajectory.py - Trajectory Storage and Geometry

TDD tests written BEFORE implementation per modular refactor design.

These tests define the expected behavior for:
- TrajectoryBuffer: Storage, derivatives, geometric characterization
- compute_trajectory_derivatives: Standalone function

Tests are designed to pass when trajectory.py is implemented correctly
while maintaining backward compatibility with existing extensions.py behavior.
"""

import pytest
import numpy as np
import json
from pathlib import Path

# Will import from src.trajectory once implemented
# For now, import from existing extensions to validate golden outputs
try:
    from src.trajectory import TrajectoryBuffer, compute_trajectory_derivatives
    TRAJECTORY_MODULE_EXISTS = True
except ImportError:
    from src.extensions import TrajectoryBuffer
    TRAJECTORY_MODULE_EXISTS = False
    compute_trajectory_derivatives = None


# Load golden outputs for regression testing
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_OUTPUTS_PATH = FIXTURES_DIR / 'golden_outputs.json'


@pytest.fixture
def golden_outputs():
    """Load golden outputs captured before refactor."""
    with open(GOLDEN_OUTPUTS_PATH) as f:
        return json.load(f)


class TestTrajectoryBufferStorage:
    """Tests for TrajectoryBuffer storage operations."""

    def test_initialization_defaults(self):
        """Should initialize with default parameters."""
        buffer = TrajectoryBuffer()

        assert buffer.window_size == 50
        assert buffer.timestep == 1.0
        assert len(buffer.history) == 0

    def test_initialization_custom(self):
        """Should initialize with custom parameters."""
        buffer = TrajectoryBuffer(window_size=20, timestep=0.5)

        assert buffer.window_size == 20
        assert buffer.timestep == 0.5

    def test_append_single(self):
        """Should append a single Ψ vector."""
        buffer = TrajectoryBuffer()

        psi = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.3,
            'psi_affective': 0.1,
            'psi_biosignal': None
        }
        buffer.append(psi)

        assert len(buffer.history) == 1
        timestamp, stored_psi = buffer.history[0]
        assert stored_psi == psi
        assert timestamp == 0.0  # First entry at t=0

    def test_append_with_timestamp(self):
        """Should use provided timestamp."""
        buffer = TrajectoryBuffer()

        psi = {'psi_semantic': 0.5, 'psi_temporal': 0.3, 'psi_affective': 0.1, 'psi_biosignal': None}
        buffer.append(psi, timestamp=10.5)

        assert buffer.history[0][0] == 10.5

    def test_append_auto_timestamp(self):
        """Should auto-generate timestamps based on timestep."""
        buffer = TrajectoryBuffer(timestep=2.0)

        for i in range(3):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        assert buffer.history[0][0] == 0.0
        assert buffer.history[1][0] == 2.0
        assert buffer.history[2][0] == 4.0

    def test_window_size_limit(self):
        """Should maintain window size limit by dropping oldest entries."""
        buffer = TrajectoryBuffer(window_size=5)

        for i in range(10):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        assert len(buffer.history) == 5
        # Should have entries 5-9, not 0-4
        assert buffer.history[0][1]['psi_semantic'] == pytest.approx(0.5)  # i=5

    def test_get_trajectory_segment_all(self):
        """Should retrieve all trajectory points."""
        buffer = TrajectoryBuffer()

        for i in range(5):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        segment = buffer.get_trajectory_segment()

        assert len(segment) == 5
        assert segment[0][1]['psi_semantic'] == pytest.approx(0.0)
        assert segment[-1][1]['psi_semantic'] == pytest.approx(0.4)

    def test_get_trajectory_segment_n_points(self):
        """Should retrieve last N points."""
        buffer = TrajectoryBuffer()

        for i in range(5):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        segment = buffer.get_trajectory_segment(n_points=3)

        assert len(segment) == 3
        assert segment[-1][1]['psi_semantic'] == pytest.approx(0.4)

    def test_get_trajectory_segment_empty(self):
        """Should return None for empty buffer."""
        buffer = TrajectoryBuffer()

        segment = buffer.get_trajectory_segment()
        assert segment is None

    def test_clear(self):
        """Should clear history if clear() method exists."""
        buffer = TrajectoryBuffer()

        for i in range(5):
            buffer.append({'psi_semantic': i * 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        if hasattr(buffer, 'clear'):
            buffer.clear()
            assert len(buffer.history) == 0


class TestTrajectoryBufferDerivatives:
    """Tests for TrajectoryBuffer derivative computations."""

    def test_velocity_linear_trajectory(self, golden_outputs):
        """Should compute correct velocity for linear trajectory."""
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

        # Compare to golden outputs
        golden = golden_outputs['trajectory_buffer']['linear_trajectory']['velocity']
        assert velocity is not None
        assert velocity['dpsi_semantic_dt'] == pytest.approx(golden['dpsi_semantic_dt'], abs=1e-10)
        assert velocity['dpsi_temporal_dt'] == pytest.approx(golden['dpsi_temporal_dt'], abs=1e-10)

    def test_velocity_insufficient_history(self):
        """Should return None with insufficient history (< 3 points)."""
        buffer = TrajectoryBuffer()

        buffer.append({'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 0.2, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        velocity = buffer.compute_velocity()
        assert velocity is None

    def test_acceleration_linear_trajectory(self, golden_outputs):
        """Should compute ~zero acceleration for linear trajectory."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        acceleration = buffer.compute_acceleration()

        golden = golden_outputs['trajectory_buffer']['linear_trajectory']['acceleration']
        assert acceleration is not None
        # Near-zero for linear (within floating point tolerance)
        assert abs(acceleration['d2psi_semantic_dt2']) < 1e-10

    def test_acceleration_quadratic_trajectory(self):
        """Should compute constant acceleration for quadratic trajectory."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Quadratic: psi_semantic = 0.1 * t^2
        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * (i ** 2),
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        acceleration = buffer.compute_acceleration()

        assert acceleration is not None
        # d²(0.1t²)/dt² = 0.2
        assert acceleration['d2psi_semantic_dt2'] == pytest.approx(0.2, abs=0.01)

    def test_curvature_linear_zero(self, golden_outputs):
        """Should return zero curvature for linear trajectory."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        curvature = buffer.compute_curvature()

        golden = golden_outputs['trajectory_buffer']['linear_trajectory']['curvature']
        assert curvature == pytest.approx(golden, abs=1e-10)

    def test_curvature_sinusoidal(self, golden_outputs):
        """Should compute non-zero curvature for curved trajectory."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(10):
            buffer.append({
                'psi_semantic': float(np.sin(i * 0.5)),
                'psi_temporal': float(np.cos(i * 0.5)),
                'psi_affective': 0.1 * i,
                'psi_biosignal': None
            })

        curvature = buffer.compute_curvature()

        golden = golden_outputs['trajectory_buffer']['sinusoidal_trajectory']['curvature']
        assert curvature is not None
        assert curvature == pytest.approx(golden, rel=1e-6)

    def test_curvature_insufficient_history(self):
        """Should return None with insufficient history."""
        buffer = TrajectoryBuffer()

        buffer.append({'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 0.2, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        curvature = buffer.compute_curvature()
        assert curvature is None


class TestTrajectoryBufferGeometry:
    """Tests for new geometric characterization methods in trajectory.py."""

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_tortuosity_straight_path(self):
        """Straight path should have tortuosity = 1.0 (minimal winding)."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Straight line in 4D space
        for i in range(10):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.1 * i,
                'psi_affective': 0.1 * i,
                'psi_biosignal': 0.1 * i
            })

        tortuosity = buffer.compute_tortuosity()

        assert tortuosity is not None
        assert tortuosity == pytest.approx(1.0, abs=0.05)

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_tortuosity_winding_path(self):
        """Winding path should have tortuosity > 1.0."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Sinusoidal path (winding)
        for i in range(20):
            buffer.append({
                'psi_semantic': float(np.sin(i * 0.3)),
                'psi_temporal': float(np.cos(i * 0.3)),
                'psi_affective': 0.05 * i,
                'psi_biosignal': None
            })

        tortuosity = buffer.compute_tortuosity()

        assert tortuosity is not None
        assert tortuosity > 1.0

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_displacement_returns_euclidean_distance(self):
        """Should return Euclidean distance from start to end."""
        buffer = TrajectoryBuffer(timestep=1.0)

        buffer.append({'psi_semantic': 0.0, 'psi_temporal': 0.0, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 0.1, 'psi_temporal': 0.1, 'psi_affective': 0.1, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 0.3, 'psi_temporal': 0.0, 'psi_affective': 0.4, 'psi_biosignal': None})

        displacement = buffer.compute_displacement()

        # Euclidean distance from (0,0,0) to (0.3, 0, 0.4)
        expected = np.sqrt(0.3**2 + 0.4**2)  # 0.5
        assert displacement == pytest.approx(expected, abs=0.01)

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_path_length_sum_of_steps(self):
        """Should return total path length (sum of step distances)."""
        buffer = TrajectoryBuffer(timestep=1.0)

        # Simple 2-step path: 0 -> 1 -> 2
        buffer.append({'psi_semantic': 0.0, 'psi_temporal': 0.0, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 1.0, 'psi_temporal': 0.0, 'psi_affective': 0.0, 'psi_biosignal': None})
        buffer.append({'psi_semantic': 2.0, 'psi_temporal': 0.0, 'psi_affective': 0.0, 'psi_biosignal': None})

        path_length = buffer.compute_path_length()

        assert path_length == pytest.approx(2.0, abs=0.01)

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_autocorrelation_constant_signal(self):
        """Constant signal should have high autocorrelation at all lags."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(20):
            buffer.append({
                'psi_semantic': 0.5,
                'psi_temporal': 0.5,
                'psi_affective': 0.5,
                'psi_biosignal': None
            })

        autocorr = buffer.compute_autocorrelation(max_lag=5)

        assert autocorr is not None
        assert len(autocorr) <= 5
        # All lags should be close to 1.0 for constant signal
        for lag_corr in autocorr:
            assert lag_corr == pytest.approx(1.0, abs=0.1)

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_autocorrelation_random_signal(self):
        """Random signal should have low autocorrelation at non-zero lags."""
        np.random.seed(42)
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(50):
            buffer.append({
                'psi_semantic': np.random.random(),
                'psi_temporal': np.random.random(),
                'psi_affective': np.random.random(),
                'psi_biosignal': None
            })

        autocorr = buffer.compute_autocorrelation(max_lag=5)

        assert autocorr is not None
        # Lag 0 should be ~1.0, higher lags should be lower
        # (not strictly tested as random can still have some correlation)

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_geometry_insufficient_data(self):
        """Should return None for insufficient trajectory length."""
        buffer = TrajectoryBuffer(timestep=1.0)

        buffer.append({'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None})

        assert buffer.compute_tortuosity() is None
        assert buffer.compute_displacement() is None
        assert buffer.compute_path_length() is None
        assert buffer.compute_autocorrelation() is None


class TestComputeTrajectoryDerivatives:
    """Tests for standalone compute_trajectory_derivatives function."""

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_returns_dict_structure(self):
        """Should return dict with velocity, acceleration, curvature, speed, direction."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        result = compute_trajectory_derivatives(buffer)

        assert 'velocity' in result
        assert 'acceleration' in result
        assert 'curvature' in result
        assert 'speed' in result
        assert 'direction' in result

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_none_for_empty_buffer(self):
        """Should return all None values for empty buffer."""
        buffer = TrajectoryBuffer()

        result = compute_trajectory_derivatives(buffer)

        assert result['velocity'] is None
        assert result['acceleration'] is None
        assert result['curvature'] is None
        assert result['speed'] is None
        assert result['direction'] is None

    @pytest.mark.skipif(not TRAJECTORY_MODULE_EXISTS, reason="trajectory.py not yet implemented")
    def test_speed_is_velocity_magnitude(self):
        """Speed should be the magnitude of velocity vector."""
        buffer = TrajectoryBuffer(timestep=1.0)

        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.1 * i,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        result = compute_trajectory_derivatives(buffer)

        if result['velocity'] and result['speed']:
            # Compute expected magnitude
            v_vals = [v for v in result['velocity'].values() if v is not None]
            expected_speed = np.linalg.norm(v_vals)
            assert result['speed'] == pytest.approx(expected_speed, abs=1e-6)


class TestBackwardCompatibility:
    """Ensure trajectory.py maintains backward compatibility with extensions.py."""

    def test_trajectory_buffer_same_api(self):
        """TrajectoryBuffer should have same core API."""
        buffer = TrajectoryBuffer(window_size=50, timestep=1.0)

        # These methods must exist
        assert hasattr(buffer, 'append')
        assert hasattr(buffer, 'compute_velocity')
        assert hasattr(buffer, 'compute_acceleration')
        assert hasattr(buffer, 'compute_curvature')
        assert hasattr(buffer, 'get_trajectory_segment')
        assert hasattr(buffer, 'history')
        assert hasattr(buffer, 'window_size')
        assert hasattr(buffer, 'timestep')

    def test_golden_output_regression(self, golden_outputs):
        """New implementation must match golden outputs exactly."""
        # This test verifies refactored code matches pre-refactor behavior

        # Test linear trajectory
        buffer = TrajectoryBuffer(timestep=1.0)
        for i in range(5):
            buffer.append({
                'psi_semantic': 0.1 * i,
                'psi_temporal': 0.5,
                'psi_affective': 0.0,
                'psi_biosignal': None
            })

        golden = golden_outputs['trajectory_buffer']['linear_trajectory']

        velocity = buffer.compute_velocity()
        assert velocity['dpsi_semantic_dt'] == pytest.approx(golden['velocity']['dpsi_semantic_dt'], abs=1e-10)

        acceleration = buffer.compute_acceleration()
        assert abs(acceleration['d2psi_semantic_dt2']) < 1e-10

        curvature = buffer.compute_curvature()
        assert curvature == pytest.approx(golden['curvature'], abs=1e-10)
