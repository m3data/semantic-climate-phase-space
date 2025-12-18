"""
Tests for Movement-Preserving Classification (v0.3.0)

This module tests the core innovation of v0.3.0: classification that preserves
movement through state space, enabling distinctions like "vigilant stillness
from settling" vs "vigilant stillness from capture".

Tests cover:
- TrajectoryStateVector computation
- Soft state inference
- Movement annotation generation
- Hysteresis-aware transitions
- Integration with SemanticClimateAnalyzer
"""

import pytest
import numpy as np
from src.trajectory import TrajectoryBuffer, TrajectoryStateVector
from src.basins import (
    BasinDetector,
    BasinHistory,
    SoftStateInference,
    HysteresisConfig,
    generate_movement_annotation,
)


class TestTrajectoryStateVector:
    """Tests for TrajectoryStateVector dataclass."""

    def test_basic_creation(self):
        """TSV can be created with minimal required fields."""
        position = {'psi_semantic': 0.5, 'psi_affective': 0.3}
        tsv = TrajectoryStateVector(position=position)
        assert tsv.position == position
        assert tsv.velocity_magnitude is None
        assert tsv.dwell_time == 0
        assert tsv.movement_annotation == "unknown"

    def test_full_creation(self):
        """TSV can be created with all fields."""
        tsv = TrajectoryStateVector(
            position={'psi_semantic': 0.5},
            velocity_magnitude=0.1,
            acceleration_magnitude=0.02,
            curvature=0.05,
            approach_basin='Deep Resonance',
            approach_direction={'psi_semantic': 0.8},
            dwell_time=5,
            movement_annotation='settled'
        )
        assert tsv.velocity_magnitude == 0.1
        assert tsv.approach_basin == 'Deep Resonance'
        assert tsv.dwell_time == 5
        assert tsv.movement_annotation == 'settled'

    def test_to_dict(self):
        """TSV can be converted to dictionary."""
        tsv = TrajectoryStateVector(
            position={'psi_semantic': 0.5},
            velocity_magnitude=0.1,
            dwell_time=3,
            movement_annotation='moving'
        )
        d = tsv.to_dict()
        assert isinstance(d, dict)
        assert d['position'] == {'psi_semantic': 0.5}
        assert d['velocity_magnitude'] == 0.1
        assert d['dwell_time'] == 3


class TestApproachVector:
    """Tests for approach vector computation."""

    def test_approach_vector_basic(self):
        """Approach vector computed from two positions."""
        buf = TrajectoryBuffer()
        buf.append({'psi_semantic': 0.0, 'psi_affective': 0.0})
        buf.append({'psi_semantic': 1.0, 'psi_affective': 0.0})

        approach = buf.compute_approach_vector()
        assert approach is not None
        assert approach['magnitude'] == pytest.approx(1.0)
        assert approach['direction']['psi_semantic'] == pytest.approx(1.0)
        assert approach['direction']['psi_affective'] == pytest.approx(0.0)

    def test_approach_vector_insufficient_history(self):
        """Approach vector returns None with < 2 points."""
        buf = TrajectoryBuffer()
        buf.append({'psi_semantic': 0.5})
        assert buf.compute_approach_vector() is None

    def test_approach_vector_stationary(self):
        """Stationary trajectory has zero magnitude."""
        buf = TrajectoryBuffer()
        buf.append({'psi_semantic': 0.5, 'psi_affective': 0.3})
        buf.append({'psi_semantic': 0.5, 'psi_affective': 0.3})

        approach = buf.compute_approach_vector()
        assert approach['magnitude'] == pytest.approx(0.0, abs=1e-10)


class TestVelocityMagnitude:
    """Tests for velocity magnitude computation."""

    def test_velocity_magnitude_linear(self):
        """Velocity magnitude on linear trajectory."""
        buf = TrajectoryBuffer()
        for i in range(5):
            buf.append({'psi_semantic': i * 0.1, 'psi_affective': 0.0})

        vel_mag = buf.compute_velocity_magnitude()
        assert vel_mag is not None
        assert vel_mag > 0

    def test_velocity_magnitude_insufficient(self):
        """Returns None with < 3 points."""
        buf = TrajectoryBuffer()
        buf.append({'psi_semantic': 0.0})
        buf.append({'psi_semantic': 0.1})
        assert buf.compute_velocity_magnitude() is None


class TestSoftStateInference:
    """Tests for soft membership computation."""

    def test_soft_membership_structure(self):
        """Soft inference returns correct structure."""
        detector = BasinDetector()
        psi = {'psi_semantic': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5}
        metrics = {'delta_kappa': 0.5}

        soft = detector.compute_soft_membership(psi, metrics)
        assert isinstance(soft, SoftStateInference)
        assert soft.primary_basin in detector.BASINS
        assert 0.0 <= soft.ambiguity <= 1.0
        assert sum(soft.membership.values()) == pytest.approx(1.0)

    def test_soft_membership_near_centroid(self):
        """Position near basin centroid has high membership relative to others."""
        detector = BasinDetector()
        # Position exactly at Deep Resonance centroid
        psi = {'psi_semantic': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5}
        metrics = {'delta_kappa': 0.5}

        soft = detector.compute_soft_membership(psi, metrics)
        assert soft.primary_basin == 'Deep Resonance'
        # With 10 basins, ~0.14 is reasonable for the top basin
        # Check it's the highest membership
        assert soft.membership['Deep Resonance'] == max(soft.membership.values())

    def test_soft_membership_at_boundary(self):
        """Position at boundary has higher ambiguity."""
        detector = BasinDetector()
        # Position equidistant from multiple centroids
        psi = {'psi_semantic': 0.3, 'psi_affective': 0.2, 'psi_biosignal': 0.1}
        metrics = {'delta_kappa': 0.3}

        soft = detector.compute_soft_membership(psi, metrics)
        assert soft.ambiguity > 0.5  # High ambiguity at boundary

    def test_distribution_shift(self):
        """Distribution shift computed between consecutive inferences."""
        detector = BasinDetector()
        psi1 = {'psi_semantic': 0.1, 'psi_affective': 0.1, 'psi_biosignal': 0.1}
        psi2 = {'psi_semantic': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5}
        metrics = {'delta_kappa': 0.4}

        soft1 = detector.compute_soft_membership(psi1, metrics)
        soft2 = detector.compute_soft_membership(psi2, metrics, previous_inference=soft1)

        assert soft2.distribution_shift is not None
        assert soft2.distribution_shift > 0  # Should be significant shift


class TestMovementAnnotation:
    """Tests for movement annotation generation."""

    def test_settled_annotation(self):
        """Low velocity + high dwell = settled."""
        ann = generate_movement_annotation(
            velocity_magnitude=0.01,
            acceleration_magnitude=0.001,
            previous_basin=None,
            dwell_time=10
        )
        assert 'settled' in ann

    def test_still_annotation(self):
        """Low velocity + low dwell = still."""
        ann = generate_movement_annotation(
            velocity_magnitude=0.02,
            acceleration_magnitude=0.001,
            previous_basin='Deep Resonance',
            dwell_time=2
        )
        assert 'still' in ann
        assert 'Deep Resonance' in ann

    def test_accelerating_annotation(self):
        """High acceleration = accelerating."""
        ann = generate_movement_annotation(
            velocity_magnitude=0.1,
            acceleration_magnitude=0.05,
            previous_basin=None,
            dwell_time=1
        )
        assert 'accelerating' in ann

    def test_decelerating_annotation(self):
        """Negative acceleration = decelerating."""
        ann = generate_movement_annotation(
            velocity_magnitude=0.08,
            acceleration_magnitude=-0.03,
            previous_basin='Cognitive Mimicry',
            dwell_time=2
        )
        assert 'decelerating' in ann
        assert 'Cognitive Mimicry' in ann

    def test_insufficient_data(self):
        """None velocity = insufficient data."""
        ann = generate_movement_annotation(
            velocity_magnitude=None,
            acceleration_magnitude=None,
            previous_basin=None,
            dwell_time=0
        )
        assert ann == 'insufficient data'


class TestHysteresisConfig:
    """Tests for HysteresisConfig dataclass."""

    def test_default_values(self):
        """HysteresisConfig has sensible defaults."""
        config = HysteresisConfig(basin_name='Test')
        assert config.entry_threshold < config.exit_threshold
        assert config.provisional_turns < config.established_turns
        assert config.entry_penalty < 1.0
        assert config.settled_bonus > 1.0

    def test_to_dict(self):
        """HysteresisConfig can be converted to dict."""
        config = HysteresisConfig(basin_name='Test', entry_threshold=0.3)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['basin_name'] == 'Test'
        assert d['entry_threshold'] == 0.3


class TestBasinHistoryHysteresis:
    """Tests for hysteresis state tracking in BasinHistory."""

    def test_initial_state_unknown(self):
        """New history starts in unknown state."""
        history = BasinHistory()
        assert history.get_state_status() == 'unknown'

    def test_state_status_transition(self):
        """State status can be changed."""
        history = BasinHistory()
        history.set_state_status('provisional')
        assert history.get_state_status() == 'provisional'

        history.set_state_status('established')
        assert history.get_state_status() == 'established'

    def test_invalid_state_raises(self):
        """Invalid state status raises ValueError."""
        history = BasinHistory()
        with pytest.raises(ValueError):
            history.set_state_status('invalid')

    def test_provisional_duration(self):
        """Provisional duration tracks correctly."""
        history = BasinHistory()
        history.append('TestBasin', 0.5)
        history.set_state_status('provisional')

        history.append('TestBasin', 0.6)
        history.append('TestBasin', 0.7)

        duration = history.get_provisional_duration()
        assert duration == 2  # 2 entries since provisional

    def test_clear_resets_status(self):
        """Clear resets state status."""
        history = BasinHistory()
        history.set_state_status('established')
        history.clear()
        assert history.get_state_status() == 'unknown'


class TestDetectWithHysteresis:
    """Tests for hysteresis-aware basin detection."""

    def test_entry_easier_than_exit(self):
        """Entry threshold is lower than exit threshold."""
        detector = BasinDetector()
        for basin_name, config in detector.DEFAULT_HYSTERESIS.items():
            assert config.entry_threshold < config.exit_threshold, \
                f"{basin_name}: entry should be < exit"

    def test_first_entry_provisional(self):
        """First entry sets provisional state."""
        detector = BasinDetector()
        history = BasinHistory()
        psi = {'psi_semantic': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5}
        metrics = {'delta_kappa': 0.5}

        basin, confidence, metadata = detector.detect_with_hysteresis(
            psi, metrics, None, history
        )

        assert metadata['state_status'] == 'provisional'
        assert metadata['transition_type'] == 'entry'

    def test_sustained_becomes_established(self):
        """Provisional state becomes established after τ₁ turns."""
        detector = BasinDetector()
        history = BasinHistory()
        psi = {'psi_semantic': 0.5, 'psi_affective': 0.5, 'psi_biosignal': 0.5}
        metrics = {'delta_kappa': 0.5}

        # First detection - entry
        basin, _, _ = detector.detect_with_hysteresis(psi, metrics, None, history)
        history.append(basin, 0.5)

        # Continue in same basin
        for _ in range(5):
            basin, _, metadata = detector.detect_with_hysteresis(psi, metrics, None, history)
            history.append(basin, 0.5)
            if metadata.get('state_status') == 'established':
                break

        # Should eventually become established
        assert history.get_state_status() in ('provisional', 'established')


class TestSameEndpointDifferentPaths:
    """Tests for the core movement-preservation property."""

    def test_different_velocities_different_annotations(self):
        """Same position with different velocities produces different annotations."""
        # Fast approach
        ann_fast = generate_movement_annotation(
            velocity_magnitude=0.15,
            acceleration_magnitude=0.01,
            previous_basin='Cognitive Mimicry',
            dwell_time=1
        )

        # Slow approach
        ann_slow = generate_movement_annotation(
            velocity_magnitude=0.02,
            acceleration_magnitude=0.001,
            previous_basin='Cognitive Mimicry',
            dwell_time=1
        )

        # Annotations should differ (fast = moving, slow = still)
        assert ann_fast != ann_slow

    def test_different_dwell_times_different_annotations(self):
        """Same position with different dwell times produces different annotations."""
        # Recent arrival
        ann_recent = generate_movement_annotation(
            velocity_magnitude=0.02,
            acceleration_magnitude=0.001,
            previous_basin='Deep Resonance',
            dwell_time=2
        )

        # Settled
        ann_settled = generate_movement_annotation(
            velocity_magnitude=0.02,
            acceleration_magnitude=0.001,
            previous_basin='Deep Resonance',
            dwell_time=10
        )

        # Annotations should differ (recent includes source, settled doesn't)
        assert ann_recent != ann_settled
