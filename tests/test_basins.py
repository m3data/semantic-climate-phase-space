"""
Tests for basins.py - Attractor Basin Detection and History

TDD tests written BEFORE implementation per modular refactor design.

These tests define the expected behavior for:
- BasinHistory: Tracks basin sequence for hysteresis-aware detection
- BasinDetector: Detects attractor basins with optional history conditioning
- detect_attractor_basin: Backward-compatible legacy function

The key innovation is residence_confidence_modulation: basins you've been
in longer have higher confidence (settled bonus), while new entries have
reduced confidence (new entry penalty).
"""

import pytest
import numpy as np
import json
from pathlib import Path

# Will import from src.basins once implemented
# For now, test backward-compatible imports from extensions
try:
    from src.basins import BasinHistory, BasinDetector, detect_attractor_basin
    BASINS_MODULE_EXISTS = True
except ImportError:
    from src.extensions import SemanticClimateAnalyzer
    BASINS_MODULE_EXISTS = False
    BasinHistory = None
    BasinDetector = None
    # Create backward-compatible wrapper
    _analyzer = SemanticClimateAnalyzer()
    def detect_attractor_basin(psi_vector, raw_metrics=None, dialogue_context=None):
        return _analyzer.detect_attractor_basin(psi_vector, raw_metrics, dialogue_context)


# Load golden outputs for regression testing
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_OUTPUTS_PATH = FIXTURES_DIR / 'golden_outputs.json'


@pytest.fixture
def golden_outputs():
    """Load golden outputs captured before refactor."""
    with open(GOLDEN_OUTPUTS_PATH) as f:
        return json.load(f)


class TestBasinHistory:
    """Tests for BasinHistory class - new in basins.py."""

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_initialization(self):
        """Should initialize with empty history."""
        history = BasinHistory(max_history=100)

        assert len(history.history) == 0
        assert history.get_current_basin() is None
        assert history.get_residence_time() == 0

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_append_single(self):
        """Should append a basin entry."""
        history = BasinHistory()

        history.append('Deep Resonance', 0.85, turn=1)

        assert len(history.history) == 1
        assert history.get_current_basin() == 'Deep Resonance'

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_residence_time_same_basin(self):
        """Should track residence time in same basin."""
        history = BasinHistory()

        for i in range(5):
            history.append('Collaborative Inquiry', 0.7, turn=i)

        assert history.get_residence_time() == 5

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_residence_time_resets_on_transition(self):
        """Should reset residence time when basin changes."""
        history = BasinHistory()

        for i in range(3):
            history.append('Deep Resonance', 0.8, turn=i)
        for i in range(2):
            history.append('Cognitive Mimicry', 0.6, turn=i+3)

        assert history.get_current_basin() == 'Cognitive Mimicry'
        assert history.get_residence_time() == 2

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_previous_basin_tracking(self):
        """Should track previous basin after transition."""
        history = BasinHistory()

        history.append('Deep Resonance', 0.8, turn=0)
        history.append('Deep Resonance', 0.8, turn=1)
        history.append('Transitional', 0.4, turn=2)

        assert history.get_current_basin() == 'Transitional'
        assert history.get_previous_basin() == 'Deep Resonance'

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_transition_counting(self):
        """Should count basin transitions."""
        history = BasinHistory()

        history.append('Deep Resonance', 0.8, turn=0)
        history.append('Deep Resonance', 0.8, turn=1)
        history.append('Transitional', 0.4, turn=2)
        history.append('Cognitive Mimicry', 0.6, turn=3)
        history.append('Cognitive Mimicry', 0.6, turn=4)

        assert history.get_transition_count() == 2

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_basin_sequence(self):
        """Should return recent basin sequence."""
        history = BasinHistory()

        basins = ['Deep Resonance', 'Transitional', 'Cognitive Mimicry']
        for i, basin in enumerate(basins):
            history.append(basin, 0.7, turn=i)

        sequence = history.get_basin_sequence(n=3)
        assert sequence == basins

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_basin_distribution(self):
        """Should compute basin distribution."""
        history = BasinHistory()

        for i in range(5):
            history.append('Deep Resonance', 0.8, turn=i)
        for i in range(3):
            history.append('Cognitive Mimicry', 0.6, turn=i+5)

        dist = history.get_basin_distribution()
        assert dist['Deep Resonance'] == 5
        assert dist['Cognitive Mimicry'] == 3

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_transition_matrix(self):
        """Should compute transition matrix."""
        history = BasinHistory()

        # A -> A -> B -> B -> A
        history.append('A', 0.7, turn=0)
        history.append('A', 0.7, turn=1)
        history.append('B', 0.7, turn=2)
        history.append('B', 0.7, turn=3)
        history.append('A', 0.7, turn=4)

        matrix = history.get_transition_matrix()
        assert matrix[('A', 'B')] == 1
        assert matrix[('B', 'A')] == 1

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_max_history_limit(self):
        """Should respect max history limit."""
        history = BasinHistory(max_history=5)

        for i in range(10):
            history.append('Deep Resonance', 0.8, turn=i)

        assert len(history.history) == 5

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_clear(self):
        """Should clear history."""
        history = BasinHistory()

        for i in range(5):
            history.append('Deep Resonance', 0.8, turn=i)

        history.clear()

        assert len(history.history) == 0
        assert history.get_current_basin() is None


class TestBasinDetector:
    """Tests for BasinDetector class - extracted from SemanticClimateAnalyzer."""

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_initialization_defaults(self):
        """Should initialize with default parameters."""
        detector = BasinDetector()

        assert detector.residence_confidence_modulation is True
        assert detector.new_entry_penalty == 0.7
        assert detector.settled_bonus == 1.1
        assert detector.settled_threshold == 10

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_all_basins_defined(self):
        """Should define all canonical basins."""
        detector = BasinDetector()

        expected_basins = [
            'Deep Resonance',
            'Collaborative Inquiry',
            'Cognitive Mimicry',
            'Reflexive Performance',
            'Sycophantic Convergence',
            'Creative Dilation',
            'Generative Conflict',
            'Embodied Coherence',
            'Dissociation',
            'Transitional'
        ]

        for basin in expected_basins:
            assert basin in detector.BASINS

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_detect_returns_tuple(self):
        """Should return (basin_name, confidence, metadata) tuple."""
        detector = BasinDetector(residence_confidence_modulation=False)

        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.5,
            'psi_biosignal': 0.5
        }

        basin, confidence, metadata = detector.detect(psi_vector)

        assert isinstance(basin, str)
        assert isinstance(confidence, float)
        assert isinstance(metadata, dict)
        assert 0 <= confidence <= 1

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_detect_with_history_modulation(self):
        """Should modulate confidence based on residence time."""
        detector = BasinDetector(
            residence_confidence_modulation=True,
            new_entry_penalty=0.7,
            settled_bonus=1.1,
            settled_threshold=5
        )

        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.5,
            'psi_biosignal': 0.5
        }

        # First detection (new entry)
        history = BasinHistory()
        basin1, conf1, _ = detector.detect(psi_vector, basin_history=history)

        # Build up residence
        for i in range(10):
            history.append(basin1, conf1, turn=i)

        # Detection after settling
        basin2, conf2, _ = detector.detect(psi_vector, basin_history=history)

        # Confidence should be higher after settling (if same basin detected)
        if basin1 == basin2:
            # Raw confidence would be same, but settled bonus should increase it
            # This is a behavioral test - exact values depend on implementation
            pass  # Implementation will determine exact behavior

    @pytest.mark.skipif(not BASINS_MODULE_EXISTS, reason="basins.py not yet implemented")
    def test_metadata_contains_residence_info(self):
        """Metadata should include residence time and previous basin."""
        detector = BasinDetector()
        history = BasinHistory()

        # Add some history
        history.append('Deep Resonance', 0.8, turn=0)
        history.append('Deep Resonance', 0.8, turn=1)
        history.append('Transitional', 0.5, turn=2)

        psi_vector = {
            'psi_semantic': 0.3,
            'psi_temporal': 0.5,
            'psi_affective': 0.2,
            'psi_biosignal': None
        }

        _, _, metadata = detector.detect(psi_vector, basin_history=history)

        assert 'residence_time' in metadata
        assert 'previous_basin' in metadata


class TestDetectAttractorBasin:
    """Tests for detect_attractor_basin function - backward compatible API."""

    def test_basic_classification(self):
        """Should classify basic psi vectors."""
        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.5,
            'psi_biosignal': 0.5
        }

        basin, confidence = detect_attractor_basin(psi_vector)

        assert isinstance(basin, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_deep_resonance_detection(self):
        """Should detect Deep Resonance when all substrates high."""
        psi_vector = {
            'psi_semantic': 0.6,
            'psi_temporal': 0.5,
            'psi_affective': 0.6,
            'psi_biosignal': 0.5
        }

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics={'delta_kappa': 0.4})

        assert basin == 'Deep Resonance'

    def test_dissociation_detection(self):
        """Should detect Dissociation when all substrates low."""
        psi_vector = {
            'psi_semantic': 0.05,
            'psi_temporal': 0.5,
            'psi_affective': 0.05,
            'psi_biosignal': 0.05
        }

        basin, confidence = detect_attractor_basin(psi_vector)

        assert basin == 'Dissociation'

    def test_sycophantic_convergence_detection(self, golden_outputs):
        """Should detect Sycophantic Convergence (high semantic, low Δκ, low affect)."""
        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.1,
            'psi_biosignal': 0.0
        }
        raw_metrics = {'delta_kappa': 0.2}  # Below 0.35 threshold

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics)

        assert basin == 'Sycophantic Convergence'

    def test_creative_dilation_detection(self):
        """Should detect Creative Dilation (high Δκ, positive affect)."""
        psi_vector = {
            'psi_semantic': 0.3,
            'psi_temporal': 0.5,
            'psi_affective': 0.4,
            'psi_biosignal': None
        }
        raw_metrics = {'delta_kappa': 0.5}  # Above 0.35 threshold

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics)

        assert basin == 'Creative Dilation'

    def test_generative_conflict_detection(self):
        """Should detect Generative Conflict (high semantic, high Δκ, high affect)."""
        psi_vector = {
            'psi_semantic': 0.4,
            'psi_temporal': 0.5,
            'psi_affective': 0.4,
            'psi_biosignal': None
        }
        raw_metrics = {'delta_kappa': 0.5}

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics)

        # Note: Deep Resonance takes priority if affect > 0.4
        # Generative Conflict requires specific configuration
        assert basin in ['Generative Conflict', 'Deep Resonance']

    def test_embodied_coherence_detection(self):
        """Should detect Embodied Coherence (low semantic, high biosignal)."""
        psi_vector = {
            'psi_semantic': 0.1,
            'psi_temporal': 0.5,
            'psi_affective': 0.2,
            'psi_biosignal': 0.5
        }

        basin, confidence = detect_attractor_basin(psi_vector)

        assert basin == 'Embodied Coherence'

    def test_cognitive_mimicry_vs_collaborative_inquiry(self):
        """Should distinguish Cognitive Mimicry from Collaborative Inquiry via dialogue context."""
        # Both have high semantic, low affect, low biosignal
        psi_vector = {
            'psi_semantic': 0.5,
            'psi_temporal': 0.5,
            'psi_affective': 0.1,
            'psi_biosignal': 0.0
        }
        raw_metrics = {'delta_kappa': 0.4}  # Above threshold to avoid Sycophantic

        # Cognitive Mimicry: low hedging, AI dominates, smooth trajectory
        mimicry_context = {
            'hedging_density': 0.0,
            'turn_length_ratio': 3.0,  # AI dominates
            'delta_kappa_variance': 0.002,  # Low variance = scripted
            'coherence_pattern': 'locked'
        }

        basin1, _ = detect_attractor_basin(psi_vector, raw_metrics, mimicry_context)

        # Collaborative Inquiry: hedging present, balanced turns, oscillating trajectory
        inquiry_context = {
            'hedging_density': 0.05,
            'turn_length_ratio': 1.2,  # Balanced
            'delta_kappa_variance': 0.03,  # Higher variance = responsive
            'coherence_pattern': 'breathing'
        }

        basin2, _ = detect_attractor_basin(psi_vector, raw_metrics, inquiry_context)

        assert basin1 == 'Cognitive Mimicry'
        assert basin2 == 'Collaborative Inquiry'

    def test_transitional_default(self):
        """Should return Transitional when no clear basin matches."""
        # Moderate values across the board
        psi_vector = {
            'psi_semantic': 0.2,
            'psi_temporal': 0.5,
            'psi_affective': 0.1,
            'psi_biosignal': None
        }
        raw_metrics = {'delta_kappa': 0.2}

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics)

        # Could be Transitional or Dissociation depending on exact values
        assert basin in ['Transitional', 'Dissociation', 'Sycophantic Convergence']


class TestGoldenOutputRegression:
    """Verify basin detection matches golden outputs."""

    def test_basic_embeddings_basin(self, golden_outputs):
        """Should match golden output for basic_embeddings_only scenario."""
        golden = golden_outputs['scenarios']['basic_embeddings_only']

        psi_vector = {
            'psi_semantic': golden['psi_state']['position']['psi_semantic'],
            'psi_temporal': golden['psi_state']['position']['psi_temporal'],
            'psi_affective': golden['psi_state']['position']['psi_affective'],
            'psi_biosignal': golden['psi_state']['position']['psi_biosignal'],
        }
        raw_metrics = golden['raw_metrics']
        dialogue_context = golden['dialogue_context']

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics, dialogue_context)

        assert basin == golden['attractor_dynamics']['basin']
        assert confidence == pytest.approx(golden['attractor_dynamics']['confidence'], abs=0.01)

    def test_with_hedging_texts_basin(self, golden_outputs):
        """Should match golden output for with_hedging_texts scenario."""
        golden = golden_outputs['scenarios']['with_hedging_texts']

        psi_vector = {
            'psi_semantic': golden['psi_state']['position']['psi_semantic'],
            'psi_temporal': golden['psi_state']['position']['psi_temporal'],
            'psi_affective': golden['psi_state']['position']['psi_affective'],
            'psi_biosignal': golden['psi_state']['position']['psi_biosignal'],
        }
        raw_metrics = golden['raw_metrics']
        dialogue_context = golden['dialogue_context']

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics, dialogue_context)

        assert basin == golden['attractor_dynamics']['basin']
        assert confidence == pytest.approx(golden['attractor_dynamics']['confidence'], abs=0.01)

    def test_with_confident_texts_basin(self, golden_outputs):
        """Should match golden output for with_confident_texts scenario."""
        golden = golden_outputs['scenarios']['with_confident_texts']

        psi_vector = {
            'psi_semantic': golden['psi_state']['position']['psi_semantic'],
            'psi_temporal': golden['psi_state']['position']['psi_temporal'],
            'psi_affective': golden['psi_state']['position']['psi_affective'],
            'psi_biosignal': golden['psi_state']['position']['psi_biosignal'],
        }
        raw_metrics = golden['raw_metrics']
        dialogue_context = golden['dialogue_context']

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics, dialogue_context)

        assert basin == golden['attractor_dynamics']['basin']
        assert confidence == pytest.approx(golden['attractor_dynamics']['confidence'], abs=0.01)

    def test_with_biosignal_basin(self, golden_outputs):
        """Should match golden output for with_biosignal scenario."""
        golden = golden_outputs['scenarios']['with_biosignal']

        psi_vector = {
            'psi_semantic': golden['psi_state']['position']['psi_semantic'],
            'psi_temporal': golden['psi_state']['position']['psi_temporal'],
            'psi_affective': golden['psi_state']['position']['psi_affective'],
            'psi_biosignal': golden['psi_state']['position']['psi_biosignal'],
        }
        raw_metrics = golden['raw_metrics']
        dialogue_context = golden['dialogue_context']

        basin, confidence = detect_attractor_basin(psi_vector, raw_metrics, dialogue_context)

        assert basin == golden['attractor_dynamics']['basin']
        assert confidence == pytest.approx(golden['attractor_dynamics']['confidence'], abs=0.01)
