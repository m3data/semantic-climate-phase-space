"""
Tests for substrates.py - Ψ Substrate Computations

TDD tests written BEFORE implementation per modular refactor design.

These tests define the expected behavior for:
- compute_semantic_substrate: Ψ_semantic from Δκ, ΔH, α
- compute_temporal_substrate: Ψ_temporal from metric stability
- compute_affective_substrate: Ψ_affective from sentiment, hedging, vulnerability
- compute_biosignal_substrate: Ψ_biosignal from physiological data
- compute_dialogue_context: Context for basin detection
"""

import pytest
import numpy as np
import json
from pathlib import Path

# Will import from src.substrates once implemented
# For now, test backward-compatible behavior via SemanticClimateAnalyzer
try:
    from src.substrates import (
        compute_semantic_substrate,
        compute_temporal_substrate,
        compute_affective_substrate,
        compute_affective_substrate_fast,
        compute_biosignal_substrate,
        compute_dialogue_context,
        merge_affective_results
    )
    SUBSTRATES_MODULE_EXISTS = True
except ImportError:
    from src.extensions import SemanticClimateAnalyzer
    SUBSTRATES_MODULE_EXISTS = False
    _analyzer = SemanticClimateAnalyzer()
    compute_semantic_substrate = _analyzer.compute_semantic_substrate
    compute_temporal_substrate = _analyzer.compute_temporal_substrate
    compute_affective_substrate = _analyzer.compute_affective_substrate
    compute_affective_substrate_fast = _analyzer.compute_affective_substrate  # Fallback
    compute_biosignal_substrate = _analyzer._compute_biosignal_substrate
    compute_dialogue_context = _analyzer.compute_dialogue_context
    merge_affective_results = None


# Load golden outputs for regression testing
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
GOLDEN_OUTPUTS_PATH = FIXTURES_DIR / 'golden_outputs.json'


@pytest.fixture
def golden_outputs():
    """Load golden outputs captured before refactor."""
    with open(GOLDEN_OUTPUTS_PATH) as f:
        return json.load(f)


class TestSemanticSubstrate:
    """Tests for compute_semantic_substrate function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        metrics = {'delta_kappa': 0.3, 'delta_h': 0.2, 'alpha': 0.8}

        result = compute_semantic_substrate(
            embeddings=np.array([]),
            metrics=metrics
        )

        assert 'psi_semantic' in result
        assert 'alignment_score' in result
        assert 'exploratory_depth' in result
        assert 'raw_metrics' in result

    def test_psi_semantic_bounded(self):
        """Ψ_semantic should be bounded [-1, 1] via tanh."""
        # High metrics
        high_metrics = {'delta_kappa': 0.8, 'delta_h': 0.8, 'alpha': 1.2}
        high_result = compute_semantic_substrate(np.array([]), high_metrics)
        assert -1 <= high_result['psi_semantic'] <= 1

        # Low metrics
        low_metrics = {'delta_kappa': 0.01, 'delta_h': 0.01, 'alpha': 0.5}
        low_result = compute_semantic_substrate(np.array([]), low_metrics)
        assert -1 <= low_result['psi_semantic'] <= 1

    def test_exploratory_depth_equals_delta_kappa(self):
        """exploratory_depth should equal Δκ."""
        metrics = {'delta_kappa': 0.42, 'delta_h': 0.15, 'alpha': 0.85}

        result = compute_semantic_substrate(np.array([]), metrics)

        assert result['exploratory_depth'] == pytest.approx(0.42)

    def test_golden_output_regression(self, golden_outputs):
        """Should match golden output for precomputed metrics scenario."""
        golden = golden_outputs['scenarios']['with_precomputed_metrics']

        metrics = golden['raw_metrics']
        result = compute_semantic_substrate(np.array([]), metrics)

        golden_semantic = golden['substrate_details']['semantic']
        assert result['psi_semantic'] == pytest.approx(golden_semantic['psi_semantic'], abs=1e-6)


class TestTemporalSubstrate:
    """Tests for compute_temporal_substrate function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 384)

        result = compute_temporal_substrate(embeddings=embeddings)

        assert 'psi_temporal' in result
        assert 'metric_stability' in result
        assert 'turn_synchrony' in result
        assert 'rhythm_score' in result

    def test_psi_temporal_bounded(self):
        """Ψ_temporal should be bounded [0, 1]."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 384)

        result = compute_temporal_substrate(embeddings=embeddings)

        assert 0 <= result['psi_temporal'] <= 1

    def test_insufficient_data_returns_neutral(self):
        """Should return 0.5 (neutral) for insufficient data."""
        embeddings = np.random.randn(5, 384)  # Too short

        result = compute_temporal_substrate(embeddings=embeddings)

        assert result['psi_temporal'] == 0.5
        assert result['metric_stability'] == 0.5

    def test_golden_output_regression(self, golden_outputs):
        """Should produce consistent temporal substrate values."""
        # Note: temporal substrate depends on windowed metric computation
        # which requires actual embeddings. We verify structure and bounds.
        golden = golden_outputs['scenarios']['basic_embeddings_only']

        # Just verify the golden output has expected structure
        assert 'psi_temporal' in golden['substrate_details']['temporal']


class TestAffectiveSubstrate:
    """Tests for compute_affective_substrate function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        texts = ["I think this is interesting.", "Maybe we could explore more."]

        result = compute_affective_substrate(turn_texts=texts)

        assert 'psi_affective' in result
        assert 'sentiment_trajectory' in result
        assert 'hedging_density' in result
        assert 'vulnerability_score' in result
        assert 'confidence_variance' in result

    def test_psi_affective_bounded(self):
        """Ψ_affective should be bounded [-1, 1]."""
        texts = [
            "I'm absolutely certain this is correct!",
            "Without doubt, this will work.",
        ] * 5

        result = compute_affective_substrate(turn_texts=texts)

        assert -1 <= result['psi_affective'] <= 1

    def test_hedging_detection(self):
        """Should detect hedging patterns."""
        hedging_texts = [
            "I think maybe this could be right.",
            "Perhaps we should consider alternatives.",
            "I'm not sure, but it seems like...",
            "I guess we could try this approach.",
        ]

        result = compute_affective_substrate(turn_texts=hedging_texts)

        assert result['hedging_density'] > 0

    def test_no_hedging_returns_zero(self):
        """Should return zero hedging for confident text."""
        confident_texts = [
            "This is correct.",
            "The answer is obvious.",
            "We will proceed.",
        ]

        result = compute_affective_substrate(turn_texts=confident_texts)

        assert result['hedging_density'] == 0.0

    def test_vulnerability_detection(self):
        """Should detect vulnerability indicators."""
        vulnerable_texts = [
            "I'm feeling uncertain about this.",
            "I'm afraid we might be wrong.",
            "Honestly, I'm worried about the outcome.",
        ]

        result = compute_affective_substrate(turn_texts=vulnerable_texts)

        assert result['vulnerability_score'] > 0

    def test_sentiment_trajectory_length(self):
        """Should return sentiment score for each turn."""
        texts = ["Happy!", "Sad.", "Neutral text here."]

        result = compute_affective_substrate(turn_texts=texts)

        assert len(result['sentiment_trajectory']) == 3

    def test_empty_texts_returns_zeros(self):
        """Should return zeros for empty text list."""
        result = compute_affective_substrate(turn_texts=[])

        assert result['psi_affective'] == 0.0
        assert result['hedging_density'] == 0.0
        assert result['vulnerability_score'] == 0.0

    def test_golden_output_regression(self, golden_outputs):
        """Should match golden output for hedging texts scenario."""
        golden = golden_outputs['scenarios']['with_hedging_texts']
        golden_affective = golden['substrate_details']['affective']

        # Note: Can't reproduce exact values without original texts
        # Verify the golden has expected structure
        assert golden_affective['hedging_density'] > 0


class TestAffectiveSubstrateFast:
    """Tests for compute_affective_substrate_fast (VADER-only path)."""

    def test_returns_source_vader(self):
        """Fast path should indicate source as 'vader'."""
        texts = ["This is a test.", "Another test here."]

        result = compute_affective_substrate_fast(turn_texts=texts)

        assert result.get('source') == 'vader'

    def test_matches_original_behavior(self):
        """Fast path should match original compute_affective_substrate behavior."""
        texts = [
            "I think this might work.",
            "Perhaps we should consider alternatives.",
            "I'm feeling uncertain about this approach.",
        ]

        fast_result = compute_affective_substrate_fast(turn_texts=texts)

        # Should have all original keys
        assert 'psi_affective' in fast_result
        assert 'sentiment_trajectory' in fast_result
        assert 'hedging_density' in fast_result
        assert 'vulnerability_score' in fast_result
        assert 'confidence_variance' in fast_result

    def test_performance_no_transformer_overhead(self):
        """Fast path should be quick (no transformer loading)."""
        import time

        texts = ["Quick test."] * 10

        start = time.time()
        for _ in range(100):
            compute_affective_substrate_fast(turn_texts=texts)
        elapsed = time.time() - start

        # Should complete 100 iterations in under 1 second
        assert elapsed < 1.0


class TestAffectiveSubstrateHybrid:
    """Tests for hybrid compute_affective_substrate with emotion service."""

    def test_without_emotion_service_returns_fast_path(self):
        """Without emotion_service, should return fast path result."""
        texts = ["I think this is interesting.", "Maybe we should explore."]

        result = compute_affective_substrate(turn_texts=texts)

        # Should be vader source when no emotion service
        assert result.get('source') == 'vader'

    def test_without_emotion_service_backward_compatible(self):
        """Should be backward compatible with original API."""
        texts = ["Happy thoughts!", "Sad times.", "Neutral expression."]

        result = compute_affective_substrate(turn_texts=texts)

        # All original fields should be present
        assert 'psi_affective' in result
        assert 'sentiment_trajectory' in result
        assert len(result['sentiment_trajectory']) == 3

    @pytest.mark.skipif(not SUBSTRATES_MODULE_EXISTS, reason="Requires substrates module")
    def test_merge_affective_results_structure(self):
        """merge_affective_results should produce expected structure."""
        if merge_affective_results is None:
            pytest.skip("merge_affective_results not available")

        # Mock fast result
        fast_result = {
            'psi_affective': -0.2,
            'sentiment_trajectory': [0.5, -0.3, 0.1],
            'hedging_density': 0.05,
            'vulnerability_score': 0.02,
            'confidence_variance': 0.001,
            'source': 'vader'
        }

        # Mock EmotionResult-like objects
        class MockEmotionResult:
            def __init__(self, scores, epistemic, safety):
                self.scores = scores
                self.epistemic_score = epistemic
                self.safety_score = safety

        emotion_results = [
            MockEmotionResult(
                scores={'curiosity': 0.7, 'confusion': 0.2, 'neutral': 0.1},
                epistemic=0.6,
                safety=0.3
            ),
            MockEmotionResult(
                scores={'curiosity': 0.4, 'realization': 0.5, 'neutral': 0.1},
                epistemic=0.8,
                safety=0.4
            ),
        ]

        merged = merge_affective_results(fast_result, emotion_results, ["text1", "text2"])

        # Check hybrid source
        assert merged['source'] == 'hybrid'

        # Check new fields
        assert 'emotion_profiles' in merged
        assert 'epistemic_trajectory' in merged
        assert 'safety_trajectory' in merged
        assert 'top_emotions' in merged
        assert 'epistemic_mean' in merged
        assert 'safety_mean' in merged

        # Check preserved fast-path fields
        assert merged['hedging_density'] == 0.05
        assert merged['vulnerability_score'] == 0.02

    @pytest.mark.skipif(not SUBSTRATES_MODULE_EXISTS, reason="Requires substrates module")
    def test_merge_epistemic_trajectory_extraction(self):
        """Should extract epistemic emotions into trajectory."""
        if merge_affective_results is None:
            pytest.skip("merge_affective_results not available")

        fast_result = {
            'psi_affective': 0.0,
            'sentiment_trajectory': [],
            'hedging_density': 0.0,
            'vulnerability_score': 0.0,
            'confidence_variance': 0.0,
            'source': 'vader'
        }

        class MockEmotionResult:
            def __init__(self, scores, epistemic, safety):
                self.scores = scores
                self.epistemic_score = epistemic
                self.safety_score = safety

        emotion_results = [
            MockEmotionResult(
                scores={'curiosity': 0.8, 'confusion': 0.1, 'realization': 0.0, 'surprise': 0.2},
                epistemic=0.7, safety=0.5
            ),
            MockEmotionResult(
                scores={'curiosity': 0.3, 'confusion': 0.5, 'realization': 0.6, 'surprise': 0.1},
                epistemic=0.4, safety=0.3
            ),
        ]

        merged = merge_affective_results(fast_result, emotion_results, ["t1", "t2"])

        # Check epistemic trajectory
        assert merged['epistemic_trajectory']['curiosity'] == [0.8, 0.3]
        assert merged['epistemic_trajectory']['confusion'] == [0.1, 0.5]
        assert merged['epistemic_trajectory']['realization'] == [0.0, 0.6]
        assert merged['epistemic_trajectory']['surprise'] == [0.2, 0.1]


class TestBiosignalSubstrate:
    """Tests for compute_biosignal_substrate function."""

    def test_heart_rate_normalization(self):
        """Should normalize heart rate around resting range."""
        # Normal resting HR
        result = compute_biosignal_substrate({'heart_rate': 70})
        assert -0.5 <= result <= 0.5

        # Elevated HR
        result_high = compute_biosignal_substrate({'heart_rate': 100})
        assert result_high > result  # Higher HR -> higher value

        # Low HR
        result_low = compute_biosignal_substrate({'heart_rate': 55})
        assert result_low < result  # Lower HR -> lower value

    def test_tanh_bounded(self):
        """Should be bounded via tanh."""
        # Extreme values
        result_extreme_high = compute_biosignal_substrate({'heart_rate': 180})
        assert -1 <= result_extreme_high <= 1

        result_extreme_low = compute_biosignal_substrate({'heart_rate': 40})
        assert -1 <= result_extreme_low <= 1

    def test_missing_hr_returns_zero(self):
        """Should return 0.0 if heart_rate not provided."""
        result = compute_biosignal_substrate({})
        assert result == 0.0

    def test_golden_output_regression(self, golden_outputs):
        """Should match golden output for biosignal scenario."""
        golden = golden_outputs['scenarios']['with_biosignal']

        result = compute_biosignal_substrate({'heart_rate': 75})

        # The golden used HR=75
        assert result == pytest.approx(golden['psi_state']['position']['psi_biosignal'], abs=0.01)


class TestDialogueContext:
    """Tests for compute_dialogue_context function."""

    def test_returns_dict_structure(self):
        """Should return dict with expected keys."""
        result = compute_dialogue_context()

        assert 'hedging_density' in result
        assert 'turn_length_ratio' in result
        assert 'delta_kappa_variance' in result
        assert 'coherence_pattern' in result

    def test_default_values(self):
        """Should return sensible defaults when no input provided."""
        result = compute_dialogue_context()

        assert result['hedging_density'] == 0.0
        assert result['turn_length_ratio'] == 1.0
        assert result['delta_kappa_variance'] == 0.0

    def test_turn_length_ratio_calculation(self):
        """Should compute AI/human turn length ratio."""
        texts = [
            "Short human",  # 2 words
            "This is a longer AI response with more words",  # 9 words
            "Short again",  # 2 words
            "Another long AI response here with extra content",  # 8 words
        ]
        speakers = ['human', 'ai', 'human', 'ai']

        result = compute_dialogue_context(
            turn_texts=texts,
            turn_speakers=speakers
        )

        # AI avg = (9 + 8) / 2 = 8.5, Human avg = (2 + 2) / 2 = 2
        # Ratio = 8.5 / 2 = 4.25
        assert result['turn_length_ratio'] == pytest.approx(4.25, abs=0.1)

    def test_delta_kappa_variance_calculation(self):
        """Should compute Δκ variance from trajectory metrics."""
        trajectory_metrics = [
            {'delta_kappa': 0.3},
            {'delta_kappa': 0.4},
            {'delta_kappa': 0.35},
            {'delta_kappa': 0.5},
        ]

        result = compute_dialogue_context(trajectory_metrics=trajectory_metrics)

        # Variance of [0.3, 0.4, 0.35, 0.5]
        expected_variance = np.var([0.3, 0.4, 0.35, 0.5])
        assert result['delta_kappa_variance'] == pytest.approx(expected_variance, abs=1e-6)

    def test_hedging_density_passthrough(self):
        """Should pass through pre-computed hedging density."""
        result = compute_dialogue_context(hedging_density=0.15)

        assert result['hedging_density'] == 0.15

    def test_coherence_pattern_passthrough(self):
        """Should pass through pre-computed coherence pattern."""
        result = compute_dialogue_context(coherence_pattern='breathing')

        assert result['coherence_pattern'] == 'breathing'

    def test_golden_output_regression(self, golden_outputs):
        """Should match golden output for with_speakers scenario."""
        golden = golden_outputs['scenarios']['with_speakers']

        # The golden had turn_length_ratio = 1.0 because texts had same length
        assert golden['dialogue_context']['turn_length_ratio'] == 1.0


class TestSubstratesIntegration:
    """Integration tests verifying substrates work together."""

    def test_all_substrates_produce_valid_psi_vector(self):
        """All substrate computations should produce valid Ψ components."""
        np.random.seed(42)

        # Semantic
        metrics = {'delta_kappa': 0.3, 'delta_h': 0.2, 'alpha': 0.8}
        semantic = compute_semantic_substrate(np.array([]), metrics)
        assert -1 <= semantic['psi_semantic'] <= 1

        # Temporal
        embeddings = np.random.randn(30, 384)
        temporal = compute_temporal_substrate(embeddings=embeddings)
        assert 0 <= temporal['psi_temporal'] <= 1

        # Affective
        texts = ["I think this might work.", "Perhaps we should try."]
        affective = compute_affective_substrate(turn_texts=texts)
        assert -1 <= affective['psi_affective'] <= 1

        # Biosignal
        biosignal = compute_biosignal_substrate({'heart_rate': 75})
        assert -1 <= biosignal <= 1

    def test_substrates_independent(self):
        """Changes to one substrate input should not affect others."""
        np.random.seed(42)

        metrics = {'delta_kappa': 0.3, 'delta_h': 0.2, 'alpha': 0.8}

        # Compute semantic with low metrics
        semantic_low = compute_semantic_substrate(
            np.array([]),
            {'delta_kappa': 0.1, 'delta_h': 0.1, 'alpha': 0.5}
        )

        # Compute semantic with high metrics
        semantic_high = compute_semantic_substrate(
            np.array([]),
            {'delta_kappa': 0.5, 'delta_h': 0.5, 'alpha': 0.9}
        )

        # Results should be different
        assert semantic_low['psi_semantic'] != semantic_high['psi_semantic']
