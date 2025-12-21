"""
Semantic complexity metrics calculation service.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Implements trajectory-aware coupling mode detection that captures
the movement of meaning in human-AI dialogue, distinguishing between
progressive and regressive dynamics.

This module wraps the clean src/ implementation for web app use.

Updated 2025-12-21: Added GoEmotions-based emotion analysis via EmotionService.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Import from clean src/ package (Phase 4 port)
# Add parent directory to path for src imports when running from backend/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src import SemanticClimateAnalyzer
import numpy as np

# Import EmotionService - handle both package and direct execution contexts
try:
    from .emotion_service import EmotionService
except ImportError:
    from emotion_service import EmotionService


@dataclass
class SemanticCoherence:
    """
    Measures how meaning moves through the dialogue.

    Based on autocorrelation of semantic shifts:
    - Negative autocorr (-1 to -0.3): Healthy alternation (breathing rhythm)
    - Near-zero (-0.3 to 0.3): Transitional or unstable
    - Positive (0.3 to 1): Lock-in / repetitive patterns

    Combined with variance to detect fragmentation vs coherent exploration.
    """
    autocorrelation: float       # Lag-1 autocorr of semantic shifts
    variance: float              # Variance in shift magnitudes
    pattern: str                 # 'breathing', 'transitional', 'locked', 'fragmented'
    coherence_score: float       # Normalized score [0, 1] where 1 = maximally coherent


@dataclass
class CouplingMode:
    """
    Represents a coupling mode with trajectory information.

    Movement of meaning matters - a mode captures both position in
    metric space AND direction of travel.
    """
    mode: str                    # Primary mode classification
    trajectory: str              # 'warming', 'cooling', 'stable', 'oscillating'
    compound_label: str          # e.g., 'Generative-Warming', 'Sycophantic-Regressive'
    epistemic_risk: str          # 'low', 'moderate', 'high', 'critical'
    risk_factors: list           # List of specific risk indicators
    confidence: float            # How cleanly this fits the classification
    coherence: Optional[SemanticCoherence] = None  # Semantic coherence analysis


class MetricsService:
    """Wrapper around SemanticClimateAnalyzer for web app."""

    def __init__(self, enable_goemotions: bool = True, debug_timing: bool = False):
        """
        Initialize metrics service.

        Args:
            enable_goemotions: If True, loads GoEmotions model for rich
                emotion analysis. Model loads lazily on first use.
                Set False for faster startup / lower memory.
            debug_timing: If True, prints timing information for performance debugging.
        """
        self.debug_timing = debug_timing

        # Initialize emotion service (lazy load to avoid slow startup)
        if enable_goemotions:
            self.emotion_service = EmotionService(lazy_load=True)
        else:
            self.emotion_service = None

        self.analyzer = SemanticClimateAnalyzer(
            emotion_service=self.emotion_service,
            debug_timing=debug_timing
        )

    def analyze(
        self,
        embeddings: list,
        turn_texts: list = None,
        turn_speakers: list = None,
        previous_metrics: dict = None,
        semantic_shifts: list = None,
        biosignal_data: dict = None,
        trajectory_metrics: list = None,
        coherence_pattern: str = None
    ) -> dict:
        """
        Analyze dialogue embeddings and return climate data with vector Psi.

        Args:
            embeddings: List of embedding vectors
            turn_texts: Optional list of turn text strings (for affective substrate)
            turn_speakers: Optional list of speaker labels ('human', 'ai') for turn asymmetry
            previous_metrics: Optional dict with previous dk, dh, alpha for trajectory calc
            semantic_shifts: Optional list of turn-to-turn semantic shift values
            biosignal_data: Optional dict with heart_rate for biosignal substrate
            trajectory_metrics: Optional list of per-window metrics dicts for Δκ variance
            coherence_pattern: Optional pre-computed coherence pattern string

        Returns:
            {
                'metrics': {'delta_kappa': ..., 'alpha': ..., 'delta_h': ...},
                'climate': {'temperature': ..., 'humidity': ..., 'pressure': ...},
                'mode': 'Resonant' | 'Generative' | 'Contemplative' | ...,
                'coupling_mode': {
                    'mode': str,
                    'trajectory': str,
                    'compound_label': str,
                    'epistemic_risk': str,
                    'risk_factors': list,
                    'confidence': float,
                    'coherence': {...}  # Semantic coherence analysis
                },
                'complexity_detected': bool,
                'psi_vector': {'semantic': ..., 'temporal': ..., 'affective': ..., 'biosignal': ...},
                'attractor_basin': {'name': ..., 'confidence': ...},
                'affective_substrate': {...} or None
            }
        """
        if self.debug_timing:
            import time
            _t0 = time.time()

        # Use Morgoulis's analyzer (via SemanticClimateAnalyzer which extends it)
        results = self.analyzer.calculate_all_metrics(embeddings)
        if self.debug_timing:
            print(f"  [TIMING] core_metrics: {int((time.time()-_t0)*1000)}ms", flush=True)

        # Interpret as climate
        climate = {
            'temperature': float(results['delta_kappa']),
            'humidity': float(results['delta_h']),
            'pressure': float(results['alpha']),
            'temp_level': self._classify_level(results['delta_kappa'], [0.20, 0.35]),
            'humidity_level': self._classify_level(results['delta_h'], [0.08, 0.12]),
            'pressure_level': self._classify_pressure(results['alpha'])
        }

        # Calculate trends if we have previous metrics
        dk_trend = 0.0
        dh_trend = 0.0
        alpha_trend = 0.0

        if previous_metrics:
            dk_trend = results['delta_kappa'] - previous_metrics.get('delta_kappa', results['delta_kappa'])
            dh_trend = results['delta_h'] - previous_metrics.get('delta_h', results['delta_h'])
            alpha_trend = results['alpha'] - previous_metrics.get('alpha', results['alpha'])

        # Compute semantic coherence if shifts provided
        coherence = None
        if semantic_shifts and len(semantic_shifts) >= 4:
            coherence = self.compute_semantic_coherence(semantic_shifts)

        # Detect trajectory-aware coupling mode (with coherence)
        coupling_mode = self.detect_coupling_mode(
            results['delta_kappa'],
            results['delta_h'],
            results['alpha'],
            dk_trend,
            dh_trend,
            alpha_trend,
            coherence
        )

        # Legacy mode field for backward compatibility
        mode = coupling_mode.mode

        # Calculate vector Psi (Coupling Coefficient)
        psi_vector = None
        attractor_basin = None
        affective_substrate = None
        psi_composite = None

        # Determine coherence pattern for basin detection
        # Use provided pattern, or derive from computed coherence
        effective_coherence_pattern = coherence_pattern
        if effective_coherence_pattern is None and coherence is not None:
            effective_coherence_pattern = coherence.pattern

        try:
            if self.debug_timing:
                import time
                _t2 = time.time()
            psi_result = self.analyzer.compute_coupling_coefficient(
                dialogue_embeddings=embeddings,
                turn_texts=turn_texts,
                turn_speakers=turn_speakers,
                metrics=results,
                trajectory_metrics=trajectory_metrics,
                biosignal_data=biosignal_data,
                coherence_pattern=effective_coherence_pattern
            )
            if self.debug_timing:
                print(f"  [TIMING] coupling_coeff: {int((time.time()-_t2)*1000)}ms (turns={len(turn_texts) if turn_texts else 0})", flush=True)

            # Extract vector components
            psi_position = psi_result['psi_state']['position']
            psi_vector = {
                'semantic': float(psi_position['psi_semantic']),
                'temporal': float(psi_position['psi_temporal']),
                'affective': float(psi_position['psi_affective']) if psi_position['psi_affective'] is not None else None,
                'biosignal': psi_position['psi_biosignal']
            }

            # Extract attractor basin
            attractor_basin = {
                'name': psi_result['attractor_dynamics']['basin'],
                'confidence': float(psi_result['attractor_dynamics']['confidence'])
            }

            # Extract affective substrate details if available
            if 'affective' in psi_result['substrate_details'] and psi_result['substrate_details']['affective'] is not None:
                aff = psi_result['substrate_details']['affective']
                affective_substrate = {
                    'hedging_density': float(aff['hedging_density']),
                    'vulnerability_score': float(aff['vulnerability_score']),
                    'confidence_variance': float(aff['confidence_variance']),
                    'sentiment_trajectory_length': len(aff['sentiment_trajectory']),
                    'source': aff.get('source', 'vader')
                }
                # Include GoEmotions data if available (hybrid mode)
                if aff.get('source') == 'hybrid':
                    affective_substrate['epistemic_trajectory'] = aff.get('epistemic_trajectory')
                    affective_substrate['safety_trajectory'] = aff.get('safety_trajectory')
                    affective_substrate['top_emotions'] = aff.get('top_emotions')
                    affective_substrate['epistemic_mean'] = aff.get('epistemic_mean')
                    affective_substrate['safety_mean'] = aff.get('safety_mean')

            psi_composite = float(psi_result['psi'])

            # Extract movement-preserving classification data (v0.3.0+)
            movement_aware_label = psi_result.get('movement_aware_label')
            trajectory_state_vector = psi_result.get('trajectory_state_vector')
            soft_state_inference = psi_result.get('soft_state_inference')

        except Exception as e:
            print(f"[ERROR] Failed to calculate vector Psi: {e}")
            import traceback
            traceback.print_exc()
            movement_aware_label = None
            trajectory_state_vector = None
            soft_state_inference = None

        # Build coherence dict for response
        coherence_data = None
        if coupling_mode.coherence:
            coherence_data = {
                'autocorrelation': coupling_mode.coherence.autocorrelation,
                'variance': coupling_mode.coherence.variance,
                'pattern': coupling_mode.coherence.pattern,
                'coherence_score': coupling_mode.coherence.coherence_score
            }

        response = {
            'metrics': {
                'delta_kappa': float(results['delta_kappa']),
                'alpha': float(results['alpha']),
                'delta_h': float(results['delta_h']),
                'confidence_intervals': {
                    'delta_kappa': [float(x) for x in results['delta_kappa_ci']],
                    'alpha': [float(x) for x in results['alpha_ci']],
                    'delta_h': [float(x) for x in results['delta_h_ci']]
                },
                'trends': {
                    'delta_kappa_trend': dk_trend,
                    'delta_h_trend': dh_trend,
                    'alpha_trend': alpha_trend
                }
            },
            'climate': climate,
            'mode': mode,  # Legacy: base mode for backward compatibility
            'coupling_mode': {
                'mode': coupling_mode.mode,
                'trajectory': coupling_mode.trajectory,
                'compound_label': coupling_mode.compound_label,
                'epistemic_risk': coupling_mode.epistemic_risk,
                'risk_factors': coupling_mode.risk_factors,
                'confidence': coupling_mode.confidence,
                'coherence': coherence_data
            },
            'complexity_detected': results['summary']['cognitive_complexity_detected'],
            'psi_vector': psi_vector,
            'psi_composite': psi_composite,
            'attractor_basin': attractor_basin,
            'affective_substrate': affective_substrate,
            # Movement-preserving classification (v0.3.0+)
            'movement_aware_label': movement_aware_label,
            'trajectory_state_vector': trajectory_state_vector,
            'soft_state_inference': soft_state_inference
        }

        return response

    def _classify_level(self, value: float, thresholds: list) -> str:
        """Classify metric value as LOW/MEDIUM/HIGH."""
        if value < thresholds[0]:
            return 'LOW'
        elif value < thresholds[1]:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _classify_pressure(self, alpha: float) -> str:
        """Classify alpha (pressure) as LOW/OPTIMAL/HIGH."""
        if alpha < 0.70:
            return 'LOW'
        elif 0.70 <= alpha <= 0.90:
            return 'OPTIMAL'
        else:
            return 'HIGH'

    def compute_semantic_coherence(self, semantic_shifts: list) -> SemanticCoherence:
        """
        Compute semantic coherence from the pattern of meaning movement.

        Meaning moves - and HOW it moves reveals the coupling quality.
        Healthy dialogue has a "breathing" rhythm: big shift -> consolidation -> big shift.
        This manifests as negative autocorrelation in the shift series.

        Args:
            semantic_shifts: List of turn-to-turn semantic shift values

        Returns:
            SemanticCoherence with pattern classification and score
        """
        if len(semantic_shifts) < 4:
            # Not enough data for meaningful autocorrelation
            return SemanticCoherence(
                autocorrelation=0.0,
                variance=0.0,
                pattern='insufficient_data',
                coherence_score=0.5
            )

        shifts = np.array(semantic_shifts)

        # Compute lag-1 autocorrelation
        try:
            autocorr = np.corrcoef(shifts[:-1], shifts[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        except Exception:
            autocorr = 0.0

        # Compute variance (normalized by mean to get coefficient of variation)
        variance = float(np.var(shifts))
        mean_shift = float(np.mean(shifts))
        cv = variance / mean_shift if mean_shift > 0 else 0

        # Classify pattern based on autocorrelation and variance
        if autocorr < -0.3:
            # Healthy breathing rhythm
            pattern = 'breathing'
            # Score: stronger negative autocorr = better coherence
            coherence_score = min(1.0, 0.7 + abs(autocorr) * 0.5)
        elif autocorr > 0.3:
            # Lock-in / repetitive
            pattern = 'locked'
            # Score: strong positive autocorr indicates problematic lock-in
            coherence_score = max(0.0, 0.5 - (autocorr - 0.3) * 0.7)
        elif cv > 0.5 and abs(autocorr) < 0.2:
            # High variance + no clear pattern = fragmentation
            pattern = 'fragmented'
            coherence_score = max(0.0, 0.4 - cv * 0.3)
        else:
            # Transitional state
            pattern = 'transitional'
            coherence_score = 0.5 + (0.3 - abs(autocorr)) * 0.3

        return SemanticCoherence(
            autocorrelation=round(autocorr, 4),
            variance=round(variance, 4),
            pattern=pattern,
            coherence_score=round(coherence_score, 3)
        )

    def _classify_trajectory(
        self,
        dk_trend: float,
        dh_trend: float,
        alpha_trend: float,
        threshold: float = 0.02
    ) -> Tuple[str, Dict[str, str]]:
        """
        Classify the trajectory of meaning movement.

        Args:
            dk_trend: Rate of change in semantic curvature
            dh_trend: Rate of change in entropy shift
            alpha_trend: Rate of change in fractal coherence
            threshold: Minimum change to register as movement

        Returns:
            (trajectory_label, component_directions)
        """
        directions = {
            'curvature': 'rising' if dk_trend > threshold else ('falling' if dk_trend < -threshold else 'stable'),
            'entropy': 'rising' if dh_trend > threshold else ('falling' if dh_trend < -threshold else 'stable'),
            'coherence': 'rising' if alpha_trend > threshold else ('falling' if alpha_trend < -threshold else 'stable')
        }

        # Determine overall trajectory
        rising_count = sum(1 for d in directions.values() if d == 'rising')
        falling_count = sum(1 for d in directions.values() if d == 'falling')

        # Check for oscillation (mixed signals)
        if rising_count > 0 and falling_count > 0:
            trajectory = 'oscillating'
        elif rising_count >= 2:
            trajectory = 'warming'
        elif falling_count >= 2:
            trajectory = 'cooling'
        elif directions['curvature'] == 'rising':
            # Curvature rising alone indicates warming toward complexity
            trajectory = 'warming'
        elif directions['curvature'] == 'falling':
            trajectory = 'cooling'
        else:
            trajectory = 'stable'

        return trajectory, directions

    def _assess_epistemic_risk(
        self,
        mode: str,
        trajectory: str,
        dk: float,
        dh: float,
        dk_trend: float,
        coherence: Optional[SemanticCoherence] = None
    ) -> Tuple[str, list]:
        """
        Assess epistemic risk based on mode, trajectory, and coherence.

        Progressive sycophancy (moving toward depth) is lower risk than
        regressive sycophancy (moving toward compliance/error).

        Coherence pattern modulates risk assessment:
        - 'breathing' pattern reduces risk (healthy rhythm)
        - 'locked' pattern increases risk (repetitive)
        - 'fragmented' pattern increases risk (incoherent)

        Returns:
            (risk_level, risk_factors)
        """
        risk_factors = []

        # Sycophantic patterns
        if mode == 'Sycophantic':
            if trajectory == 'cooling' or trajectory == 'stable':
                risk_factors.append('regressive_sycophancy')
                risk_factors.append('epistemic_enclosure_risk')
                if dk < 0.10:
                    risk_factors.append('minimal_cognitive_engagement')
                    return 'critical', risk_factors
                return 'high', risk_factors
            elif trajectory == 'warming':
                risk_factors.append('progressive_sycophancy')
                return 'moderate', risk_factors

        # Low curvature with falling trajectory
        if dk < 0.25 and dk_trend < -0.02:
            risk_factors.append('complexity_collapse')
            return 'high', risk_factors

        # Chaotic patterns
        if mode == 'Chaotic':
            risk_factors.append('conceptual_instability')
            if dh > 0.30:
                risk_factors.append('semantic_fragmentation')
                return 'high', risk_factors
            return 'moderate', risk_factors

        # Dissociative patterns - but check coherence first
        if mode == 'Dissociative':
            if coherence and coherence.pattern == 'breathing':
                # Not truly incoherent - semantic thread exists
                risk_factors.append('metric_instability')
                risk_factors.append('semantic_continuity_present')
                return 'low', risk_factors
            else:
                risk_factors.append('incoherent_coupling')
                return 'moderate', risk_factors

        # Liminal mode (reclassified from Dissociative with coherence)
        if mode == 'Liminal':
            risk_factors.append('edge_exploration')
            return 'low', risk_factors

        # Transitional mode
        if mode == 'Transitional':
            risk_factors.append('mode_transition')
            return 'low', risk_factors

        # Check coherence patterns for additional risk signals
        if coherence:
            if coherence.pattern == 'locked':
                risk_factors.append('repetitive_pattern')
                # Locked pattern in otherwise healthy mode is concerning
                if mode in ['Resonant', 'Generative', 'Contemplative']:
                    return 'moderate', risk_factors
            elif coherence.pattern == 'fragmented':
                risk_factors.append('semantic_fragmentation')
                return 'moderate', risk_factors

        # Healthy modes
        if mode in ['Resonant', 'Generative', 'Contemplative', 'Dialectical', 'Emergent', 'Liminal', 'Transitional']:
            if trajectory == 'cooling' and dk_trend < -0.05:
                risk_factors.append('engagement_declining')
                return 'moderate', risk_factors
            return 'low', risk_factors

        return 'low', risk_factors

    def _detect_mode(self, dk: float, dh: float, alpha: float) -> str:
        """
        Detect coupling mode based on metric values (position only).

        This is the base mode without trajectory. Use detect_coupling_mode()
        for full trajectory-aware classification.

        Taxonomy based on empirical analysis of SCM experiments:
        - Sycophantic: Minimal exploration, compliant
        - Exploratory: Early-phase probing, metrics not yet stabilized
        - Resonant: Optimal coupling, co-emergent flow
        - Generative: High curvature, low entropy - building new ground
        - Contemplative: Very high curvature, minimal branching - dense meaning
        - Emergent: High curvature, moderate entropy - complexity developing
        - Dialectical: Productive tension with developing coherence
        - Chaotic: Confirmed high entropy with LOW coherence - actual fragmentation
        - Dissociative: Genuinely incoherent (rare)

        Tuned 2025-12-09:
        - Added Exploratory for early-phase (α=0.5 boundary condition)
        - Chaotic now requires confirmed low α (not boundary condition)
        - Dialectical expanded to capture productive tension with healthy coherence
        """
        # Sycophantic: minimal exploration, structured compliance
        if dk < 0.20 and dh < 0.08:
            return 'Sycophantic'

        # Exploratory: early-phase conversation, metrics haven't stabilized
        # α = 0.5 is the DFA boundary condition (insufficient data for fractal analysis)
        # This is natural for early turns — probing, establishing rapport, finding topic
        # Not pathological, just nascent coupling
        if alpha == 0.5:
            return 'Exploratory'

        # Chaotic: requires HIGH entropy AND confirmed low coherence - actual fragmentation
        # Key insight: high ΔH with healthy α is productive tension, not chaos
        # Only triggers when α has emerged from boundary and shows genuine incoherence
        if dh > 0.40 and alpha < 0.55:
            return 'Chaotic'

        # Dialectical: productive tension, challenge-response dynamics
        # Expanded range - high curvature + moderate-high entropy + developing coherence
        # This is the "working through" zone - semantic bending with reorganization
        if dk > 0.50 and 0.20 <= dh <= 0.70 and alpha >= 0.55:
            return 'Dialectical'

        # Contemplative: very high curvature, minimal entropy branching
        # Dense sustained meaning-making (e.g., deep philosophical dialogue)
        if dk > 0.70 and dh < 0.10:
            return 'Contemplative'

        # Emergent: high curvature with moderate entropy, coherence developing
        # Complexity is developing but hasn't reached full dialectical tension
        if dk > 0.65 and 0.10 <= dh < 0.25 and alpha >= 0.50:
            return 'Emergent'

        # Generative: mid-high curvature, low entropy - building new conceptual ground
        if 0.50 <= dk <= 0.75 and dh < 0.12:
            return 'Generative'

        # Resonant: optimal coupling, co-emergent flow
        if 0.35 <= dk <= 0.65 and 0.08 <= dh <= 0.20 and alpha >= 0.45:
            return 'Resonant'

        # Dissociative: genuinely incoherent coupling (should now be rare)
        return 'Dissociative'

    def detect_coupling_mode(
        self,
        dk: float,
        dh: float,
        alpha: float,
        dk_trend: float = 0.0,
        dh_trend: float = 0.0,
        alpha_trend: float = 0.0,
        coherence: Optional[SemanticCoherence] = None
    ) -> CouplingMode:
        """
        Full trajectory-aware coupling mode detection.

        Movement of meaning matters - this captures both WHERE the conversation
        is in metric space AND which direction it's moving.

        Args:
            dk: Current semantic curvature (delta_kappa)
            dh: Current entropy shift (delta_H)
            alpha: Current fractal coherence (alpha)
            dk_trend: Rate of change in delta_kappa
            dh_trend: Rate of change in delta_H
            alpha_trend: Rate of change in alpha
            coherence: Optional SemanticCoherence from autocorrelation analysis

        Returns:
            CouplingMode with full trajectory and risk assessment
        """
        # Get base mode from position
        base_mode = self._detect_mode(dk, dh, alpha)

        # Classify trajectory
        trajectory, directions = self._classify_trajectory(dk_trend, dh_trend, alpha_trend)

        # Use coherence to refine Dissociative classification
        # Key insight: Dissociative with "breathing" coherence is actually Liminal
        if base_mode == 'Dissociative' and coherence:
            if coherence.pattern == 'breathing' and coherence.coherence_score > 0.6:
                # Not truly dissociative - there's semantic continuity
                # Reclassify based on curvature
                if dk > 0.65:
                    base_mode = 'Liminal'  # High complexity edge exploration
                else:
                    base_mode = 'Transitional'  # Moving between modes

        # Assess epistemic risk (now considers coherence)
        risk_level, risk_factors = self._assess_epistemic_risk(
            base_mode, trajectory, dk, dh, dk_trend, coherence
        )

        # Build compound label
        if base_mode == 'Sycophantic':
            if trajectory in ['warming']:
                compound_label = 'Sycophantic-Progressive'
            else:
                compound_label = 'Sycophantic-Regressive'
        elif trajectory == 'stable':
            compound_label = base_mode
        else:
            compound_label = f"{base_mode}-{trajectory.capitalize()}"

        # Calculate confidence (how cleanly this fits the classification)
        confidence = self._calculate_mode_confidence(dk, dh, alpha, base_mode)

        # Boost confidence if coherence confirms the pattern
        if coherence and coherence.coherence_score > 0.7:
            confidence = min(1.0, confidence + 0.1)

        return CouplingMode(
            mode=base_mode,
            trajectory=trajectory,
            compound_label=compound_label,
            epistemic_risk=risk_level,
            risk_factors=risk_factors,
            confidence=confidence,
            coherence=coherence
        )

    def _calculate_mode_confidence(
        self,
        dk: float,
        dh: float,
        alpha: float,
        mode: str
    ) -> float:
        """
        Calculate how cleanly the metrics fit the assigned mode.

        Returns a value between 0 and 1, where 1 means perfectly centered
        in the mode's expected range.
        """
        # Define ideal centers for each mode
        mode_centers = {
            'Sycophantic': {'dk': 0.10, 'dh': 0.04, 'alpha': 0.80},
            'Resonant': {'dk': 0.50, 'dh': 0.13, 'alpha': 0.70},
            'Generative': {'dk': 0.62, 'dh': 0.04, 'alpha': 0.60},
            'Contemplative': {'dk': 0.80, 'dh': 0.02, 'alpha': 0.65},
            'Emergent': {'dk': 0.72, 'dh': 0.08, 'alpha': 0.65},
            'Dialectical': {'dk': 0.52, 'dh': 0.18, 'alpha': 0.75},
            'Chaotic': {'dk': 0.70, 'dh': 0.35, 'alpha': 0.35},
            'Dissociative': {'dk': 0.45, 'dh': 0.10, 'alpha': 0.50},
            'Liminal': {'dk': 0.75, 'dh': 0.12, 'alpha': 0.60},  # Edge exploration
            'Transitional': {'dk': 0.50, 'dh': 0.10, 'alpha': 0.55}  # Moving between modes
        }

        if mode not in mode_centers:
            return 0.5

        center = mode_centers[mode]

        # Calculate normalized distance from center
        dk_dist = abs(dk - center['dk']) / 0.5  # Normalize by max expected deviation
        dh_dist = abs(dh - center['dh']) / 0.2
        alpha_dist = abs(alpha - center['alpha']) / 0.4

        # Convert distance to confidence (closer = higher confidence)
        avg_dist = (dk_dist + dh_dist + alpha_dist) / 3
        confidence = max(0.0, min(1.0, 1.0 - avg_dist))

        return round(confidence, 3)
