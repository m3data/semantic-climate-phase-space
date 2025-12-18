"""
Attractor Basin Detection and History Module

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

This module provides attractor basin detection with optional history tracking
for hysteresis-aware classification in Semantic Climate Phase Space.

Classes:
    BasinHistory: Tracks basin sequence, residence time, transitions
    BasinDetector: Detects attractor basins with optional history conditioning

Functions:
    detect_attractor_basin: Backward-compatible legacy function

Basin Taxonomy (v2):
    1. Deep Resonance: All substrates high
    2. Collaborative Inquiry: Genuine co-exploration with mutual uncertainty
    3. Cognitive Mimicry: Model performs engagement without genuine uncertainty
    4. Reflexive Performance: Model appears to self-examine but pattern-matches
    5. Sycophantic Convergence: High alignment + low Δκ, low affect
    6. Creative Dilation: Divergent + high Δκ, high affect
    7. Generative Conflict: High divergent semantic, high affect
    8. Embodied Coherence: Low semantic, high biosignal
    9. Dissociation: All substrates low
    10. Transitional: No clear basin
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import numpy as np

__all__ = [
    'BasinHistory',
    'BasinDetector',
    'SoftStateInference',
    'HysteresisConfig',
    'detect_attractor_basin',
    'generate_movement_annotation',
]


@dataclass
class SoftStateInference:
    """
    Weighted membership across modes/phases.

    Replaces hard thresholds with probability-like weights. State changes
    are inferred from distribution shifts, not single threshold crossings.

    This addresses the core problem: threshold cuts discard movement.
    Soft membership preserves ambiguity at boundaries.

    Attributes:
        membership: {basin_name: weight} for all basins, sum to 1.0
        primary_basin: Basin with highest membership weight
        secondary_basin: Basin with second highest weight (if within margin)
        ambiguity: 1 - (max_weight - second_weight), high = uncertain classification
        distribution_shift: KL divergence from previous timestep (if available)
    """
    membership: Dict[str, float]
    primary_basin: str
    secondary_basin: Optional[str] = None
    ambiguity: float = 0.0
    distribution_shift: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HysteresisConfig:
    """
    Per-basin entry/exit threshold configuration.

    Entry thresholds are lower than exit thresholds, making it easier
    to enter a state than to leave it. This prevents oscillation at
    boundaries and respects the principle that threshold crossing ≠
    genuine state transition.

    Attributes:
        basin_name: Name of the basin this config applies to
        entry_threshold: Confidence threshold to enter this basin (lower)
        exit_threshold: Confidence threshold to exit this basin (higher)
        provisional_turns: Turns before provisional confirmation (τ₁)
        established_turns: Turns before established state (τ₂)
        entry_penalty: Confidence multiplier on new entry (< 1.0)
        settled_bonus: Confidence multiplier after settling (> 1.0)
    """
    basin_name: str
    entry_threshold: float = 0.3
    exit_threshold: float = 0.4
    provisional_turns: int = 3
    established_turns: int = 10
    entry_penalty: float = 0.7
    settled_bonus: float = 1.1

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BasinHistory:
    """
    Tracks basin sequence for hysteresis-aware detection.

    Enables:
    - Residence time computation
    - Approach path tracking
    - Transition counting
    - Confidence modulation by history

    Attributes:
        history: List of (turn, basin, confidence) tuples
        max_history: Maximum entries to retain
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize basin history.

        Args:
            max_history: Maximum number of entries to retain
        """
        self.max_history = max_history
        self.history: List[Tuple[int, str, float]] = []
        self._current_basin: Optional[str] = None
        self._previous_basin: Optional[str] = None
        self._basin_entry_turn: int = 0
        self._transition_count: int = 0
        # Hysteresis state tracking (v0.3.0+)
        self._state_status: str = 'unknown'  # 'unknown', 'provisional', 'established'
        self._provisional_since: int = 0  # Turn when provisional state began

    def append(self, basin: str, confidence: float, turn: int = None) -> None:
        """
        Add a basin entry to history.

        Args:
            basin: Basin name
            confidence: Detection confidence
            turn: Turn number (auto-increments if not provided)
        """
        if turn is None:
            turn = len(self.history)

        # Track transitions
        if self._current_basin is not None and basin != self._current_basin:
            self._previous_basin = self._current_basin
            self._basin_entry_turn = turn
            self._transition_count += 1

        if self._current_basin is None:
            self._basin_entry_turn = turn

        self._current_basin = basin
        self.history.append((turn, basin, confidence))

        # Maintain max history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_current_basin(self) -> Optional[str]:
        """Get the most recent basin."""
        return self._current_basin

    def get_residence_time(self) -> int:
        """
        Get number of consecutive turns in current basin.

        Returns:
            int: Number of turns in current basin (0 if no history)
        """
        if not self.history:
            return 0

        current = self._current_basin
        count = 0

        for turn, basin, _ in reversed(self.history):
            if basin == current:
                count += 1
            else:
                break

        return count

    def get_previous_basin(self) -> Optional[str]:
        """Get the basin before the current one."""
        return self._previous_basin

    def get_transition_count(self) -> int:
        """Get total number of basin transitions."""
        return self._transition_count

    def get_basin_sequence(self, n: int = None) -> List[str]:
        """
        Get sequence of recent basins.

        Args:
            n: Number of entries to return (None = all)

        Returns:
            List[str]: Basin names in order
        """
        if n is None:
            return [basin for _, basin, _ in self.history]
        return [basin for _, basin, _ in self.history[-n:]]

    def get_basin_distribution(self) -> Dict[str, int]:
        """
        Compute basin visit counts.

        Returns:
            Dict[str, int]: Count of visits to each basin
        """
        dist = defaultdict(int)
        for _, basin, _ in self.history:
            dist[basin] += 1
        return dict(dist)

    def get_transition_matrix(self) -> Dict[Tuple[str, str], int]:
        """
        Compute transition counts between basins.

        Returns:
            Dict[(from_basin, to_basin), count]: Transition counts
        """
        matrix = defaultdict(int)

        for i in range(1, len(self.history)):
            prev_basin = self.history[i-1][1]
            curr_basin = self.history[i][1]
            if prev_basin != curr_basin:
                matrix[(prev_basin, curr_basin)] += 1

        return dict(matrix)

    def get_state_status(self) -> str:
        """
        Get current state status for hysteresis tracking.

        Returns:
            str: 'unknown', 'provisional', or 'established'
        """
        return self._state_status

    def set_state_status(self, status: str, turn: int = None) -> None:
        """
        Set state status for hysteresis tracking.

        Args:
            status: 'unknown', 'provisional', or 'established'
            turn: Turn number when status changed (auto-computed if None)
        """
        if status not in ('unknown', 'provisional', 'established'):
            raise ValueError(f"Invalid state status: {status}")

        if status == 'provisional' and self._state_status != 'provisional':
            self._provisional_since = turn if turn is not None else len(self.history)

        self._state_status = status

    def get_provisional_duration(self) -> int:
        """
        Get turns in provisional state.

        Returns:
            int: Number of turns since provisional began (0 if not provisional)
        """
        if self._state_status != 'provisional':
            return 0
        return len(self.history) - self._provisional_since

    def clear(self) -> None:
        """Clear all history."""
        self.history = []
        self._current_basin = None
        self._previous_basin = None
        self._basin_entry_turn = 0
        self._transition_count = 0
        self._state_status = 'unknown'
        self._provisional_since = 0


class BasinDetector:
    """
    Detects attractor basins with optional history conditioning.

    Implements residence confidence modulation: basins you've been in
    longer have higher confidence (settled bonus), while new entries
    have reduced confidence (new entry penalty).

    Now supports soft membership computation for movement-preserving
    classification (v0.3.0+).
    """

    # Canonical basin definitions
    BASINS = [
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

    # Basin centroids in (psi_semantic, psi_affective, delta_kappa, psi_biosignal) space
    # Derived from threshold analysis in _classify_basin()
    BASIN_CENTROIDS = {
        'Deep Resonance': {'psi_semantic': 0.5, 'psi_affective': 0.5, 'delta_kappa': 0.5, 'psi_biosignal': 0.5},
        'Collaborative Inquiry': {'psi_semantic': 0.4, 'psi_affective': 0.1, 'delta_kappa': 0.4, 'psi_biosignal': 0.0},
        'Cognitive Mimicry': {'psi_semantic': 0.5, 'psi_affective': 0.1, 'delta_kappa': 0.3, 'psi_biosignal': 0.0},
        'Reflexive Performance': {'psi_semantic': 0.4, 'psi_affective': 0.1, 'delta_kappa': 0.35, 'psi_biosignal': 0.0},
        'Sycophantic Convergence': {'psi_semantic': 0.5, 'psi_affective': 0.1, 'delta_kappa': 0.2, 'psi_biosignal': 0.0},
        'Creative Dilation': {'psi_semantic': 0.3, 'psi_affective': 0.4, 'delta_kappa': 0.5, 'psi_biosignal': 0.0},
        'Generative Conflict': {'psi_semantic': 0.4, 'psi_affective': 0.4, 'delta_kappa': 0.5, 'psi_biosignal': 0.0},
        'Embodied Coherence': {'psi_semantic': 0.1, 'psi_affective': 0.2, 'delta_kappa': 0.3, 'psi_biosignal': 0.5},
        'Dissociation': {'psi_semantic': 0.1, 'psi_affective': 0.1, 'delta_kappa': 0.1, 'psi_biosignal': 0.1},
        'Transitional': {'psi_semantic': 0.25, 'psi_affective': 0.25, 'delta_kappa': 0.3, 'psi_biosignal': 0.0},
    }

    # Default hysteresis configuration per basin
    DEFAULT_HYSTERESIS = {
        'Deep Resonance': HysteresisConfig('Deep Resonance', 0.35, 0.45, 3, 10, 0.7, 1.1),
        'Collaborative Inquiry': HysteresisConfig('Collaborative Inquiry', 0.30, 0.40, 3, 8, 0.75, 1.1),
        'Cognitive Mimicry': HysteresisConfig('Cognitive Mimicry', 0.30, 0.40, 2, 5, 0.8, 1.05),
        'Reflexive Performance': HysteresisConfig('Reflexive Performance', 0.30, 0.40, 2, 5, 0.8, 1.05),
        'Sycophantic Convergence': HysteresisConfig('Sycophantic Convergence', 0.25, 0.35, 2, 5, 0.8, 1.05),
        'Creative Dilation': HysteresisConfig('Creative Dilation', 0.30, 0.40, 3, 8, 0.75, 1.1),
        'Generative Conflict': HysteresisConfig('Generative Conflict', 0.30, 0.40, 3, 8, 0.75, 1.1),
        'Embodied Coherence': HysteresisConfig('Embodied Coherence', 0.30, 0.40, 3, 10, 0.7, 1.15),
        'Dissociation': HysteresisConfig('Dissociation', 0.25, 0.35, 3, 8, 0.75, 1.1),
        'Transitional': HysteresisConfig('Transitional', 0.20, 0.30, 2, 5, 0.85, 1.0),
    }

    def __init__(
        self,
        residence_confidence_modulation: bool = True,
        new_entry_penalty: float = 0.7,
        settled_bonus: float = 1.1,
        settled_threshold: int = 10
    ):
        """
        Initialize basin detector.

        Args:
            residence_confidence_modulation: Whether to adjust confidence by residence
            new_entry_penalty: Multiplier for confidence on first entry (< 1.0)
            settled_bonus: Multiplier for confidence after settling (> 1.0)
            settled_threshold: Turns to wait before applying settled bonus
        """
        self.residence_confidence_modulation = residence_confidence_modulation
        self.new_entry_penalty = new_entry_penalty
        self.settled_bonus = settled_bonus
        self.settled_threshold = settled_threshold

    def detect(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: dict = None,
        basin_history: BasinHistory = None
    ) -> Tuple[str, float, dict]:
        """
        Classify phase-space position into attractor basin.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            raw_metrics: Optional dict with delta_kappa, delta_h, alpha
            dialogue_context: Optional dict with hedging_density, turn_length_ratio, etc.
            basin_history: Optional BasinHistory for hysteresis-aware detection

        Returns:
            tuple: (basin_name: str, confidence: float, metadata: dict)
            metadata includes: residence_time, previous_basin, raw_confidence
        """
        # Get raw basin classification
        basin_name, raw_confidence = self._classify_basin(
            psi_vector, raw_metrics, dialogue_context
        )

        # Build metadata
        metadata = {
            'raw_confidence': raw_confidence,
            'residence_time': 0,
            'previous_basin': None
        }

        # Apply residence confidence modulation
        final_confidence = raw_confidence

        if basin_history is not None:
            metadata['residence_time'] = basin_history.get_residence_time()
            metadata['previous_basin'] = basin_history.get_previous_basin()

            if self.residence_confidence_modulation:
                current_basin = basin_history.get_current_basin()

                if current_basin is None:
                    # First entry ever - apply new entry penalty
                    final_confidence = raw_confidence * self.new_entry_penalty
                elif basin_name != current_basin:
                    # Transitioning to new basin - apply new entry penalty
                    final_confidence = raw_confidence * self.new_entry_penalty
                elif metadata['residence_time'] >= self.settled_threshold:
                    # Settled in basin - apply settled bonus
                    final_confidence = min(1.0, raw_confidence * self.settled_bonus)

        return (basin_name, float(final_confidence), metadata)

    def detect_with_hysteresis(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: dict = None,
        basin_history: BasinHistory = None
    ) -> Tuple[str, float, dict]:
        """
        Hysteresis-aware basin detection.

        This extends detect() with entry/exit threshold asymmetry and
        provisional → established state tracking. Entry thresholds are
        lower than exit thresholds, making it easier to enter a state
        than to leave it.

        State machine:
            UNKNOWN → PROVISIONAL (on entry) → ESTABLISHED (after τ₂ turns)
            ESTABLISHED → PROVISIONAL (on potential exit) → UNKNOWN (confirmed exit)
            PROVISIONAL → (revert if not sustained)

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            raw_metrics: Optional dict with delta_kappa, delta_h, alpha
            dialogue_context: Optional dict with hedging_density, turn_length_ratio, etc.
            basin_history: BasinHistory for state tracking (required for hysteresis)

        Returns:
            tuple: (basin_name: str, confidence: float, metadata: dict)
            metadata includes: residence_time, previous_basin, raw_confidence,
                             state_status, transition_type
        """
        # Get raw classification
        basin_name, raw_confidence = self._classify_basin(
            psi_vector, raw_metrics, dialogue_context
        )

        # Build metadata
        metadata = {
            'raw_confidence': raw_confidence,
            'residence_time': 0,
            'previous_basin': None,
            'state_status': 'unknown',
            'transition_type': None  # 'entry', 'exit', 'sustained', None
        }

        # Without history, fall back to basic detection
        if basin_history is None:
            return (basin_name, float(raw_confidence), metadata)

        # Get current state
        current_basin = basin_history.get_current_basin()
        state_status = basin_history.get_state_status()
        residence_time = basin_history.get_residence_time()

        metadata['residence_time'] = residence_time
        metadata['previous_basin'] = basin_history.get_previous_basin()
        metadata['state_status'] = state_status

        # Get hysteresis config for relevant basins
        current_config = self.DEFAULT_HYSTERESIS.get(
            current_basin,
            HysteresisConfig(current_basin or 'unknown')
        )
        proposed_config = self.DEFAULT_HYSTERESIS.get(
            basin_name,
            HysteresisConfig(basin_name)
        )

        # Apply hysteresis logic
        final_basin = basin_name
        final_confidence = raw_confidence

        if current_basin is None:
            # First entry - use entry threshold
            if raw_confidence >= proposed_config.entry_threshold:
                final_basin = basin_name
                final_confidence = raw_confidence * proposed_config.entry_penalty
                basin_history.set_state_status('provisional')
                metadata['transition_type'] = 'entry'
                metadata['state_status'] = 'provisional'
            else:
                final_basin = 'Transitional'
                final_confidence = 0.3
                metadata['state_status'] = 'unknown'

        elif basin_name == current_basin:
            # Staying in same basin - check for establishment
            final_basin = current_basin
            final_confidence = raw_confidence

            if state_status == 'provisional':
                prov_duration = basin_history.get_provisional_duration()
                if prov_duration >= proposed_config.provisional_turns:
                    basin_history.set_state_status('established')
                    metadata['state_status'] = 'established'
                    metadata['transition_type'] = 'sustained'

            if state_status == 'established' and residence_time >= current_config.established_turns:
                final_confidence = min(1.0, raw_confidence * current_config.settled_bonus)

        else:
            # Potential transition to different basin
            if state_status == 'established':
                # Established state - need to cross exit threshold to leave
                if raw_confidence < current_config.exit_threshold:
                    # Can't exit yet - stay in current basin
                    final_basin = current_basin
                    final_confidence = current_config.exit_threshold * 0.9
                    metadata['transition_type'] = None
                else:
                    # Crossing exit threshold - go to provisional for new basin
                    final_basin = basin_name
                    final_confidence = raw_confidence * proposed_config.entry_penalty
                    basin_history.set_state_status('provisional')
                    metadata['state_status'] = 'provisional'
                    metadata['transition_type'] = 'exit'
            else:
                # Provisional or unknown - easier to transition
                if raw_confidence >= proposed_config.entry_threshold:
                    final_basin = basin_name
                    final_confidence = raw_confidence * proposed_config.entry_penalty
                    basin_history.set_state_status('provisional')
                    metadata['state_status'] = 'provisional'
                    metadata['transition_type'] = 'entry'
                else:
                    # Revert to current or transitional
                    if current_basin:
                        final_basin = current_basin
                        final_confidence = raw_confidence
                    else:
                        final_basin = 'Transitional'
                        final_confidence = 0.3

        return (final_basin, float(final_confidence), metadata)

    def _classify_basin(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: dict = None
    ) -> Tuple[str, float]:
        """
        Core basin classification logic.

        This implements the refined taxonomy (v2) that distinguishes
        performative from genuine engagement.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            raw_metrics: Optional dict with delta_kappa, delta_h, alpha
            dialogue_context: Optional dict with hedging_density, turn_length_ratio, etc.

        Returns:
            tuple: (basin_name: str, confidence: float)
        """
        sem = psi_vector.get('psi_semantic', 0.0) or 0.0
        temp = psi_vector.get('psi_temporal', 0.5) or 0.5
        aff = psi_vector.get('psi_affective', 0.0) or 0.0
        bio = psi_vector.get('psi_biosignal', 0.0) or 0.0

        delta_kappa = raw_metrics.get('delta_kappa', 0.0) if raw_metrics else 0.0

        # Extract dialogue context for refined classification
        ctx = dialogue_context or {}
        hedging = ctx.get('hedging_density', 0.0)
        turn_ratio = ctx.get('turn_length_ratio', 1.0)  # AI/human, >1 = AI dominates
        dk_variance = ctx.get('delta_kappa_variance', 0.0)
        coherence_pattern = ctx.get('coherence_pattern', 'transitional')

        # === HIGH-CERTAINTY BASINS (check first) ===

        # Basin 7: Deep Resonance (all substrates high)
        if sem > 0.4 and aff > 0.4 and (bio > 0.4 or bio == 0.0):
            confidence = min(abs(sem), abs(aff)) if bio == 0.0 else min(abs(sem), abs(aff), abs(bio))
            return ("Deep Resonance", float(confidence))

        # Basin 9: Dissociation (all substrates low)
        if abs(sem) < 0.2 and abs(aff) < 0.2 and abs(bio) < 0.2:
            confidence = 1.0 - max(abs(sem), abs(aff), abs(bio))
            return ("Dissociation", float(confidence))

        # Basin 8: Embodied Coherence (body leads, semantic follows)
        if abs(sem) < 0.3 and bio > 0.3:
            confidence = abs(bio) * (1.0 - abs(sem) / 0.3)
            return ("Embodied Coherence", float(confidence))

        # === SEMANTIC-ACTIVE, HIGH-AFFECT BASINS ===

        # Basin 6: Generative Conflict (productive tension)
        if abs(sem) > 0.3 and delta_kappa > 0.35 and aff > 0.3:
            confidence = min((delta_kappa / 0.7), abs(aff))
            return ("Generative Conflict", float(confidence))

        # Basin 5: Creative Dilation (expansive exploration with feeling)
        if delta_kappa > 0.35 and aff > 0.3:
            confidence = (delta_kappa / 0.7) * abs(aff)
            return ("Creative Dilation", float(confidence))

        # Basin 4: Sycophantic Convergence (agreeable flattening)
        # Check BEFORE refined basins - low Δκ is the key discriminator
        if sem > 0.3 and delta_kappa < 0.35 and aff < 0.2:
            confidence = sem * (1.0 - delta_kappa / 0.35) * (1.0 - abs(aff))
            return ("Sycophantic Convergence", float(confidence))

        # === SEMANTIC-ACTIVE, LOW-AFFECT BASINS (the refined territory) ===

        # This is where we distinguish mimicry from inquiry
        if abs(sem) > 0.3 and aff < 0.2 and bio < 0.2:
            # All three basins share: high semantic, low affect, low biosignal
            # Discriminators: hedging, turn asymmetry, Δκ variance, coherence

            # Collaborative Inquiry indicators:
            # - Hedging present (genuine uncertainty)
            # - Balanced turn lengths (mutual contribution)
            # - Oscillating Δκ (responsive trajectory)
            # - Breathing coherence pattern
            inquiry_score = 0.0
            if hedging > 0.02:  # Some hedging present
                inquiry_score += 0.3
            if 0.5 < turn_ratio < 2.0:  # Relatively balanced
                inquiry_score += 0.3
            if dk_variance > 0.01:  # Responsive oscillation
                inquiry_score += 0.2
            if coherence_pattern == 'breathing':
                inquiry_score += 0.2

            # Cognitive Mimicry indicators:
            # - Low hedging (confident performance)
            # - AI dominates turn length
            # - Smooth Δκ (scripted trajectory)
            # - Locked or transitional coherence
            mimicry_score = 0.0
            if hedging < 0.01:  # Very low hedging
                mimicry_score += 0.3
            if turn_ratio > 2.0:  # AI dominates
                mimicry_score += 0.3
            if dk_variance < 0.005:  # Smooth/scripted
                mimicry_score += 0.2
            if coherence_pattern in ('locked', 'transitional'):
                mimicry_score += 0.2

            # Reflexive Performance indicators:
            # - Moderate hedging (performed uncertainty)
            # - AI dominates but with theatrical pauses
            # - Medium Δκ variance (scripted oscillation)
            reflexive_score = 0.0
            if 0.01 <= hedging <= 0.03:  # Performed hedging
                reflexive_score += 0.3
            if turn_ratio > 1.5:  # AI still dominates
                reflexive_score += 0.2
            if 0.005 <= dk_variance <= 0.015:  # Scripted oscillation
                reflexive_score += 0.3
            if coherence_pattern == 'transitional':
                reflexive_score += 0.2

            # Classify based on highest score
            scores = {
                'Collaborative Inquiry': inquiry_score,
                'Cognitive Mimicry': mimicry_score,
                'Reflexive Performance': reflexive_score
            }
            best_basin = max(scores, key=scores.get)
            best_score = scores[best_basin]

            # If scores are close, we're in ambiguous territory
            sorted_scores = sorted(scores.values(), reverse=True)
            if sorted_scores[0] - sorted_scores[1] < 0.1:
                # Ambiguous - default based on semantic strength
                confidence = abs(sem) * 0.5  # Lower confidence
                return (best_basin, float(confidence))
            else:
                confidence = abs(sem) * (0.5 + best_score * 0.5)
                return (best_basin, float(confidence))

        # === DEFAULT: Transitional ===
        magnitudes = {
            'semantic': abs(sem),
            'affective': abs(aff),
            'biosignal': abs(bio),
            'temporal': abs(temp - 0.5)
        }

        dominant = max(magnitudes, key=magnitudes.get)
        confidence = magnitudes[dominant]

        if dominant == 'semantic' and delta_kappa > 0.35:
            return ("Creative Dilation", float(confidence))
        elif dominant == 'affective':
            return ("Generative Conflict" if delta_kappa > 0.35 else "Cognitive Mimicry", float(confidence))
        elif dominant == 'biosignal':
            return ("Embodied Coherence", float(confidence))
        else:
            return ("Transitional", 0.3)

    def compute_soft_membership(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        temperature: float = 1.0,
        previous_inference: 'SoftStateInference' = None
    ) -> SoftStateInference:
        """
        Compute weighted membership across all basins.

        Uses softmax on negative squared distances to basin centroids.
        This replaces hard threshold cuts with probability-like weights,
        preserving ambiguity at boundaries.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            raw_metrics: Optional dict with delta_kappa (used in centroid distance)
            temperature: Softmax temperature (lower = sharper, higher = softer)
            previous_inference: Previous SoftStateInference for distribution shift

        Returns:
            SoftStateInference with membership weights across all basins
        """
        # Build current position vector
        sem = psi_vector.get('psi_semantic', 0.0) or 0.0
        aff = psi_vector.get('psi_affective', 0.0) or 0.0
        bio = psi_vector.get('psi_biosignal', 0.0) or 0.0
        dk = raw_metrics.get('delta_kappa', 0.0) if raw_metrics else 0.0

        position = np.array([sem, aff, dk, bio])

        # Compute squared distances to each centroid
        distances = {}
        for basin_name, centroid in self.BASIN_CENTROIDS.items():
            centroid_vec = np.array([
                centroid['psi_semantic'],
                centroid['psi_affective'],
                centroid['delta_kappa'],
                centroid['psi_biosignal']
            ])
            distances[basin_name] = np.sum((position - centroid_vec) ** 2)

        # Apply softmax: weight_i = exp(-d_i / T) / sum(exp(-d_j / T))
        # Use negative distances so closer = higher weight
        exp_weights = {}
        max_neg_dist = max(-d for d in distances.values())  # For numerical stability

        for basin_name, dist in distances.items():
            exp_weights[basin_name] = np.exp((-dist - max_neg_dist) / temperature)

        total = sum(exp_weights.values())
        membership = {k: float(v / total) for k, v in exp_weights.items()}

        # Find primary and secondary basins
        sorted_basins = sorted(membership.items(), key=lambda x: x[1], reverse=True)
        primary_basin = sorted_basins[0][0]
        primary_weight = sorted_basins[0][1]

        secondary_basin = None
        secondary_weight = 0.0
        if len(sorted_basins) > 1:
            secondary_basin = sorted_basins[1][0]
            secondary_weight = sorted_basins[1][1]

        # Compute ambiguity: high when top two are close
        ambiguity = 1.0 - (primary_weight - secondary_weight)

        # Compute distribution shift (KL divergence) if previous inference available
        distribution_shift = None
        if previous_inference is not None:
            # KL(P || Q) = sum(P * log(P/Q))
            kl = 0.0
            epsilon = 1e-10  # Avoid log(0)
            for basin_name in membership:
                p = membership[basin_name]
                q = previous_inference.membership.get(basin_name, epsilon)
                if p > epsilon:
                    kl += p * np.log((p + epsilon) / (q + epsilon))
            distribution_shift = float(kl)

        return SoftStateInference(
            membership=membership,
            primary_basin=primary_basin,
            secondary_basin=secondary_basin,
            ambiguity=float(ambiguity),
            distribution_shift=distribution_shift
        )


def generate_movement_annotation(
    velocity_magnitude: Optional[float],
    acceleration_magnitude: Optional[float],
    previous_basin: Optional[str],
    dwell_time: int,
    velocity_threshold: float = 0.05,
    acceleration_threshold: float = 0.02,
    settled_threshold: int = 5
) -> str:
    """
    Generate human-readable movement annotation.

    This is the key to movement-preserving classification: the annotation
    encodes HOW you arrived at a state, not just WHERE you are.

    Args:
        velocity_magnitude: ||dΨ/dt|| - speed through phase space
        acceleration_magnitude: ||d²Ψ/dt²|| - rate of velocity change
        previous_basin: Basin transitioned from (if any)
        dwell_time: Consecutive turns in current region
        velocity_threshold: Below this is considered "still"
        acceleration_threshold: Above this is significant acceleration
        settled_threshold: Dwell time above this is "settled"

    Returns:
        str: Human-readable annotation like "from settling", "accelerating toward"

    Examples:
        - Low velocity, high dwell, from Deep Resonance → "settled from Deep Resonance"
        - High velocity, positive acceleration → "accelerating"
        - Low velocity, negative acceleration, from Cognitive Mimicry → "settling from Cognitive Mimicry"
        - Very low velocity, high dwell → "stable"
    """
    parts = []

    # Check if we have enough data
    if velocity_magnitude is None:
        return "insufficient data"

    # Determine movement state
    is_still = velocity_magnitude < velocity_threshold
    is_settled = is_still and dwell_time >= settled_threshold

    if is_settled:
        parts.append("settled")
    elif is_still:
        parts.append("still")
    else:
        # Moving - check acceleration
        if acceleration_magnitude is not None:
            if acceleration_magnitude > acceleration_threshold:
                parts.append("accelerating")
            elif acceleration_magnitude < -acceleration_threshold:
                parts.append("decelerating")
            else:
                parts.append("moving")
        else:
            parts.append("moving")

    # Add approach context if transitioning
    if previous_basin is not None and dwell_time < settled_threshold:
        # Recently transitioned - include source
        parts.append(f"from {previous_basin}")

    # Join parts
    if not parts:
        return "unknown"

    return " ".join(parts)


def detect_attractor_basin(
    psi_vector: dict,
    raw_metrics: dict = None,
    dialogue_context: dict = None
) -> Tuple[str, float]:
    """
    Classify phase-space position into attractor basins.

    This is the backward-compatible function that wraps BasinDetector.
    It does not use history conditioning.

    Args:
        psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
        raw_metrics: Optional dict with delta_kappa, delta_h, alpha
        dialogue_context: Optional dict with hedging_density, turn_length_ratio, etc.

    Returns:
        tuple: (basin_name: str, confidence: float)
    """
    detector = BasinDetector(residence_confidence_modulation=False)
    basin, confidence, _ = detector.detect(psi_vector, raw_metrics, dialogue_context)
    return (basin, confidence)
