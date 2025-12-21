"""
Semantic Climate Analyzer - Orchestrator Module

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

This module provides the SemanticClimateAnalyzer class, a thin orchestrator
that wires together the modular components:
- trajectory.py: TrajectoryBuffer, compute_trajectory_derivatives
- substrates.py: compute_*_substrate functions
- basins.py: BasinDetector, BasinHistory
- integrity.py: IntegrityAnalyzer

This replaces the monolithic SemanticClimateAnalyzer in extensions.py with
a cleaner composition-based design while maintaining full backward compatibility.

Usage:
    from src.analyzer import SemanticClimateAnalyzer

    analyzer = SemanticClimateAnalyzer(track_history=True, compute_integrity=True)
    result = analyzer.compute_coupling_coefficient(embeddings, turn_texts=texts)

    # Access trajectory integrity
    if result['trajectory_integrity']:
        print(f"Integrity: {result['trajectory_integrity']['integrity_label']}")

    # Access enhanced attractor dynamics
    print(f"Residence time: {result['attractor_dynamics']['residence_time']}")
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

from .core_metrics import SemanticComplexityAnalyzer
from .trajectory import TrajectoryBuffer, TrajectoryStateVector, compute_trajectory_derivatives
from .substrates import (
    compute_semantic_substrate,
    compute_temporal_substrate,
    compute_affective_substrate,
    compute_biosignal_substrate,
    compute_dialogue_context,
)

# Type hint for optional EmotionService without importing
if TYPE_CHECKING:
    pass  # EmotionService imported dynamically to avoid circular deps
from .basins import (
    BasinDetector,
    BasinHistory,
    SoftStateInference,
    generate_movement_annotation,
)
from .integrity import IntegrityAnalyzer

__all__ = [
    'SemanticClimateAnalyzer',
]


class SemanticClimateAnalyzer(SemanticComplexityAnalyzer):
    """
    Extended analyzer for Semantic Climate Model.

    Inherits from SemanticComplexityAnalyzer (Morgoulis, 2025):
    - Semantic Curvature (Δκ)
    - Fractal Similarity Score (α)
    - Entropy Shift (ΔH)

    Adds:
    - Vector Ψ representation (semantic, temporal, affective, biosignal)
    - Attractor basin detection with hysteresis-aware history
    - Trajectory dynamics (velocity, acceleration, curvature)
    - Trajectory integrity computation
    - Flow field properties (stubs)

    This is the Phase 2 orchestrator that wires together the modular components
    from trajectory.py, substrates.py, basins.py, and integrity.py.
    """

    def __init__(
        self,
        random_state: int = 42,
        bootstrap_iterations: int = 1000,
        track_history: bool = True,
        compute_integrity: bool = True,
        emotion_service=None,
        debug_timing: bool = False
    ):
        """
        Initialize the extended analyzer.

        Args:
            random_state: Random seed for reproducibility
            bootstrap_iterations: Number of bootstrap samples for CI
            track_history: Whether to maintain trajectory and basin history
            compute_integrity: Whether to compute trajectory integrity
            emotion_service: Optional EmotionService for GoEmotions-based
                affective analysis. If None, uses VADER-only fast path.
            debug_timing: Whether to print timing information for performance debugging
        """
        super().__init__(random_state=random_state, bootstrap_iterations=bootstrap_iterations)

        self.track_history = track_history
        self.compute_integrity = compute_integrity
        self.emotion_service = emotion_service
        self.debug_timing = debug_timing

        # Initialize components based on flags
        if track_history:
            self.trajectory = TrajectoryBuffer()
            self.basin_history = BasinHistory()
        else:
            self.trajectory = None
            self.basin_history = None

        self.basin_detector = BasinDetector()

        if compute_integrity:
            self.integrity_analyzer = IntegrityAnalyzer()
        else:
            self.integrity_analyzer = None

        # Movement-preserving classification state (v0.3.0+)
        self._previous_soft_inference: Optional[SoftStateInference] = None

    def reset_history(self) -> None:
        """
        Clear trajectory and basin history.

        Call this between unrelated dialogues to prevent history
        from one dialogue affecting analysis of another.
        """
        if self.trajectory is not None:
            self.trajectory.clear()
        if self.basin_history is not None:
            self.basin_history.clear()
        self._previous_soft_inference = None

    def compute_coupling_coefficient(
        self,
        dialogue_embeddings: np.ndarray = None,
        turn_texts: List[str] = None,
        turn_speakers: List[str] = None,
        timestamps: List[float] = None,
        metrics: dict = None,
        trajectory_history: List[dict] = None,
        trajectory_metrics: List[dict] = None,
        biosignal_data: dict = None,
        temporal_window_size: int = 10,
        coherence_pattern: str = None
    ) -> dict:
        """
        Calculate 4D coupling coefficient vector with dynamical properties.

        Args:
            dialogue_embeddings: Sequence of embeddings (required if metrics not provided)
            turn_texts: List of conversational turn texts for affective substrate
            turn_speakers: List of speaker labels ('human', 'ai') for turn asymmetry
            timestamps: Optional turn timestamps for temporal synchrony
            metrics: Pre-computed semantic climate metrics (optional)
            trajectory_history: List of past Ψ vectors for dynamics (legacy, ignored if track_history=True)
            trajectory_metrics: List of per-window metrics dicts for Δκ variance
            biosignal_data: Physiological signals for biosignal substrate
            temporal_window_size: Window size for temporal coherence calculation
            coherence_pattern: Pre-computed coherence pattern ('breathing', etc.)

        Returns:
            dict: Extended structure with:
                - psi: Composite score (backward compatible)
                - psi_state: position, velocity, acceleration
                - trajectory_dynamics: curvature, speed, direction, path_length
                - attractor_dynamics: basin, confidence, pull, depth, escape, stability,
                                      residence_time, previous_basin, raw_confidence
                - trajectory_integrity: autocorrelation, tortuosity, recurrence_rate,
                                        integrity_score, integrity_label, sufficient_data
                - flow_field: damping, turbulence, bifurcation, lyapunov
                - raw_metrics: Δκ, ΔH, α
                - substrate_details: per-substrate breakdowns
                - dialogue_context: context used for basin detection
        """
        # === 1. Calculate raw metrics if not provided ===
        if metrics is None:
            if dialogue_embeddings is None:
                raise ValueError("Must provide dialogue_embeddings or metrics")
            metrics = self.calculate_all_metrics(dialogue_embeddings)

        # === 2. Calculate substrate components ===

        # Semantic substrate (from Δκ, ΔH, α)
        semantic_result = compute_semantic_substrate(
            embeddings=dialogue_embeddings if dialogue_embeddings is not None else np.array([]),
            metrics=metrics
        )

        # Temporal substrate (from metric stability + timing if available)
        temporal_result = compute_temporal_substrate(
            embeddings=dialogue_embeddings,
            timestamps=timestamps,
            metrics=metrics,
            window_size=temporal_window_size
        )

        # Affective substrate (from turn texts if available)
        # Uses hybrid VADER + GoEmotions if emotion_service is configured
        if turn_texts is not None and len(turn_texts) > 0:
            if self.debug_timing:
                import time as _time
                _ta = _time.time()
            affective_result = compute_affective_substrate(
                turn_texts=turn_texts,
                embeddings=dialogue_embeddings,
                emotion_service=self.emotion_service
            )
            if self.debug_timing:
                print(f"    [TIMING] affective_substrate: {int((_time.time()-_ta)*1000)}ms", flush=True)
        else:
            affective_result = {
                'psi_affective': 0.0,
                'sentiment_trajectory': [],
                'hedging_density': 0.0,
                'vulnerability_score': 0.0,
                'confidence_variance': 0.0,
                'source': 'none'
            }

        # Biosignal substrate
        if biosignal_data is not None:
            psi_biosignal = compute_biosignal_substrate(biosignal_data)
        else:
            psi_biosignal = None

        # === 3. Build Ψ vector ===
        psi_vector = {
            'psi_semantic': semantic_result['psi_semantic'],
            'psi_temporal': temporal_result['psi_temporal'],
            'psi_affective': affective_result['psi_affective'],
            'psi_biosignal': psi_biosignal
        }

        # === 4. Update internal trajectory if tracking ===
        if self.track_history and self.trajectory is not None:
            self.trajectory.append(psi_vector)

        # === 5. Compute dialogue context for refined basin detection ===
        if self.debug_timing:
            import time as _time
            _tc = _time.time()
        dialogue_context = compute_dialogue_context(
            turn_texts=turn_texts,
            turn_speakers=turn_speakers,
            trajectory_metrics=trajectory_metrics,
            coherence_pattern=coherence_pattern,
            hedging_density=affective_result.get('hedging_density', 0.0)
        )
        if self.debug_timing:
            print(f"    [TIMING] dialogue_context: {int((_time.time()-_tc)*1000)}ms", flush=True)

        # === 6. Detect attractor basin ===
        if self.debug_timing:
            _td = _time.time()
        if self.track_history and self.basin_history is not None:
            # Use hysteresis-aware detection with history
            basin_name, basin_confidence, basin_metadata = self.basin_detector.detect(
                psi_vector=psi_vector,
                raw_metrics=metrics,
                dialogue_context=dialogue_context,
                basin_history=self.basin_history
            )
            # Update basin history
            self.basin_history.append(basin_name, basin_confidence)
        else:
            # Legacy mode: no history conditioning
            basin_name, basin_confidence, basin_metadata = self.basin_detector.detect(
                psi_vector=psi_vector,
                raw_metrics=metrics,
                dialogue_context=dialogue_context,
                basin_history=None
            )
        if self.debug_timing:
            print(f"    [TIMING] basin_detect: {int((_time.time()-_td)*1000)}ms", flush=True)

        # === 7. Calculate trajectory dynamics ===
        if self.debug_timing:
            _te = _time.time()
        # Use internal trajectory if tracking, otherwise build from legacy parameter
        if self.track_history and self.trajectory is not None:
            trajectory_buffer = self.trajectory
        elif trajectory_history is not None:
            # Legacy mode: build temporary buffer from provided history
            trajectory_buffer = TrajectoryBuffer()
            for psi in trajectory_history:
                trajectory_buffer.append(psi)
        else:
            trajectory_buffer = None

        trajectory_deriv = compute_trajectory_derivatives(trajectory_buffer)
        attractor_dyn = self._compute_attractor_dynamics(psi_vector, basin_name, trajectory_buffer)
        flow_field = self._compute_flow_field_properties(psi_vector, trajectory_buffer)
        if self.debug_timing:
            print(f"    [TIMING] traj_dynamics: {int((_time.time()-_te)*1000)}ms", flush=True)

        # === 7.5. Movement-preserving classification (v0.3.0+) ===
        if self.debug_timing:
            _tf = _time.time()
        # Compute soft membership
        soft_inference = self.basin_detector.compute_soft_membership(
            psi_vector=psi_vector,
            raw_metrics=metrics,
            previous_inference=self._previous_soft_inference
        )
        self._previous_soft_inference = soft_inference

        # Compute velocity and acceleration magnitudes
        velocity_magnitude = None
        acceleration_magnitude = None
        if trajectory_buffer is not None:
            velocity_magnitude = trajectory_buffer.compute_velocity_magnitude()
            acceleration_magnitude = trajectory_buffer.compute_acceleration_magnitude()

        # Get approach vector
        approach_vector = None
        approach_direction = None
        if trajectory_buffer is not None:
            approach_vector = trajectory_buffer.compute_approach_vector()
            if approach_vector is not None:
                approach_direction = approach_vector.get('direction')

        # Generate movement annotation
        movement_annotation = generate_movement_annotation(
            velocity_magnitude=velocity_magnitude,
            acceleration_magnitude=acceleration_magnitude,
            previous_basin=basin_metadata.get('previous_basin'),
            dwell_time=basin_metadata.get('residence_time', 0)
        )

        # Build TrajectoryStateVector
        trajectory_state_vector = TrajectoryStateVector(
            position=psi_vector,
            velocity_magnitude=velocity_magnitude,
            acceleration_magnitude=acceleration_magnitude,
            curvature=trajectory_deriv.get('curvature'),
            approach_basin=basin_metadata.get('previous_basin'),
            approach_direction=approach_direction,
            dwell_time=basin_metadata.get('residence_time', 0),
            movement_annotation=movement_annotation
        )
        if self.debug_timing:
            print(f"    [TIMING] movement_class: {int((_time.time()-_tf)*1000)}ms", flush=True)

        # === 8. Compute trajectory integrity ===
        if self.debug_timing:
            _tg = _time.time()
        if self.compute_integrity and self.integrity_analyzer is not None and trajectory_buffer is not None:
            trajectory_integrity = self.integrity_analyzer.compute(
                trajectory=trajectory_buffer,
                basin_history=self.basin_history
            )
        else:
            trajectory_integrity = None
        if self.debug_timing:
            print(f"    [TIMING] integrity: {int((_time.time()-_tg)*1000)}ms", flush=True)

        # === 9. Composite Ψ (scalar for backward compatibility) ===
        if psi_biosignal is not None:
            psi_composite = (
                0.4 * psi_vector['psi_semantic'] +
                0.3 * psi_vector['psi_temporal'] +
                0.2 * psi_vector['psi_affective'] +
                0.1 * psi_biosignal
            )
        else:
            psi_composite = (
                0.5 * psi_vector['psi_semantic'] +
                0.3 * psi_vector['psi_temporal'] +
                0.2 * psi_vector['psi_affective']
            )

        # === 10. Build extended return structure ===
        return {
            'psi': float(psi_composite),
            'psi_state': {
                'position': psi_vector,
                'velocity': trajectory_deriv['velocity'],
                'acceleration': trajectory_deriv['acceleration']
            },
            'trajectory_dynamics': {
                'curvature': trajectory_deriv['curvature'],
                'speed': trajectory_deriv['speed'],
                'direction': trajectory_deriv['direction'],
                'path_length': trajectory_buffer.compute_path_length() if trajectory_buffer else None
            },
            'attractor_dynamics': {
                'basin': basin_name,
                'confidence': float(basin_confidence),
                'residence_time': basin_metadata.get('residence_time', 0),
                'previous_basin': basin_metadata.get('previous_basin'),
                'raw_confidence': basin_metadata.get('raw_confidence', basin_confidence),
                'pull_strength': attractor_dyn['pull_strength'],
                'basin_depth': attractor_dyn['basin_depth'],
                'escape_velocity': attractor_dyn['escape_velocity'],
                'basin_stability': attractor_dyn['basin_stability']
            },
            'trajectory_integrity': trajectory_integrity,
            'flow_field': {
                'damping_coefficient': flow_field['damping_coefficient'],
                'turbulence': flow_field['turbulence'],
                'bifurcation_proximity': flow_field['bifurcation_proximity'],
                'lyapunov_exponent': flow_field['lyapunov_exponent']
            },
            'raw_metrics': {
                'delta_kappa': float(metrics['delta_kappa']),
                'delta_h': float(metrics['delta_h']),
                'alpha': float(metrics['alpha'])
            },
            'substrate_details': {
                'semantic': semantic_result,
                'temporal': temporal_result,
                'affective': affective_result,
                'biosignal': {'psi_biosignal': psi_biosignal}
            },
            'dialogue_context': dialogue_context,
            # === Movement-preserving classification (v0.3.0+) ===
            'trajectory_state_vector': trajectory_state_vector.to_dict(),
            'soft_state_inference': soft_inference.to_dict(),
            'movement_aware_label': f"{basin_name} ({movement_annotation})"
        }

    def _compute_attractor_dynamics(
        self,
        psi_vector: dict,
        basin_name: str,
        trajectory_buffer: TrajectoryBuffer = None
    ) -> dict:
        """
        Calculate attractor basin dynamics.

        Args:
            psi_vector: Current Ψ position in phase-space
            basin_name: Detected attractor basin name
            trajectory_buffer: Optional trajectory history

        Returns:
            dict: {
                'pull_strength': float or None,
                'basin_depth': float or None,
                'escape_velocity': float or None,
                'basin_stability': float or None
            }
        """
        # Placeholder for future implementation
        return {
            'pull_strength': None,
            'basin_depth': None,
            'escape_velocity': None,
            'basin_stability': None
        }

    def _compute_flow_field_properties(
        self,
        psi_vector: dict,
        trajectory_buffer: TrajectoryBuffer = None
    ) -> dict:
        """
        Calculate flow field properties.

        Args:
            psi_vector: Current Ψ position in phase-space
            trajectory_buffer: Optional trajectory history

        Returns:
            dict: {
                'damping_coefficient': float or None,
                'turbulence': float or None,
                'bifurcation_proximity': float or None,
                'lyapunov_exponent': float or None
            }
        """
        # Placeholder for future implementation
        return {
            'damping_coefficient': None,
            'turbulence': None,
            'bifurcation_proximity': None,
            'lyapunov_exponent': None
        }
