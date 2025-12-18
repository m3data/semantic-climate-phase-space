"""
Semantic Climate Phase Space - Extensions Module (DEPRECATED)

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

DEPRECATION NOTICE (v0.2.0):
    This module is deprecated and will be removed in v0.3.0.

    Migration guide:
    - Instead of: from src.extensions import SemanticClimateAnalyzer
      Use:        from src import SemanticClimateAnalyzer

    - Instead of: from src.extensions import TrajectoryBuffer
      Use:        from src import TrajectoryBuffer

    - The new SemanticClimateAnalyzer from src.analyzer provides:
      - Same compute_coupling_coefficient() interface
      - New track_history and compute_integrity constructor options
      - New trajectory_integrity output field
      - Enhanced attractor_dynamics with residence_time

    For direct access to modular components:
    - from src import compute_dialogue_context  (was method, now function)
    - from src import BasinDetector  (replaces detect_attractor_basin method)
    - from src import IntegrityAnalyzer  (new in v0.2.0)

This module extends Morgoulis (2025) 4D Semantic Coupling Framework with:
- Vector Ψ representation (4D phase-space)
- Trajectory buffer for temporal dynamics
- Attractor basin detection
- Substrate computation (semantic, temporal, affective, biosignal)

These extensions build on the core metrics (Δκ, α, ΔH) to provide a richer
dynamical systems perspective on human-AI dialogue.

Dependencies:
    - src.core_metrics.SemanticComplexityAnalyzer (Morgoulis, 2025, MIT)
    - vaderSentiment (for affective substrate)
"""
import warnings

warnings.warn(
    "Importing from src.extensions is deprecated and will be removed in v0.3.0. "
    "Use 'from src import SemanticClimateAnalyzer' instead. "
    "See extensions.py docstring for full migration guide.",
    DeprecationWarning,
    stacklevel=2
)

from typing import Optional, List, Dict, Tuple
import numpy as np
import re

from .core_metrics import SemanticComplexityAnalyzer

__all__ = [
    'TrajectoryBuffer',
    'SemanticClimateAnalyzer',
]


class TrajectoryBuffer:
    """
    Windowed storage of Ψ trajectory for derivative calculation.

    Stores recent history of Ψ vectors to enable computation of:
    - Velocity (dΨ/dt): First derivative
    - Acceleration (d²Ψ/dt²): Second derivative
    - Curvature: Rate of directional change
    - Attractor pull strength: Force toward basin center

    Phase 1: Stub implementation with placeholders
    Phase 3: Full implementation with circular buffer and derivative methods
    """

    def __init__(self, window_size: int = 50, timestep: float = 1.0):
        """
        Initialize trajectory buffer.

        Args:
            window_size: Maximum number of Ψ observations to retain
            timestep: Time interval between observations (default: 1 turn)
        """
        self.window_size = window_size
        self.timestep = timestep
        self.history: List[Tuple[float, dict]] = []

    def append(self, psi_vector: dict, timestamp: float = None) -> None:
        """
        Add new Ψ observation to buffer.

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            timestamp: Optional timestamp (if None, use sequential numbering)
        """
        if timestamp is None:
            timestamp = float(len(self.history)) * self.timestep

        self.history.append((timestamp, psi_vector))

        # Maintain window size
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def compute_velocity(self) -> Optional[dict]:
        """
        Calculate dΨ/dt via finite differences.

        Returns:
            dict: {
                'dpsi_semantic_dt': float,
                'dpsi_affective_dt': float,
                'dpsi_temporal_dt': float,
                'dpsi_biosignal_dt': float
            } or None if insufficient history
        """
        if len(self.history) < 3:
            return None

        # Central difference: (Ψ[t+1] - Ψ[t-1]) / (2 * dt)
        t_prev, psi_prev = self.history[-3]
        t_next, psi_next = self.history[-1]
        dt = t_next - t_prev

        if dt == 0:
            return None

        velocity = {}
        for key in ['psi_semantic', 'psi_temporal', 'psi_affective', 'psi_biosignal']:
            prev_val = psi_prev.get(key)
            next_val = psi_next.get(key)

            if prev_val is not None and next_val is not None:
                velocity[f'd{key}_dt'] = (next_val - prev_val) / dt
            else:
                velocity[f'd{key}_dt'] = None

        return velocity

    def compute_acceleration(self) -> Optional[dict]:
        """
        Calculate d²Ψ/dt² via second-order finite differences.

        Returns:
            dict: {
                'd2psi_semantic_dt2': float,
                'd2psi_affective_dt2': float,
                'd2psi_temporal_dt2': float,
                'd2psi_biosignal_dt2': float
            } or None if insufficient history
        """
        if len(self.history) < 3:
            return None

        # Second derivative: (Ψ[t+1] - 2*Ψ[t] + Ψ[t-1]) / dt²
        t_prev, psi_prev = self.history[-3]
        t_curr, psi_curr = self.history[-2]
        t_next, psi_next = self.history[-1]

        dt = (t_next - t_prev) / 2  # Average timestep

        if dt == 0:
            return None

        acceleration = {}
        for key in ['psi_semantic', 'psi_temporal', 'psi_affective', 'psi_biosignal']:
            prev_val = psi_prev.get(key)
            curr_val = psi_curr.get(key)
            next_val = psi_next.get(key)

            if all(v is not None for v in [prev_val, curr_val, next_val]):
                acceleration[f'd2{key}_dt2'] = (next_val - 2*curr_val + prev_val) / (dt ** 2)
            else:
                acceleration[f'd2{key}_dt2'] = None

        return acceleration

    def compute_curvature(self) -> Optional[float]:
        """
        Calculate trajectory curvature (rate of directional change).

        Returns:
            float: Curvature magnitude or None if insufficient history
        """
        velocity = self.compute_velocity()
        acceleration = self.compute_acceleration()

        if velocity is None or acceleration is None:
            return None

        # Extract numeric values only
        v_vals = [v for v in velocity.values() if v is not None]
        a_vals = [v for v in acceleration.values() if v is not None]

        if len(v_vals) < 2 or len(a_vals) < 2:
            return None

        v = np.array(v_vals)
        a = np.array(a_vals[:len(v_vals)])  # Match dimensions

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return None

        # Curvature = ||v × a|| / ||v||³ (generalized for N dimensions)
        # In N-D, use: sqrt(||v||² ||a||² - (v·a)²) / ||v||³
        cross_magnitude = np.sqrt(
            np.linalg.norm(v)**2 * np.linalg.norm(a)**2 - np.dot(v, a)**2
        )

        return float(cross_magnitude / (v_norm ** 3))

    def get_trajectory_segment(self, n_points: int = None) -> Optional[List[Tuple[float, dict]]]:
        """
        Retrieve recent trajectory segment.

        Args:
            n_points: Number of recent points to retrieve (None = all)

        Returns:
            list: List of (timestamp, psi_vector) tuples or None if empty
        """
        if len(self.history) == 0:
            return None

        if n_points is None:
            return self.history.copy()

        return self.history[-n_points:]


class SemanticClimateAnalyzer(SemanticComplexityAnalyzer):
    """
    Extended analyzer for Semantic Climate Model.

    Inherits from SemanticComplexityAnalyzer (Morgoulis, 2025):
    - Semantic Curvature (Δκ)
    - Fractal Similarity Score (α)
    - Entropy Shift (ΔH)

    Adds:
    - Vector Ψ representation (semantic, temporal, affective, biosignal)
    - Attractor basin detection
    - Trajectory dynamics (velocity, acceleration, curvature)
    - Flow field properties
    """

    def __init__(self, random_state: int = 42, bootstrap_iterations: int = 1000):
        """
        Initialize the extended analyzer.

        Args:
            random_state: Random seed for reproducibility
            bootstrap_iterations: Number of bootstrap samples for CI
        """
        super().__init__(random_state=random_state, bootstrap_iterations=bootstrap_iterations)

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
            dialogue_embeddings: Sequence of embeddings (required)
            turn_texts: List of conversational turn texts for affective substrate
            turn_speakers: List of speaker labels ('human', 'ai') for turn asymmetry
            timestamps: Optional turn timestamps for temporal synchrony
            metrics: Pre-computed semantic climate metrics (optional)
            trajectory_history: List of past Ψ vectors for dynamics
            trajectory_metrics: List of per-window metrics dicts for Δκ variance
            biosignal_data: Physiological signals for biosignal substrate
            temporal_window_size: Window size for temporal coherence calculation
            coherence_pattern: Pre-computed coherence pattern ('breathing', etc.)

        Returns:
            dict: Extended structure with:
                - psi: Composite score (backward compatible)
                - psi_state: position, velocity, acceleration
                - trajectory_dynamics: curvature, speed, direction, path_length
                - attractor_dynamics: basin, confidence, pull, depth, escape, stability
                - flow_field: damping, turbulence, bifurcation, lyapunov
                - raw_metrics: Δκ, ΔH, α
                - substrate_details: per-substrate breakdowns
                - dialogue_context: context used for basin detection
        """
        # === 1. Calculate raw metrics if not provided ===
        if metrics is None:
            if dialogue_embeddings is None:
                raise ValueError("Must provide dialogue_embeddings")
            metrics = self.calculate_all_metrics(dialogue_embeddings)

        # === 2. Calculate substrate components ===

        # Semantic substrate (from Δκ, ΔH, α)
        semantic_result = self.compute_semantic_substrate(
            embeddings=dialogue_embeddings if dialogue_embeddings is not None else np.array([]),
            metrics=metrics
        )

        # Temporal substrate (from metric stability + timing if available)
        temporal_result = self.compute_temporal_substrate(
            embeddings=dialogue_embeddings,
            timestamps=timestamps,
            metrics=metrics,
            temporal_window_size=temporal_window_size
        )

        # Affective substrate (from turn texts if available)
        if turn_texts is not None and len(turn_texts) > 0:
            affective_result = self.compute_affective_substrate(
                turn_texts=turn_texts,
                embeddings=dialogue_embeddings
            )
        else:
            affective_result = {
                'psi_affective': 0.0,
                'sentiment_trajectory': [],
                'hedging_density': 0.0,
                'vulnerability_score': 0.0,
                'confidence_variance': 0.0
            }

        # Biosignal substrate
        if biosignal_data is not None:
            psi_biosignal = self._compute_biosignal_substrate(biosignal_data)
        else:
            psi_biosignal = None

        # === 3. Build Ψ vector ===
        psi_vector = {
            'psi_semantic': semantic_result['psi_semantic'],
            'psi_temporal': temporal_result['psi_temporal'],
            'psi_affective': affective_result['psi_affective'],
            'psi_biosignal': psi_biosignal
        }

        # === 4. Compute dialogue context for refined basin detection ===
        dialogue_context = self.compute_dialogue_context(
            turn_texts=turn_texts,
            turn_speakers=turn_speakers,
            trajectory_metrics=trajectory_metrics,
            coherence_pattern=coherence_pattern,
            hedging_density=affective_result.get('hedging_density', 0.0)
        )

        # === 5. Detect attractor basin ===
        basin_name, basin_confidence = self.detect_attractor_basin(
            psi_vector=psi_vector,
            raw_metrics=metrics,
            dialogue_context=dialogue_context
        )

        # === 6. Calculate dynamics ===
        trajectory_buffer = None
        if trajectory_history is not None:
            trajectory_buffer = TrajectoryBuffer()
            for psi in trajectory_history:
                trajectory_buffer.append(psi)

        trajectory_deriv = self.compute_trajectory_derivatives(trajectory_buffer)
        attractor_dyn = self.compute_attractor_dynamics(psi_vector, basin_name, trajectory_buffer)
        flow_field = self.compute_flow_field_properties(psi_vector, trajectory_buffer)

        # === 7. Composite Ψ (scalar for backward compatibility) ===
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

        # === 8. Build extended return structure ===
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
                'path_length': None
            },
            'attractor_dynamics': {
                'basin': basin_name,
                'confidence': float(basin_confidence),
                'pull_strength': attractor_dyn['pull_strength'],
                'basin_depth': attractor_dyn['basin_depth'],
                'escape_velocity': attractor_dyn['escape_velocity'],
                'basin_stability': attractor_dyn['basin_stability']
            },
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
            'dialogue_context': dialogue_context
        }

    def _compute_biosignal_substrate(self, biosignal_data: dict) -> float:
        """
        Compute biosignal substrate from physiological data.

        Args:
            biosignal_data: Dict with heart_rate, hrv, gsr, etc.

        Returns:
            float: Normalized biosignal substrate value
        """
        # Basic implementation - can be extended
        hr = biosignal_data.get('heart_rate')
        if hr is not None:
            # Normalize HR around resting (60-100 bpm typical)
            hr_normalized = (hr - 80) / 40  # Maps ~60-100 to [-0.5, 0.5]
            return float(np.tanh(hr_normalized))
        return 0.0

    def compute_dialogue_context(
        self,
        turn_texts: List[str] = None,
        turn_speakers: List[str] = None,
        trajectory_metrics: List[dict] = None,
        coherence_pattern: str = None,
        hedging_density: float = 0.0
    ) -> dict:
        """
        Compute dialogue context for refined basin detection.

        This provides the discriminating features needed to distinguish:
        - Cognitive Mimicry (performed engagement)
        - Collaborative Inquiry (genuine co-exploration)
        - Reflexive Performance (performed self-examination)

        Args:
            turn_texts: List of turn text strings
            turn_speakers: List of speaker labels ('human', 'ai', 'user', 'assistant')
            trajectory_metrics: List of per-window metric dicts with 'delta_kappa'
            coherence_pattern: Pre-computed coherence pattern string
            hedging_density: Pre-computed hedging density from affective substrate

        Returns:
            dict: {
                'hedging_density': float,
                'turn_length_ratio': float (AI avg / human avg),
                'delta_kappa_variance': float,
                'coherence_pattern': str
            }
        """
        context = {
            'hedging_density': hedging_density,
            'turn_length_ratio': 1.0,
            'delta_kappa_variance': 0.0,
            'coherence_pattern': coherence_pattern or 'transitional'
        }

        # Compute turn length ratio (AI / human)
        if turn_texts is not None and turn_speakers is not None:
            if len(turn_texts) == len(turn_speakers):
                ai_lengths = []
                human_lengths = []

                for text, speaker in zip(turn_texts, turn_speakers):
                    # Normalize speaker labels
                    speaker_lower = speaker.lower() if speaker else ''
                    word_count = len(text.split()) if text else 0

                    if speaker_lower in ('ai', 'assistant', 'model', 'claude'):
                        ai_lengths.append(word_count)
                    elif speaker_lower in ('human', 'user', 'participant'):
                        human_lengths.append(word_count)

                if ai_lengths and human_lengths:
                    ai_avg = np.mean(ai_lengths)
                    human_avg = np.mean(human_lengths)
                    if human_avg > 0:
                        context['turn_length_ratio'] = float(ai_avg / human_avg)

        # Compute Δκ variance across trajectory windows
        if trajectory_metrics is not None and len(trajectory_metrics) >= 2:
            dk_values = [
                m.get('delta_kappa', 0.0)
                for m in trajectory_metrics
                if m is not None and 'delta_kappa' in m
            ]
            if len(dk_values) >= 2:
                context['delta_kappa_variance'] = float(np.var(dk_values))

        return context

    def compute_affective_substrate(
        self,
        turn_texts: List[str],
        embeddings: np.ndarray = None
    ) -> dict:
        """
        Calculate Ψ_affective from conversational text.

        Measures emotional safety, openness, vulnerability, and epistemic stance.

        Args:
            turn_texts: List of conversational turn texts
            embeddings: Optional embeddings (reserved for future)

        Returns:
            dict: {
                'psi_affective': Composite affective substrate score [-1, 1]
                'sentiment_trajectory': List of per-turn sentiment scores
                'hedging_density': Proportion of hedging markers
                'vulnerability_score': Vulnerability indicator density
                'confidence_variance': Variability in confidence expression
            }
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            # Fallback if vaderSentiment not installed
            return {
                'psi_affective': 0.0,
                'sentiment_trajectory': [],
                'hedging_density': 0.0,
                'vulnerability_score': 0.0,
                'confidence_variance': 0.0
            }

        if not turn_texts or len(turn_texts) == 0:
            return {
                'psi_affective': 0.0,
                'sentiment_trajectory': [],
                'hedging_density': 0.0,
                'vulnerability_score': 0.0,
                'confidence_variance': 0.0
            }

        vader = SentimentIntensityAnalyzer()

        # === 1. Sentiment Trajectory Analysis ===
        sentiment_scores = []
        for text in turn_texts:
            scores = vader.polarity_scores(text)
            sentiment_scores.append(scores['compound'])

        sentiment_variance = float(np.var(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0

        # === 2. Hedging Pattern Detection ===
        hedging_patterns = [
            r'\b(I think|I guess|I suppose|maybe|perhaps|possibly|probably|might|could be|seems like|sort of|kind of)\b',
            r'\b(I\'m not sure|I wonder|I feel like|it appears|it seems)\b',
            r'\b(arguably|presumably|apparently|seemingly)\b'
        ]

        total_words = 0
        hedging_count = 0

        for text in turn_texts:
            words = text.split()
            total_words += len(words)
            for pattern in hedging_patterns:
                hedging_count += len(re.findall(pattern, text, re.IGNORECASE))

        hedging_density = float(hedging_count / max(total_words, 1))

        # === 3. Vulnerability Indicators ===
        vulnerability_patterns = [
            r'\b(I feel|I\'m feeling|I felt)\b',
            r'\b(I\'m|I am)\s+(scared|worried|afraid|anxious|nervous|uncertain|confused|overwhelmed)\b',
            r'\b(my|I)\s+(fear|worry|concern|anxiety|doubt)\b',
            r'\b(honestly|to be honest|truthfully|frankly)\b',
            r'\b(I don\'t know|I\'m struggling|I\'m not sure|I\'m uncertain)\b'
        ]

        emotion_words = [
            'afraid', 'angry', 'anxious', 'confused', 'disappointed', 'excited',
            'frustrated', 'grateful', 'happy', 'hopeful', 'lonely', 'sad',
            'scared', 'surprised', 'uncertain', 'worried'
        ]

        vulnerability_count = 0
        emotion_count = 0

        for text in turn_texts:
            text_lower = text.lower()
            for pattern in vulnerability_patterns:
                vulnerability_count += len(re.findall(pattern, text, re.IGNORECASE))
            for emotion in emotion_words:
                if re.search(r'\b' + emotion + r'\b', text_lower):
                    emotion_count += 1

        vulnerability_score = float((vulnerability_count + emotion_count) / max(total_words, 1))

        # === 4. Confidence Markers ===
        confidence_patterns = [
            r'\b(definitely|certainly|absolutely|clearly|obviously|undoubtedly)\b',
            r'\b(I\'m certain|I\'m sure|I know|without doubt|no question)\b',
            r'\b(always|never|must|will)\b'
        ]

        confidence_counts = []
        for text in turn_texts:
            turn_confidence = 0
            turn_words = len(text.split())
            for pattern in confidence_patterns:
                turn_confidence += len(re.findall(pattern, text, re.IGNORECASE))
            confidence_counts.append(turn_confidence / max(turn_words, 1))

        confidence_variance = float(np.var(confidence_counts)) if len(confidence_counts) > 1 else 0.0

        # === Composite Ψ_affective Calculation ===
        sentiment_norm = min(sentiment_variance / 0.5, 1.0)
        hedging_norm = min(hedging_density / 0.1, 1.0)
        vulnerability_norm = min(vulnerability_score / 0.05, 1.0)
        confidence_norm = min(confidence_variance / 0.01, 1.0)

        psi_affective_raw = (
            0.3 * sentiment_norm +
            0.3 * hedging_norm +
            0.3 * vulnerability_norm +
            0.1 * confidence_norm
        )

        psi_affective = np.tanh(2 * (psi_affective_raw - 0.5))

        return {
            'psi_affective': float(psi_affective),
            'sentiment_trajectory': sentiment_scores,
            'hedging_density': float(hedging_density),
            'vulnerability_score': float(vulnerability_score),
            'confidence_variance': float(confidence_variance)
        }

    def compute_semantic_substrate(
        self,
        embeddings: np.ndarray,
        metrics: dict
    ) -> dict:
        """
        Calculate Ψ_semantic from semantic climate metrics.

        Combines Δκ, ΔH, α into composite semantic substrate score.

        Args:
            embeddings: Dialogue embeddings (reserved for future alignment calc)
            metrics: Pre-computed semantic climate metrics (Δκ, ΔH, α)

        Returns:
            dict: {
                'psi_semantic': Composite semantic substrate score [-1, 1]
                'alignment_score': Alignment vs divergence (None for now)
                'exploratory_depth': Δκ (semantic curvature as exploration measure)
                'raw_metrics': Dict of Δκ, ΔH, α values
            }
        """
        delta_kappa = metrics['delta_kappa']
        delta_h = metrics['delta_h']
        alpha = metrics['alpha']

        # PC1 approximation with equal weights
        weights = np.array([0.577, 0.577, 0.577])

        # Standardize metrics (z-score relative to typical ranges)
        # RECALIBRATED 2025-12-08 for fixed metric implementations:
        # - Δκ (local curvature): Range ~[0, 0.5], center ~0.15, std ~0.15
        # - ΔH (JS divergence): Range ~[0, 0.5], center ~0.15, std ~0.15
        # - α (DFA on semantic velocity): Range ~[0.5, 1.5], center ~0.8, std ~0.3
        metric_std = np.array([
            (delta_kappa - 0.15) / 0.15,  # Local curvature has lower typical values
            (delta_h - 0.15) / 0.15,      # JS divergence similar to old entropy diff
            (alpha - 0.8) / 0.3           # Wider range with semantic velocity DFA
        ])

        psi_semantic_raw = np.dot(metric_std, weights) / np.linalg.norm(weights)
        psi_semantic = np.tanh(psi_semantic_raw)

        return {
            'psi_semantic': float(psi_semantic),
            'alignment_score': None,
            'exploratory_depth': float(delta_kappa),
            'raw_metrics': {
                'delta_kappa': float(delta_kappa),
                'delta_h': float(delta_h),
                'alpha': float(alpha)
            }
        }

    def compute_temporal_substrate(
        self,
        embeddings: np.ndarray,
        timestamps: List[float] = None,
        metrics: dict = None,
        temporal_window_size: int = 10
    ) -> dict:
        """
        Calculate Ψ_temporal from metric stability and timing patterns.

        Args:
            embeddings: Dialogue embeddings for windowed analysis
            timestamps: Optional turn timestamps for synchrony calculation
            metrics: Pre-computed full-dialogue metrics (optional)
            temporal_window_size: Window size for stability calculation

        Returns:
            dict: {
                'psi_temporal': Composite temporal substrate score [0, 1]
                'metric_stability': Inverse coefficient of variation
                'turn_synchrony': Timing regularity (None for now)
                'rhythm_score': Entrainment measure (None for now)
            }
        """
        if embeddings is not None and len(embeddings) >= temporal_window_size * 2:
            window_metrics = []

            for i in range(len(embeddings) - temporal_window_size + 1):
                window_embs = embeddings[i:i+temporal_window_size]
                if len(window_embs) >= 6:
                    try:
                        window_result = self.calculate_all_metrics(window_embs)
                        window_metrics.append([
                            window_result['delta_kappa'],
                            window_result['delta_h'],
                            window_result['alpha']
                        ])
                    except:
                        pass

            if len(window_metrics) >= 3:
                window_array = np.array(window_metrics)
                window_array_std = (window_array - np.mean(window_array, axis=0)) / (np.std(window_array, axis=0) + 1e-10)

                weights = np.array([0.577, 0.577, 0.577])
                loadings = weights / np.linalg.norm(weights)
                window_psi = np.dot(window_array_std, loadings)

                cv = np.std(window_psi) / (np.abs(np.mean(window_psi)) + 1e-10)
                psi_temporal = 1 / (1 + cv)
                metric_stability = psi_temporal
            else:
                psi_temporal = 0.5
                metric_stability = 0.5
        else:
            psi_temporal = 0.5
            metric_stability = 0.5

        return {
            'psi_temporal': float(psi_temporal),
            'metric_stability': float(metric_stability),
            'turn_synchrony': None,
            'rhythm_score': None
        }

    def detect_attractor_basin(
        self,
        psi_vector: dict,
        raw_metrics: dict = None,
        dialogue_context: dict = None
    ) -> Tuple[str, float]:
        """
        Classify phase-space position into attractor basins.

        Refined taxonomy (v2) distinguishes performative from genuine engagement:

        Basins:
        1. Cognitive Mimicry: Model performs engagement without genuine uncertainty
           - High semantic, low affect, smooth Δκ, AI dominates turn length
        2. Collaborative Inquiry: Genuine co-exploration with mutual uncertainty
           - High semantic, low affect, oscillating Δκ, balanced turns, hedging present
        3. Reflexive Performance: Model appears to self-examine but pattern-matches
           - High semantic, performed uncertainty, scripted oscillation
        4. Sycophantic Convergence: High alignment + low Δκ, low affect
        5. Creative Dilation: Divergent + high Δκ, high affect
        6. Generative Conflict: High divergent semantic, high affect
        7. Deep Resonance: All substrates high
        8. Embodied Coherence: Low semantic, high biosignal
        9. Dissociation: All substrates low

        Args:
            psi_vector: Dict with psi_semantic, psi_temporal, psi_affective, psi_biosignal
            raw_metrics: Optional dict with delta_kappa, delta_h, alpha
            dialogue_context: Optional dict with:
                - hedging_density: float (from affective substrate)
                - turn_length_ratio: float (AI avg / human avg, >1 = AI dominates)
                - delta_kappa_variance: float (variance across windows)
                - coherence_pattern: str ('breathing', 'transitional', 'locked', 'fragmented')

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

    def compute_trajectory_derivatives(
        self,
        trajectory_buffer: TrajectoryBuffer = None
    ) -> dict:
        """
        Calculate trajectory derivatives (velocity, acceleration, curvature).

        Args:
            trajectory_buffer: TrajectoryBuffer with recent Ψ history

        Returns:
            dict: {
                'velocity': dict or None,
                'acceleration': dict or None,
                'curvature': float or None,
                'speed': float or None,
                'direction': dict or None
            }
        """
        if trajectory_buffer is None:
            return {
                'velocity': None,
                'acceleration': None,
                'curvature': None,
                'speed': None,
                'direction': None
            }

        velocity = trajectory_buffer.compute_velocity()
        acceleration = trajectory_buffer.compute_acceleration()
        curvature = trajectory_buffer.compute_curvature()

        # Calculate speed and direction from velocity
        speed = None
        direction = None

        if velocity is not None:
            v_vals = [v for v in velocity.values() if v is not None]
            if v_vals:
                speed = float(np.linalg.norm(v_vals))
                if speed > 1e-10:
                    direction = {k: v / speed for k, v in velocity.items() if v is not None}

        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'curvature': curvature,
            'speed': speed,
            'direction': direction
        }

    def compute_attractor_dynamics(
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

    def compute_flow_field_properties(
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
