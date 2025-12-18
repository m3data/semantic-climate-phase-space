"""
Ψ Substrate Computation Module

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

This module provides substrate computation functions for the Semantic Climate
Phase Space model. Each substrate represents one dimension of the Ψ vector:
- Semantic: Derived from Δκ, ΔH, α metrics
- Temporal: Derived from metric stability over time
- Affective: Derived from sentiment, hedging, vulnerability patterns
- Biosignal: Derived from physiological data

Functions:
    compute_semantic_substrate: Ψ_semantic from core metrics
    compute_temporal_substrate: Ψ_temporal from metric stability
    compute_affective_substrate: Ψ_affective from text analysis
    compute_biosignal_substrate: Ψ_biosignal from physio data
    compute_dialogue_context: Context features for basin detection
"""

from typing import List, Optional
import numpy as np
import re

__all__ = [
    'compute_semantic_substrate',
    'compute_temporal_substrate',
    'compute_affective_substrate',
    'compute_biosignal_substrate',
    'compute_dialogue_context',
]


def compute_semantic_substrate(
    embeddings: np.ndarray,
    metrics: dict
) -> dict:
    """
    Calculate Ψ_semantic from semantic climate metrics.

    Combines Δκ, ΔH, α into composite semantic substrate score using
    a PC1-like weighting with equal contributions.

    Args:
        embeddings: Dialogue embeddings (reserved for future alignment calc)
        metrics: Pre-computed semantic climate metrics with:
            - delta_kappa: Semantic curvature
            - delta_h: Entropy shift
            - alpha: Fractal similarity

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
    embeddings: np.ndarray,
    timestamps: List[float] = None,
    metrics: dict = None,
    window_size: int = 10
) -> dict:
    """
    Calculate Ψ_temporal from metric stability and timing patterns.

    Measures how consistently the semantic climate metrics behave over
    sliding windows. High stability indicates a settled/predictable dynamic.

    Args:
        embeddings: Dialogue embeddings for windowed analysis
        timestamps: Optional turn timestamps for synchrony calculation
        metrics: Pre-computed full-dialogue metrics (optional)
        window_size: Window size for stability calculation (default: 10)

    Returns:
        dict: {
            'psi_temporal': Composite temporal substrate score [0, 1]
            'metric_stability': Inverse coefficient of variation
            'turn_synchrony': Timing regularity (None for now)
            'rhythm_score': Entrainment measure (None for now)
        }
    """
    # Import here to avoid circular dependency
    from .core_metrics import SemanticComplexityAnalyzer

    if embeddings is not None and len(embeddings) >= window_size * 2:
        analyzer = SemanticComplexityAnalyzer()
        window_metrics = []

        for i in range(len(embeddings) - window_size + 1):
            window_embs = embeddings[i:i+window_size]
            if len(window_embs) >= 6:
                try:
                    window_result = analyzer.calculate_all_metrics(window_embs)
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


def compute_affective_substrate(
    turn_texts: List[str],
    embeddings: np.ndarray = None
) -> dict:
    """
    Calculate Ψ_affective from conversational text.

    Measures emotional safety, openness, vulnerability, and epistemic stance
    through sentiment analysis, hedging detection, and vulnerability indicators.

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


def compute_biosignal_substrate(biosignal_data: dict) -> float:
    """
    Compute biosignal substrate from physiological data.

    Currently supports heart rate normalization. Future versions will
    incorporate HRV, GSR, and other biosignals.

    Args:
        biosignal_data: Dict with heart_rate, hrv, gsr, etc.

    Returns:
        float: Normalized biosignal substrate value [-1, 1]
    """
    hr = biosignal_data.get('heart_rate')
    if hr is not None:
        # Normalize HR around resting (60-100 bpm typical)
        hr_normalized = (hr - 80) / 40  # Maps ~60-100 to [-0.5, 0.5]
        return float(np.tanh(hr_normalized))
    return 0.0


def compute_dialogue_context(
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
