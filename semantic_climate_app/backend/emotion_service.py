"""
GoEmotions-based emotion analysis service.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Provides fine-grained emotion detection using GoEmotions (28 categories)
as an async complement to VADER's fast lexicon-based sentiment.

Model: SamLowe/roberta-base-go_emotions
- Multi-label classification (emotions can co-occur)
- 28 outputs: 27 emotions + neutral
- RoBERTa-base architecture (~125M params)

Usage:
    service = EmotionService()
    emotions = service.analyze("I'm curious about this but also a bit nervous")
    # {'curiosity': 0.82, 'nervousness': 0.45, 'neutral': 0.12, ...}
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from threading import Thread, Lock
from queue import Queue, Empty
import time


# GoEmotions label set (27 emotions + neutral)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Semantic groupings for Psi_affective computation
EPISTEMIC_EMOTIONS = ['curiosity', 'confusion', 'realization', 'surprise']
SAFETY_POSITIVE = ['caring', 'gratitude', 'love', 'admiration', 'approval', 'relief']
SAFETY_NEGATIVE = ['nervousness', 'fear', 'embarrassment', 'grief', 'sadness']
VALENCE_POSITIVE = ['joy', 'excitement', 'amusement', 'optimism', 'pride']
VALENCE_NEGATIVE = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'remorse']


@dataclass
class EmotionResult:
    """Result from emotion analysis."""
    text: str
    scores: Dict[str, float]  # All 28 emotion scores
    top_emotions: List[tuple]  # Top N (label, score) pairs
    epistemic_score: float  # Aggregate epistemic signal
    safety_score: float  # Safety+ minus Safety- (felt safety)
    valence: float  # Positive minus negative valence
    timestamp: float  # When analysis completed


class EmotionService:
    """
    GoEmotions-based emotion analysis with async support.

    Provides:
    - Synchronous single/batch analysis
    - Background async processing with callback
    - Caching to avoid repeat inference
    """

    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        device: str = "cpu",
        lazy_load: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize emotion service.

        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda"
            lazy_load: If True, defer model loading until first use
            cache_size: Max cached results (LRU eviction)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._loaded = False
        self._load_lock = Lock()

        # Simple cache: text -> EmotionResult
        self._cache: Dict[str, EmotionResult] = {}
        self._cache_size = cache_size
        self._cache_order: List[str] = []  # For LRU eviction

        # Async processing
        self._queue: Queue = Queue()
        self._worker_thread: Optional[Thread] = None
        self._running = False

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the GoEmotions model."""
        with self._load_lock:
            if self._loaded:
                return

            print(f"Loading GoEmotions model: {self.model_name}...")
            start = time.time()

            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all labels
                device=0 if self.device == "cuda" else -1
            )

            self._loaded = True
            elapsed = time.time() - start
            print(f"  GoEmotions model loaded ({elapsed:.1f}s)")

    def _ensure_loaded(self):
        """Ensure model is loaded before use."""
        if not self._loaded:
            self._load_model()

    def _compute_aggregates(self, scores: Dict[str, float]) -> tuple:
        """
        Compute aggregate scores from raw emotion scores.

        Returns:
            (epistemic_score, safety_score, valence)
        """
        # Epistemic: curiosity + realization - confusion (confusion is ambiguous)
        epistemic = (
            scores.get('curiosity', 0) * 1.0 +
            scores.get('realization', 0) * 1.0 +
            scores.get('surprise', 0) * 0.5 -
            scores.get('confusion', 0) * 0.3  # Confusion can be productive
        )

        # Safety: positive relational emotions minus threat emotions
        safety_pos = sum(scores.get(e, 0) for e in SAFETY_POSITIVE)
        safety_neg = sum(scores.get(e, 0) for e in SAFETY_NEGATIVE)
        safety = safety_pos - safety_neg

        # Valence: positive minus negative
        valence_pos = sum(scores.get(e, 0) for e in VALENCE_POSITIVE)
        valence_neg = sum(scores.get(e, 0) for e in VALENCE_NEGATIVE)
        valence = valence_pos - valence_neg

        return epistemic, safety, valence

    def _to_result(self, text: str, raw_output: List[Dict]) -> EmotionResult:
        """Convert pipeline output to EmotionResult."""
        # Pipeline returns [{'label': 'curiosity', 'score': 0.82}, ...]
        scores = {item['label']: item['score'] for item in raw_output}

        # Ensure all labels present
        for label in EMOTION_LABELS:
            if label not in scores:
                scores[label] = 0.0

        # Top emotions (above threshold)
        top = sorted(
            [(k, v) for k, v in scores.items() if v > 0.1],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        epistemic, safety, valence = self._compute_aggregates(scores)

        return EmotionResult(
            text=text,
            scores=scores,
            top_emotions=top,
            epistemic_score=float(epistemic),
            safety_score=float(safety),
            valence=float(valence),
            timestamp=time.time()
        )

    def _cache_result(self, text: str, result: EmotionResult):
        """Add result to cache with LRU eviction."""
        if text in self._cache:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[text] = result
        self._cache_order.append(text)

    def analyze(self, text: str, use_cache: bool = True) -> EmotionResult:
        """
        Analyze emotions in text (synchronous).

        Args:
            text: Input text
            use_cache: Whether to use/update cache

        Returns:
            EmotionResult with scores and aggregates
        """
        # Check cache
        if use_cache and text in self._cache:
            return self._cache[text]

        self._ensure_loaded()

        # Run inference
        raw_output = self._pipeline(text)[0]  # [0] because single input
        result = self._to_result(text, raw_output)

        if use_cache:
            self._cache_result(text, result)

        return result

    def analyze_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[EmotionResult]:
        """
        Analyze emotions in multiple texts (synchronous batch).

        Args:
            texts: List of input texts
            use_cache: Whether to use/update cache

        Returns:
            List of EmotionResult
        """
        self._ensure_loaded()

        results = []
        texts_to_process = []
        indices_to_process = []

        # Check cache first
        for i, text in enumerate(texts):
            if use_cache and text in self._cache:
                results.append(self._cache[text])
            else:
                results.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)

        # Batch process uncached
        if texts_to_process:
            raw_outputs = self._pipeline(texts_to_process)

            for text, raw_output, idx in zip(
                texts_to_process, raw_outputs, indices_to_process
            ):
                result = self._to_result(text, raw_output)
                results[idx] = result

                if use_cache:
                    self._cache_result(text, result)

        return results

    # === Async Processing ===

    def _worker_loop(self):
        """Background worker for async processing."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break

                text, callback = item
                result = self.analyze(text)

                if callback:
                    callback(result)

            except Empty:
                continue
            except Exception as e:
                print(f"EmotionService worker error: {e}")

    def start_async(self):
        """Start background processing thread."""
        if self._worker_thread is not None:
            return

        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_async(self):
        """Stop background processing thread."""
        self._running = False
        self._queue.put(None)  # Sentinel
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def analyze_async(
        self,
        text: str,
        callback: callable = None
    ):
        """
        Queue text for async analysis.

        Args:
            text: Input text
            callback: Function to call with EmotionResult when ready
        """
        if not self._running:
            self.start_async()

        self._queue.put((text, callback))

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def cache_stats(self) -> Dict:
        """Return cache statistics."""
        return {
            'size': len(self._cache),
            'capacity': self._cache_size,
            'hit_rate': None  # Could track if needed
        }


def extract_epistemic_trajectory(
    results: List[EmotionResult]
) -> Dict[str, List[float]]:
    """
    Extract epistemic emotion trajectories from a sequence of results.

    Returns:
        Dict with lists for each epistemic emotion over time
    """
    trajectory = {emotion: [] for emotion in EPISTEMIC_EMOTIONS}
    trajectory['epistemic_composite'] = []

    for result in results:
        for emotion in EPISTEMIC_EMOTIONS:
            trajectory[emotion].append(result.scores.get(emotion, 0.0))
        trajectory['epistemic_composite'].append(result.epistemic_score)

    return trajectory


def extract_safety_trajectory(
    results: List[EmotionResult]
) -> Dict[str, List[float]]:
    """
    Extract safety-related emotion trajectories.

    Returns:
        Dict with safety_positive, safety_negative, and net safety over time
    """
    trajectory = {
        'safety_positive': [],
        'safety_negative': [],
        'safety_net': []
    }

    for result in results:
        pos = sum(result.scores.get(e, 0) for e in SAFETY_POSITIVE)
        neg = sum(result.scores.get(e, 0) for e in SAFETY_NEGATIVE)

        trajectory['safety_positive'].append(pos)
        trajectory['safety_negative'].append(neg)
        trajectory['safety_net'].append(result.safety_score)

    return trajectory
