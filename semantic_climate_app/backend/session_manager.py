"""
Dialogue session management.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.
"""

from collections import deque
from datetime import datetime
import numpy as np
import uuid
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

from src.schema import get_versions_dict, EXPORT_SCHEMA_VERSION


class DialogueTurn:
    """A single turn in the dialogue."""

    def __init__(self, speaker: str, text: str, embedding: np.ndarray):
        self.speaker = speaker  # 'human' or 'ai'
        self.text = text
        self.embedding = embedding
        self.timestamp = datetime.now()

        # Context snapshots (populated after turn is added)
        self.window_entropy = None
        self.semantic_shift = None
        self.coupling_trend = None

        # Latency tracking (for isolating signal from noise)
        # human turns: compose_time_ms = time human spent composing (signal)
        # ai turns: api_latency_ms = time waiting for API response (noise)
        #           generation_time_ms = token generation time (partial signal)
        self.compose_time_ms = None      # Human: time since last AI response
        self.api_latency_ms = None       # AI: time to first token / full response
        self.token_count = None          # AI: response length for normalization

    def to_dict(self, include_context: bool = True):
        """Export turn without embedding (for JSON serialization)."""
        data = {
            'speaker': self.speaker,
            'text': self.text,
            'timestamp': self.timestamp.isoformat()
        }

        # Include latency data
        if self.speaker == 'human' and self.compose_time_ms is not None:
            data['compose_time_ms'] = self.compose_time_ms
        elif self.speaker == 'ai':
            if self.api_latency_ms is not None:
                data['api_latency_ms'] = self.api_latency_ms
            if self.token_count is not None:
                data['token_count'] = self.token_count

        if include_context:
            data['context'] = {
                'window_entropy': float(self.window_entropy) if self.window_entropy is not None else None,
                'semantic_shift': float(self.semantic_shift) if self.semantic_shift is not None else None,
                'coupling_trend': self.coupling_trend
            }

        return data


class SessionManager:
    """Manages a dialogue session with rolling window."""

    def __init__(self, min_turns: int = 10, max_turns: int = 50, embedding_model: str = "all-mpnet-base-v2"):
        """
        Initialize session.

        Args:
            min_turns: Minimum turns before analysis starts
            max_turns: Maximum turns to keep in window
            embedding_model: Name of the embedding model being used
        """
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.turns = deque(maxlen=max_turns)
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.metrics_history = []
        self.model_name = None
        self.embedding_model = embedding_model
        self.thresholds_version = "v1_narrative_adjusted"

        # Biosignal stream from EBS (1Hz phase dynamics)
        self.biosignal_stream = []
        self.latest_biosignal = None
        self.ebs_session_id = None

        # ECP context (when launched from Field Journal)
        self.ecp_session_id = None
        self.ecp_experiment_type = None

    def add_turn(
        self,
        speaker: str,
        text: str,
        embedding: np.ndarray,
        compose_time_ms: int = None,
        api_latency_ms: int = None
    ):
        """
        Add a turn to the session with context snapshots and latency tracking.

        Args:
            speaker: 'human' or 'ai'
            text: The turn text
            embedding: Semantic embedding vector
            compose_time_ms: (human turns) Time spent composing since last AI response
            api_latency_ms: (ai turns) Time waiting for API response
        """
        turn = DialogueTurn(speaker, text, embedding)

        # Calculate context features
        turn.window_entropy = self._calculate_window_entropy(text, window_size=50)
        turn.semantic_shift = self._calculate_semantic_shift(embedding)
        turn.coupling_trend = self._calculate_coupling_trend()

        # Add latency tracking
        if speaker == 'human' and compose_time_ms is not None:
            turn.compose_time_ms = compose_time_ms
        elif speaker == 'ai':
            if api_latency_ms is not None:
                turn.api_latency_ms = api_latency_ms
            # Rough token count for normalization (whitespace split)
            turn.token_count = len(text.split())

        self.turns.append(turn)

    def _calculate_window_entropy(self, text: str, window_size: int = 50) -> float:
        """
        Calculate Shannon entropy over last N tokens in the conversation.

        Args:
            text: Current turn text
            window_size: Number of recent tokens to consider

        Returns:
            Shannon entropy (bits)
        """
        # Collect recent tokens from last few turns + current
        recent_tokens = []

        # Add tokens from recent turns (up to window_size)
        for past_turn in reversed(list(self.turns)):
            tokens = past_turn.text.lower().split()
            recent_tokens = tokens + recent_tokens
            if len(recent_tokens) >= window_size:
                break

        # Add current turn tokens
        current_tokens = text.lower().split()
        recent_tokens.extend(current_tokens)

        # Take last window_size tokens
        recent_tokens = recent_tokens[-window_size:]

        if len(recent_tokens) == 0:
            return 0.0

        # Calculate Shannon entropy
        token_counts = Counter(recent_tokens)
        total = len(recent_tokens)
        entropy = 0.0

        for count in token_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_semantic_shift(self, current_embedding: np.ndarray) -> float:
        """
        Calculate cosine distance to previous turn's embedding.

        Args:
            current_embedding: Current turn embedding

        Returns:
            Semantic shift (1 - cosine_similarity), or None if first turn
        """
        if len(self.turns) == 0:
            return None

        previous_embedding = self.turns[-1].embedding

        # Handle zero vectors
        if np.linalg.norm(current_embedding) == 0 or np.linalg.norm(previous_embedding) == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = cosine_similarity(
            current_embedding.reshape(1, -1),
            previous_embedding.reshape(1, -1)
        )[0, 0]

        # Return distance (1 - similarity)
        return float(1.0 - similarity)

    def _calculate_coupling_trend(self) -> dict:
        """
        Calculate coupling trend from recent metrics history.

        Returns:
            Dict with trend indicators or None if insufficient data.
            Includes trajectory-aware coupling mode information when available.
        """
        if len(self.metrics_history) < 2:
            return None

        # Get last 2 metrics for simple trend
        recent = self.metrics_history[-2:]

        delta_kappa_trend = recent[1]['metrics']['delta_kappa'] - recent[0]['metrics']['delta_kappa']
        delta_h_trend = recent[1]['metrics']['delta_h'] - recent[0]['metrics']['delta_h']
        alpha_trend = recent[1]['metrics']['alpha'] - recent[0]['metrics']['alpha']

        # Check mode stability (use compound_label if available, fall back to mode)
        current_mode = recent[1].get('coupling_mode', {}).get('compound_label') or recent[1].get('mode')
        previous_mode = recent[0].get('coupling_mode', {}).get('compound_label') or recent[0].get('mode')
        mode_stable = current_mode == previous_mode

        result = {
            'delta_kappa_trend': float(delta_kappa_trend),
            'delta_h_trend': float(delta_h_trend),
            'alpha_trend': float(alpha_trend),
            'mode_stable': mode_stable,
            'mode': recent[1].get('mode')  # Legacy base mode
        }

        # Include enhanced coupling mode data if available
        if 'coupling_mode' in recent[1]:
            cm = recent[1]['coupling_mode']
            result['coupling_mode'] = {
                'compound_label': cm.get('compound_label'),
                'trajectory': cm.get('trajectory'),
                'epistemic_risk': cm.get('epistemic_risk'),
                'risk_factors': cm.get('risk_factors', [])
            }

        return result

    def can_analyze(self) -> bool:
        """Check if we have enough turns for analysis."""
        return len(self.turns) >= self.min_turns

    def get_embeddings(self) -> list:
        """Get all embeddings in current window."""
        return [turn.embedding for turn in self.turns]

    def get_turn_texts(self) -> list:
        """Get all turn texts in current window (for affective substrate)."""
        return [turn.text for turn in self.turns]

    def get_semantic_shifts(self) -> list:
        """
        Get all semantic shift values for coherence calculation.

        Returns:
            List of turn-to-turn semantic shift values (floats).
            Only includes turns where semantic_shift was computed.
        """
        shifts = []
        for turn in self.turns:
            if turn.semantic_shift is not None:
                shifts.append(turn.semantic_shift)
        return shifts

    def get_turn_count(self) -> int:
        """Get current number of turns."""
        return len(self.turns)

    def get_previous_metrics(self) -> dict:
        """
        Get previous metrics for trajectory calculation.

        Returns:
            Dict with delta_kappa, delta_h, alpha from last analysis,
            or None if no previous metrics exist.
        """
        if not self.metrics_history:
            return None

        last = self.metrics_history[-1]
        return {
            'delta_kappa': last['metrics']['delta_kappa'],
            'delta_h': last['metrics']['delta_h'],
            'alpha': last['metrics']['alpha']
        }

    def add_metrics_result(self, metrics: dict):
        """Store metrics result in history."""
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'turn_count': len(self.turns),
            **metrics
        })

    def set_model(self, model_name: str):
        """Set the LLM model being used."""
        self.model_name = model_name

    def add_biosignal_sample(self, phase_data: dict):
        """
        Store incoming phase dynamics from EBS.

        Args:
            phase_data: Dict with ts, coherence, position, velocity_mag, etc.
        """
        sample = {
            "ts": phase_data.get("ts"),
            "coherence": phase_data.get("coherence", 0),
            "hr": phase_data.get("hr", 0),
            "position": phase_data.get("position", [0, 0, 0]),
            "velocity_mag": phase_data.get("velocity_mag", 0),
            "curvature": phase_data.get("curvature", 0),
            "stability": phase_data.get("stability", 0),
            "phase_label": phase_data.get("phase_label", "")
        }
        self.biosignal_stream.append(sample)
        self.latest_biosignal = sample

    def get_latest_biosignal(self) -> dict:
        """Get most recent biosignal sample (for psi_biosignal)."""
        return self.latest_biosignal

    def get_biosignal_window(self, seconds: int = 30) -> list:
        """
        Get recent biosignal samples for windowed analysis.

        Args:
            seconds: Number of seconds of data to return (approx, 1Hz)

        Returns:
            List of recent biosignal samples
        """
        # Since we sample at 1Hz, seconds ~ samples
        return self.biosignal_stream[-seconds:] if self.biosignal_stream else []

    def set_ebs_session(self, session_id: str):
        """Set the EBS session ID for coupling."""
        self.ebs_session_id = session_id

    def set_ecp_context(self, session_id: str, experiment_type: str = None):
        """
        Set the ECP session context for correlation with Field Journal.

        Args:
            session_id: The ECP experiment ID from Field Journal
            experiment_type: The experiment type (e.g., 'human-ai-coupling', 'semantic-only')
        """
        self.ecp_session_id = session_id
        self.ecp_experiment_type = experiment_type

    def export_session(self) -> dict:
        """
        Export session data for analysis.

        Returns structured data with conversation, metrics, biosignal, and metadata.
        Includes version tracking for all analysis modules.
        """
        return {
            "metadata": {
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat(),
                "exported_at": datetime.now().isoformat(),
                "schema_version": EXPORT_SCHEMA_VERSION,
                "versions": get_versions_dict(),
                "model": self.model_name or "unknown",
                "embedding_model": self.embedding_model,
                "thresholds_version": self.thresholds_version,
                "total_turns": len(self.turns),
                "min_turns": self.min_turns,
                "max_turns": self.max_turns,
                "ebs_session_id": self.ebs_session_id,
                "biosignal_samples": len(self.biosignal_stream),
                # ECP Field Journal correlation
                "ecp_session_id": self.ecp_session_id,
                "ecp_experiment_type": self.ecp_experiment_type
            },
            "conversation": [
                {
                    "turn": idx + 1,
                    **turn.to_dict()
                }
                for idx, turn in enumerate(self.turns)
            ],
            "metrics_history": self.metrics_history,
            "biosignal_stream": self.biosignal_stream
        }

    def reset(self):
        """Clear session and generate new ID."""
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.turns.clear()
        self.metrics_history.clear()
        self.model_name = None
        self.biosignal_stream = []
        self.latest_biosignal = None
        self.ebs_session_id = None
        self.ecp_session_id = None
        self.ecp_experiment_type = None
