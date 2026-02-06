"""
Safety-Gated Coupling Feedback — Observation-Only Phase.

Evaluates epistemic risk from MetricsService output and generates
gate decisions (what WOULD be injected into LLM context). Logs all
decisions to an audit trail for research analysis.

Does NOT inject into LLM context or modify frontend display.
See research/theory/DESIGN_safety-gated-coupling-feedback.md for
the full architecture.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class GateDecision:
    """A single gate evaluation result."""
    turn: int
    timestamp: str
    risk_level: str                     # 'low', 'moderate', 'high', 'critical'
    risk_factors: list
    gate_output: str                    # The text that WOULD be injected
    metrics_snapshot: dict              # dk, dh, alpha at this turn
    attractor_basin: Optional[str]      # Basin label if available
    coherence_pattern: Optional[str]    # 'breathing', 'locked', 'fragmented', 'transitional'

    def to_dict(self) -> dict:
        return asdict(self)


# Gate output templates — from the design doc's 4-tier system.
# These are what WOULD be prepended to the LLM context at each level.

GATE_TEMPLATES = {
    'low': (
        "[CLIMATE] Breathing pattern. Generative flow. Coherence steady."
    ),
    'moderate': (
        "[CLIMATE NOTICE] Pattern shifting toward lock-in. "
        "Hold current complexity. Do not introduce new frameworks this turn."
    ),
    'high': (
        "[CLIMATE CONSTRAINT] Epistemic risk elevated.\n"
        "Risk factors: {risk_factors}.\n"
        "REQUIRED:\n"
        "1. Acknowledge the current thread honestly.\n"
        "2. Simplify your next response.\n"
        "3. Check understanding before continuing.\n"
        "This constraint is visible to your conversation partner."
    ),
    'critical': (
        "[CLIMATE HALT] Critical epistemic risk.\n"
        "STOP the current line of reasoning.\n"
        "Ask your conversation partner what they need right now.\n"
        "Do not continue until they respond.\n"
        "Full metrics are visible to your partner."
    ),
}


class SafetyGate:
    """
    Observation-only safety gate for coupling feedback.

    Evaluates every metrics result, classifies risk, generates what
    the gate output WOULD be, and logs everything. Does not inject
    into LLM context (that's a future phase).
    """

    def __init__(self):
        self._audit_trail: list[GateDecision] = []

    def evaluate(self, metrics_result: dict, turn_count: int) -> GateDecision:
        """
        Evaluate a metrics result and produce a gate decision.

        Args:
            metrics_result: Output from MetricsService.analyze()
            turn_count: Current dialogue turn count

        Returns:
            GateDecision with risk classification and gate output text
        """
        # Extract fields from metrics_result (all already computed by MetricsService)
        coupling_mode = metrics_result.get('coupling_mode', {})
        risk_level = coupling_mode.get('epistemic_risk', 'low')
        risk_factors = coupling_mode.get('risk_factors', [])

        # Coherence pattern (nested under coupling_mode.coherence)
        coherence_data = coupling_mode.get('coherence')
        coherence_pattern = None
        if coherence_data and isinstance(coherence_data, dict):
            coherence_pattern = coherence_data.get('pattern')

        # Attractor basin
        attractor_data = metrics_result.get('attractor_basin')
        attractor_basin = None
        if attractor_data and isinstance(attractor_data, dict):
            attractor_basin = attractor_data.get('basin')
        elif isinstance(attractor_data, str):
            attractor_basin = attractor_data

        # Metrics snapshot
        metrics = metrics_result.get('metrics', {})
        metrics_snapshot = {
            'dk': metrics.get('delta_kappa'),
            'dh': metrics.get('delta_h'),
            'alpha': metrics.get('alpha'),
        }

        # Generate gate output text from template
        gate_output = self._render_gate_output(risk_level, risk_factors)

        decision = GateDecision(
            turn=turn_count,
            timestamp=datetime.now().isoformat(),
            risk_level=risk_level,
            risk_factors=risk_factors,
            gate_output=gate_output,
            metrics_snapshot=metrics_snapshot,
            attractor_basin=attractor_basin,
            coherence_pattern=coherence_pattern,
        )

        self._audit_trail.append(decision)
        return decision

    def get_latest(self) -> Optional[GateDecision]:
        """Return the most recent gate decision, or None."""
        if not self._audit_trail:
            return None
        return self._audit_trail[-1]

    def get_audit_trail(self) -> list[dict]:
        """Return full audit trail as serializable dicts."""
        return [d.to_dict() for d in self._audit_trail]

    def reset(self):
        """Clear the audit trail (e.g. on session reset)."""
        self._audit_trail.clear()

    @staticmethod
    def _render_gate_output(risk_level: str, risk_factors: list) -> str:
        """Render the gate output text for a given risk level."""
        template = GATE_TEMPLATES.get(risk_level, GATE_TEMPLATES['low'])

        if '{risk_factors}' in template:
            factors_str = ', '.join(risk_factors) if risk_factors else 'unspecified'
            return template.format(risk_factors=factors_str)

        return template
