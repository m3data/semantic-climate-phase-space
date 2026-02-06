"""
Tests for SafetyGate — observation-only coupling feedback.

Tests use synthetic metrics_result dicts that mirror the structure
produced by MetricsService.analyze().
"""

import sys
from pathlib import Path

# Add backend to path so we can import safety_gate directly
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_climate_app" / "backend"))

from safety_gate import SafetyGate, GateDecision, GATE_TEMPLATES


def _make_metrics_result(
    risk_level: str = "low",
    risk_factors: list = None,
    coherence_pattern: str = None,
    attractor_basin: str = None,
    dk: float = 0.35,
    dh: float = 0.10,
    alpha: float = 0.80,
):
    """Build a synthetic metrics_result dict matching MetricsService output."""
    coherence_data = None
    if coherence_pattern:
        coherence_data = {
            "pattern": coherence_pattern,
            "autocorrelation": 0.1,
            "shift_variance": 0.05,
            "semantic_coherence": 0.7,
        }

    attractor_data = None
    if attractor_basin:
        attractor_data = {
            "basin": attractor_basin,
            "residence_time": 3,
            "transition": None,
        }

    return {
        "metrics": {
            "delta_kappa": dk,
            "delta_h": dh,
            "alpha": alpha,
        },
        "coupling_mode": {
            "mode": "Generative",
            "trajectory": "stable",
            "compound_label": "Generative-Stable",
            "epistemic_risk": risk_level,
            "risk_factors": risk_factors or [],
            "confidence": 0.85,
            "coherence": coherence_data,
        },
        "attractor_basin": attractor_data,
    }


# --- GateDecision tests ---

class TestGateDecision:
    def test_to_dict_roundtrip(self):
        d = GateDecision(
            turn=5,
            timestamp="2026-02-06T14:00:00",
            risk_level="low",
            risk_factors=[],
            gate_output="test",
            metrics_snapshot={"dk": 0.3, "dh": 0.1, "alpha": 0.8},
            attractor_basin="Generative",
            coherence_pattern="breathing",
        )
        result = d.to_dict()
        assert result["turn"] == 5
        assert result["risk_level"] == "low"
        assert result["attractor_basin"] == "Generative"
        assert result["coherence_pattern"] == "breathing"
        assert isinstance(result["metrics_snapshot"], dict)


# --- SafetyGate.evaluate tests ---

class TestSafetyGateEvaluate:
    def test_low_risk(self):
        gate = SafetyGate()
        result = _make_metrics_result(risk_level="low")
        decision = gate.evaluate(result, turn_count=12)

        assert decision.risk_level == "low"
        assert "[CLIMATE]" in decision.gate_output
        assert "Breathing pattern" in decision.gate_output

    def test_moderate_risk(self):
        gate = SafetyGate()
        result = _make_metrics_result(
            risk_level="moderate",
            risk_factors=["progressive_sycophancy"],
        )
        decision = gate.evaluate(result, turn_count=14)

        assert decision.risk_level == "moderate"
        assert "[CLIMATE NOTICE]" in decision.gate_output
        assert "lock-in" in decision.gate_output

    def test_high_risk_with_factors(self):
        gate = SafetyGate()
        result = _make_metrics_result(
            risk_level="high",
            risk_factors=["complexity_collapse", "engagement_declining"],
            dk=0.15,
        )
        decision = gate.evaluate(result, turn_count=18)

        assert decision.risk_level == "high"
        assert "[CLIMATE CONSTRAINT]" in decision.gate_output
        assert "complexity_collapse" in decision.gate_output
        assert "engagement_declining" in decision.gate_output
        assert decision.metrics_snapshot["dk"] == 0.15

    def test_critical_risk(self):
        gate = SafetyGate()
        result = _make_metrics_result(
            risk_level="critical",
            risk_factors=["regressive_sycophancy", "epistemic_enclosure_risk", "minimal_cognitive_engagement"],
            dk=0.05,
            dh=0.02,
        )
        decision = gate.evaluate(result, turn_count=22)

        assert decision.risk_level == "critical"
        assert "[CLIMATE HALT]" in decision.gate_output
        assert "STOP" in decision.gate_output

    def test_extracts_coherence_pattern(self):
        gate = SafetyGate()
        result = _make_metrics_result(
            risk_level="low",
            coherence_pattern="breathing",
        )
        decision = gate.evaluate(result, turn_count=10)

        assert decision.coherence_pattern == "breathing"

    def test_extracts_attractor_basin(self):
        gate = SafetyGate()
        result = _make_metrics_result(
            risk_level="low",
            attractor_basin="Contemplative",
        )
        decision = gate.evaluate(result, turn_count=10)

        assert decision.attractor_basin == "Contemplative"

    def test_missing_coherence_is_none(self):
        gate = SafetyGate()
        result = _make_metrics_result(risk_level="low")
        decision = gate.evaluate(result, turn_count=10)

        assert decision.coherence_pattern is None

    def test_missing_attractor_is_none(self):
        gate = SafetyGate()
        result = _make_metrics_result(risk_level="low")
        decision = gate.evaluate(result, turn_count=10)

        assert decision.attractor_basin is None

    def test_metrics_snapshot_captured(self):
        gate = SafetyGate()
        result = _make_metrics_result(dk=0.42, dh=0.18, alpha=0.75)
        decision = gate.evaluate(result, turn_count=10)

        assert decision.metrics_snapshot["dk"] == 0.42
        assert decision.metrics_snapshot["dh"] == 0.18
        assert decision.metrics_snapshot["alpha"] == 0.75

    def test_turn_count_stored(self):
        gate = SafetyGate()
        result = _make_metrics_result()
        decision = gate.evaluate(result, turn_count=37)

        assert decision.turn == 37

    def test_unknown_risk_defaults_to_low_template(self):
        gate = SafetyGate()
        result = _make_metrics_result(risk_level="unknown_level")
        decision = gate.evaluate(result, turn_count=10)

        assert "[CLIMATE]" in decision.gate_output

    def test_high_risk_with_no_factors(self):
        gate = SafetyGate()
        result = _make_metrics_result(risk_level="high", risk_factors=[])
        decision = gate.evaluate(result, turn_count=10)

        assert "unspecified" in decision.gate_output


# --- Audit trail tests ---

class TestSafetyGateAuditTrail:
    def test_starts_empty(self):
        gate = SafetyGate()
        assert gate.get_audit_trail() == []
        assert gate.get_latest() is None

    def test_accumulates_decisions(self):
        gate = SafetyGate()
        for i in range(5):
            gate.evaluate(_make_metrics_result(), turn_count=10 + i * 2)

        trail = gate.get_audit_trail()
        assert len(trail) == 5
        assert trail[0]["turn"] == 10
        assert trail[4]["turn"] == 18

    def test_get_latest_returns_most_recent(self):
        gate = SafetyGate()
        gate.evaluate(_make_metrics_result(risk_level="low"), turn_count=10)
        gate.evaluate(_make_metrics_result(risk_level="high", risk_factors=["test"]), turn_count=12)

        latest = gate.get_latest()
        assert latest.risk_level == "high"
        assert latest.turn == 12

    def test_reset_clears_trail(self):
        gate = SafetyGate()
        gate.evaluate(_make_metrics_result(), turn_count=10)
        gate.evaluate(_make_metrics_result(), turn_count=12)

        gate.reset()

        assert gate.get_audit_trail() == []
        assert gate.get_latest() is None

    def test_audit_trail_entries_are_dicts(self):
        gate = SafetyGate()
        gate.evaluate(_make_metrics_result(), turn_count=10)

        trail = gate.get_audit_trail()
        assert isinstance(trail[0], dict)
        assert "risk_level" in trail[0]
        assert "gate_output" in trail[0]
        assert "timestamp" in trail[0]


# --- Template rendering tests ---

class TestGateTemplates:
    def test_all_levels_have_templates(self):
        assert "low" in GATE_TEMPLATES
        assert "moderate" in GATE_TEMPLATES
        assert "high" in GATE_TEMPLATES
        assert "critical" in GATE_TEMPLATES

    def test_high_template_formats_risk_factors(self):
        gate = SafetyGate()
        output = gate._render_gate_output("high", ["factor_a", "factor_b"])
        assert "factor_a, factor_b" in output

    def test_low_template_no_formatting_needed(self):
        gate = SafetyGate()
        output = gate._render_gate_output("low", [])
        assert output == GATE_TEMPLATES["low"]
