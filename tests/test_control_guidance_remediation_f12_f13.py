"""Deterministic offline F12/F13 reward decomposition tests."""
import json
from pathlib import Path
from uav_vpp_guidance.evaluation.f12_f13_reward_evidence import run_experiment
ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"

def test_reward_decomposition_reports_step_terminal_safety_and_saturation_terms():
    result = run_experiment()["default"]
    assert result["aggregate"]["terminal_reward"] == 400.0
    assert result["safety_trigger_occupancy"] == 2 / 3
    assert result["saturation_occupancy"] == 1 / 3
    assert result["aggregate"]["reward_total"] == sum(row["reward_total"] for row in result["per_step"])

def test_alive_and_closing_counterfactual_isolated_without_changing_other_terms():
    report = run_experiment(); base, off = report["default"]["aggregate"], report["counterfactual_disable_alive_and_closing"]["aggregate"]
    assert off["reward_alive"] == 0.0 and off["reward_closing"] == 0.0
    for key in ("reward_range", "reward_angle", "reward_safety", "reward_saturation", "reward_smooth", "terminal_reward"):
        assert off[key] == base[key]

def test_f12_f13_gate_retains_default_reward_pending_ablations_and_strict_backend_evidence():
    gate = json.loads((SPEC / "evidence/f12_f13_reward_evidence_bundle/f12_f13_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_default_reward_weights"
