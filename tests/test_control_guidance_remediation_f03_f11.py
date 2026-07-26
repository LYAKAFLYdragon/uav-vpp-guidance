"""Deterministic offline tests for F03/F11 dynamics evidence."""
import json
from pathlib import Path
import yaml
from uav_vpp_guidance.evaluation.f03_f11_dynamics_evidence import frozen_topologies, run_experiment

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"

def _guidance():
    return yaml.safe_load((ROOT / "config/guidance.yaml").read_text(encoding="utf-8"))["guidance"]

def test_frozen_topology_enumerates_enabled_and_disabled_stages():
    topology = frozen_topologies(_guidance())
    assert [stage.name for stage in topology["jsbsim_pn"]] == ["pn_los_rate_filter", "environment_command_filter", "low_level_command_filter"]
    assert topology["disabled"] == {"los_internal_filter": True, "command_post_processor": True, "lift_compensation": True}

def test_each_candidate_has_delay_phase_peak_saturation_and_slew_evidence():
    report = run_experiment(_guidance())
    assert set(report["candidates"]) == {"pn_filter_tuning_alpha_0_5", "filter_consolidation_remove_low_level_stage", "actuator_slew_limit_2_per_s", "bank_compensation_at_45_deg"}
    for result in [report["baseline"], *report["candidates"].values()]:
        assert len(result["spectrum"]) == 5
        assert result["step"]["peak_command"] <= 7.0
        assert result["step"]["max_slew_per_s"] >= 0.0
        assert all(item["amplitude_ratio"] >= 0.0 for item in result["spectrum"])

def test_slew_candidate_bounds_output_slew_without_changing_f04_limits():
    report = run_experiment(_guidance())
    assert report["candidates"]["actuator_slew_limit_2_per_s"]["step"]["max_slew_per_s"] <= 2.0 + 1e-9
    assert report["f04_limits"] == {"nz": [-2.0, 7.0], "roll_rate_radps": [-1.5, 1.5], "throttle": [0.4, 0.9]}

def test_artifact_gate_retains_baseline_and_exposes_blockers():
    gate = json.loads((SPEC / "evidence/f03_f11_dynamics_bundle/f03_f11_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_frozen_defaults"
