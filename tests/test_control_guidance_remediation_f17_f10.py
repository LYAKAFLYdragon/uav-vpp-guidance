"""Deterministic offline F17/F10 evidence tests; runtime defaults remain untouched."""
import json
from math import pi
from pathlib import Path

import pytest

from uav_vpp_guidance.evaluation.f17_f10_evidence import (
    ArbitrationInput,
    F04_NZ_LIMITS,
    HybridSwitchParameters,
    OfflineRangeSwitch,
    f10_sweep,
    f17_sweep,
    offline_arbitrate,
    qbar_gain_scales,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"


def test_f17_frozen_hysteresis_and_dwell_hold_until_three_consecutive_desired_steps():
    model = OfflineRangeSwitch(HybridSwitchParameters())
    assert model.step(2750.0)["active_law"] == "pn"  # still inside hysteresis band
    assert model.step(2400.0)["active_law"] == "pn"
    assert model.step(2400.0)["active_law"] == "pn"
    switched = model.step(2400.0)
    assert switched["active_law"] == "los" and switched["changed"]


def test_f17_sweep_reports_delay_travel_chatter_and_missed_opportunities_at_closing_speeds():
    report = f17_sweep()
    steady = report["baseline"]["steady_closing"]["300.0"]
    jitter = report["baseline"]["threshold_jitter"]["100.0"]
    assert steady["switch_delay_steps"] == 2
    assert steady["range_travelled_during_delay_m"] == pytest.approx(120.0)
    assert jitter["chatter_count"] == 0
    assert jitter["missed_switch_opportunities"] >= 0


def test_f10_gain_scaling_is_finite_and_reacts_to_qbar_altitude_and_aoa_sweeps():
    low_qbar = qbar_gain_scales(140.0, 5000.0, 0.05)
    high_qbar = qbar_gain_scales(350.0, 5000.0, 0.05)
    high_altitude = qbar_gain_scales(250.0, 14000.0, 0.05)
    high_aoa = qbar_gain_scales(250.0, 5000.0, 0.4)
    assert low_qbar["qbar_scale"] > high_qbar["qbar_scale"]
    assert high_altitude["altitude_schedule_active"]
    assert high_aoa["aoa_schedule_active"] and high_aoa["nz_proportional_scale"] > 1.0


def test_f10_offline_arbitration_declares_frozen_priority_and_preserves_f04_limits():
    result = offline_arbitrate(ArbitrationInput(7.0, 1.0, 800.0, 140.0, pi * 85.0 / 180.0, 0.05))
    assert result["order"] == ["guidance", "stall_limit", "bank_recovery", "altitude_hold", "final_clip"]
    assert result["flags"]["guidance"] and result["flags"]["stall_limited"]
    assert result["flags"]["bank_recovery"] and result["flags"]["altitude_hold"]
    assert F04_NZ_LIMITS[0] <= result["nz_cmd"] <= result["nz_upper_limit"] <= F04_NZ_LIMITS[1]
    assert abs(result["roll_rate_cmd"]) <= 1.5


def test_f10_sweep_has_named_candidate_but_does_not_select_it_at_runtime():
    report = f10_sweep()
    candidate = report["scheduled_candidate"]["high_aoa"]
    assert candidate["candidate_name"] == "qbar_altitude_aoa_gain_scheduled_controller"
    assert candidate["runtime_selected"] is False
    assert report["baseline_fixed_mapping"]["bank_and_altitude"]["flags"]["bank_recovery"]


def test_f17_f10_gate_requires_paired_matrix_and_strict_jsbsim_before_runtime_change():
    gate = json.loads((SPEC / "evidence/f17_f10_evidence_bundle/f17_f10_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "needs_more_evidence"
    assert gate["protected_path_allowlist"] == []
    assert gate["runtime_disposition"] == "preserve_frozen_dwell_hysteresis_and_controller_defaults"
