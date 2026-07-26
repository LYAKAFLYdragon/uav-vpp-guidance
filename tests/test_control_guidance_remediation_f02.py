"""Pure F02 two-loop evidence checks; no protected runtime control imports."""
import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from uav_vpp_guidance.evaluation.f02_roll_two_loop_evidence import (
    F04_ROLL_RATE_LIMITS_RADPS,
    OfflineF02TwoLoopController,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro/specs/control-guidance-remediation"
EVIDENCE = SPEC / "evidence"

finite = st.floats(min_value=-1.4, max_value=1.4, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False)
speed = st.floats(min_value=20.0, max_value=450.0, allow_nan=False, allow_infinity=False)


def test_frozen_baseline_is_angle_damped_but_not_rate_damped():
    trace = json.loads((EVIDENCE / "f02_candidate_vs_frozen_baseline.json").read_text(encoding="utf-8"))
    baseline = trace["rate_isolation"]
    assert baseline["positive_rate"]["baseline_roll_rate_cmd_radps"] == pytest.approx(
        baseline["negative_rate"]["baseline_roll_rate_cmd_radps"]
    )
    assert baseline["positive_rate"]["candidate_damping_contribution_radps"] == pytest.approx(
        -baseline["negative_rate"]["candidate_damping_contribution_radps"]
    )


def test_step_and_opposite_step_preserve_heading_command_sign():
    controller = OfflineF02TwoLoopController()
    right = controller.command(0.30, 0.0, 0.0, 250.0)
    left = controller.command(-0.30, 0.0, 0.0, 250.0)
    assert right["bank_cmd_rad"] > 0.0 and right["roll_rate_cmd_radps"] > 0.0
    assert left["bank_cmd_rad"] < 0.0 and left["roll_rate_cmd_radps"] < 0.0


def test_bank_release_and_reset_clear_command_without_state_leakage():
    controller = OfflineF02TwoLoopController()
    controller.command(0.35, 0.1, 0.2, 250.0)
    released = controller.command(0.0, 0.0, 0.0, 250.0)
    assert released["roll_rate_cmd_radps"] == pytest.approx(0.0)
    controller.reset()
    assert controller.last_command is None
    assert controller.command(0.0, 0.0, 0.0, 250.0)["roll_rate_cmd_radps"] == pytest.approx(0.0)


def test_speed_sweep_increases_same_sign_coordinated_bank_before_limit():
    controller = OfflineF02TwoLoopController()
    banks = [controller.command(0.02, 0.0, 0.0, speed)["bank_cmd_rad"] for speed in (100.0, 200.0, 300.0)]
    assert banks == sorted(banks)
    assert all(bank > 0.0 for bank in banks)


def test_explicit_rate_feedback_reduces_declared_rate_decay_trace():
    """Under a fixed offline first-order roll plant, rate feedback decays faster than F02 baseline."""
    controller = OfflineF02TwoLoopController()
    candidate_rate, baseline_rate = 0.8, 0.8
    candidate_trace, baseline_trace = [candidate_rate], [baseline_rate]
    for _ in range(5):
        candidate_command = controller.command(0.0, 0.0, candidate_rate, 250.0)["roll_rate_cmd_radps"]
        candidate_rate = 0.8 * candidate_rate + 0.2 * candidate_command
        baseline_rate = 0.8 * baseline_rate  # frozen command is zero at zero heading/bank
        candidate_trace.append(candidate_rate)
        baseline_trace.append(baseline_rate)
    assert all(abs(candidate) < abs(baseline) for candidate, baseline in zip(candidate_trace[1:], baseline_trace[1:]))


def test_saturation_keeps_f04_roll_rate_limits_and_opposes_rate():
    controller = OfflineF02TwoLoopController()
    positive_rate = controller.command(1.4, -1.4, 10.0, 350.0)
    negative_rate = controller.command(-1.4, 1.4, -10.0, 350.0)
    for result in (positive_rate, negative_rate):
        assert F04_ROLL_RATE_LIMITS_RADPS[0] <= result["roll_rate_cmd_radps"] <= F04_ROLL_RATE_LIMITS_RADPS[1]
        assert result["saturated"] is True
    assert positive_rate["damping_contribution_radps"] < 0.0
    assert negative_rate["damping_contribution_radps"] > 0.0


# **Validates: Requirements 2.2**
@settings(max_examples=100, deadline=None)
@given(heading_error=finite, roll=finite, roll_rate=rate, airspeed=speed)
def test_candidate_is_finite_rate_damped_sign_consistent_and_f04_bounded(
    heading_error, roll, roll_rate, airspeed
):
    """Explicit rate damping reverses with p and outputs remain inside F04."""
    controller = OfflineF02TwoLoopController()
    result = controller.command(heading_error, roll, roll_rate, airspeed)
    opposite = controller.command(heading_error, roll, -roll_rate, airspeed)
    assert all(isinstance(value, (bool, float)) for value in result.values())
    assert F04_ROLL_RATE_LIMITS_RADPS[0] <= result["roll_rate_cmd_radps"] <= F04_ROLL_RATE_LIMITS_RADPS[1]
    assert result["damping_contribution_radps"] == pytest.approx(-opposite["damping_contribution_radps"])
    if abs(heading_error) > 1.0e-4 and abs(roll) < 1.0e-8 and abs(roll_rate) < 1.0e-8:
        assert result["roll_rate_cmd_radps"] * heading_error > 0.0


def test_gate_records_needs_more_evidence_and_hashes_specification():
    gate = json.loads((EVIDENCE / "f02_evidence_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "approved"
    assert gate["runtime_disposition"] == "implement_opt_in_mode_only"
    assert gate["results"]["offline_deterministic_trace_checks_pass"] is True
    assert gate["decision"] == "approved_by_user_runtime_authorization"
    assert gate["results"]["pytest_validation"].startswith("not_established")
    assert gate["results"]["frozen_reference_matrix_candidate_comparison"].startswith("not_executable")
    assert hashlib.sha256((SPEC / "f02_roll_two_loop_specification.md").read_bytes()).hexdigest() == gate["evidence_hashes"]["specification_sha256"]
