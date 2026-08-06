from __future__ import annotations

from scripts.run_control_guidance_f03_f06_f11_strict_runtime_preflight import (
    apply_slew,
    mode_spec,
    jsonable,
)
from scripts.verify_control_guidance_f03_f06_f11_strict_runtime_preflight import verify_pair


def test_probe_modes_change_only_declared_dimension():
    assert mode_spec("pn_alpha_0_3", {})["pn_alpha"] == 0.3
    assert mode_spec("pn_alpha_0_5", {})["pn_alpha"] == 0.5
    assert mode_spec("single_rate_5hz", {})["low_ticks"] == 20
    assert mode_spec("multirate_10_20_100hz", {}) == {
        "law": "los", "high_ticks": 10, "guidance_ticks": 5, "low_ticks": 1, "slew_per_s": None
    }
    assert mode_spec("slew_2_norm_per_s", {})["slew_per_s"] == 2.0


def test_probe_slew_is_symmetric_and_bounded():
    output, limited = apply_slew(
        {"fcs/aileron-cmd-norm": 1.0, "fcs/elevator-cmd-norm": -1.0},
        {"fcs/aileron-cmd-norm": 0.0, "fcs/elevator-cmd-norm": 0.0},
        2.0,
        0.01,
    )
    assert limited is True
    assert output["fcs/aileron-cmd-norm"] == 0.02
    assert output["fcs/elevator-cmd-norm"] == -0.02


def test_jsonable_preserves_strict_boolean_fields():
    value = jsonable({"strict_backend": True, "backend_fallback_occurred": False})
    assert value == {"strict_backend": True, "backend_fallback_occurred": False}
    assert isinstance(value["strict_backend"], bool)


def test_verifier_rejects_target_hash_mismatch_without_authorizing_runtime():
    row = {
        "pre_action_state_hash": "s", "target_state_hash": "t", "raw_command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.0},
        "fcs_command": {}, "actuator_actual": {"elevator_pos_rad": 0.0, "aileron_pos_rad": 0.0, "left_aileron_pos_rad": 0.0, "right_aileron_pos_rad": 0.0, "throttle_pos_norm": 0.7},
        "own_state": {}, "target_state": {}, "geometry": {"range_m": 1.0, "range_rate_mps": 0.0, "ata_deg": 0.0, "aa_deg": 0.0}, "p_rps": 0.0, "nz_g": 1.0, "controller_saturated": False, "rate_limited": False,
    }
    rows = [dict(row, target_state_hash=f"t{i}") for i in range(200)]
    episode = {"backend": "jsbsim", "strict_backend": True, "backend_fallback_occurred": False, "reset_count": 1, "mid_episode_reset_count": 0, "initial_state_hash": "same", "trajectory": rows}
    result = verify_pair({"pair_id": "x", "finding_id": "F06", "baseline": episode, "candidate": dict(episode, trajectory=[dict(row, target_state_hash="different") for row in rows])})
    assert "target_trajectory_hash_mismatch" in result["errors"]
