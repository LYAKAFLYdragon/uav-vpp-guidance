from __future__ import annotations

from scripts.run_control_guidance_f01_f02_optin_paired_regression import (
    jsonable,
    mode_config,
    stable_hash,
)
from scripts.verify_control_guidance_f01_f02_optin_paired_regression import verify_pair


def test_mode_config_is_explicit_and_records_override():
    base = {"params": {"remediation_mode": "legacy"}}
    candidate = mode_config(base, "f01_f02_remediated")
    assert candidate["params"]["remediation_mode"] == "f01_f02_remediated"
    assert base["params"]["remediation_mode"] == "legacy"
    assert candidate["provenance"]["config_overrides"][0]["key"] == "params.remediation_mode"


def test_state_hash_is_order_independent_for_jsonable_mapping():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_jsonable_preserves_boolean_contract_fields():
    value = jsonable({"strict_backend": True, "backend_fallback_occurred": False})
    assert value == {"strict_backend": True, "backend_fallback_occurred": False}
    assert isinstance(value["strict_backend"], bool)
    assert isinstance(value["backend_fallback_occurred"], bool)


def test_verifier_rejects_non_paired_reset_and_wrong_mode():
    rows = [{
        "pre_action_state_hash": "state",
        "target_state_hash": "target",
        "raw_command": {"nz_cmd": 1.0, "roll_rate_cmd": 0.1},
        "filtered_command": {}, "fcs_command": {},
        "actuator_actual": {
            "elevator_pos_rad": 0.0, "aileron_pos_rad": 0.0,
            "left_aileron_pos_rad": 0.0, "right_aileron_pos_rad": 0.0,
            "rudder_pos_rad": 0.0, "throttle_pos_norm": 0.7,
        },
        "own_state": {}, "target_state": {},
        "geometry": {"range_m": 1.0, "range_rate_mps": 0.0, "closing_speed_mps": 0.0, "ata_deg": 0.0, "aa_deg": 0.0},
        "p_rps": 0.0, "nz_g": 1.0, "clipping": False,
    } for _ in range(48)]
    episode = {
        "mode": "f01_f02_remediated", "backend": "jsbsim", "strict_backend": True,
        "backend_fallback_occurred": False, "reset_count": 2, "mid_episode_reset_count": 1,
        "initial_state_hash": "state", "trajectory": rows,
    }
    pair = {
        "pair_id": "test", "scenario_id": "test", "variant": "f01_only",
        "baseline": dict(episode, mode="legacy", reset_count=1, mid_episode_reset_count=0),
        "candidate": episode,
    }
    result = verify_pair(pair)
    assert "candidate:reset_count" in result["errors"]
    assert "candidate:mid_episode_reset" in result["errors"]
