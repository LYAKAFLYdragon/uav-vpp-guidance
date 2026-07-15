from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent


def _load_runner():
    path = ROOT / "scripts" / "run_thesis_five_state_reentry_v10_action_authority.py"
    spec = importlib.util.spec_from_file_location("v10_action_authority_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_verifier():
    path = ROOT / "scripts" / "verify_thesis_five_state_reentry_v10_action_authority.py"
    spec = importlib.util.spec_from_file_location("v10_action_authority_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _step() -> dict:
    return {
        "vp_forward_bias_m": 10.0,
        "vp_lateral_bias_m": -5.0,
        "vp_vertical_bias_m": None,
        "nz_cmd": 1.5,
        "actual_nz_g": 1.2,
        "ego_attack_aoa_deg": 25.0,
        "guidance_command": {"nz_cmd": 1.5, "roll_rate_cmd": 0.1, "throttle_cmd": 0.8},
        "applied_delta": [0.0, 0.0, 0.25],
        "altitude_diff_observation": 0.05,
    }


def test_v10_records_unobserved_vertical_bias_as_explicit_null_not_nan():
    module = _load_runner()
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v10_action_authority_design.yaml").read_text(encoding="utf-8"))

    result = module._annotate_telemetry({"window": [_step()]}, config)

    step = result["window"][0]
    assert step["vp_vertical_bias_m"] is None
    assert step["vp_vertical_bias_observed"] is False
    assert step["vertical_response_proxy"]["applied_vertical_action"] == pytest.approx(0.25)
    assert step["guidance_command"]["throttle_cmd"] == pytest.approx(0.8)


def test_v10_fails_closed_when_required_telemetry_is_missing():
    module = _load_runner()
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v10_action_authority_design.yaml").read_text(encoding="utf-8"))
    result = {"window": [_step()]}
    result["window"][0]["actual_nz_g"] = None

    with pytest.raises(module.V10RuntimeError, match="actual_nz_g"):
        module._annotate_telemetry(result, config)


def test_v10_design_keeps_training_and_reward_changes_locked():
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v10_action_authority_design.yaml").read_text(encoding="utf-8"))

    assert config["authorization"]["execution_permitted"] is False
    assert config["authorization"]["training_permitted"] is False
    assert config["authorization"]["reward_change_permitted"] is False
    assert config["telemetry_contract"]["backend_unobserved"] == ["vp_vertical_bias_m"]


def test_v10_verifier_tolerates_json_float_round_trip_but_not_material_difference():
    module = _load_verifier()

    assert module._equivalent({"value": 0.2499999995529652}, {"value": 0.24999999955296515})
    assert not module._equivalent({"value": 0.25}, {"value": 0.2501})
