from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from uav_vpp_guidance.evaluation.thesis_reentry_v12_intent_controllability import (
    INTENT_FIELDS,
    component_errors,
    summarize_records,
)
from uav_vpp_guidance.hierarchy.shared_intent_profiles import compile_profile


ROOT = Path(__file__).resolve().parent.parent


def test_component_errors_are_zero_at_the_frozen_profile_target():
    target = compile_profile("range_extension")["physical_targets"]

    errors = component_errors(target, "range_extension")

    assert errors.tolist() == pytest.approx([0.0] * 6)
    assert len(INTENT_FIELDS) == 6


def _record(method: str, opponent: str, key: str, errors: list[float]) -> dict:
    step = {"after_component_errors": errors}
    return {
        "method": method,
        "opponent": opponent,
        "pair_key": key,
        "handoff_reached": True,
        "contract_passed": True,
        "window_steps": 20,
        "handoff_state_hash": "same",
        "ego_failure": False,
        "window": [step] * 20,
    }


def test_full_factorial_summary_requires_multiobjective_compatibility_in_two_opponents():
    methods = (
        "frozen_fixed_head_on", "range_only", "turn_only", "vertical_only",
        "range_turn", "range_vertical", "turn_vertical", "range_turn_vertical",
    )
    records = []
    for opponent in ("expert", "end_to_end", "independent_ppo_vpp"):
        for index in range(2):
            key = f"{opponent}-{index}"
            records.append(_record("frozen_fixed_head_on", opponent, key, [0.2] * 6))
            for method in methods[1:]:
                errors = [0.18, 0.18, 0.18, 0.18, 0.2, 0.2] if opponent != "independent_ppo_vpp" else [0.25] * 6
                records.append(_record(method, opponent, key, errors))
    summary = summarize_records(
        records,
        scenarios_per_opponent=2,
        min_window_steps=20,
        min_pairs=2,
        max_high_priority_worsening=0.02,
        max_any_component_worsening=0.05,
        min_component_improvement=0.01,
        min_improved_components=2,
        min_weighted_error_improvement=0.01,
        safety_delta_max=0.05,
        profile_name="range_extension",
    )

    assert summary["passes"] is True
    assert "range_turn_vertical" in summary["local_multiobjective_candidates"]
    assert summary["matrix"]["range_turn_vertical"]["expert"]["mean_component_error_delta_vs_reference"][INTENT_FIELDS[2]] == pytest.approx(-0.02)


def test_v12_design_is_nonlearning_and_has_full_factorial_matrix(tmp_path: Path):
    path = ROOT / "scripts" / "run_thesis_five_state_reentry_v12_intent_controllability.py"
    spec = importlib.util.spec_from_file_location("v12_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT / "config/experiment/thesis_five_state_reentry_v12_intent_controllability_design.yaml").read_text(encoding="utf-8"))

    assert config["authorization"]["execution_permitted"] is False
    assert config["authorization"]["training_permitted"] is False
    assert config["authorization"]["reward_change_permitted"] is False
    assert len(config["methods"]) == 8
    config["outputs"]["root"] = str(tmp_path / "v12_output")
    temporary_config = tmp_path / "v12_design.yaml"
    temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = module.validate_design(temporary_config)
    assert result["planned_records"] == 288


def test_v12_verifier_only_tolerates_serialization_scale_float_noise():
    path = ROOT / "scripts" / "verify_thesis_five_state_reentry_v12_intent_controllability.py"
    spec = importlib.util.spec_from_file_location("v12_verifier", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module._equivalent({"value": 0.2499999999999}, {"value": 0.25})
    assert not module._equivalent({"value": 0.25}, {"value": 0.2501})
