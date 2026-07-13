from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "config" / "experiment" / "jsbsim_hrl_noncanonical_thesis_taxonomy_routing3_base.yaml"
EXPERT = ROOT / "config" / "experiment" / "jsbsim_hrl_noncanonical_thesis_taxonomy_routing3_expert.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_resolved(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    resolved = {}
    for include in payload.pop("includes", []) or []:
        resolved = _merge(resolved, _load_resolved((path.parent / include).resolve()))
    return _merge(resolved, payload)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("routing3_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routing3_config_is_explicitly_noncanonical_and_has_exactly_three_methods():
    resolved = _load_resolved(BASE)
    expected = [
        "canonical_ppo_high_level_policy",
        "noncanonical_forced_first_crossing_then_ppo",
        "crossing_vpp_specialist_no_routing",
    ]
    assert resolved["noncanonical_routing_only_causal_validation"][
        "canonical_family_untouched"
    ] is True
    assert resolved["run_defaults"]["main_methods"] == expected
    assert resolved["run_defaults"]["ablation_methods"] == expected
    assert resolved["run_defaults"]["tasks"] == ["head_on"]
    override = resolved["methods"]["noncanonical_forced_first_crossing_then_ppo"][
        "config_overrides"
    ]
    assert override == {
        "commander.initial_macro_mode_override.enabled": True,
        "commander.initial_macro_mode_override.task_name": "head_on",
        "commander.initial_macro_mode_override.mode_id": 1,
        "commander.initial_macro_mode_override.reason": "noncanonical_forced_first_crossing_then_ppo",
    }


def test_routing3_manifest_materializes_the_three_preregistered_head_on_scenarios():
    runner = _load_runner_module()
    materialized = runner._materialize_scenario_manifest(_load_resolved(BASE), BASE)
    scenarios = materialized["tasks"]["head_on"]["task"]["scenarios"]
    assert materialized["scenario_manifest_resolution"]["task_counts"] == {"head_on": 3}
    assert {scenario["name"] for scenario in scenarios} == {
        "taxonomy30_head_on_above400_neg",
        "taxonomy30_head_on_below400_pos",
        "taxonomy30_head_on_level_neg",
    }
    assert {scenario["metadata"]["scenario_seed"] for scenario in scenarios} == {
        73007,
        73008,
        73010,
    }


def test_forced_method_alone_receives_the_initial_macro_override():
    runner = _load_runner_module()
    resolved = _load_resolved(EXPERT)
    materialized = runner._materialize_scenario_manifest(resolved, EXPERT)
    canonical = runner.build_eval_config(
        materialized,
        "canonical_ppo_high_level_policy",
        "head_on",
        backend="jsbsim",
        opponent_stage="expert",
    )
    forced = runner.build_eval_config(
        materialized,
        "noncanonical_forced_first_crossing_then_ppo",
        "head_on",
        backend="jsbsim",
        opponent_stage="expert",
    )
    assert "initial_macro_mode_override" not in canonical.get("commander", {})
    assert forced["commander"]["initial_macro_mode_override"] == {
        "enabled": True,
        "task_name": "head_on",
        "mode_id": 1,
        "reason": "noncanonical_forced_first_crossing_then_ppo",
    }
