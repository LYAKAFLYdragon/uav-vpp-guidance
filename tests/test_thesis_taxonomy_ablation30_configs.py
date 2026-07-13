from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_taxonomy_ablation30_base.yaml"
EXPERT = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_taxonomy_ablation30_expert.yaml"
END_TO_END = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_taxonomy_ablation30_end_to_end.yaml"
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
    spec = importlib.util.spec_from_file_location("taxonomy_ablation30_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_config_freezes_exactly_four_methods_and_the_rq4_crossing_contract():
    resolved = _load_resolved(BASE)
    expected = [
        "canonical_ppo_high_level_policy",
        "legacy_static_oracle_task_gate",
        "head_on_vpp_specialist_no_routing",
        "crossing_vpp_specialist_no_routing",
    ]
    assert resolved["run_defaults"]["main_methods"] == expected
    assert resolved["run_defaults"]["ablation_methods"] == expected
    assert resolved["thesis_taxonomy_ablation30"]["fixed_controls_bypass"] == {
        "high_level_routing": True,
        "recovery_override": True,
    }
    crossing = resolved["methods"]["crossing_vpp_specialist_no_routing"]
    assert crossing["agent_type"] == "ppo"
    assert crossing["config_overrides"] == {"observation.include_task_type": False}
    assert crossing["observation_contract"]["checkpoint_obs_dim"] == 18
    assert crossing["observation_contract"]["include_task_type"] is False


def test_manifest_materialization_gives_the_static_oracle_its_preregistered_task_groups():
    runner = _load_runner_module()
    materialized = runner._materialize_scenario_manifest(_load_resolved(BASE), BASE)

    assert materialized["scenario_manifest_resolution"]["task_counts"] == {
        "head_on": 24,
        "crossing_feasible": 6,
    }
    assert len(materialized["tasks"]["head_on"]["task"]["scenarios"]) == 24
    assert len(materialized["tasks"]["crossing_feasible"]["task"]["scenarios"]) == 6
    for task_name, task_def in materialized["tasks"].items():
        assert {
            scenario["metadata"]["task_registry_key"]
            for scenario in task_def["task"]["scenarios"]
        } == {task_name}


def test_split_configs_only_change_opponent_stage_and_output_identity():
    expert = _load_resolved(EXPERT)
    end_to_end = _load_resolved(END_TO_END)

    assert expert["opponent_stage"] == "expert"
    assert end_to_end["opponent_stage"] == "end_to_end"
    assert expert["run_defaults"]["output_id"] != end_to_end["run_defaults"]["output_id"]
    assert expert["run_defaults"]["main_methods"] == end_to_end["run_defaults"][
        "main_methods"
    ]
    assert expert["scenario_manifest"] == end_to_end["scenario_manifest"]
