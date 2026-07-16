from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION_CONTRACT = ROOT / "config" / "experiment" / "decision_contracts" / "thesis_dualskill_taxonomy60_no_go_v3.yaml"
RUNNER = ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"


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


def _load_runner():
    spec = importlib.util.spec_from_file_location("taxonomy60_formal_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_config_materializes_exactly_the_frozen_nine_hundred_record_universe():
    runner = _load_runner()
    resolved = _load_resolved(CONFIG)
    materialized = runner._materialize_scenario_manifest(resolved, CONFIG)
    protocol = materialized["thesis_dualskill_taxonomy60_formal"]

    assert protocol["source_id"] == "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
    assert protocol["expected_episodes"] == 900
    assert materialized["scenario_manifest_resolution"]["task_counts"] == {
        "head_on": 48,
        "crossing_feasible": 12,
    }
    assert materialized["run_defaults"]["main_methods"] == protocol["methods"]
    assert materialized["env"]["strict_backend"] is True
    assert materialized["run_defaults"]["n_episodes"] == 1


def test_formal_config_builds_all_method_task_opponent_combinations_without_cli_tuning():
    runner = _load_runner()
    materialized = runner._materialize_scenario_manifest(_load_resolved(CONFIG), CONFIG)
    protocol = materialized["thesis_dualskill_taxonomy60_formal"]

    for opponent in protocol["opponents"]:
        for method in protocol["methods"]:
            for task in materialized["run_defaults"]["tasks"]:
                config = runner.build_eval_config(
                    materialized,
                    method,
                    task,
                    backend="jsbsim",
                    opponent_stage=opponent,
                )
                assert config["backend"] == "jsbsim"
                assert config["env"]["strict_backend"] is True
                assert config["opponent_stage"] == opponent

    canonical = runner.build_eval_config(
        materialized, "canonical_ppo", "crossing_feasible", "jsbsim", "expert"
    )
    unlocked = runner.build_eval_config(
        materialized,
        "ppo_unbootstrapped_crossing",
        "crossing_feasible",
        "jsbsim",
        "expert",
    )
    assert canonical["commander"]["crossing_initial_task_bootstrap"]["enabled"] is True
    assert unlocked["commander"]["crossing_initial_task_bootstrap"]["enabled"] is False


def test_formal_manifest_and_decision_contract_freeze_the_prediction_warmup_boundary():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    contract = yaml.safe_load(DECISION_CONTRACT.read_text(encoding="utf-8"))
    expected = {
        "warmup_fallback_high_level_steps": 9,
        "first_required_valid_high_level_step": 10,
        "fallback_after_warmup_permitted": False,
    }

    assert manifest["manifest_version"] == 4
    assert {
        key: manifest["prediction_receipt_contract"][key] for key in expected
    } == expected
    actual_contract = contract["formal_validity"]["prediction_receipt_contract"]
    assert {key: actual_contract[key] for key in expected} == expected
    assert "first nine high-level steps" in actual_contract["interpretation"]
    assert "not backend or checkpoint fallback" in actual_contract["interpretation"]
