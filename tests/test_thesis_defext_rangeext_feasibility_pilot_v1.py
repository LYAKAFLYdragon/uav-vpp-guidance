from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config" / "experiment" / "thesis_defext_rangeext_feasibility_pilot_v1.yaml"
DEV = ROOT / "config" / "experiment" / "manifests" / "thesis_defext_rangeext_feasibility_pilot_v1_dev12.yaml"
HELDOUT = ROOT / "config" / "experiment" / "manifests" / "thesis_defext_rangeext_feasibility_pilot_v1_heldout24.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_preregistration_is_design_only_and_fixes_skill_profile_and_methods():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["source_id"] == "THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1"
    assert config["authorization"]["training_permitted"] is False
    assert config["authorization"]["pilot_execution_permitted"] is False
    assert config["fixed_contract"]["skill"] == "defensive_extension"
    assert config["fixed_contract"]["profile"] == "range_extension"
    assert config["fixed_contract"]["routing_enabled"] is False
    assert config["fixed_contract"]["attack_zone"]["enabled"] is True
    assert config["fixed_contract"]["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert set(config["methods"]) == {
        "candidate_fixed_defensive_extension",
        "frozen_fixed_head_on",
        "frozen_fixed_crossing",
    }
    assert config["methods"]["candidate_fixed_defensive_extension"]["checkpoint"] is None
    assert config["proposed_training"]["ppo"]["value_coef"] == 0.5
    assert config["proposed_training"]["ppo"]["max_grad_norm"] == 0.5
    assert config["proposed_training"]["policy"]["hidden_dim"] == 128
    assert config["proposed_training"]["checkpoint_selection_rule"] == "minimize_worst_opponent_mean_paired_intent_loss_auc20_delta_vs_head_on"


def test_dev12_and_heldout24_are_reproducible_and_disjoint_from_v2():
    builder = _load(
        SCRIPTS / "build_thesis_defext_rangeext_feasibility_pilot_v1_manifests.py",
        "defext_manifest_builder",
    )
    dev = yaml.safe_load(DEV.read_text(encoding="utf-8"))
    heldout = yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))
    assert len(dev["scenarios"]) == 12
    assert len(heldout["scenarios"]) == 24
    assert builder.payload_sha256(dev) == dev["integrity"]["payload_sha256"]
    assert builder.payload_sha256(heldout) == heldout["integrity"]["payload_sha256"]
    dev_signatures = {
        builder.package_signature(item)
        for item in dev["generation_contract"]["packages"]
    }
    heldout_signatures = {
        builder.package_signature(item)
        for item in heldout["generation_contract"]["packages"]
    }
    assert not (dev_signatures & heldout_signatures)
    assert not ((dev_signatures | heldout_signatures) & builder.V2_SIGNATURES)
    assert all(
        scenario["metadata"]["taxonomy_geometry_state"] == "disadvantage"
        and scenario["metadata"]["continuous_run_in"] is True
        and scenario["metadata"]["routing_enabled"] is False
        for scenario in dev["scenarios"] + heldout["scenarios"]
    )


def test_gate_and_stop_rule_preserve_three_opponent_causal_decision():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["opponents"]["order"] == [
        "expert",
        "end_to_end",
        "independent_ppo_vpp",
    ]
    assert config["opponents"]["pooled_gate"] == "prohibited"
    assert config["gates"]["per_opponent_contract_minimum"] == {
        "qualifying_paired_episodes": 2,
        "valid_target_steps": 20,
        "distinct_scenario_signatures": 2,
        "distinct_mirror_signs": 2,
        "all_66d_and_3d_finite": True,
        "no_padding": True,
        "no_prediction_fallback": True,
        "no_reset": True,
        "no_backend_fallback": True,
        "no_checkpoint_fallback": True,
    }
    assert config["gates"]["safety_noninferiority_vs_head_on"]["ego_crash_oob_rate_delta_max"] == 0.05
    assert config["gates"]["practical_improvement_vs_head_on"]["paired_intent_loss_auc20_delta_max"] == -0.05
    assert config["gates"]["practical_improvement_vs_best_existing_specialist"]["paired_intent_loss_auc20_delta_max"] == -0.02
    assert config["safety_stop_rule"]["after_abort"]["add_training_steps"] == "prohibited"
    assert config["safety_stop_rule"]["after_abort"]["rerun_same_source_id"] == "prohibited"


def test_independent_review_passes_without_authorizing_execution(tmp_path: Path):
    reviewer = _load(
        SCRIPTS / "review_thesis_defext_rangeext_feasibility_pilot_v1.py",
        "defext_prereg_reviewer",
    )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["outputs"]["root"] = str(
        tmp_path / "defensive_extension_range_extension_feasibility_v1"
    )
    config_path = tmp_path / "pilot_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = reviewer.review(config_path)
    assert result["passed"] is True
    assert result["training_authorized"] is False
    assert result["pilot_execution_authorized"] is False
