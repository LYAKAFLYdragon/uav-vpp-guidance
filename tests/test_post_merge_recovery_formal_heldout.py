from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"

FORMAL_HELDOUT = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_heldout.yaml"
)
RECOVERY_MANIFEST60 = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot.yaml"
)
RECOVERY_MANIFEST90 = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest90_pilot.yaml"
)

ORACLE_METHOD = "oracle_task_gate"
COMMANDER_METHOD = "commander_post_merge_recovery3"


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_resolved(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    includes = payload.pop("includes", []) or []
    merged = {}
    for include in includes:
        merged = _deep_merge(merged, _load_yaml_resolved((path.parent / include).resolve()))
    return _deep_merge(merged, payload)


def _extract_head_on_tuples(cfg_path: Path):
    cfg = _load_yaml_resolved(cfg_path)
    scenarios = cfg["tasks"]["head_on"]["task"]["scenarios"]
    tuples = set()
    for scenario in scenarios:
        own = scenario["own_init"]
        target = scenario["target_init"]
        range_x = int(round(float(target["position_m"][0])))
        lateral_y = int(round(float(target["position_m"][1])))
        altitude_diff = int(round(float(target["position_m"][2] - own["position_m"][2])))
        own_speed = int(round(float(own["velocity_mps"])))
        target_speed = int(round(float(target["velocity_mps"])))
        tuples.add((range_x, own_speed, target_speed, altitude_diff, lateral_y))
    return tuples


def test_recovery_formal_heldout_uses_frozen_two_method_family():
    resolved = _load_yaml_resolved(FORMAL_HELDOUT)

    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_heldout"
    )
    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert resolved["run_defaults"]["main_methods"] == [
        ORACLE_METHOD,
        COMMANDER_METHOD,
    ]
    assert resolved["run_defaults"]["ablation_methods"] == [
        ORACLE_METHOD,
        COMMANDER_METHOD,
    ]
    assert resolved["methods"][COMMANDER_METHOD]["agent_type"] == (
        "hierarchical_commander"
    )


def test_recovery_formal_heldout_keeps_disjoint_formal_scope():
    resolved = _load_yaml_resolved(FORMAL_HELDOUT)

    head_on_scenarios = resolved["tasks"]["head_on"]["task"]["scenarios"]
    crossing_scenarios = resolved["tasks"]["crossing_feasible"]["task"]["scenarios"]
    assert len(head_on_scenarios) == 36
    assert len(crossing_scenarios) == 4
    assert {scenario["metadata"]["manifest_family"] for scenario in head_on_scenarios} == {
        "formal_heldout"
    }

    heldout_tuples = _extract_head_on_tuples(FORMAL_HELDOUT)
    pilot_tuples = _extract_head_on_tuples(RECOVERY_MANIFEST60) | _extract_head_on_tuples(
        RECOVERY_MANIFEST90
    )
    assert not heldout_tuples & pilot_tuples


def test_recovery_formal_heldout_carries_current_commander_guards():
    resolved = _load_yaml_resolved(FORMAL_HELDOUT)

    assert resolved["commander"]["num_modes"] == 3
    assert resolved["commander"]["modes"][2] == {
        "id": 2,
        "name": "post_merge_recovery_specialist",
        "specialist_key": "post_merge_recovery",
        "source_specialist_key": "head_on",
        "specialist_profile": "post_merge_recovery",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_"
            "tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted/"
            "checkpoints/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_"
            "tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted.yaml"
        ),
    }
    assert resolved["commander"][
        "head_on_post_merge_reopened_crossing_target_threat_clamp"
    ] == {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_require_no_attack_zone": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
    }
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "predicted_target_forward_scale_override"
    ] == 0.25
