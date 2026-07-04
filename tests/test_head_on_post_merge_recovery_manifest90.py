from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"


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


RECOVERY_MANIFEST90 = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest90_pilot.yaml"
)
BASE_MANIFEST90 = (
    CONFIG_DIR
    / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest90_pilot.yaml"
)
RECOVERY_MANIFEST60 = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot.yaml"
)


def test_recovery_manifest90_uses_current_two_method_family():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST90)

    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest90_pilot"
    )
    assert resolved["run_defaults"]["main_methods"] == [
        "oracle_task_gate",
        "commander_post_merge_recovery3",
    ]
    assert resolved["run_defaults"]["ablation_methods"] == [
        "oracle_task_gate",
        "commander_post_merge_recovery3",
    ]


def test_recovery_manifest90_preserves_manifest90_task_scope():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST90)
    base = _load_yaml_resolved(BASE_MANIFEST90)
    manifest60 = _load_yaml_resolved(RECOVERY_MANIFEST60)

    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert resolved["tasks"]["head_on"] == base["tasks"]["head_on"]
    assert resolved["tasks"]["crossing_feasible"] == base["tasks"]["crossing_feasible"]
    assert len(resolved["tasks"]["head_on"]["task"]["scenarios"]) == 90
    assert len(resolved["tasks"]["crossing_feasible"]["task"]["scenarios"]) == 4

    manifest60_names = {
        scenario["name"]
        for scenario in manifest60["tasks"]["head_on"]["task"]["scenarios"]
    }
    manifest90_names = {
        scenario["name"]
        for scenario in resolved["tasks"]["head_on"]["task"]["scenarios"]
    }
    assert manifest60_names.issubset(manifest90_names)


def test_recovery_manifest90_keeps_minrange_only_post_merge_guard():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST90)

    assert resolved["commander"][
        "head_on_post_merge_reopened_crossing_overdeep_clamp"
    ]["head_on_min_range_so_far_m"] == 180.0
    assert resolved["commander"][
        "head_on_post_merge_reopened_crossing_target_threat_clamp"
    ] == {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 0,
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "pre_threat_entry_guard_enabled": True,
        "pre_threat_require_no_attack_zone": True,
        "pre_threat_min_range_m": 4500.0,
        "pre_threat_min_range_rate_mps": 50.0,
        "pre_threat_max_vp_forward_bias_m": -2500.0,
        "pre_threat_min_abs_vp_lateral_to_range_ratio": 1.5,
    }
    assert resolved["commander"]["head_on_post_merge_first_recovery_entry_hold"] == {
        "enabled": True,
        "task_name": "head_on",
        "head_on_mode_id": 0,
        "recovery_mode_id": 2,
        "hold_window_steps": 8,
        "allowed_reasons": [
            "close_range_reengagement_crossing",
            "overdeep_low_lateral_reopened_head_on",
            "pre_threat_opening_overlateral_negative_forward_head_on",
        ],
    }
    assert resolved["commander"]["head_on_post_merge_recovery_hold"] == {
        "enabled": True,
        "task_name": "head_on",
        "recovery_mode_id": 2,
        "min_range_m": 4500.0,
        "require_no_attack_zone": True,
        "min_abs_vp_lateral_bias_m": 6000.0,
        "min_abs_vp_lateral_to_range_ratio": 1.4,
    }
    assert resolved["methods"]["commander_post_merge_recovery3"]["agent_type"] == (
        "hierarchical_commander"
    )
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
