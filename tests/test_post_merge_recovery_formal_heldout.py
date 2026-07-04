from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"

FORMAL_HELDOUT = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_heldout.yaml"
)
FORMAL_ORACLE_ONLY_AUDIT = (
    CONFIG_DIR
    / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_oracle_only_headon_audit.yaml"
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


def test_recovery_formal_heldout_keeps_minrange_only_commander_guard():
    resolved = _load_yaml_resolved(FORMAL_HELDOUT)

    assert resolved["commander"]["num_modes"] == 3
    assert resolved["commander"][
        "head_on_post_merge_reopened_crossing_overdeep_clamp"
    ] == {
        "enabled": True,
        "task_name": "head_on",
        "crossing_mode_id": 1,
        "forced_mode_id": 2,
        "close_range_max_range_m": 3000.0,
        "min_range_m": 4500.0,
        "min_negative_vp_forward_bias_m": 8000.0,
        "max_abs_vp_lateral_to_range_ratio": 1.5,
        "head_on_mode_id": 0,
        "head_on_min_range_so_far_m": 180.0,
        "head_on_min_vp_lateral_bias_m": -1000.0,
    }
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
        "head_on_mode_id": 0,
        "head_on_forced_mode_id": 2,
        "head_on_min_vp_lateral_bias_m": -7900.0,
        "head_on_max_vp_forward_bias_m": 2000.0,
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
        "hold_window_steps_by_reason": {
            "close_range_reengagement_crossing": 0,
            "overdeep_low_lateral_reopened_head_on": 0,
            "pre_threat_opening_overlateral_negative_forward_head_on": 0,
        },
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
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "predicted_target_forward_scale_override"
    ] == 0.25
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_io_entry_lateral_sign_hold_enabled"
    ] is True
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_io_entry_lateral_sign_hold_release_vp_lateral_bias_m"
    ] == 250.0
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "entry_window_steps"
    ] == 8
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_io_entry_window_min_abs"
    ] == 0.18
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_ll_entry_window_overdeep_vp_forward_bias_m_max"
    ] == -1800.0
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_ll_entry_window_floor"
    ] == -0.05
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_cd_entry_window_descending_altitude_delta_m_min_abs"
    ] == 15.0
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "tactical_basis_action_cd_entry_window_min"
    ] == 0.30
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "predicted_target_forward_scale_entry_window_override"
    ] == 0.0
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max"
    ] == 2000.0
    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"][
        "reason_overrides"
    ] == {
        "close_range_reengagement_crossing": {
            "conditions": {
                "max_aa_deg": 129.0,
                "max_previous_vp_forward_bias_m": 0.0,
                "any": [
                    {"min_abs_previous_vp_lateral_bias_m": 145.0},
                    {"max_range_rate_mps": 275.0},
                ],
            },
            "overrides": {
                "tactical_basis_action_io_entry_lateral_sign_hold_enabled": False,
                "tactical_basis_action_io_entry_window_min_abs": 0.0,
                "tactical_basis_action_cd_scale": 0.0,
                "tactical_basis_action_cd_bias": 0.0,
            },
        }
    }


def test_recovery_formal_oracle_only_audit_is_exact_five_scenario_subset():
    resolved = _load_yaml_resolved(FORMAL_ORACLE_ONLY_AUDIT)
    head_on_scenarios = resolved["tasks"]["head_on"]["task"]["scenarios"]

    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_formal_oracle_only_headon_audit"
    )
    assert resolved["run_defaults"]["tasks"] == ["head_on"]
    assert len(head_on_scenarios) == 5
    assert [scenario["name"] for scenario in head_on_scenarios] == [
        "head_on_formal_r3300_v245_v255_altm450",
        "head_on_formal_r1600_v195_v195_alt0_ypos100",
        "head_on_formal_r2300_v225_v215_altp250_ypos100",
        "head_on_formal_r3300_v245_v255_altm450_ypos100",
        "head_on_formal_r2700_v245_v235_altp350_yneg100",
    ]
    assert {
        scenario["metadata"]["manifest_family"] for scenario in head_on_scenarios
    } == {"formal_oracle_only_audit"}
