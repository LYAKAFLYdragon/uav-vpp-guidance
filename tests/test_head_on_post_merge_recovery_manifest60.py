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


RECOVERY_MANIFEST60 = (
    CONFIG_DIR / "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot.yaml"
)
BASE_MANIFEST60 = (
    CONFIG_DIR
    / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot.yaml"
)


def test_recovery_manifest60_uses_canonical_two_method_family():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST60)

    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot"
    )
    assert resolved["run_defaults"]["main_methods"] == [
        "oracle_task_gate",
        "commander_post_merge_recovery3",
    ]
    assert resolved["run_defaults"]["ablation_methods"] == [
        "oracle_task_gate",
        "commander_post_merge_recovery3",
    ]


def test_recovery_manifest60_preserves_manifest60_task_scope():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST60)
    base = _load_yaml_resolved(BASE_MANIFEST60)

    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert resolved["tasks"]["head_on"] == base["tasks"]["head_on"]
    assert resolved["tasks"]["crossing_feasible"] == base["tasks"]["crossing_feasible"]
    assert len(resolved["tasks"]["head_on"]["task"]["scenarios"]) == 60
    assert len(resolved["tasks"]["crossing_feasible"]["task"]["scenarios"]) == 4


def test_recovery_manifest60_uses_minrange_only_recovery_profile_and_commander_method():
    resolved = _load_yaml_resolved(RECOVERY_MANIFEST60)

    assert resolved["virtual_point"]["post_merge_tactical_basis_recovery_profile"] == {
        "enabled": True,
        "task_name": "head_on",
        "anchor_mode": "predicted_target",
        "specialist_profile_names": ["post_merge_recovery"],
        "activate_on_blend_release_recovery": True,
        "require_first_pass_complete": True,
        "require_range_opening": True,
        "require_range_opening_for_specialist_profile": False,
        "require_no_attack_zone": True,
        "max_altitude_m": 5200.0,
        "tactical_basis_action_ll_scale": 0.35,
        "tactical_basis_action_ll_min": -0.25,
        "tactical_basis_action_ll_max": 0.35,
        "tactical_basis_action_io_scale": 0.55,
        "tactical_basis_action_io_abs_max": 0.6,
        "tactical_basis_action_io_entry_lateral_sign_hold_enabled": True,
        "tactical_basis_action_io_entry_lateral_sign_hold_release_vp_lateral_bias_m": 250.0,
        "tactical_basis_action_cd_scale": 0.5,
        "tactical_basis_action_cd_bias": 0.2,
        "tactical_basis_action_cd_min": 0.0,
        "tactical_basis_action_cd_max": 0.75,
        "entry_window_steps": 8,
        "tactical_basis_action_io_entry_window_min_abs": 0.18,
        "tactical_basis_action_ll_entry_window_overdeep_vp_forward_bias_m_max": -1800.0,
        "tactical_basis_action_ll_entry_window_floor": -0.05,
        "tactical_basis_action_cd_entry_window_descending_altitude_delta_m_min_abs": 15.0,
        "tactical_basis_action_cd_entry_window_min": 0.30,
        "predicted_target_forward_scale_entry_window_override": 0.0,
        "predicted_target_forward_scale_entry_window_overdeep_vp_forward_bias_m_max": 2000.0,
        "predicted_target_forward_scale_override": 0.25,
        "reason_overrides": {
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
        },
    }
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

    assert resolved["methods"]["commander_post_merge_recovery3"] == {
        "label": "Hierarchical Commander MVP (balanced tail eval10, 3-mode post-merge recovery bootstrap)",
        "agent_type": "hierarchical_commander",
        "checkpoint": (
            "outputs/diagnostics/"
            "hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_"
            "oracle_imitation_warmstart_balanced_tail_checkpoint_retrospective_expert_10seed_20260701_"
            "checkpoint_snapshots/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_hierarchical_commander_ppo_reset075_tactical_basis_3mode_post_merge_"
            "recovery_bootstrap.yaml"
        ),
        "prediction_variant": "learned",
    }
