from __future__ import annotations

import copy
from pathlib import Path

import yaml
from uav_vpp_guidance.training.train_prediction_vpp_ppo import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"


def _load_yaml(name: str):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


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


def test_prediction_jsbsim_compare_config_only_changes_prediction_block_and_anchor_mode():
    no_prediction = _load_yaml("train_no_prediction_vpp_ppo_jsbsim_compare.yaml")
    prediction = _load_yaml("train_prediction_vpp_ppo_jsbsim_compare.yaml")

    assert prediction["experiment"]["name"] == "prediction_vpp_ppo_jsbsim_compare"
    assert prediction["backend"] == no_prediction["backend"] == "jsbsim"
    assert prediction["env"] == no_prediction["env"]
    assert prediction["virtual_point"]["dynamics_aware"] is True
    assert prediction["virtual_point"]["dynamics_aware"] == no_prediction["virtual_point"]["dynamics_aware"]
    assert prediction["virtual_point"]["lookahead_steps"] == 5
    assert prediction["virtual_point"]["lookahead_steps"] == no_prediction["virtual_point"]["lookahead_steps"]
    assert prediction["virtual_point"]["anchor_mode"] == "predicted_target"
    assert no_prediction["virtual_point"]["anchor_mode"] == "current_target"

    assert prediction["trajectory_prediction"]["enabled"] is True
    assert prediction["trajectory_prediction"]["integration"]["anchor_mode"] == "predicted_target"
    assert prediction["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 1.0

    prediction_clone = copy.deepcopy(prediction)
    no_prediction_clone = copy.deepcopy(no_prediction)
    prediction_clone["experiment"]["name"] = no_prediction_clone["experiment"]["name"]
    prediction_clone["trajectory_prediction"] = no_prediction_clone["trajectory_prediction"]
    prediction_clone["virtual_point"]["anchor_mode"] = no_prediction_clone["virtual_point"]["anchor_mode"]
    assert prediction_clone == no_prediction_clone


def test_reset075_training_config_only_adds_validated_task_conditioned_vpp_overrides():
    long_lh1p0 = _load_yaml_resolved(
        CONFIG_DIR / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0.yaml"
    )
    reset075 = _load_yaml_resolved(
        CONFIG_DIR / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075.yaml"
    )

    assert reset075["experiment"]["name"] == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075"
    assert reset075["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 1.0
    assert reset075["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }
    assert reset075["virtual_point"]["post_merge_predicted_target_blend_by_task"] == {
        "head_on": 0.5,
        "crossing_feasible": 1.0,
    }
    assert reset075["virtual_point"]["post_merge_predicted_target_forward_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert reset075["virtual_point"][
        "post_merge_predicted_target_forward_scale_release_scale_by_task"
    ] == {"head_on": 0.75, "crossing_feasible": 1.0}
    assert reset075["virtual_point"][
        "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task"
    ] == {"head_on": 19}
    assert reset075["virtual_point"][
        "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task"
    ] == {"head_on": True, "crossing_feasible": False}
    assert reset075["virtual_point"]["close_range_anchor_mode_by_task"] == {
        "head_on": "current_target",
        "crossing_feasible": "predicted_target",
    }
    assert reset075["virtual_point"]["close_range_anchor_trigger_range_m_by_task"] == {
        "head_on": 1000.0,
        "crossing_feasible": 1000.0,
    }
    assert reset075["virtual_point"]["close_range_anchor_alignment_angle_deg_max_by_task"] == {
        "head_on": 10.0,
        "crossing_feasible": 180.0,
    }
    assert reset075["virtual_point"]["close_range_anchor_release_on_post_merge_by_task"] == {
        "head_on": True,
        "crossing_feasible": False,
    }

    reset075_clone = copy.deepcopy(reset075)
    reset075_clone["experiment"]["name"] = long_lh1p0["experiment"]["name"]
    for key in (
        "offset_frame_by_task",
        "post_merge_predicted_target_blend_by_task",
        "post_merge_predicted_target_forward_scale_by_task",
        "post_merge_predicted_target_forward_scale_release_scale_by_task",
        "post_merge_predicted_target_forward_scale_release_ego_only_streak_steps_by_task",
        "post_merge_predicted_target_forward_scale_release_reset_on_streak_break_by_task",
        "close_range_anchor_mode_by_task",
        "close_range_anchor_trigger_range_m_by_task",
        "close_range_anchor_alignment_angle_deg_max_by_task",
        "close_range_anchor_release_on_post_merge_by_task",
    ):
        reset075_clone["virtual_point"].pop(key)
    assert reset075_clone == long_lh1p0


def test_reset075_heldout_config_scopes_formal_methods_and_tasks():
    heldout = _load_yaml("jsbsim_hrl_reset075_heldout.yaml")

    assert heldout["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert heldout["experiment"]["name"] == "jsbsim_hrl_reset075_heldout"
    assert heldout["run_defaults"]["main_methods"] == [
        "no_prediction_vpp",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075",
    ]
    assert heldout["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert heldout["methods"]["prediction_vpp_jsbsim_compare_long_lh1p0_reset075"] == {
        "label": "Prediction VPP + HRL (JSBSim compare, lh1p0, reset075)",
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075/checkpoints/last.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_reset075_no_mode_switch_training_config_only_disables_mode_switch():
    reset075 = _load_yaml_resolved(
        CONFIG_DIR / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075.yaml"
    )
    no_mode_switch = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch.yaml"
    )

    assert (
        no_mode_switch["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch"
    )
    assert no_mode_switch["guidance"]["mode_switch"]["enabled"] is False

    no_mode_switch_clone = copy.deepcopy(no_mode_switch)
    no_mode_switch_clone["experiment"]["name"] = reset075["experiment"]["name"]
    no_mode_switch_clone["guidance"]["mode_switch"]["enabled"] = reset075["guidance"][
        "mode_switch"
    ]["enabled"]
    assert no_mode_switch_clone == reset075


def test_reset075_no_mode_switch_heldout_config_scopes_formal_methods_and_tasks():
    heldout = _load_yaml("jsbsim_hrl_reset075_no_mode_switch_heldout.yaml")

    assert heldout["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert heldout["experiment"]["name"] == "jsbsim_hrl_reset075_no_mode_switch_heldout"
    assert heldout["run_defaults"]["main_methods"] == [
        "no_prediction_vpp",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch",
    ]
    assert heldout["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert heldout["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch"
    ] == {
        "label": "Prediction VPP + HRL (JSBSim compare, lh1p0, reset075, no mode switch)",
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch/checkpoints/last.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_reset075_no_mode_switch_longscale00_training_config_only_adds_longitudinal_scale_override():
    no_mode_switch = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch.yaml"
    )
    longscale00 = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml"
    )

    assert (
        longscale00["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"
    )
    assert longscale00["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }

    longscale00_clone = copy.deepcopy(longscale00)
    longscale00_clone["experiment"]["name"] = no_mode_switch["experiment"]["name"]
    longscale00_clone["virtual_point"].pop("longitudinal_scale_by_task")
    assert longscale00_clone == no_mode_switch


def test_reset075_no_mode_switch_longscale00_heldout_config_scopes_formal_methods_and_tasks():
    heldout = _load_yaml("jsbsim_hrl_reset075_no_mode_switch_longscale00_heldout.yaml")

    assert heldout["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert heldout["experiment"]["name"] == "jsbsim_hrl_reset075_no_mode_switch_longscale00_heldout"
    assert heldout["run_defaults"]["main_methods"] == [
        "no_prediction_vpp",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
    ]
    assert heldout["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert heldout["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00"
    ] == {
        "label": (
            "Prediction VPP + HRL (JSBSim compare, lh1p0, reset075, no mode switch, longscale00)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/checkpoints/last.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_tactical_basis_headon_mvp_training_config_preserves_3d_action_head_and_task_extents():
    resolved = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp.yaml"
        )
    )

    assert resolved["policy"]["action_dim"] == 3
    assert resolved["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert resolved["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] == {
        "head_on": 1600.0,
        "crossing_feasible": 300.0,
    }
    assert resolved["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"][
        "head_on"
    ] > resolved["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"][
        "crossing_feasible"
    ]


def test_tactical_basis_headon_mvp_comparison_config_reuses_combat_only_scope():
    pilot = _load_yaml(
        "jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp.yaml"
    )

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["main_methods"] == [
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp",
    ]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp"
    ] == {
        "label": "Prediction VPP + HRL (tactical basis MVP, reset075 no mode switch baseline)",
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp/checkpoints/last.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_reset075_no_mode_switch_smoke_pilot_config_scopes_combat_only_methods_and_tasks():
    pilot = _load_yaml("jsbsim_hrl_reset075_no_mode_switch_smoke_pilot.yaml")

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["experiment"]["name"] == "jsbsim_hrl_reset075_no_mode_switch_smoke_pilot"
    assert pilot["run_defaults"]["main_methods"] == [
        "no_prediction_vpp",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_smoke",
    ]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["formal_small_seeds"] == [0, 1, 2]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_smoke"
    ] == {
        "label": (
            "Prediction VPP + HRL (JSBSim compare, lh1p0, reset075, no mode switch, smoke)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_smoke_check/checkpoints/last.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_prediction_training_loader_resolves_nested_includes_for_reset075():
    cfg = load_experiment_config(
        str(CONFIG_DIR / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075.yaml")
    )

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["trajectory_prediction"]["predictor_type"] == "lstm"
    assert cfg["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 1.0
    assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
    assert cfg["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }


def test_prediction_training_loader_resolves_nested_includes_for_reset075_no_mode_switch():
    cfg = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch.yaml"
        )
    )

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["trajectory_prediction"]["predictor_type"] == "lstm"
    assert cfg["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 1.0
    assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
    assert cfg["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }
    assert cfg["guidance"]["mode_switch"]["enabled"] is False


def test_prediction_training_loader_resolves_nested_includes_for_reset075_no_mode_switch_longscale00():
    cfg = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00.yaml"
        )
    )

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["trajectory_prediction"]["predictor_type"] == "lstm"
    assert cfg["trajectory_prediction"]["prediction"]["lookahead_time_s"] == 1.0
    assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
    assert cfg["virtual_point"]["offset_frame_by_task"] == {
        "head_on": "target_velocity",
        "crossing_feasible": "world_neu",
    }
    assert cfg["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert cfg["guidance"]["mode_switch"]["enabled"] is False


def test_reset075_no_mode_switch_longscale00_combat_finetune_config_adds_combat_scope_and_warm_start():
    combat = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat_finetune.yaml"
    )

    assert (
        combat["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat_finetune"
    )
    assert combat["attack_zone"]["enabled"] is True
    assert combat["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert combat["combat_finetune"]["comparison_config_path"] == (
        "config/experiment/jsbsim_hrl_comparison.yaml"
    )
    assert combat["combat_finetune"]["tasks"] == [
        "head_on",
        "crossing_feasible",
    ]
    assert combat["combat_finetune"]["opponent_stages"] == [
        "expert",
        "end_to_end",
    ]
    assert combat["combat_finetune"]["selection_metric"] == "win_rate"
    assert combat["warm_start"] == {
        "enabled": True,
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/"
            "checkpoints/last.pt"
        ),
        "step": 0,
        "load_optimizer": False,
        "strict_dims": True,
    }
    assert combat["ppo"]["total_timesteps"] == 16384
    assert combat["evaluation"]["eval_interval"] == 4096
    assert combat["evaluation"]["eval_episodes"] == 1
    assert combat["evaluation"]["seeds"] == [480, 481, 482]


def test_prediction_training_loader_resolves_nested_includes_for_combat_finetune():
    cfg = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat_finetune.yaml"
        )
    )

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["trajectory_prediction"]["predictor_type"] == "lstm"
    assert cfg["virtual_point"]["anchor_mode"] == "predicted_target"
    assert cfg["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 0.0,
        "crossing_feasible": 1.0,
    }
    assert cfg["guidance"]["mode_switch"]["enabled"] is False
    assert cfg["attack_zone"]["enabled"] is True
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert cfg["combat_finetune"]["tasks"] == ["head_on", "crossing_feasible"]


def test_tactical_basis_headon_mvp_combat_finetune_config_keeps_tactical_semantics_and_baseline_warm_start():
    combat = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml"
    )

    assert (
        combat["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune"
    )
    assert combat["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert combat["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] == {
        "head_on": 1600.0,
        "crossing_feasible": 300.0,
    }
    assert combat["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 1.0,
        "crossing_feasible": 1.0,
    }
    assert combat["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert combat["combat_finetune"]["tasks"] == ["head_on", "crossing_feasible"]
    assert combat["combat_finetune"]["opponent_stages"] == ["expert", "end_to_end"]
    assert combat["warm_start"] == {
        "enabled": True,
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/"
            "checkpoints/last.pt"
        ),
        "step": 0,
        "load_optimizer": False,
        "strict_dims": True,
    }
    assert combat["ppo"]["total_timesteps"] == 16384
    assert combat["evaluation"]["seeds"] == [480, 481, 482]


def test_tactical_basis_headon_mvp_combat_finetune_comparison_config_points_to_best_checkpoint():
    pilot = _load_yaml(
        "jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_best.yaml"
    )

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["main_methods"] == [
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_best",
    ]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_best"
    ] == {
        "label": (
            "Prediction VPP + HRL (tactical basis combat finetune best, reset075 no mode switch longscale00)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune/checkpoints/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_prediction_training_loader_resolves_nested_includes_for_tactical_basis_combat_finetune():
    cfg = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune.yaml"
        )
    )

    assert cfg["trajectory_prediction"]["enabled"] is True
    assert cfg["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert cfg["virtual_point"]["longitudinal_scale_by_task"] == {
        "head_on": 1.0,
        "crossing_feasible": 1.0,
    }
    assert cfg["guidance"]["mode_switch"]["enabled"] is False
    assert cfg["attack_zone"]["enabled"] is True
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert cfg["combat_finetune"]["tasks"] == ["head_on", "crossing_feasible"]


def test_tactical_basis_headon_mvp_combat_finetune_narrow_extents_config_reduces_forward_and_vertical_amplitudes():
    cfg = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml"
    )

    assert (
        cfg["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents"
    )
    assert cfg["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert cfg["virtual_point"]["tactical_basis_lead_lag_extent_m_by_task"] == {
        "head_on": 300.0,
        "crossing_feasible": 250.0,
    }
    assert cfg["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] == {
        "head_on": 600.0,
        "crossing_feasible": 300.0,
    }
    assert cfg["virtual_point"]["tactical_basis_climb_descent_extent_m_by_task"] == {
        "head_on": 100.0,
        "crossing_feasible": 75.0,
    }
    assert cfg["warm_start"]["checkpoint"] == (
        "outputs/experiments/"
        "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00/"
        "checkpoints/last.pt"
    )


def test_tactical_basis_headon_mvp_combat_finetune_narrow_extents_comparison_config_points_to_best_checkpoint():
    pilot = _load_yaml(
        "jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best.yaml"
    )

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["main_methods"] == [
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best",
    ]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents_best"
    ] == {
        "label": (
            "Prediction VPP + HRL (tactical basis combat finetune narrow extents best, reset075 no mode switch longscale00)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents/checkpoints/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_config_keeps_narrow_head_on_and_restores_crossing_long_vertical_extents():
    cfg = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml"
    )

    assert (
        cfg["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored"
    )
    assert cfg["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert cfg["virtual_point"]["tactical_basis_lead_lag_extent_m_by_task"] == {
        "head_on": 300.0,
        "crossing_feasible": 800.0,
    }
    assert cfg["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] == {
        "head_on": 600.0,
        "crossing_feasible": 300.0,
    }
    assert cfg["virtual_point"]["tactical_basis_climb_descent_extent_m_by_task"] == {
        "head_on": 100.0,
        "crossing_feasible": 300.0,
    }
    assert cfg["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert cfg["combat_finetune"]["tasks"] == ["head_on", "crossing_feasible"]
    assert cfg["combat_finetune"]["opponent_stages"] == ["expert", "end_to_end"]
    assert cfg["combat_finetune"]["selection_metric"] == {
        "mode": "lane_gated",
        "objective": "overall.win_rate",
        "fallback_metric": "overall.mean_return",
        "gates": [
            {
                "task": "head_on",
                "opponent_stage": "expert",
                "metric": "win_rate",
                "min_value": 0.5,
            },
            {
                "task": "head_on",
                "opponent_stage": "end_to_end",
                "metric": "crash_rate",
                "max_value": 0.5,
            },
        ],
    }
    assert cfg["warm_start"]["checkpoint"] == (
        "outputs/experiments/"
        "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents/"
        "checkpoints/step_12288.pt"
    )


def test_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_comparison_config_points_to_best_checkpoint():
    pilot = _load_yaml(
        "jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best.yaml"
    )

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["main_methods"] == [
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best",
    ]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored_best"
    ] == {
        "label": (
            "Prediction VPP + HRL (tactical basis combat finetune mixed crossing-restored best, reset075 no mode switch longscale00)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored/checkpoints/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_crossing_restored.yaml"
        ),
        "prediction_variant": "learned",
    }


def test_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_config_adds_end_to_end_win_gate_and_more_conservative_head_on_extents():
    cfg = _load_yaml_resolved(
        CONFIG_DIR
        / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon.yaml"
    )

    assert (
        cfg["experiment"]["name"]
        == "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon"
    )
    assert cfg["virtual_point"]["action_semantics"] == "tactical_basis_v1"
    assert cfg["virtual_point"]["tactical_basis_lead_lag_extent_m_by_task"] == {
        "head_on": 250.0,
        "crossing_feasible": 800.0,
    }
    assert cfg["virtual_point"]["tactical_basis_inside_outside_extent_m_by_task"] == {
        "head_on": 350.0,
        "crossing_feasible": 300.0,
    }
    assert cfg["virtual_point"]["tactical_basis_climb_descent_extent_m_by_task"] == {
        "head_on": 60.0,
        "crossing_feasible": 300.0,
    }
    assert cfg["combat_finetune"]["selection_metric"] == {
        "mode": "lane_gated",
        "objective": "overall.win_rate",
        "fallback_metric": "overall.mean_return",
        "gates": [
            {
                "task": "head_on",
                "opponent_stage": "expert",
                "metric": "win_rate",
                "min_value": 0.5,
            },
            {
                "task": "head_on",
                "opponent_stage": "end_to_end",
                "metric": "crash_rate",
                "max_value": 0.5,
            },
            {
                "task": "head_on",
                "opponent_stage": "end_to_end",
                "metric": "win_rate",
                "min_value": 0.5,
            },
        ],
    }
    assert cfg["warm_start"]["checkpoint"] == (
        "outputs/experiments/"
        "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_narrow_extents/"
        "checkpoints/step_12288.pt"
    )


def test_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_comparison_config_points_to_best_checkpoint():
    pilot = _load_yaml(
        "jsbsim_hrl_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_best.yaml"
    )

    assert pilot["includes"] == ["./jsbsim_hrl_combat_only_pilot.yaml"]
    assert pilot["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]
    assert pilot["run_defaults"]["main_methods"] == [
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00",
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_best",
    ]
    assert pilot["methods"][
        "prediction_vpp_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon_best"
    ] == {
        "label": (
            "Prediction VPP + HRL (tactical basis combat finetune mixed quality-gated conservative head_on best, reset075 no mode switch longscale00)"
        ),
        "agent_type": "ppo",
        "checkpoint": (
            "outputs/experiments/"
            "prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon/checkpoints/best.pt"
        ),
        "config_path": (
            "config/experiment/"
            "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_quality_gated_conservative_headon.yaml"
        ),
        "prediction_variant": "learned",
    }
