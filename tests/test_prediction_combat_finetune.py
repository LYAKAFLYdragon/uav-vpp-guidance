from __future__ import annotations

from pathlib import Path

import torch

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.provenance import get_config_overrides
from uav_vpp_guidance.training.train_prediction_vpp_combat_finetune import (
    build_lane_training_config,
    compute_selection_result,
    resolve_combat_finetune_plan,
)
from uav_vpp_guidance.training.train_prediction_vpp_ppo import (
    load_experiment_config,
    load_warm_start_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"


def _agent_config():
    return {
        "policy": {"hidden_sizes": [16], "activation": "tanh", "action_dim": 3},
        "ppo": {"rollout_steps": 8, "minibatch_size": 4, "learning_rate": 3e-4},
    }


def test_resolve_combat_finetune_plan_scopes_only_requested_task_opponent_lanes():
    config = load_experiment_config(
        str(
            CONFIG_DIR
            / "train_prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_combat_finetune.yaml"
        )
    )

    plan = resolve_combat_finetune_plan(config)

    assert plan["tasks"] == ["head_on", "crossing_feasible"]
    assert plan["opponent_stages"] == ["expert", "end_to_end"]
    assert {lane["lane_id"] for lane in plan["lanes"]} == {
        "head_on::expert",
        "head_on::end_to_end",
        "crossing_feasible::expert",
        "crossing_feasible::end_to_end",
    }
    assert all(lane["weight"] == 1.0 for lane in plan["lanes"])
    assert {
        lane["task_name"]: lane["scenario_names"] for lane in plan["lanes"]
    }["head_on"] == ["head_on_minimal"]
    assert {
        lane["task_name"]: lane["scenario_names"] for lane in plan["lanes"]
    }["crossing_feasible"] == ["crossing_feasible_left_1500m"]


def test_build_lane_training_config_records_provenance_for_task_opponent_and_attack_zone():
    base_config = {
        "attack_zone": {"enabled": False},
        "virtual_point": {"anchor_mode": "predicted_target"},
        "trajectory_prediction": {"enabled": True},
    }
    lane_spec = {
        "lane_id": "head_on::expert",
        "task_name": "head_on",
        "task_config": {
            "name": "head_on",
            "env_class": "CloseRangeTrackingEnv",
            "scenarios": [{"name": "head_on_minimal"}],
        },
        "opponent_stage": "expert",
        "opponent_entry": {"stage": "expert", "type": "rule_based", "config": {}},
    }

    lane_config = build_lane_training_config(base_config, lane_spec)
    overrides = get_config_overrides(lane_config)
    override_keys = {item["key"] for item in overrides}

    assert lane_config["task"]["name"] == "head_on"
    assert lane_config["opponent_stage"] == "expert"
    assert lane_config["opponent"]["stage"] == "expert"
    assert lane_config["attack_zone"]["enabled"] is True
    assert lane_config["attack_zone"]["close_range_max_aoa_deg"] == 60.0
    assert override_keys == {
        "task",
        "opponent_stage",
        "opponent",
        "attack_zone.enabled",
        "attack_zone.close_range_max_aoa_deg",
    }


def test_prediction_training_warm_start_helper_loads_policy_weights(tmp_path):
    source = PPOAgent(obs_dim=16, action_dim=3, config=_agent_config(), device="cpu")
    with torch.no_grad():
        for param in source.network.parameters():
            param.fill_(0.234)

    checkpoint_path = tmp_path / "warm_start.pt"
    source.save(str(checkpoint_path))

    target = PPOAgent(obs_dim=16, action_dim=3, config=_agent_config(), device="cpu")
    meta = load_warm_start_checkpoint(
        target,
        checkpoint_path=str(checkpoint_path),
        strict_dims=True,
        load_optimizer=False,
    )

    assert meta["loaded"] is True
    assert meta["checkpoint_path"] == str(checkpoint_path)
    assert meta["optimizer_loaded"] is False
    for param in target.network.parameters():
        assert torch.allclose(param, torch.full_like(param, 0.234))


def test_compute_selection_result_uses_overall_metric_for_simple_selection():
    eval_result = {
        "overall": {"win_rate": 0.75, "mean_return": -12.0},
        "by_lane": [],
    }

    result = compute_selection_result(eval_result, "win_rate")

    assert result["selection_metric"] == "win_rate"
    assert result["selection_objective_metric"] == "overall.win_rate"
    assert result["selection_score"] == 0.75
    assert result["selection_gate_passed"] is True
    assert result["selection_gate_total"] == 0


def test_compute_selection_result_lane_gated_prioritizes_gate_satisfaction():
    gated_cfg = {
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
    failing = {
        "overall": {"win_rate": 0.90, "mean_return": -10.0},
        "by_lane": [
            {"task": "head_on", "opponent_stage": "expert", "win_rate": 1.0, "crash_rate": 0.0},
            {"task": "head_on", "opponent_stage": "end_to_end", "win_rate": 0.0, "crash_rate": 1.0},
        ],
    }
    passing = {
        "overall": {"win_rate": 0.60, "mean_return": -20.0},
        "by_lane": [
            {"task": "head_on", "opponent_stage": "expert", "win_rate": 2.0 / 3.0, "crash_rate": 0.0},
            {"task": "head_on", "opponent_stage": "end_to_end", "win_rate": 1.0 / 3.0, "crash_rate": 1.0 / 3.0},
        ],
    }

    failing_result = compute_selection_result(failing, gated_cfg)
    passing_result = compute_selection_result(passing, gated_cfg)

    assert failing_result["selection_gate_passed"] is False
    assert failing_result["selection_gate_pass_count"] == 1
    assert failing_result["selection_gate_total"] == 2
    assert '"lane_id": "head_on::end_to_end"' in failing_result["selection_gate_failures"]

    assert passing_result["selection_gate_passed"] is True
    assert passing_result["selection_gate_pass_count"] == 2
    assert passing_result["selection_gate_total"] == 2
    assert passing_result["selection_score"] > 1_000_000.0
    assert passing_result["selection_score"] > failing_result["selection_score"]


def test_compute_selection_result_lane_gated_can_require_end_to_end_head_on_win_rate():
    gated_cfg = {
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
    quality_failing = {
        "overall": {"win_rate": 0.75, "mean_return": -10.0},
        "by_lane": [
            {"task": "head_on", "opponent_stage": "expert", "win_rate": 1.0, "crash_rate": 0.0},
            {"task": "head_on", "opponent_stage": "end_to_end", "win_rate": 1.0 / 3.0, "crash_rate": 0.0},
        ],
    }
    quality_passing = {
        "overall": {"win_rate": 0.60, "mean_return": -20.0},
        "by_lane": [
            {"task": "head_on", "opponent_stage": "expert", "win_rate": 2.0 / 3.0, "crash_rate": 0.0},
            {"task": "head_on", "opponent_stage": "end_to_end", "win_rate": 2.0 / 3.0, "crash_rate": 1.0 / 3.0},
        ],
    }

    failing_result = compute_selection_result(quality_failing, gated_cfg)
    passing_result = compute_selection_result(quality_passing, gated_cfg)

    assert failing_result["selection_gate_passed"] is False
    assert failing_result["selection_gate_pass_count"] == 2
    assert failing_result["selection_gate_total"] == 3
    assert '"lane_id": "head_on::end_to_end"' in failing_result["selection_gate_failures"]
    assert '"metric": "win_rate"' in failing_result["selection_gate_failures"]

    assert passing_result["selection_gate_passed"] is True
    assert passing_result["selection_gate_pass_count"] == 3
    assert passing_result["selection_gate_total"] == 3
    assert passing_result["selection_score"] > failing_result["selection_score"]
