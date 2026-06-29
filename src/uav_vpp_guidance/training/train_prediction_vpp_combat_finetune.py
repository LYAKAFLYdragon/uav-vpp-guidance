"""Combat-only finetune entrypoint for the reset075 prediction VPP mainline.

This training loop keeps the current prediction-VPP policy architecture and
fine-tunes it on a narrow combat-only distribution:

- tasks: ``head_on`` and ``crossing_feasible``
- opponents: ``expert`` and ``end_to_end``
- backend: JSBSim

The implementation intentionally reuses the existing PPO agent and
CloseRangeTrackingEnv, but samples from a small pool of task/opponent lanes
instead of the generic intercept-style scenario dictionary used by the base
prediction-VPP trainer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.provenance import (
    get_config_overrides,
    record_config_override_if_changed,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.metrics.combat_evaluator import compute_combat_metrics
from uav_vpp_guidance.training.train_prediction_vpp_ppo import (
    load_experiment_config,
    load_warm_start_checkpoint,
)
from uav_vpp_guidance.trajectory_prediction._telemetry import (
    PredictorHealthAccumulator,
)
from uav_vpp_guidance.trajectory_prediction.config_validator import (
    validate_full_config,
)
from uav_vpp_guidance.utils.seed import set_seed


SCRIPT_NAME = "train_prediction_vpp_combat_finetune.py"
CRASH_REASONS = {"crash", "ego_crash_or_out_of_bounds", "out_of_bounds"}
TIMEOUT_REASONS = {
    "timeout",
    "timeout_draw",
    "timeout_hp_advantage",
    "timeout_hp_disadvantage",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (_repo_root() / path).resolve()


def _import_class(class_path: str):
    normalized = str(class_path)
    if normalized.startswith("src."):
        normalized = normalized[len("src.") :]
    module_name, class_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_opponent_entry(
    comparison_config: Dict[str, Any],
    opponent_stage: str,
) -> Dict[str, Any]:
    stage = str(opponent_stage or "none")
    if stage == "none":
        return {"stage": "none", "type": "none"}

    registry = comparison_config.get("opponent_registry", {})
    if stage not in registry:
        raise KeyError(f"Unknown opponent_stage: {stage}")

    entry = copy.deepcopy(registry[stage])
    entry["stage"] = stage
    if entry.get("type") == "curriculum":
        stages = list(entry.get("stages", []))
        if not stages:
            raise ValueError("curriculum opponent requires at least one stage")
        resolved_stage = entry.get("active_stage") or stages[-1]
        resolved = _resolve_opponent_entry(comparison_config, resolved_stage)
        resolved["stage"] = stage
        resolved["resolved_stage"] = resolved_stage
        resolved["curriculum_config"] = entry
        return resolved
    return entry


def resolve_combat_finetune_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve task/opponent lanes from the shared combat comparison catalog."""
    combat_cfg = config.get("combat_finetune", {})
    comparison_config_path = combat_cfg.get("comparison_config_path")
    if not comparison_config_path:
        raise ValueError(
            "combat_finetune.comparison_config_path is required for combat finetune"
        )

    comparison_path = _resolve_path(str(comparison_config_path))
    comparison_config = load_experiment_config(str(comparison_path))

    task_names = list(combat_cfg.get("tasks", []))
    opponent_stages = list(combat_cfg.get("opponent_stages", []))
    if not task_names:
        raise ValueError("combat_finetune.tasks must contain at least one task")
    if not opponent_stages:
        raise ValueError(
            "combat_finetune.opponent_stages must contain at least one stage"
        )

    task_weights = combat_cfg.get("task_weights", {}) or {}
    opponent_weights = combat_cfg.get("opponent_stage_weights", {}) or {}

    lanes = []
    task_registry = comparison_config.get("tasks", {})
    for task_name in task_names:
        if task_name not in task_registry:
            raise KeyError(f"Unknown combat_finetune task: {task_name}")

        task_entry = copy.deepcopy(task_registry[task_name])
        task_config = copy.deepcopy(task_entry.get("task", {}))
        if task_config.get("env_class") != "CloseRangeTrackingEnv":
            raise ValueError(
                "combat finetune currently only supports CloseRangeTrackingEnv "
                f"tasks, got {task_config.get('env_class')!r} for {task_name}"
            )

        scenarios = list(task_config.get("scenarios", []))
        if not scenarios:
            raise ValueError(f"Task {task_name} must define at least one scenario")
        scenario_names = [
            str(item.get("name", f"scenario_{idx}"))
            for idx, item in enumerate(scenarios)
        ]

        for opponent_stage in opponent_stages:
            opponent_entry = _resolve_opponent_entry(
                comparison_config,
                opponent_stage,
            )
            weight = float(task_weights.get(task_name, 1.0)) * float(
                opponent_weights.get(opponent_stage, 1.0)
            )
            if weight <= 0.0:
                raise ValueError(
                    f"Non-positive lane weight for {task_name}/{opponent_stage}: "
                    f"{weight}"
                )

            lanes.append(
                {
                    "lane_id": f"{task_name}::{opponent_stage}",
                    "task_name": task_name,
                    "task_label": task_entry.get("label", task_name),
                    "task_config": task_config,
                    "scenario_names": scenario_names,
                    "opponent_stage": opponent_stage,
                    "opponent_entry": opponent_entry,
                    "opponent_type": opponent_entry.get("type", "unknown"),
                    "weight": weight,
                }
            )

    return {
        "comparison_config_path": str(comparison_path),
        "tasks": task_names,
        "opponent_stages": opponent_stages,
        "lanes": lanes,
    }


def build_lane_training_config(
    base_config: Dict[str, Any],
    lane_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a per-lane environment config and record all YAML mutations."""
    config = copy.deepcopy(base_config)
    source = f"{SCRIPT_NAME}:lane[{lane_spec['lane_id']}]"

    task_config = copy.deepcopy(lane_spec["task_config"])
    task_config["name"] = lane_spec["task_name"]
    old_task = config.get("task")
    config["task"] = task_config
    record_config_override_if_changed(
        config,
        "task",
        task_config,
        old_value=old_task,
        source=source,
    )

    old_stage = config.get("opponent_stage")
    config["opponent_stage"] = lane_spec["opponent_stage"]
    record_config_override_if_changed(
        config,
        "opponent_stage",
        lane_spec["opponent_stage"],
        old_value=old_stage,
        source=source,
    )

    opponent_entry = copy.deepcopy(lane_spec["opponent_entry"])
    old_opponent = config.get("opponent")
    config["opponent"] = opponent_entry
    record_config_override_if_changed(
        config,
        "opponent",
        opponent_entry,
        old_value=old_opponent,
        source=source,
    )

    if "attack_zone" not in config:
        config["attack_zone"] = {}
    attack_zone_cfg = config["attack_zone"]
    old_enabled = attack_zone_cfg.get("enabled")
    attack_zone_cfg["enabled"] = True
    record_config_override_if_changed(
        config,
        "attack_zone.enabled",
        True,
        old_value=old_enabled,
        source=source,
    )

    close_range_max_aoa_deg = attack_zone_cfg.get("close_range_max_aoa_deg")
    if close_range_max_aoa_deg is None:
        close_range_max_aoa_deg = 60.0
        attack_zone_cfg["close_range_max_aoa_deg"] = close_range_max_aoa_deg
        record_config_override_if_changed(
            config,
            "attack_zone.close_range_max_aoa_deg",
            close_range_max_aoa_deg,
            old_value=None,
            source=source,
        )

    return config


def _build_opponent_policy(opponent_entry: Dict[str, Any]):
    entry = copy.deepcopy(opponent_entry)
    if not entry or entry.get("stage") == "none":
        return None

    class_path = entry.get("class")
    if not class_path:
        typ = entry.get("type")
        if typ == "rule_based":
            class_path = "uav_vpp_guidance.envs.expert_opponent.ExpertOpponent"
        elif typ == "neural":
            class_path = "uav_vpp_guidance.envs.end_to_end_opponent.EndToEndOpponent"
        else:
            raise ValueError(f"Opponent entry missing class: {entry}")

    checkpoint = entry.get("checkpoint")
    if checkpoint is not None:
        entry["checkpoint"] = str(_resolve_path(str(checkpoint)))

    cls = _import_class(class_path)
    kwargs = {}
    if "checkpoint" in entry:
        kwargs["checkpoint_path"] = entry["checkpoint"]
    if "config" in entry:
        kwargs["config"] = entry.get("config")
    if "device" in entry:
        kwargs["device"] = entry.get("device")
    if "invert_observation" in entry:
        kwargs["invert_observation"] = entry.get("invert_observation")
    return cls(**kwargs)


def _instantiate_lane_envs(plan: Dict[str, Any], base_config: Dict[str, Any]):
    lane_envs = []
    for lane_spec in plan["lanes"]:
        lane_config = build_lane_training_config(base_config, lane_spec)
        opponent_entry = copy.deepcopy(lane_spec["opponent_entry"])
        env = CloseRangeTrackingEnv(
            lane_config,
            opponent_policy=_build_opponent_policy(opponent_entry),
            opponent_config=opponent_entry,
        )
        scenario_pool = [
            {
                "name": str(item.get("name", f"scenario_{idx}")),
                "scenario": copy.deepcopy(item),
            }
            for idx, item in enumerate(lane_spec["task_config"].get("scenarios", []))
        ]
        lane_envs.append(
            {
                **copy.deepcopy(lane_spec),
                "config": lane_config,
                "scenario_pool": scenario_pool,
                "env": env,
            }
        )
    return lane_envs


def _sample_lane_reset(lane_envs, rng: np.random.Generator):
    weights = np.asarray([lane["weight"] for lane in lane_envs], dtype=np.float64)
    weights = weights / np.sum(weights)
    lane_idx = int(rng.choice(len(lane_envs), p=weights))
    lane = lane_envs[lane_idx]
    scenario_idx = int(rng.integers(0, len(lane["scenario_pool"])))
    scenario_entry = lane["scenario_pool"][scenario_idx]
    seed = int(rng.integers(0, 1_000_000))
    obs = lane["env"].reset(scenario=scenario_entry["scenario"], seed=seed)
    return lane, scenario_entry, seed, obs


def _summarize_episode_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    returns = [float(item["return"]) for item in rows]
    lengths = [float(item["length"]) for item in rows]
    final_ranges = [
        float(item["final_range_m"])
        for item in rows
        if np.isfinite(item.get("final_range_m", np.nan))
    ]
    final_atas = [
        float(item["final_ata_deg"])
        for item in rows
        if np.isfinite(item.get("final_ata_deg", np.nan))
    ]
    crash_rate = float(
        np.mean([item.get("is_crash", False) for item in rows])
    ) if rows else np.nan
    out_of_bounds_rate = float(
        np.mean([item.get("is_out_of_bounds", False) for item in rows])
    ) if rows else np.nan
    timeout_rate = float(
        np.mean([item.get("is_timeout", False) for item in rows])
    ) if rows else np.nan

    combat = compute_combat_metrics(rows)
    return {
        "num_episodes": len(rows),
        "mean_return": float(np.mean(returns)) if returns else np.nan,
        "std_return": float(np.std(returns)) if returns else np.nan,
        "mean_length": float(np.mean(lengths)) if lengths else np.nan,
        "crash_rate": crash_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "timeout_rate": timeout_rate,
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else np.nan,
        "mean_final_ata_deg": float(np.mean(final_atas)) if final_atas else np.nan,
        **combat,
    }


def _lane_id(task: str, opponent_stage: str) -> str:
    return f"{task}::{opponent_stage}"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _selection_metric_label(selection_cfg: Any) -> str:
    if isinstance(selection_cfg, str):
        return selection_cfg
    if not isinstance(selection_cfg, dict):
        return "win_rate"

    mode = str(selection_cfg.get("mode", "simple"))
    if mode == "lane_gated":
        objective = str(selection_cfg.get("objective", "overall.win_rate"))
        return f"lane_gated:{objective}"

    metric = selection_cfg.get("metric") or selection_cfg.get("objective") or "win_rate"
    return str(metric)


def _build_lane_summary_index(eval_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        _lane_id(str(row.get("task", "")), str(row.get("opponent_stage", ""))): row
        for row in eval_result.get("by_lane", [])
    }


def _resolve_selection_metric_value(
    eval_result: Dict[str, Any],
    metric_path: str,
) -> float:
    path = str(metric_path or "overall.win_rate")
    lane_lookup = _build_lane_summary_index(eval_result)

    if path.startswith("overall."):
        return _safe_float(eval_result.get("overall", {}).get(path.split(".", 1)[1]))

    if "::" in path and "." in path:
        lane_key, lane_metric = path.rsplit(".", 1)
        lane_row = lane_lookup.get(lane_key)
        if lane_row is None:
            return np.nan
        return _safe_float(lane_row.get(lane_metric))

    return _safe_float(eval_result.get("overall", {}).get(path))


def _resolve_gate_lane_id(gate_cfg: Dict[str, Any]) -> str:
    lane_id = gate_cfg.get("lane_id")
    if lane_id:
        return str(lane_id)

    task = gate_cfg.get("task")
    opponent_stage = gate_cfg.get("opponent_stage")
    if task is None or opponent_stage is None:
        raise ValueError(
            "lane_gated selection gates require lane_id or task+opponent_stage"
        )
    return _lane_id(str(task), str(opponent_stage))


def _evaluate_lane_gate(
    lane_lookup: Dict[str, Dict[str, Any]],
    gate_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    lane_key = _resolve_gate_lane_id(gate_cfg)
    metric_name = str(gate_cfg.get("metric", "win_rate"))
    lane_row = lane_lookup.get(lane_key)
    metric_value = _safe_float(
        lane_row.get(metric_name) if lane_row is not None else np.nan
    )

    passed = True
    violation = 0.0
    failure_reasons: List[str] = []

    if not np.isfinite(metric_value):
        passed = False
        violation = float(gate_cfg.get("missing_penalty", 1.0))
        failure_reasons.append("missing_metric")
    else:
        if "min_value" in gate_cfg:
            min_value = float(gate_cfg["min_value"])
            if metric_value < min_value:
                passed = False
                violation += min_value - metric_value
                failure_reasons.append("below_min")
        if "max_value" in gate_cfg:
            max_value = float(gate_cfg["max_value"])
            if metric_value > max_value:
                passed = False
                violation += metric_value - max_value
                failure_reasons.append("above_max")

    return {
        "lane_id": lane_key,
        "metric": metric_name,
        "metric_value": metric_value,
        "min_value": gate_cfg.get("min_value"),
        "max_value": gate_cfg.get("max_value"),
        "passed": passed,
        "violation": float(violation),
        "failure_reasons": failure_reasons,
    }


def compute_selection_result(
    eval_result: Dict[str, Any],
    selection_cfg: Any,
) -> Dict[str, Any]:
    label = _selection_metric_label(selection_cfg)

    if isinstance(selection_cfg, dict):
        mode = str(selection_cfg.get("mode", "simple"))
    else:
        mode = "simple"

    if mode == "lane_gated":
        objective_metric = str(selection_cfg.get("objective", "overall.win_rate"))
        fallback_metric = str(
            selection_cfg.get("fallback_metric", "overall.mean_return")
        )
        objective_value = _resolve_selection_metric_value(eval_result, objective_metric)
        fallback_value = _resolve_selection_metric_value(eval_result, fallback_metric)
        if not np.isfinite(objective_value):
            objective_value = fallback_value

        lane_lookup = _build_lane_summary_index(eval_result)
        gate_results = [
            _evaluate_lane_gate(lane_lookup, gate_cfg)
            for gate_cfg in list(selection_cfg.get("gates", []))
        ]
        gate_total = len(gate_results)
        gate_pass_count = sum(1 for gate in gate_results if gate["passed"])
        gate_violation_total = float(sum(gate["violation"] for gate in gate_results))
        gate_passed = gate_pass_count == gate_total if gate_total else True

        if gate_passed:
            gate_bonus = float(selection_cfg.get("gate_bonus", 1_000_000.0))
            selection_score = gate_bonus + float(objective_value)
        else:
            pass_fraction = (
                float(gate_pass_count) / float(gate_total) if gate_total else 0.0
            )
            selection_score = pass_fraction * 1000.0 - gate_violation_total

        if not np.isfinite(selection_score):
            selection_score = -float("inf")

        return {
            "selection_metric": label,
            "selection_score": float(selection_score),
            "selection_objective_metric": objective_metric,
            "selection_objective_value": float(objective_value)
            if np.isfinite(objective_value)
            else np.nan,
            "selection_gate_passed": bool(gate_passed),
            "selection_gate_pass_count": int(gate_pass_count),
            "selection_gate_total": int(gate_total),
            "selection_gate_violation_total": float(gate_violation_total),
            "selection_gate_failures": json.dumps(
                [gate for gate in gate_results if not gate["passed"]],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

    metric_path = str(selection_cfg or "win_rate")
    if "." not in metric_path:
        metric_path = f"overall.{metric_path}"
    objective_value = _resolve_selection_metric_value(eval_result, metric_path)
    if not np.isfinite(objective_value):
        objective_value = _resolve_selection_metric_value(
            eval_result,
            "overall.mean_return",
        )

    return {
        "selection_metric": label,
        "selection_score": float(objective_value)
        if np.isfinite(objective_value)
        else -float("inf"),
        "selection_objective_metric": metric_path,
        "selection_objective_value": float(objective_value)
        if np.isfinite(objective_value)
        else np.nan,
        "selection_gate_passed": True,
        "selection_gate_pass_count": 0,
        "selection_gate_total": 0,
        "selection_gate_violation_total": 0.0,
        "selection_gate_failures": "[]",
    }


def run_combat_evaluation(
    lane_envs,
    agent,
    eval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    seeds = list(eval_cfg.get("seeds", [0, 1, 2]))
    num_episodes = int(eval_cfg.get("eval_episodes", 1))
    lane_rows = []
    all_rows = []

    for lane in lane_envs:
        rows = []
        env = lane["env"]
        for seed in seeds:
            for ep in range(num_episodes):
                ep_seed = int(seed) * 10000 + ep
                rng = np.random.default_rng(
                    ep_seed + (abs(hash(lane["lane_id"])) % 100000)
                )
                scenario_entry = lane["scenario_pool"][
                    int(rng.integers(0, len(lane["scenario_pool"])))
                ]
                obs = env.reset(scenario=scenario_entry["scenario"], seed=ep_seed)

                ep_reward = 0.0
                ep_length = 0
                min_range = float("inf")
                final_range = np.nan
                final_ata = np.nan
                reason = "timeout"
                info = {}

                for _step in range(env.max_steps):
                    action = agent.get_deterministic_action(obs["observation_vector"])
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    ep_length += 1

                    rel_state = obs.get("relative_state", {})
                    range_m = float(rel_state.get("range_m", np.nan))
                    ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                    if np.isfinite(range_m):
                        min_range = min(min_range, range_m)
                        final_range = range_m
                    final_ata = ata_deg

                    if terminated or truncated:
                        reason = info.get("reason", "unknown")
                        break

                combat_reason = str(info.get("combat_reason", reason))
                row = {
                    "task": lane["task_name"],
                    "opponent_stage": lane["opponent_stage"],
                    "scenario": scenario_entry["name"],
                    "seed": int(seed),
                    "episode": ep,
                    "return": ep_reward,
                    "length": ep_length,
                    "min_range_m": min_range,
                    "final_range_m": final_range,
                    "final_ata_deg": final_ata,
                    "termination_reason": reason,
                    "combat_reason": combat_reason,
                    "combat_outcome": info.get("combat_outcome"),
                    "combat_success": bool(info.get("combat_success", False)),
                    "is_crash": combat_reason in CRASH_REASONS,
                    "is_out_of_bounds": combat_reason in {
                        "out_of_bounds",
                        "ego_crash_or_out_of_bounds",
                    },
                    "is_timeout": combat_reason in TIMEOUT_REASONS,
                    "ego_hp": info.get("ego_hp"),
                    "target_hp": info.get("target_hp"),
                    "combat_time_to_kill": info.get("combat_time_to_kill"),
                    "combat_initial_hp": info.get("combat_initial_hp", 100.0),
                }
                rows.append(row)
                all_rows.append(row)

        summary = _summarize_episode_rows(rows)
        summary["task"] = lane["task_name"]
        summary["opponent_stage"] = lane["opponent_stage"]
        lane_rows.append(summary)

    overall = _summarize_episode_rows(all_rows)
    return {
        "overall": overall,
        "by_lane": lane_rows,
        "episodes": all_rows,
    }


def train_combat_finetune(config: Dict[str, Any], output_dir: str, smoke: bool = False):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    import yaml

    config_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    plan = resolve_combat_finetune_plan(config)
    serializable_plan = copy.deepcopy(plan)
    for lane in serializable_plan["lanes"]:
        lane_config = build_lane_training_config(config, lane)
        lane["config_overrides"] = get_config_overrides(lane_config)
    plan_path = os.path.join(output_dir, "combat_finetune_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(serializable_plan, f, indent=2, ensure_ascii=False)

    train_lanes = _instantiate_lane_envs(plan, config)
    eval_lanes = _instantiate_lane_envs(plan, config)

    try:
        ppo_cfg = config.get("ppo", {})
        total_timesteps = int(ppo_cfg.get("total_timesteps", 200000))
        rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
        eval_interval = int(config.get("evaluation", {}).get("eval_interval", 10000))
        save_interval = int(config.get("checkpoint", {}).get("save_interval", 10000))
        save_best = bool(config.get("checkpoint", {}).get("save_best", True))
        save_last = bool(config.get("checkpoint", {}).get("save_last", True))

        sample_obs = None
        obs_dim = None
        for lane in train_lanes:
            seed = 0
            scenario = lane["scenario_pool"][0]["scenario"]
            obs = lane["env"].reset(scenario=scenario, seed=seed)
            current_dim = int(obs["observation_vector"].shape[0])
            if sample_obs is None:
                sample_obs = obs
                obs_dim = current_dim
            elif current_dim != obs_dim:
                raise ValueError(
                    "All combat finetune lanes must share the same observation dim, "
                    f"got {obs_dim} and {current_dim}"
                )

        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        device = ppo_cfg.get("device", "cpu")
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
        )
        print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
        print(f"Network parameters: {agent.network.count_parameters()}")

        resume_step = 0
        warm_start_cfg = config.get("warm_start", {})
        if warm_start_cfg.get("enabled", False):
            checkpoint_path = (
                warm_start_cfg.get("checkpoint")
                or warm_start_cfg.get("checkpoint_path")
            )
            if not checkpoint_path:
                raise ValueError(
                    "warm_start.enabled=true requires warm_start.checkpoint"
                )
            warm_start_meta = load_warm_start_checkpoint(
                agent,
                checkpoint_path=checkpoint_path,
                strict_dims=bool(warm_start_cfg.get("strict_dims", True)),
                load_optimizer=bool(warm_start_cfg.get("load_optimizer", False)),
            )
            resume_step = int(warm_start_cfg.get("step", 0))
            print(
                f"Warm-start loaded from {warm_start_meta['checkpoint_path']}: "
                f"obs_dim={warm_start_meta['checkpoint_obs_dim']}, "
                f"action_dim={warm_start_meta['checkpoint_action_dim']}, "
                f"resume_step={resume_step}, "
                f"optimizer_loaded={warm_start_meta['optimizer_loaded']}"
            )

        if smoke:
            total_timesteps = resume_step + 512 if resume_step > 0 else 512
            rollout_steps = 128
            eval_interval = 256
            save_interval = 256
            print("[SMOKE] Running combat finetune smoke mode")
            print(
                f"  total_timesteps={total_timesteps}, rollout_steps={rollout_steps}"
            )

        episode_log_path = os.path.join(log_dir, "episode_train_log.csv")
        update_log_path = os.path.join(log_dir, "update_train_log.csv")
        eval_log_path = os.path.join(log_dir, "eval_log.csv")
        eval_lane_log_path = os.path.join(log_dir, "eval_lane_log.csv")

        episode_fieldnames = [
            "step",
            "episode",
            "task",
            "opponent_stage",
            "scenario",
            "episode_seed",
            "episode_return",
            "episode_length",
            "combat_outcome",
            "combat_success",
            "crash",
            "out_of_bounds",
            "timeout",
            "mean_range",
            "final_range",
            "final_ata",
            "ego_hp",
            "target_hp",
            "prediction_valid_rate",
            "fallback_rate",
            "post_warmup_fallback_rate",
            "warmup_fallback_rate",
            "runtime_fallback_rate",
            "predictor_init_failed_count",
            "unknown_fallback_phase_count",
            "missing_fallback_phase_count",
            "configured_current_target_fallback_count",
            "mean_prediction_error_m",
            "median_prediction_error_m",
            "prediction_error_count",
        ]
        update_fieldnames = [
            "step",
            "update_num",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "learning_rate",
        ]
        eval_fieldnames = [
            "step",
            "scope",
            "num_episodes",
            "mean_return",
            "std_return",
            "mean_length",
            "win_rate",
            "survival_rate",
            "hp_advantage",
            "mean_time_to_kill",
            "damage_exchange_rate",
            "effective_engagement_rate",
            "damaging_win_rate",
            "mean_damage_dealt",
            "mean_damage_taken",
            "crash_rate",
            "out_of_bounds_rate",
            "timeout_rate",
            "mean_final_range_m",
            "mean_final_ata_deg",
            "selection_metric",
            "selection_score",
            "selection_objective_metric",
            "selection_objective_value",
            "selection_gate_passed",
            "selection_gate_pass_count",
            "selection_gate_total",
            "selection_gate_violation_total",
            "selection_gate_failures",
        ]
        eval_lane_fieldnames = ["step", "task", "opponent_stage"] + [
            field for field in eval_fieldnames if field not in {"step", "scope"}
        ]

        rng = np.random.default_rng(config.get("experiment", {}).get("seed", 0))
        current_lane, current_scenario, current_seed, obs = _sample_lane_reset(
            train_lanes,
            rng,
        )
        current_env = current_lane["env"]

        global_step = resume_step
        episode_count = 0
        best_score = -float("inf")
        selection_metric_cfg = config.get("combat_finetune", {}).get(
            "selection_metric",
            "win_rate",
        )
        selection_metric_label = _selection_metric_label(selection_metric_cfg)

        episode_return = 0.0
        episode_length = 0
        episode_ranges = []
        episode_health = PredictorHealthAccumulator()
        start_time = time.time()
        update_num = 0

        with open(episode_log_path, "w", newline="", encoding="utf-8") as f_ep:
            ep_writer = csv.DictWriter(f_ep, fieldnames=episode_fieldnames)
            ep_writer.writeheader()

            with open(update_log_path, "w", newline="", encoding="utf-8") as f_up:
                up_writer = csv.DictWriter(f_up, fieldnames=update_fieldnames)
                up_writer.writeheader()

                with open(eval_log_path, "w", newline="", encoding="utf-8") as f_eval:
                    eval_writer = csv.DictWriter(f_eval, fieldnames=eval_fieldnames)
                    eval_writer.writeheader()

                    with open(
                        eval_lane_log_path,
                        "w",
                        newline="",
                        encoding="utf-8",
                    ) as f_eval_lane:
                        eval_lane_writer = csv.DictWriter(
                            f_eval_lane,
                            fieldnames=eval_lane_fieldnames,
                        )
                        eval_lane_writer.writeheader()

                        while global_step < total_timesteps:
                            for _step in range(rollout_steps):
                                obs_vec = obs["observation_vector"]
                                action, log_prob, value = agent.select_action(
                                    obs_vec,
                                    deterministic=False,
                                    store=False,
                                )
                                agent.store_transition(
                                    obs_vec,
                                    action,
                                    log_prob,
                                    0.0,
                                    False,
                                    value,
                                )

                                obs, reward, terminated, truncated, info = current_env.step(
                                    action
                                )
                                global_step += 1
                                episode_return += reward
                                episode_length += 1

                                rel_state = obs.get("relative_state", {})
                                range_m = rel_state.get("range_m", np.nan)
                                if np.isfinite(range_m):
                                    episode_ranges.append(float(range_m))
                                episode_health.step(info)

                                agent.buffer.rewards[agent.buffer.ptr - 1] = float(
                                    reward
                                )
                                agent.buffer.dones[agent.buffer.ptr - 1] = float(
                                    terminated or truncated
                                )

                                if terminated or truncated:
                                    episode_count += 1
                                    combat_reason = str(
                                        info.get("combat_reason", info.get("reason", ""))
                                    )
                                    health_rates = episode_health.rates(episode_length)
                                    ep_writer.writerow(
                                        {
                                            "step": global_step,
                                            "episode": episode_count,
                                            "task": current_lane["task_name"],
                                            "opponent_stage": current_lane[
                                                "opponent_stage"
                                            ],
                                            "scenario": current_scenario["name"],
                                            "episode_seed": current_seed,
                                            "episode_return": episode_return,
                                            "episode_length": episode_length,
                                            "combat_outcome": info.get("combat_outcome"),
                                            "combat_success": int(
                                                bool(info.get("combat_success", False))
                                            ),
                                            "crash": int(combat_reason in CRASH_REASONS),
                                            "out_of_bounds": int(
                                                combat_reason
                                                in {
                                                    "out_of_bounds",
                                                    "ego_crash_or_out_of_bounds",
                                                }
                                            ),
                                            "timeout": int(
                                                combat_reason in TIMEOUT_REASONS
                                            ),
                                            "mean_range": (
                                                float(np.mean(episode_ranges))
                                                if episode_ranges
                                                else np.nan
                                            ),
                                            "final_range": (
                                                episode_ranges[-1]
                                                if episode_ranges
                                                else np.nan
                                            ),
                                            "final_ata": float(
                                                np.rad2deg(
                                                    rel_state.get("ata_rad", 0.0)
                                                )
                                            ),
                                            "ego_hp": info.get("ego_hp"),
                                            "target_hp": info.get("target_hp"),
                                            "prediction_valid_rate": round(
                                                health_rates["prediction_valid_rate"],
                                                4,
                                            ),
                                            "fallback_rate": round(
                                                health_rates["fallback_rate"],
                                                4,
                                            ),
                                            "post_warmup_fallback_rate": round(
                                                health_rates[
                                                    "post_warmup_fallback_rate"
                                                ],
                                                4,
                                            ),
                                            "warmup_fallback_rate": round(
                                                health_rates["warmup_fallback_rate"],
                                                4,
                                            ),
                                            "runtime_fallback_rate": round(
                                                health_rates["runtime_fallback_rate"],
                                                4,
                                            ),
                                            "predictor_init_failed_count": health_rates[
                                                "predictor_init_failed_count"
                                            ],
                                            "unknown_fallback_phase_count": health_rates[
                                                "unknown_fallback_phase_count"
                                            ],
                                            "missing_fallback_phase_count": health_rates[
                                                "missing_fallback_phase_count"
                                            ],
                                            "configured_current_target_fallback_count": (
                                                health_rates[
                                                    "configured_current_target_fallback_count"
                                                ]
                                            ),
                                            "mean_prediction_error_m": (
                                                round(
                                                    health_rates[
                                                        "mean_prediction_error_m"
                                                    ],
                                                    4,
                                                )
                                                if np.isfinite(
                                                    health_rates[
                                                        "mean_prediction_error_m"
                                                    ]
                                                )
                                                else np.nan
                                            ),
                                            "median_prediction_error_m": (
                                                round(
                                                    health_rates[
                                                        "median_prediction_error_m"
                                                    ],
                                                    4,
                                                )
                                                if np.isfinite(
                                                    health_rates[
                                                        "median_prediction_error_m"
                                                    ]
                                                )
                                                else np.nan
                                            ),
                                            "prediction_error_count": health_rates[
                                                "prediction_error_count"
                                            ],
                                        }
                                    )
                                    f_ep.flush()

                                    episode_return = 0.0
                                    episode_length = 0
                                    episode_ranges = []
                                    episode_health.reset()

                                    (
                                        current_lane,
                                        current_scenario,
                                        current_seed,
                                        obs,
                                    ) = _sample_lane_reset(train_lanes, rng)
                                    current_env = current_lane["env"]

                                    if agent.buffer.full:
                                        break

                                if global_step >= total_timesteps:
                                    break

                            if agent.buffer.full or (
                                global_step >= total_timesteps and len(agent.buffer) > 0
                            ):
                                next_obs_vec = obs["observation_vector"]
                                update_stats = agent.update(next_obs=next_obs_vec)
                                update_num += 1
                                up_writer.writerow(
                                    {
                                        "step": global_step,
                                        "update_num": update_num,
                                        "policy_loss": update_stats.get(
                                            "policy_loss", ""
                                        ),
                                        "value_loss": update_stats.get(
                                            "value_loss", ""
                                        ),
                                        "entropy": update_stats.get("entropy", ""),
                                        "approx_kl": update_stats.get("approx_kl", ""),
                                        "clip_fraction": update_stats.get(
                                            "clip_fraction", ""
                                        ),
                                        "explained_variance": update_stats.get(
                                            "explained_variance", ""
                                        ),
                                        "learning_rate": update_stats.get(
                                            "learning_rate", ""
                                        ),
                                    }
                                )
                                f_up.flush()

                            if (
                                eval_interval > 0
                                and global_step % eval_interval == 0
                                and global_step > 0
                            ):
                                print(f"\n--- Combat evaluation at step {global_step} ---")
                                eval_result = run_combat_evaluation(
                                    eval_lanes,
                                    agent,
                                    config.get("evaluation", {}),
                                )
                                overall = eval_result["overall"]
                                selection_result = compute_selection_result(
                                    eval_result,
                                    selection_metric_cfg,
                                )
                                selection_score = float(
                                    selection_result["selection_score"]
                                )

                                eval_writer.writerow(
                                    {
                                        "step": global_step,
                                        "scope": "overall",
                                        "num_episodes": overall.get("num_episodes"),
                                        "mean_return": overall.get("mean_return"),
                                        "std_return": overall.get("std_return"),
                                        "mean_length": overall.get("mean_length"),
                                        "win_rate": overall.get("win_rate"),
                                        "survival_rate": overall.get(
                                            "survival_rate"
                                        ),
                                        "hp_advantage": overall.get("hp_advantage"),
                                        "mean_time_to_kill": overall.get(
                                            "mean_time_to_kill"
                                        ),
                                        "damage_exchange_rate": overall.get(
                                            "damage_exchange_rate"
                                        ),
                                        "effective_engagement_rate": overall.get(
                                            "effective_engagement_rate"
                                        ),
                                        "damaging_win_rate": overall.get(
                                            "damaging_win_rate"
                                        ),
                                        "mean_damage_dealt": overall.get(
                                            "mean_damage_dealt"
                                        ),
                                        "mean_damage_taken": overall.get(
                                            "mean_damage_taken"
                                        ),
                                        "crash_rate": overall.get("crash_rate"),
                                        "out_of_bounds_rate": overall.get(
                                            "out_of_bounds_rate"
                                        ),
                                        "timeout_rate": overall.get("timeout_rate"),
                                        "mean_final_range_m": overall.get(
                                            "mean_final_range_m"
                                        ),
                                        "mean_final_ata_deg": overall.get(
                                            "mean_final_ata_deg"
                                        ),
                                        "selection_metric": selection_result.get(
                                            "selection_metric",
                                            selection_metric_label,
                                        ),
                                        "selection_score": selection_score,
                                        "selection_objective_metric": (
                                            selection_result.get(
                                                "selection_objective_metric"
                                            )
                                        ),
                                        "selection_objective_value": (
                                            selection_result.get(
                                                "selection_objective_value"
                                            )
                                        ),
                                        "selection_gate_passed": (
                                            selection_result.get(
                                                "selection_gate_passed"
                                            )
                                        ),
                                        "selection_gate_pass_count": (
                                            selection_result.get(
                                                "selection_gate_pass_count"
                                            )
                                        ),
                                        "selection_gate_total": (
                                            selection_result.get(
                                                "selection_gate_total"
                                            )
                                        ),
                                        "selection_gate_violation_total": (
                                            selection_result.get(
                                                "selection_gate_violation_total"
                                            )
                                        ),
                                        "selection_gate_failures": (
                                            selection_result.get(
                                                "selection_gate_failures"
                                            )
                                        ),
                                    }
                                )
                                f_eval.flush()

                                for lane_row in eval_result["by_lane"]:
                                    payload = {
                                        "step": global_step,
                                        "task": lane_row["task"],
                                        "opponent_stage": lane_row[
                                            "opponent_stage"
                                        ],
                                        "num_episodes": lane_row.get("num_episodes"),
                                        "mean_return": lane_row.get("mean_return"),
                                        "std_return": lane_row.get("std_return"),
                                        "mean_length": lane_row.get("mean_length"),
                                        "win_rate": lane_row.get("win_rate"),
                                        "survival_rate": lane_row.get(
                                            "survival_rate"
                                        ),
                                        "hp_advantage": lane_row.get(
                                            "hp_advantage"
                                        ),
                                        "mean_time_to_kill": lane_row.get(
                                            "mean_time_to_kill"
                                        ),
                                        "damage_exchange_rate": lane_row.get(
                                            "damage_exchange_rate"
                                        ),
                                        "effective_engagement_rate": lane_row.get(
                                            "effective_engagement_rate"
                                        ),
                                        "damaging_win_rate": lane_row.get(
                                            "damaging_win_rate"
                                        ),
                                        "mean_damage_dealt": lane_row.get(
                                            "mean_damage_dealt"
                                        ),
                                        "mean_damage_taken": lane_row.get(
                                            "mean_damage_taken"
                                        ),
                                        "crash_rate": lane_row.get("crash_rate"),
                                        "out_of_bounds_rate": lane_row.get(
                                            "out_of_bounds_rate"
                                        ),
                                        "timeout_rate": lane_row.get("timeout_rate"),
                                        "mean_final_range_m": lane_row.get(
                                            "mean_final_range_m"
                                        ),
                                        "mean_final_ata_deg": lane_row.get(
                                            "mean_final_ata_deg"
                                        ),
                                        "selection_metric": selection_result.get(
                                            "selection_metric",
                                            selection_metric_label,
                                        ),
                                        "selection_score": selection_score,
                                        "selection_objective_metric": (
                                            selection_result.get(
                                                "selection_objective_metric"
                                            )
                                        ),
                                        "selection_objective_value": (
                                            selection_result.get(
                                                "selection_objective_value"
                                            )
                                        ),
                                        "selection_gate_passed": (
                                            selection_result.get(
                                                "selection_gate_passed"
                                            )
                                        ),
                                        "selection_gate_pass_count": (
                                            selection_result.get(
                                                "selection_gate_pass_count"
                                            )
                                        ),
                                        "selection_gate_total": (
                                            selection_result.get(
                                                "selection_gate_total"
                                            )
                                        ),
                                        "selection_gate_violation_total": (
                                            selection_result.get(
                                                "selection_gate_violation_total"
                                            )
                                        ),
                                        "selection_gate_failures": (
                                            selection_result.get(
                                                "selection_gate_failures"
                                            )
                                        ),
                                    }
                                    eval_lane_writer.writerow(payload)
                                f_eval_lane.flush()

                                print(
                                    f"Eval win_rate={overall.get('win_rate', np.nan):.3f} | "
                                    f"survival={overall.get('survival_rate', np.nan):.3f} | "
                                    f"crash={overall.get('crash_rate', np.nan):.3f} | "
                                    f"selection={selection_result.get('selection_metric', selection_metric_label)} "
                                    f"score={selection_score:.3f}"
                                )

                                if save_best and selection_score > best_score:
                                    best_score = selection_score
                                    best_path = os.path.join(
                                        checkpoint_dir,
                                        "best.pt",
                                    )
                                    agent.save(best_path)
                                    print(
                                        f"  -> Saved best checkpoint "
                                        f"({selection_result.get('selection_metric', selection_metric_label)}="
                                        f"{best_score:.4f})"
                                    )

                            if (
                                save_interval > 0
                                and global_step % save_interval == 0
                                and global_step > 0
                            ):
                                step_path = os.path.join(
                                    checkpoint_dir,
                                    f"step_{global_step}.pt",
                                )
                                agent.save(step_path)

                        if save_last:
                            last_path = os.path.join(checkpoint_dir, "last.pt")
                            agent.save(last_path)
                            print(f"\nSaved last checkpoint to {last_path}")

        elapsed = time.time() - start_time
        print(
            f"\nCombat finetune complete! Total steps: {global_step}, "
            f"Episodes: {episode_count}, Time: {elapsed:.1f}s"
        )
    finally:
        for lane in train_lanes:
            lane["env"].close()
        for lane in eval_lanes:
            lane["env"].close()

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Combat-only finetune for prediction VPP PPO"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to combat finetune config YAML",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (minimal finetune)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute device (default: from config).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override simulation backend (default: from config).",
    )
    parser.add_argument(
        "--use-jsbsim",
        action="store_true",
        help="Force use_jsbsim=True (equivalent to --backend jsbsim).",
    )
    parser.add_argument(
        "--comparison-config",
        type=str,
        default=None,
        help="Override combat_finetune.comparison_config_path",
    )
    parser.add_argument(
        "--combat-tasks",
        nargs="+",
        default=None,
        help="Override combat_finetune.tasks",
    )
    parser.add_argument(
        "--combat-opponents",
        nargs="+",
        default=None,
        help="Override combat_finetune.opponent_stages",
    )
    parser.add_argument(
        "--attack-zone-close-range-max-aoa-deg",
        type=float,
        default=None,
        help="Override attack_zone.close_range_max_aoa_deg",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Warm-start checkpoint override",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="Logical starting step for warm-started runs",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also load optimizer state from --resume",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    backend = args.backend
    if args.use_jsbsim:
        backend = "jsbsim"
    if backend is not None:
        old_backend = config.get("backend")
        config["backend"] = backend
        record_config_override_if_changed(
            config,
            "backend",
            backend,
            old_value=old_backend,
            source=f"{SCRIPT_NAME}:--backend",
        )
        if "env" not in config:
            config["env"] = {}
        old_env_backend = config["env"].get("backend")
        old_use_jsbsim = config["env"].get("use_jsbsim")
        config["env"]["backend"] = backend
        config["env"]["use_jsbsim"] = backend == "jsbsim"
        record_config_override_if_changed(
            config,
            "env.backend",
            backend,
            old_value=old_env_backend,
            source=f"{SCRIPT_NAME}:--backend",
        )
        record_config_override_if_changed(
            config,
            "env.use_jsbsim",
            backend == "jsbsim",
            old_value=old_use_jsbsim,
            source=f"{SCRIPT_NAME}:--backend",
        )

    if args.seed is not None:
        config.setdefault("experiment", {})
        old_seed = config["experiment"].get("seed")
        config["experiment"]["seed"] = int(args.seed)
        record_config_override_if_changed(
            config,
            "experiment.seed",
            int(args.seed),
            old_value=old_seed,
            source=f"{SCRIPT_NAME}:--seed",
        )

    if args.device is not None:
        if "ppo" not in config:
            config["ppo"] = {}
        old_device = config["ppo"].get("device")
        config["ppo"]["device"] = args.device
        record_config_override_if_changed(
            config,
            "ppo.device",
            args.device,
            old_value=old_device,
            source=f"{SCRIPT_NAME}:--device",
        )

    combat_cfg = config.setdefault("combat_finetune", {})
    if args.comparison_config is not None:
        old_value = combat_cfg.get("comparison_config_path")
        combat_cfg["comparison_config_path"] = args.comparison_config
        record_config_override_if_changed(
            config,
            "combat_finetune.comparison_config_path",
            args.comparison_config,
            old_value=old_value,
            source=f"{SCRIPT_NAME}:--comparison-config",
        )

    if args.combat_tasks is not None:
        old_value = combat_cfg.get("tasks")
        combat_cfg["tasks"] = list(args.combat_tasks)
        record_config_override_if_changed(
            config,
            "combat_finetune.tasks",
            combat_cfg["tasks"],
            old_value=old_value,
            source=f"{SCRIPT_NAME}:--combat-tasks",
        )

    if args.combat_opponents is not None:
        old_value = combat_cfg.get("opponent_stages")
        combat_cfg["opponent_stages"] = list(args.combat_opponents)
        record_config_override_if_changed(
            config,
            "combat_finetune.opponent_stages",
            combat_cfg["opponent_stages"],
            old_value=old_value,
            source=f"{SCRIPT_NAME}:--combat-opponents",
        )

    if args.attack_zone_close_range_max_aoa_deg is not None:
        if "attack_zone" not in config:
            config["attack_zone"] = {}
        old_value = config["attack_zone"].get("close_range_max_aoa_deg")
        config["attack_zone"]["close_range_max_aoa_deg"] = float(
            args.attack_zone_close_range_max_aoa_deg
        )
        record_config_override_if_changed(
            config,
            "attack_zone.close_range_max_aoa_deg",
            config["attack_zone"]["close_range_max_aoa_deg"],
            old_value=old_value,
            source=f"{SCRIPT_NAME}:--attack-zone-close-range-max-aoa-deg",
        )

    if args.resume is not None:
        old_warm_start = config.get("warm_start")
        config["warm_start"] = {
            "enabled": True,
            "checkpoint": args.resume,
            "step": int(args.resume_step),
            "load_optimizer": bool(args.resume_optimizer),
            "strict_dims": True,
        }
        record_config_override_if_changed(
            config,
            "warm_start",
            config["warm_start"],
            old_value=old_warm_start,
            source=f"{SCRIPT_NAME}:--resume",
        )

    seed = int(config.get("experiment", {}).get("seed", 0))
    set_seed(seed)

    exp_name = config.get("experiment", {}).get(
        "name",
        "prediction_vpp_combat_finetune",
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "experiments",
            exp_name,
        )
    os.makedirs(output_dir, exist_ok=True)

    on_unknown = "warn" if args.smoke else "raise"
    try:
        validate_full_config(config, on_unknown=on_unknown)
    except ValueError as exc:
        print(f"ERROR: Full config validation failed: {exc}")
        sys.exit(1)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {seed}")
    print(
        f"Combat finetune lanes: tasks={combat_cfg.get('tasks')} | "
        f"opponents={combat_cfg.get('opponent_stages')}"
    )

    train_combat_finetune(config, output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
