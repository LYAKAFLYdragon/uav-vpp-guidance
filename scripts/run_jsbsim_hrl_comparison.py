#!/usr/bin/env python3
"""Run the minimal JSBSim HRL/VPP comparison matrix.

This runner is intentionally research-facing rather than controller-facing:

* main preset: prediction VPP vs no-prediction VPP vs end-to-end RL
* ablation preset: perfect vs learned vs no vs noisy prediction
* frozen task set: head-on, feasible crossing, break-turn, sustained-turn

It writes a portable manifest, artifact contract, resolved config snapshot, raw
episode JSONs, and a compact summary CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.agents.end_to_end_ppo_agent import EndToEndPPOAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.artifact_contract import ArtifactContract
from uav_vpp_guidance.common.hash import config_sha256
from uav_vpp_guidance.common.manifest import RunManifest
from uav_vpp_guidance.common.provenance import (
    get_config_overrides,
    record_config_override_if_changed,
)
from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder, RunRecorder
from uav_vpp_guidance.metrics.combat_evaluator import compute_combat_metrics
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


SCRIPT_NAME = Path(__file__).name
STAGE_NAME = "jsbsim_hrl_comparison"
STAGE_VERSION = "0.1.0"

TASK_METRIC_POLICY = {
    "head_on": {
        "primary_success_metric": "success_rate",
        "success_semantics": "Default proximity success from CloseRangeTrackingEnv.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "time_in_envelope_s"],
    },
    "crossing_feasible": {
        "primary_success_metric": "success_rate",
        "success_semantics": "Default proximity success from CloseRangeTrackingEnv.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "time_in_envelope_s"],
    },
    "break_turn": {
        "primary_success_metric": None,
        "success_semantics": "BreakTurnEnv disables proximity success; success_rate is not a valid primary metric.",
        "tracking_metrics": ["final_range_m", "mean_range_m", "min_range_m", "time_in_envelope_s"],
    },
    "sustained_turn": {
        "primary_success_metric": None,
        "success_semantics": "SustainedTurnEnv disables proximity success; success_rate is not a valid primary metric.",
        "tracking_metrics": ["completed_orbits", "mean_range_m", "time_in_envelope_s", "mean_abs_ata_deg"],
    },
}

PREDICTION_VARIANT_SEMANTICS = {
    "none": "No trajectory prediction; VPP anchors on the current target state.",
    "learned": "Frozen learned predictor supplies the predicted target anchor for VPP; prediction features are not added to policy observation by default.",
    "noisy": "Learned prediction anchor with seeded Gaussian position noise injected before VPP generation.",
    "perfect": "Constant-velocity kinematic oracle using the current true target velocity; this is not a true future-trajectory oracle for maneuvering targets.",
}

DESIGN_NOTES = {
    "main_claim_scope": "Prediction assists VPP anchor placement. Current main preset does not add prediction features to the high-level policy observation.",
    "prediction_features_in_observation": False,
    "task_metric_policy": TASK_METRIC_POLICY,
    "prediction_variant_semantics": PREDICTION_VARIANT_SEMANTICS,
    "formal_policy": "formal-small runs are pilot evidence; paper-safe formal evidence requires a separately chosen full budget after pilot analysis.",
    "curriculum_opponent_eval_policy": "When --opponent-stage=curriculum, this runner evaluates a single resolved opponent. The YAML active_stage selects it; if omitted, the final listed stage is used.",
    "survival_rate_semantics": "survival_rate means the ego aircraft was not physically destroyed, shot down, crashed, or out-of-bounds; HP-timeout losses can still count as survived.",
}


def _load_config_with_includes(config_path: Path) -> Dict[str, Any]:
    config = load_yaml_config(str(config_path))
    includes = config.pop("includes", [])
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (config_path.parent / inc).resolve()
        if not inc_path.exists():
            inc_path = (config_path.parent / ".." / Path(inc).name).resolve()
        if not inc_path.exists():
            raise FileNotFoundError(f"Included config not found: {inc}")
        merged = merge_config(merged, _load_config_with_includes(inc_path))
    return merge_config(merged, config)


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(target)
    for key, value in source.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get_path(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = copy.deepcopy(value)


def _record_set(
    config: Dict[str, Any],
    dotted_key: str,
    value: Any,
    source: str,
) -> None:
    old_value = _get_path(config, dotted_key)
    _set_path(config, dotted_key, value)
    record_config_override_if_changed(
        config,
        key=dotted_key,
        new_value=value,
        old_value=old_value,
        source=source,
    )


def _record_block(
    config: Dict[str, Any],
    dotted_key: str,
    value: Dict[str, Any],
    source: str,
) -> None:
    old_value = _get_path(config, dotted_key)
    merged = _deep_update(old_value if isinstance(old_value, dict) else {}, value)
    _set_path(config, dotted_key, merged)
    record_config_override_if_changed(
        config,
        key=dotted_key,
        new_value=merged,
        old_value=old_value,
        source=source,
    )


def _repo_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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


def _apply_opponent_stage(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    opponent_stage: str,
) -> Dict[str, Any]:
    source = f"{SCRIPT_NAME}:opponent_stage"
    opponent_entry = _resolve_opponent_entry(comparison_config, opponent_stage)
    _record_set(config, "opponent_stage", opponent_stage, source)
    _record_set(config, "opponent", opponent_entry, source)
    if opponent_stage != "none":
        _record_set(config, "attack_zone.enabled", True, source)
    return opponent_entry


def _build_opponent_policy(config: Dict[str, Any]):
    opponent_entry = copy.deepcopy(config.get("opponent", {}))
    if not opponent_entry or opponent_entry.get("stage") == "none":
        return None

    class_path = opponent_entry.get("class")
    if not class_path:
        typ = opponent_entry.get("type")
        if typ == "rule_based":
            class_path = "uav_vpp_guidance.envs.expert_opponent.ExpertOpponent"
        elif typ == "neural":
            class_path = "uav_vpp_guidance.envs.end_to_end_opponent.EndToEndOpponent"
        else:
            raise ValueError(f"Opponent entry missing class: {opponent_entry}")

    cls = _import_class(class_path)
    checkpoint = opponent_entry.get("checkpoint")
    if checkpoint is not None:
        checkpoint_path = _repo_path(str(checkpoint))
        opponent_entry["checkpoint"] = str(checkpoint_path)
    else:
        checkpoint_path = None

    kwargs = {}
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = str(checkpoint_path)
    if "config" in opponent_entry:
        kwargs["config"] = opponent_entry.get("config")
    if "device" in opponent_entry:
        kwargs["device"] = opponent_entry.get("device")
    if "invert_observation" in opponent_entry:
        kwargs["invert_observation"] = opponent_entry.get("invert_observation")
    return cls(**kwargs)


def _load_checkpoint_config(
    checkpoint_path: Optional[Path],
    fallback_config_path: Optional[Path],
) -> Dict[str, Any]:
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        ckpt_config = checkpoint.get("config", {})
        if ckpt_config:
            return copy.deepcopy(ckpt_config)

    if fallback_config_path is not None and fallback_config_path.exists():
        return _load_config_with_includes(fallback_config_path)

    missing = checkpoint_path if checkpoint_path is not None else fallback_config_path
    raise FileNotFoundError(f"No checkpoint config or fallback config found: {missing}")


def _build_env(config: Dict[str, Any]):
    opponent_policy = _build_opponent_policy(config)
    opponent_config = copy.deepcopy(config.get("opponent", {}))
    class_name = config.get("task", {}).get("env_class", "CloseRangeTrackingEnv")
    if class_name == "SustainedTurnEnv":
        from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv

        return SustainedTurnEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
    if class_name == "BreakTurnEnv":
        from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv

        return BreakTurnEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
    if class_name == "MultiWaypointTrackingEnv":
        from uav_vpp_guidance.envs.multi_waypoint_tracking_env import (
            MultiWaypointTrackingEnv,
        )

        return MultiWaypointTrackingEnv(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )

    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

    return CloseRangeTrackingEnv(
        config,
        opponent_policy=opponent_policy,
        opponent_config=opponent_config,
    )


def _apply_backend(config: Dict[str, Any], backend: str, source: str) -> None:
    backend = str(backend).lower()
    use_jsbsim = backend == "jsbsim"
    _record_set(config, "backend", backend, source)
    _record_set(config, "env.backend", backend, source)
    _record_set(config, "env.use_jsbsim", use_jsbsim, source)
    _record_set(config, "env.strict_backend", use_jsbsim, source)


def _apply_task_and_common_overrides(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    task_name: str,
    task_def: Dict[str, Any],
    backend: str,
    max_steps_override: Optional[int],
    jsbsim_root: Optional[str],
) -> None:
    source = f"{SCRIPT_NAME}:{task_name}_eval_config"
    common_env = comparison_config.get("env", {})
    if common_env:
        _record_block(config, "env", common_env, source)
    if task_def.get("env"):
        _record_block(config, "env", task_def["env"], source)
    if comparison_config.get("low_level_controller"):
        _record_set(
            config,
            "low_level_controller",
            comparison_config["low_level_controller"],
            source,
        )
    if comparison_config.get("guidance"):
        _record_block(config, "guidance", comparison_config["guidance"], source)
    for block_name in ("limits", "reward", "observation", "attack_zone"):
        if comparison_config.get(block_name):
            _record_block(config, block_name, comparison_config[block_name], source)
        if task_def.get(block_name):
            _record_block(config, block_name, task_def[block_name], source)
    _record_set(config, "task", task_def["task"], source)
    if max_steps_override is not None:
        _record_set(config, "env.max_high_level_steps", int(max_steps_override), source)
    if jsbsim_root and str(backend).lower() == "jsbsim":
        _record_set(config, "env.legacy_project_root", jsbsim_root, source)
    _apply_backend(config, backend, source)


def _apply_prediction_variant(
    config: Dict[str, Any],
    comparison_config: Dict[str, Any],
    method_name: str,
    method_def: Dict[str, Any],
) -> None:
    source = f"{SCRIPT_NAME}:{method_name}_{method_def.get('prediction_variant', 'none')}"
    variant = str(method_def.get("prediction_variant", "none")).lower()
    pred_defaults = comparison_config.get("prediction", {})
    predictor_checkpoint = method_def.get(
        "predictor_checkpoint", pred_defaults.get("learned_predictor_checkpoint")
    )
    lookahead_time_s = float(pred_defaults.get("lookahead_time_s", 1.0))

    if method_def.get("agent_type") == "end_to_end":
        _record_set(config, "end_to_end.enabled", True, source)
        _record_set(config, "virtual_point.enabled", False, source)
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "policy.action_dim", 3, source)
        return

    _record_set(config, "end_to_end.enabled", False, source)
    _record_set(config, "virtual_point.enabled", True, source)
    _record_set(config, "virtual_point.mode", "normal", source)

    if variant == "none":
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "virtual_point.anchor_mode", "current_target", source)
        _record_set(
            config,
            "trajectory_prediction.integration.prediction_noise_std_m",
            0.0,
            source,
        )
        return

    if variant == "perfect":
        _record_set(config, "trajectory_prediction.enabled", False, source)
        _record_set(config, "virtual_point.anchor_mode", "oracle_future_position", source)
        _record_set(
            config,
            "trajectory_prediction.prediction.lookahead_time_s",
            lookahead_time_s,
            source,
        )
        _record_set(
            config,
            "trajectory_prediction.integration.prediction_noise_std_m",
            0.0,
            source,
        )
        return

    if variant not in {"learned", "noisy"}:
        raise ValueError(f"Unknown prediction_variant for {method_name}: {variant}")

    _record_set(config, "trajectory_prediction.enabled", True, source)
    _record_set(config, "trajectory_prediction.predictor_type", "lstm", source)
    _record_set(config, "trajectory_prediction.checkpoint_path", predictor_checkpoint, source)
    _record_set(config, "trajectory_prediction.strict_predictor_init", True, source)
    _record_set(config, "trajectory_prediction.device", "cpu", source)
    _record_set(config, "trajectory_prediction.freeze_predictor_during_rl", True, source)
    _record_set(
        config,
        "trajectory_prediction.prediction.lookahead_time_s",
        lookahead_time_s,
        source,
    )
    _record_set(
        config,
        "trajectory_prediction.prediction.output_mode",
        "relative_displacement",
        source,
    )
    _record_set(
        config,
        "trajectory_prediction.prediction.fallback_mode",
        "constant_velocity",
        source,
    )
    _record_set(config, "trajectory_prediction.integration.anchor_mode", "predicted_target", source)
    _record_set(
        config,
        "trajectory_prediction.integration.add_prediction_to_observation",
        False,
        source,
    )
    _record_set(config, "virtual_point.anchor_mode", "predicted_target", source)

    noise_std_m = 0.0
    if variant == "noisy":
        noise_std_m = float(pred_defaults.get("noisy_prediction_std_m", 100.0))
    _record_set(
        config,
        "trajectory_prediction.integration.prediction_noise_std_m",
        noise_std_m,
        source,
    )


def build_eval_config(
    comparison_config: Dict[str, Any],
    method_name: str,
    task_name: str,
    backend: str,
    opponent_stage: str = "none",
    max_steps_override: Optional[int] = None,
    jsbsim_root: Optional[str] = None,
) -> Dict[str, Any]:
    methods = comparison_config.get("methods", {})
    tasks = comparison_config.get("tasks", {})
    if method_name not in methods:
        raise KeyError(f"Unknown method: {method_name}")
    if task_name not in tasks:
        raise KeyError(f"Unknown task: {task_name}")

    method_def = methods[method_name]
    checkpoint_path = _repo_path(method_def.get("checkpoint"))
    fallback_config_path = _repo_path(method_def.get("config_path"))
    config = _load_checkpoint_config(checkpoint_path, fallback_config_path)
    _apply_task_and_common_overrides(
        config,
        comparison_config,
        task_name,
        tasks[task_name],
        backend=backend,
        max_steps_override=max_steps_override,
        jsbsim_root=jsbsim_root,
    )
    _apply_prediction_variant(config, comparison_config, method_name, method_def)
    _apply_opponent_stage(config, comparison_config, opponent_stage)
    return config


def _build_agent(
    method_def: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_path: Path,
    sample_obs: Dict[str, Any],
    device: str,
):
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    agent_type = method_def.get("agent_type", "ppo")
    if agent_type == "end_to_end":
        agent = EndToEndPPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    else:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(str(checkpoint_path))
    return agent


def _select_methods(config: Dict[str, Any], preset: str, methods: Optional[List[str]]) -> List[str]:
    if methods:
        return methods
    defaults = config.get("run_defaults", {})
    if preset == "ablation":
        return list(defaults.get("ablation_methods", []))
    return list(defaults.get("main_methods", []))


def _select_tasks(config: Dict[str, Any], tasks: Optional[List[str]]) -> List[str]:
    if tasks:
        return tasks
    return list(config.get("run_defaults", {}).get("tasks", []))


def _select_seeds(config: Dict[str, Any], args: argparse.Namespace) -> List[int]:
    if args.seeds is not None:
        return args.seeds
    defaults = config.get("run_defaults", {})
    if args.run_status == "smoke":
        return list(defaults.get("smoke_seeds", [0]))
    return list(defaults.get("formal_small_seeds", [0, 1, 2]))


def _task_scenarios(task_def: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    scenarios = task_def.get("task", {}).get("scenarios")
    if not scenarios:
        return [None]
    return [copy.deepcopy(s) for s in scenarios]


def _scenario_name(scenario: Optional[Dict[str, Any]], task_name: str) -> str:
    if scenario is None:
        return task_name
    return scenario.get("name", task_name)


def _finite_mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _finite_min(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.min(clean)) if clean else float("nan")


def _finite_max(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.max(clean)) if clean else float("nan")


def _finite_median(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else float("nan")


def _last_finite(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return clean[-1] if clean else float("nan")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _checkpoint_dimension_info(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    ckpt_config = checkpoint.get("config", {})
    return {
        "checkpoint_obs_dim": checkpoint.get("obs_dim"),
        "checkpoint_action_dim": checkpoint.get("action_dim"),
        "checkpoint_policy_action_dim": ckpt_config.get("policy", {}).get("action_dim")
        if isinstance(ckpt_config, dict)
        else None,
    }


def _build_observation_audit(
    method_name: str,
    task_name: str,
    method_def: Dict[str, Any],
    config: Dict[str, Any],
    checkpoint_path: Path,
    sample_obs: Dict[str, Any],
) -> Dict[str, Any]:
    obs_vec = sample_obs["observation_vector"]
    schema = copy.deepcopy(sample_obs.get("observation_schema", {}))
    dim_info = _checkpoint_dimension_info(checkpoint_path)
    obs_dim = int(obs_vec.shape[0])
    schema_dim = schema.get("dim")
    policy_action_dim = int(config.get("policy", {}).get("action_dim", 3))
    ckpt_obs_dim = dim_info.get("checkpoint_obs_dim")
    ckpt_action_dim = dim_info.get("checkpoint_action_dim")
    ckpt_policy_action_dim = dim_info.get("checkpoint_policy_action_dim")

    return {
        "method": method_name,
        "task": task_name,
        "agent_type": method_def.get("agent_type", "ppo"),
        "prediction_variant": method_def.get("prediction_variant", "none"),
        "sample_observation_dim": obs_dim,
        "schema_dim": schema_dim,
        "schema_dim_matches_vector": schema_dim in (None, obs_dim),
        "feature_count": len(schema.get("feature_names", [])),
        "checkpoint_obs_dim": ckpt_obs_dim,
        "obs_dim_matches_checkpoint": ckpt_obs_dim in (None, obs_dim),
        "policy_action_dim": policy_action_dim,
        "checkpoint_action_dim": ckpt_action_dim,
        "checkpoint_policy_action_dim": ckpt_policy_action_dim,
        "action_dim_matches_checkpoint": ckpt_action_dim in (None, policy_action_dim),
        "observation_schema": schema,
    }


def _run_episode(
    env,
    agent,
    method_name: str,
    method_def: Dict[str, Any],
    task_name: str,
    scenario: Optional[Dict[str, Any]],
    seed: int,
    episode: int,
    run_id: str,
    config: Dict[str, Any],
    git_commit: str,
    save_full: bool,
) -> Dict[str, Any]:
    config_hash = config_sha256(config)
    backend = config.get("backend", "jsbsim")
    strict_backend = bool(config.get("env", {}).get("strict_backend", backend == "jsbsim"))
    recorder = EpisodeRecorder(
        run_id=run_id,
        task=task_name,
        controller=method_name,
        seed=seed,
        episode=episode,
        config=config,
        config_sha256=config_hash,
        git_commit=git_commit,
        backend=backend,
        strict_backend=strict_backend,
        save_full=save_full,
    )

    reset_kwargs: Dict[str, Any] = {"seed": seed}
    if scenario is not None:
        reset_kwargs["scenario"] = scenario
    obs = env.reset(**reset_kwargs)
    reset_provenance = copy.deepcopy(obs.get("provenance", {}))
    observation_schema = copy.deepcopy(obs.get("observation_schema", {}))

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    dt = float(env.env_config.get("high_level_dt", 0.2))
    prediction_valid_steps = 0
    prediction_fallback_steps = 0
    prediction_errors: List[float] = []
    ranges_m: List[float] = []
    abs_atas_deg: List[float] = []
    speeds_mps: List[float] = []
    altitudes_m: List[float] = []
    saturation_flags: List[float] = []
    completed_orbits_values: List[float] = []
    combat_time_to_kill = float("nan")
    envelope_steps = 0
    success_range_m = float(config.get("env", {}).get("success_range_m", 900.0))
    success_ata_deg = float(config.get("env", {}).get("success_ata_deg", 25.0))
    backend_fallback_occurred = False

    while not (terminated or truncated):
        obs_vec = obs["observation_vector"]
        action = agent.get_deterministic_action(obs_vec)
        if method_def.get("agent_type") == "end_to_end" and hasattr(agent, "clip_action"):
            action = agent.clip_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        own_state = info.get("own_state", {})
        target_state = info.get("target_state", {})
        recorder.record_step(steps, steps * dt, own_state, target_state, info, float(reward))

        range_m = _safe_float(info.get("range_m"))
        abs_ata_deg = abs(_safe_float(info.get("ata_deg")))
        speed_mps = _safe_float(own_state.get("speed_mps"))
        altitude_m = _safe_float(own_state.get("altitude_m"))
        completed_orbits = _safe_float(info.get("completed_orbits"))
        if np.isfinite(range_m):
            ranges_m.append(range_m)
        if np.isfinite(abs_ata_deg):
            abs_atas_deg.append(abs_ata_deg)
        if np.isfinite(speed_mps):
            speeds_mps.append(speed_mps)
        if np.isfinite(altitude_m):
            altitudes_m.append(altitude_m)
        if np.isfinite(completed_orbits):
            completed_orbits_values.append(completed_orbits)
        saturation_flags.append(1.0 if bool(info.get("saturation_flag", False)) else 0.0)
        if (
            np.isfinite(range_m)
            and np.isfinite(abs_ata_deg)
            and range_m <= success_range_m
            and abs_ata_deg <= success_ata_deg
        ):
            envelope_steps += 1

        if bool(info.get("prediction_valid", False)):
            prediction_valid_steps += 1
        if bool(info.get("prediction_fallback", False)):
            prediction_fallback_steps += 1
        pred_err = info.get("prediction_error_m")
        if pred_err is not None and np.isfinite(pred_err):
            prediction_errors.append(float(pred_err))
        ttk = info.get("combat_time_to_kill")
        if np.isfinite(_safe_float(ttk)) and not np.isfinite(combat_time_to_kill):
            combat_time_to_kill = float(ttk)
        backend_fallback_occurred = backend_fallback_occurred or bool(
            info.get("backend_fallback_occurred", False)
        )

        if steps >= env.max_steps:
            truncated = True
            break

    if not info:
        own_state, target_state = env._get_current_states()
        info = {
            "own_state": own_state,
            "target_state": target_state,
            "reason": "timeout",
            "is_success": False,
        }

    own_state = info.get("own_state", {})
    record = recorder.finalize(
        steps=steps,
        total_time_s=steps * dt,
        total_reward=total_reward,
        termination_reason=info.get("reason", "timeout"),
        success=bool(info.get("is_success", False)),
        final_position_m=own_state.get("position_m", own_state.get("position_neu")),
        final_speed_mps=float(own_state.get("speed_mps", 250.0)),
        final_altitude_m=float(own_state.get("altitude_m", 5000.0)),
    )
    denom = max(1, steps)
    record.update(
        {
            "method_label": method_def.get("label", method_name),
            "agent_type": method_def.get("agent_type", "ppo"),
            "prediction_variant": method_def.get("prediction_variant", "none"),
            "opponent_stage": config.get("opponent_stage", "none"),
            "opponent_config": copy.deepcopy(config.get("opponent", {})),
            "scenario": _scenario_name(scenario, task_name),
            "scenario_metadata": copy.deepcopy(scenario.get("metadata", {})) if scenario else {},
            "observation_schema": observation_schema,
            "reset_provenance": reset_provenance,
            "config_overrides": get_config_overrides(config),
            "prediction_valid_rate": prediction_valid_steps / denom,
            "prediction_fallback_rate": prediction_fallback_steps / denom,
            "mean_prediction_error_m": _finite_mean(prediction_errors),
            "final_range_m": _last_finite(ranges_m),
            "min_range_m": _finite_min(ranges_m),
            "mean_range_m": _finite_mean(ranges_m),
            "median_range_m": _finite_median(ranges_m),
            "final_abs_ata_deg": _last_finite(abs_atas_deg),
            "mean_abs_ata_deg": _finite_mean(abs_atas_deg),
            "max_abs_ata_deg": _finite_max(abs_atas_deg),
            "time_in_envelope_s": envelope_steps * dt,
            "envelope_fraction": envelope_steps / denom,
            "final_speed_mps": _last_finite(speeds_mps),
            "mean_speed_mps": _finite_mean(speeds_mps),
            "final_altitude_m": _last_finite(altitudes_m),
            "mean_altitude_m": _finite_mean(altitudes_m),
            "saturation_fraction": _finite_mean(saturation_flags),
            "completed_orbits": _last_finite(completed_orbits_values),
            "backend_fallback_occurred": backend_fallback_occurred,
            "backend_fallback_reason": info.get("backend_fallback_reason"),
            "combat_outcome": info.get("combat_outcome"),
            "combat_success": bool(info.get("combat_success", False)),
            "win": bool(info.get("win", False)),
            "loss": bool(info.get("loss", False)),
            "draw": bool(info.get("draw", False)),
            "ego_hp": _safe_float(info.get("ego_hp"), 100.0),
            "target_hp": _safe_float(info.get("target_hp"), 100.0),
            "hp_advantage": _safe_float(info.get("hp_advantage")),
            "combat_reason": info.get("combat_reason"),
            "combat_time_to_kill": combat_time_to_kill,
            "time_to_kill": combat_time_to_kill,
            "survived": not bool(
                info.get("combat_outcome") == "loss"
                and info.get("combat_reason") in {"ego_killed", "ego_crash_or_out_of_bounds"}
            ),
            "_deprecated_tracking_mean_range_m": _finite_mean(ranges_m),
            "_deprecated_tracking_mean_abs_ata_deg": _finite_mean(abs_atas_deg),
            "_deprecated_tracking_envelope_fraction": envelope_steps / denom,
            "final_info": {
                "anchor_mode": info.get("anchor_mode"),
                "guidance_mode": info.get("guidance_mode"),
                "prediction_enabled": info.get("prediction_enabled"),
                "prediction_valid": info.get("prediction_valid"),
                "prediction_fallback": info.get("prediction_fallback"),
                "prediction_noise_applied": info.get("prediction_noise_applied"),
                "prediction_noise_std_m": info.get("prediction_noise_std_m"),
            },
        }
    )
    return record


def _write_summary_csv(output_dir: Path, records: List[Dict[str, Any]]) -> Path:
    summary_path = output_dir / "summary.csv"
    fields = [
        "method",
        "method_label",
        "agent_type",
        "prediction_variant",
        "opponent_stage",
        "task",
        "scenario",
        "seed",
        "episode",
        "backend",
        "strict_backend",
        "success",
        "combat_outcome",
        "combat_success",
        "win",
        "loss",
        "draw",
        "termination_reason",
        "steps",
        "total_time_s",
        "total_reward",
        "prediction_valid_rate",
        "prediction_fallback_rate",
        "mean_prediction_error_m",
        "final_range_m",
        "min_range_m",
        "mean_range_m",
        "median_range_m",
        "final_abs_ata_deg",
        "mean_abs_ata_deg",
        "max_abs_ata_deg",
        "time_in_envelope_s",
        "envelope_fraction",
        "final_speed_mps",
        "mean_speed_mps",
        "final_altitude_m",
        "mean_altitude_m",
        "saturation_fraction",
        "completed_orbits",
        "ego_hp",
        "target_hp",
        "hp_advantage",
        "combat_time_to_kill",
        "survived",
        "_deprecated_tracking_mean_range_m",
        "_deprecated_tracking_mean_abs_ata_deg",
        "_deprecated_tracking_envelope_fraction",
        "observation_dim",
        "checkpoint_obs_dim",
        "obs_dim_matches_checkpoint",
        "backend_fallback_occurred",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {
                "method": rec.get("controller"),
                "method_label": rec.get("method_label"),
                "agent_type": rec.get("agent_type"),
                "prediction_variant": rec.get("prediction_variant"),
                "opponent_stage": rec.get("opponent_stage"),
                "task": rec.get("task"),
                "scenario": rec.get("scenario"),
                "seed": rec.get("seed"),
                "episode": rec.get("episode"),
                "backend": rec.get("backend"),
                "strict_backend": rec.get("strict_backend"),
                "success": rec.get("success"),
                "combat_outcome": rec.get("combat_outcome"),
                "combat_success": rec.get("combat_success"),
                "win": rec.get("win"),
                "loss": rec.get("loss"),
                "draw": rec.get("draw"),
                "termination_reason": rec.get("termination_reason"),
                "steps": rec.get("steps"),
                "total_time_s": rec.get("total_time_s"),
                "total_reward": rec.get("total_reward"),
                "prediction_valid_rate": rec.get("prediction_valid_rate"),
                "prediction_fallback_rate": rec.get("prediction_fallback_rate"),
                "mean_prediction_error_m": rec.get("mean_prediction_error_m"),
                "final_range_m": rec.get("final_range_m"),
                "min_range_m": rec.get("min_range_m"),
                "mean_range_m": rec.get("mean_range_m"),
                "median_range_m": rec.get("median_range_m"),
                "final_abs_ata_deg": rec.get("final_abs_ata_deg"),
                "mean_abs_ata_deg": rec.get("mean_abs_ata_deg"),
                "max_abs_ata_deg": rec.get("max_abs_ata_deg"),
                "time_in_envelope_s": rec.get("time_in_envelope_s"),
                "envelope_fraction": rec.get("envelope_fraction"),
                "final_speed_mps": rec.get("final_speed_mps"),
                "mean_speed_mps": rec.get("mean_speed_mps"),
                "final_altitude_m": rec.get("final_altitude_m"),
                "mean_altitude_m": rec.get("mean_altitude_m"),
                "saturation_fraction": rec.get("saturation_fraction"),
                "completed_orbits": rec.get("completed_orbits"),
                "ego_hp": rec.get("ego_hp"),
                "target_hp": rec.get("target_hp"),
                "hp_advantage": rec.get("hp_advantage"),
                "combat_time_to_kill": rec.get("combat_time_to_kill"),
                "survived": rec.get("survived"),
                "_deprecated_tracking_mean_range_m": rec.get("_deprecated_tracking_mean_range_m"),
                "_deprecated_tracking_mean_abs_ata_deg": rec.get("_deprecated_tracking_mean_abs_ata_deg"),
                "_deprecated_tracking_envelope_fraction": rec.get("_deprecated_tracking_envelope_fraction"),
                "observation_dim": rec.get("observation_dim"),
                "checkpoint_obs_dim": rec.get("checkpoint_obs_dim"),
                "obs_dim_matches_checkpoint": rec.get("obs_dim_matches_checkpoint"),
                "backend_fallback_occurred": rec.get("backend_fallback_occurred"),
            }
            writer.writerow(row)
    return summary_path


def _write_aggregate_summary(output_dir: Path, records: List[Dict[str, Any]]) -> Path:
    aggregate: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in records:
        opponent_stage = rec.get("opponent_stage", "none")
        key = (rec["controller"], rec["task"], opponent_stage)
        item = aggregate.setdefault(
            key,
            {
                "method": rec["controller"],
                "task": rec["task"],
                "opponent_stage": opponent_stage,
                "episodes": 0,
                "records": [],
                "successes": 0,
                "total_rewards": [],
                "final_ranges_m": [],
                "min_ranges_m": [],
                "mean_ranges_m": [],
                "mean_abs_atas_deg": [],
                "time_in_envelope_s": [],
                "envelope_fractions": [],
                "saturation_fractions": [],
                "completed_orbits": [],
                "prediction_valid_rates": [],
                "prediction_fallback_rates": [],
                "backend_fallbacks": 0,
            },
        )
        item["episodes"] += 1
        item["records"].append(rec)
        item["successes"] += int(bool(rec.get("success", False)))
        item["total_rewards"].append(float(rec.get("total_reward", 0.0)))
        for field, target in (
            ("final_range_m", "final_ranges_m"),
            ("min_range_m", "min_ranges_m"),
            ("mean_range_m", "mean_ranges_m"),
            ("mean_abs_ata_deg", "mean_abs_atas_deg"),
            ("time_in_envelope_s", "time_in_envelope_s"),
            ("envelope_fraction", "envelope_fractions"),
            ("saturation_fraction", "saturation_fractions"),
            ("completed_orbits", "completed_orbits"),
        ):
            item[target].append(_safe_float(rec.get(field)))
        item["prediction_valid_rates"].append(float(rec.get("prediction_valid_rate", 0.0)))
        item["prediction_fallback_rates"].append(float(rec.get("prediction_fallback_rate", 0.0)))
        item["backend_fallbacks"] += int(bool(rec.get("backend_fallback_occurred", False)))

    rows = []
    for item in aggregate.values():
        episodes = max(1, int(item["episodes"]))
        combat_metrics = compute_combat_metrics(item["records"])
        rows.append(
            {
                "method": item["method"],
                "task": item["task"],
                "opponent_stage": item["opponent_stage"],
                "episodes": item["episodes"],
                "success_rate": item["successes"] / episodes,
                "combat_success_rate": combat_metrics["combat_success_rate"],
                "win_rate": combat_metrics["win_rate"],
                "survival_rate": combat_metrics["survival_rate"],
                "hp_advantage": combat_metrics["hp_advantage"],
                "mean_time_to_kill": combat_metrics["mean_time_to_kill"],
                "damage_exchange_rate": combat_metrics["damage_exchange_rate"],
                "effective_engagement_rate": combat_metrics["effective_engagement_rate"],
                "damaging_win_rate": combat_metrics["damaging_win_rate"],
                "mean_damage_dealt": combat_metrics["mean_damage_dealt"],
                "mean_damage_taken": combat_metrics["mean_damage_taken"],
                "mean_total_reward": float(np.mean(item["total_rewards"])),
                "mean_final_range_m": _finite_mean(item["final_ranges_m"]),
                "mean_min_range_m": _finite_mean(item["min_ranges_m"]),
                "mean_episode_mean_range_m": _finite_mean(item["mean_ranges_m"]),
                "mean_abs_ata_deg": _finite_mean(item["mean_abs_atas_deg"]),
                "mean_time_in_envelope_s": _finite_mean(item["time_in_envelope_s"]),
                "mean_envelope_fraction": _finite_mean(item["envelope_fractions"]),
                "mean_saturation_fraction": _finite_mean(item["saturation_fractions"]),
                "mean_completed_orbits": _finite_mean(item["completed_orbits"]),
                "mean_prediction_valid_rate": float(np.mean(item["prediction_valid_rates"])),
                "mean_prediction_fallback_rate": float(np.mean(item["prediction_fallback_rates"])),
                "backend_fallbacks": item["backend_fallbacks"],
            }
        )

    path = output_dir / "aggregate" / "method_task_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    return path


def _write_config_snapshots(
    output_dir: Path,
    configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshots: Dict[str, Dict[str, Any]] = {}
    for key, cfg in sorted(configs.items()):
        safe_key = key.replace("/", "__")
        path = manifest_dir / f"config_{safe_key}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        snapshots[key] = {
            "path": str(path),
            "sha256": config_sha256(cfg),
            "config_overrides": get_config_overrides(cfg),
        }
    return snapshots


def _checkpoint_checks(
    comparison_config: Dict[str, Any],
    methods: List[str],
    opponent_stage: str,
    allow_missing: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    checks = []
    errors = []
    for method in methods:
        method_def = comparison_config.get("methods", {}).get(method, {})
        paths = [("policy_checkpoint", method_def.get("checkpoint"))]
        variant = method_def.get("prediction_variant")
        if variant in ("learned", "noisy"):
            pred_path = method_def.get(
                "predictor_checkpoint",
                comparison_config.get("prediction", {}).get("learned_predictor_checkpoint"),
            )
            paths.append(("predictor_checkpoint", pred_path))

        for role, raw_path in paths:
            path = _repo_path(raw_path)
            exists = bool(path and path.exists())
            checks.append(
                {
                    "method": method,
                    "role": role,
                    "path": str(path) if path else None,
                    "exists": exists,
                }
            )
            if not exists and not allow_missing:
                errors.append(f"{method}:{role} missing: {path}")
    opponent_entry = _resolve_opponent_entry(comparison_config, opponent_stage)
    opponent_checkpoint = opponent_entry.get("checkpoint")
    if opponent_checkpoint:
        path = _repo_path(opponent_checkpoint)
        exists = bool(path and path.exists())
        checks.append(
            {
                "method": "__opponent__",
                "role": f"opponent_{opponent_stage}_checkpoint",
                "path": str(path) if path else None,
                "exists": exists,
            }
        )
        if not exists and not allow_missing:
            errors.append(f"opponent:{opponent_stage}:checkpoint missing: {path}")
    return checks, errors


def _contract_for_run(dry_run: bool) -> ArtifactContract:
    if dry_run:
        return ArtifactContract(
            required_files=[
                "resolved_config.yaml",
                "dry_run_checks.json",
                "design_notes.json",
                "artifact_contract.json",
                "run_manifest.json",
            ],
            required_directories=["manifests"],
        )
    return ArtifactContract(
        required_files=[
            "resolved_config.yaml",
            "summary.csv",
            "aggregate/episode_records.json",
            "aggregate/method_task_summary.json",
            "design_notes.json",
            "observation_audit.json",
            "artifact_contract.json",
            "run_manifest.json",
        ],
        required_directories=["raw", "manifests"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal JSBSim HRL/VPP comparison runner")
    parser.add_argument("--config", default="config/experiment/jsbsim_hrl_comparison.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/jsbsim_hrl_comparison")
    parser.add_argument("--preset", choices=["main", "ablation"], default="main")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--backend", choices=["jsbsim", "simple"], default=None)
    parser.add_argument(
        "--opponent-stage",
        type=str,
        default="none",
        choices=["none", "expert", "end_to_end", "curriculum"],
    )
    parser.add_argument(
        "--jsbsim-root",
        default=None,
        help="Legacy project root containing envs/JSBSim/data; recorded as env.legacy_project_root.",
    )
    parser.add_argument("--run-status", choices=["smoke", "formal-small", "formal"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help="Only for dry-run planning on machines without trained artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = _repo_path(args.config)
    if config_path is None:
        raise RuntimeError("Config path is required")
    comparison_config = _load_config_with_includes(config_path)

    methods = _select_methods(comparison_config, args.preset, args.methods)
    tasks = _select_tasks(comparison_config, args.tasks)
    seeds = _select_seeds(comparison_config, args)
    n_episodes = int(args.n_episodes or comparison_config.get("run_defaults", {}).get("n_episodes", 1))
    backend = args.backend or comparison_config.get("run_defaults", {}).get("backend", "jsbsim")
    opponent_stage = args.opponent_stage or comparison_config.get("opponent_stage", "none")
    jsbsim_root = args.jsbsim_root
    if jsbsim_root is None and str(backend).lower() == "jsbsim":
        jsbsim_root = os.environ.get("JSBSIM_ROOT") or str(REPO_ROOT)
    max_steps_override = None
    if args.run_status == "smoke":
        max_steps_override = comparison_config.get("run_defaults", {}).get("smoke_max_high_level_steps")

    run_dir = _repo_path(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = copy.deepcopy(comparison_config)
    resolved_config["selection"] = {
        "preset": args.preset,
        "methods": methods,
        "tasks": tasks,
        "seeds": seeds,
        "n_episodes": n_episodes,
        "backend": backend,
        "opponent_stage": opponent_stage,
        "jsbsim_root": jsbsim_root,
        "run_status": args.run_status,
        "dry_run": args.dry_run,
    }
    resolved_config_path = run_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            resolved_config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    design_notes = copy.deepcopy(DESIGN_NOTES)
    design_notes["selected_methods"] = methods
    design_notes["selected_tasks"] = tasks
    design_notes["selected_backend"] = backend
    design_notes["selected_opponent_stage"] = opponent_stage
    design_notes["resolved_jsbsim_root"] = jsbsim_root
    design_notes_path = run_dir / "design_notes.json"
    with open(design_notes_path, "w", encoding="utf-8") as f:
        json.dump(design_notes, f, indent=2, ensure_ascii=False)

    manifest = RunManifest(
        stage_name=STAGE_NAME,
        stage_version=STAGE_VERSION,
        command_line=sys.argv,
        output_dir=str(run_dir),
        config_path=str(config_path),
        resolved_config=resolved_config,
    )
    manifest.mark_started()
    manifest.compute_config_hash()
    manifest.record_input_file("comparison_config", config_path)
    manifest.record_output_file("resolved_config.yaml", resolved_config_path)
    manifest.record_output_file("design_notes.json", design_notes_path)
    manifest.extra.update(
        {
            "run_id": args.run_id,
            "run_status": args.run_status,
            "preset": args.preset,
            "backend": backend,
            "opponent_stage": opponent_stage,
            "jsbsim_root": jsbsim_root,
            "methods": methods,
            "tasks": tasks,
            "seeds": seeds,
            "n_episodes": n_episodes,
            "design_notes": design_notes,
        }
    )

    contract = _contract_for_run(args.dry_run)
    contract_path = contract.save(run_dir)
    manifest.record_output_file("artifact_contract.json", contract_path)

    config_snapshots: Dict[str, Dict[str, Any]] = {}
    eval_configs: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    checkpoint_checks, checkpoint_errors = _checkpoint_checks(
        comparison_config,
        methods,
        opponent_stage=opponent_stage,
        allow_missing=args.allow_missing_checkpoints and args.dry_run,
    )
    if checkpoint_errors:
        failures.extend({"phase": "checkpoint_check", "error": err} for err in checkpoint_errors)

    try:
        for method in methods:
            for task in tasks:
                cfg = build_eval_config(
                    comparison_config,
                    method,
                    task,
                    backend=backend,
                    opponent_stage=opponent_stage,
                    max_steps_override=max_steps_override,
                    jsbsim_root=jsbsim_root,
                )
                eval_configs[f"{method}/{task}"] = cfg
        config_snapshots = _write_config_snapshots(run_dir, eval_configs)
        manifest.extra["config_snapshots"] = config_snapshots
    except Exception as exc:
        failures.append(
            {
                "phase": "config_build",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    if args.dry_run:
        dry_run_path = run_dir / "dry_run_checks.json"
        with open(dry_run_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint_checks": checkpoint_checks,
                    "failures": failures,
                    "config_snapshots": config_snapshots,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        manifest.record_output_file("dry_run_checks.json", dry_run_path)
    else:
        all_records: List[Dict[str, Any]] = []
        observation_audits: Dict[str, Dict[str, Any]] = {}
        git_commit = manifest.git_info.get("short_commit") or "unknown"
        start = time.time()
        for method in methods:
            method_def = comparison_config["methods"][method]
            checkpoint_path = _repo_path(method_def.get("checkpoint"))
            if checkpoint_path is None or not checkpoint_path.exists():
                failures.append(
                    {
                        "phase": "evaluation",
                        "method": method,
                        "error": f"Checkpoint not found: {checkpoint_path}",
                    }
                )
                continue
            for task in tasks:
                task_def = comparison_config["tasks"][task]
                cfg = eval_configs.get(f"{method}/{task}")
                if cfg is None:
                    continue
                scenarios = _task_scenarios(task_def)
                try:
                    env = _build_env(cfg)
                    first_scenario = scenarios[0]
                    sample_kwargs: Dict[str, Any] = {"seed": seeds[0] if seeds else 0}
                    if first_scenario is not None:
                        sample_kwargs["scenario"] = first_scenario
                    sample_obs = env.reset(**sample_kwargs)
                    audit_key = f"{method}/{task}"
                    observation_audit = _build_observation_audit(
                        method_name=method,
                        task_name=task,
                        method_def=method_def,
                        config=cfg,
                        checkpoint_path=checkpoint_path,
                        sample_obs=sample_obs,
                    )
                    observation_audits[audit_key] = observation_audit
                    if not observation_audit["schema_dim_matches_vector"]:
                        raise RuntimeError(
                            f"Observation schema dim mismatch for {audit_key}: "
                            f"schema={observation_audit['schema_dim']} "
                            f"vector={observation_audit['sample_observation_dim']}"
                        )
                    if observation_audit["obs_dim_matches_checkpoint"] is False:
                        raise RuntimeError(
                            f"Checkpoint observation dim mismatch for {audit_key}: "
                            f"checkpoint={observation_audit['checkpoint_obs_dim']} "
                            f"vector={observation_audit['sample_observation_dim']}"
                        )
                    if observation_audit["action_dim_matches_checkpoint"] is False:
                        raise RuntimeError(
                            f"Checkpoint action dim mismatch for {audit_key}: "
                            f"checkpoint={observation_audit['checkpoint_action_dim']} "
                            f"policy={observation_audit['policy_action_dim']}"
                        )
                    agent = _build_agent(method_def, cfg, checkpoint_path, sample_obs, args.device)
                    for seed in seeds:
                        for scenario_idx, scenario in enumerate(scenarios):
                            for ep in range(n_episodes):
                                episode_id = scenario_idx * n_episodes + ep
                                ep_seed = seed * 100000 + scenario_idx * 100 + ep
                                rec = _run_episode(
                                    env=env,
                                    agent=agent,
                                    method_name=method,
                                    method_def=method_def,
                                    task_name=task,
                                    scenario=scenario,
                                    seed=ep_seed,
                                    episode=episode_id,
                                    run_id=args.run_id,
                                    config=cfg,
                                    git_commit=git_commit,
                                    save_full=not args.no_trajectory,
                                )
                                rec["observation_audit"] = copy.deepcopy(observation_audit)
                                rec["observation_dim"] = observation_audit["sample_observation_dim"]
                                rec["checkpoint_obs_dim"] = observation_audit["checkpoint_obs_dim"]
                                rec["obs_dim_matches_checkpoint"] = observation_audit["obs_dim_matches_checkpoint"]
                                all_records.append(rec)
                    env.close()
                except Exception as exc:
                    failures.append(
                        {
                            "phase": "evaluation",
                            "method": method,
                            "task": task,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

        recorder = RunRecorder(run_dir)
        for rec in all_records:
            recorder.add(rec)
        recorder.write()
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        summary_path = _write_summary_csv(run_dir, all_records)
        aggregate_path = _write_aggregate_summary(run_dir, all_records)
        observation_audit_path = run_dir / "observation_audit.json"
        with open(observation_audit_path, "w", encoding="utf-8") as f:
            json.dump({"audits": observation_audits}, f, indent=2, ensure_ascii=False)
        failures_path = run_dir / "failures.json"
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump({"failures": failures}, f, indent=2, ensure_ascii=False)

        manifest.record_output_file("summary.csv", summary_path)
        manifest.record_output_file("aggregate/episode_records.json", recorder.aggregate_dir / "episode_records.json")
        manifest.record_output_file("aggregate/method_task_summary.json", aggregate_path)
        manifest.record_output_file("observation_audit.json", observation_audit_path)
        manifest.record_output_file("failures.json", failures_path)
        manifest.artifacts_present["raw"] = (run_dir / "raw").is_dir()
        manifest.artifacts_present["manifests"] = (run_dir / "manifests").is_dir()
        manifest.extra["elapsed_seconds"] = time.time() - start
        manifest.extra["total_episodes"] = len(all_records)
        manifest.extra["observation_audits"] = observation_audits

    if args.run_status != "formal":
        manifest.add_invalid_for_paper_reason(f"run_status={args.run_status} is not final paper evidence")
    if backend != "jsbsim":
        manifest.add_invalid_for_paper_reason("backend is not jsbsim")
    if args.dry_run:
        manifest.add_invalid_for_paper_reason("dry-run does not execute simulations")
    if failures:
        manifest.add_invalid_for_paper_reason("one or more checks/evaluations failed")

    manifest.config_overrides = [
        override
        for cfg in eval_configs.values()
        for override in get_config_overrides(cfg)
    ]
    manifest.artifacts_present["manifests"] = (run_dir / "manifests").is_dir()
    manifest.artifacts_present["run_manifest.json"] = True
    manifest.save(run_dir)
    validation = contract.validate(run_dir, manifest=manifest.to_dict())
    manifest.extra["artifact_validation"] = validation
    if not validation["valid"]:
        manifest.add_invalid_for_paper_reason("artifact contract validation failed")

    if failures:
        manifest.mark_failed("; ".join(str(f.get("error")) for f in failures[:3]))
    else:
        manifest.mark_completed(paper_safe=(args.run_status == "formal" and backend == "jsbsim" and not args.dry_run))
    manifest.save(run_dir)

    print("=" * 60)
    print("JSBSim HRL comparison workflow complete")
    print(f"Output dir: {run_dir}")
    print(f"Status: {manifest.status}")
    print(f"Paper-safe: {manifest.paper_safe}")
    print(f"Artifact contract valid: {validation['valid']}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures[:5]:
            print(f"  - {failure.get('phase')}: {failure.get('error')}")
    print("=" * 60)
    return 1 if failures and not (args.dry_run and args.allow_missing_checkpoints) else 0


if __name__ == "__main__":
    raise SystemExit(main())
