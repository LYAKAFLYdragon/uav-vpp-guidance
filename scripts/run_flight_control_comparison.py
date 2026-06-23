#!/usr/bin/env python3
"""
Unified runner for the flight-control comparison benchmark.

Evaluates up to five controllers on two tasks with paired seeds:
    PPO+PID-Hybrid, PPO-FixedPID, Enhanced PID, Baseline PID, (optional) GainScheduled PID

Usage:
    python scripts/run_flight_control_comparison.py \
        --run-id fc_compare_20260621_103000 \
        --controllers ppo_pid ppo enhanced_pid baseline_pid \
        --tasks multi_waypoint sustained_turn \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
        --jobs 4 \
        --checkpoint-ppo-pid outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt \
        --checkpoint-ppo outputs/experiments/no_prediction_vpp_ppo_jsbsim/checkpoints/last.pt \
        --config-base config/experiment/compare_flight_control_base.yaml \
        --config-mw config/experiment/task_multi_waypoint.yaml \
        --config-st config/experiment/task_sustained_turn.yaml

All episodes are saved as JSON under outputs/flight_control_compare/<run_id>/raw/.
A run_manifest.json is written to outputs/flight_control_compare/<run_id>/manifests/.
"""

from __future__ import annotations

import os

# Windows/PyTorch+NumPy can initialise multiple OpenMP runtimes and abort.
# Allow the process to continue; this is a known local-workaround on this
# platform and does not affect numerical results.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Allow running as script from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.evaluation.controller_adapters import (
    ControllerAdapter,
    PPOAdapter,
    ZeroOffsetAdapter,
)
from uav_vpp_guidance.evaluation.flight_control_metrics import (
    compute_multi_waypoint_metrics,
    compute_sustained_turn_metrics,
)
from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder, RunRecorder

logger = logging.getLogger(__name__)

CONTROLLER_ALIASES = {
    "ppo_pid": "PPO+PID",
    "apic_pid": "APIC-PID",
    "ppo": "PPO",
    "enhanced_pid": "Enhanced PID",
    "robust_pid": "Robust PID",
    "baseline_pid": "Baseline PID",
    "gain_scheduled_pid": "GainScheduled PID",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_experiment_config(config_path: str) -> dict:
    """Load YAML config recursively resolving ``includes`` relative to the file."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_experiment_config(inc_full))
    return merge_config(merged, base_config)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def _deep_update(target: dict, source: dict) -> dict:
    """Recursively update ``target`` with ``source`` without mutating source."""
    out = copy.deepcopy(target)
    for key, value in source.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.getcwd())
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Controller / environment construction
# ---------------------------------------------------------------------------


def _build_env(config: dict):
    """Build the appropriate environment class from the task config."""
    task_cfg = config.get("task", {})
    class_name = task_cfg.get("env_class", "CloseRangeTrackingEnv")
    if class_name == "MultiWaypointTrackingEnv":
        from uav_vpp_guidance.envs.multi_waypoint_tracking_env import MultiWaypointTrackingEnv
        return MultiWaypointTrackingEnv(config)
    if class_name == "SustainedTurnEnv":
        from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv
        return SustainedTurnEnv(config)
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    return CloseRangeTrackingEnv(config)


def _load_checkpoint_config(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt.get("config", {})


def _build_ppo_eval_config(
    checkpoint_path: str,
    base_config: dict,
    task_config: dict,
    backend_override: Optional[str] = None,
) -> dict:
    """
    Build an evaluation config for a PPO controller.

    The checkpoint's policy/observation/virtual_point settings are preserved;
    task-specific env limits and task block are overlaid from the comparison
    configs.  Backend is inherited from the task/base config (or the CLI override)
    rather than hard-coded to JSBSim, so the same runner can evaluate on the
    simple kinematic backend.
    """
    ckpt_config = _load_checkpoint_config(checkpoint_path)
    config = copy.deepcopy(ckpt_config)

    # Overlay backend from task/base config, with optional CLI override.
    backend = backend_override or task_config.get("backend") or base_config.get("backend") or "jsbsim"
    backend = str(backend).lower()
    config["backend"] = backend
    if "env" not in config:
        config["env"] = {}
    use_jsbsim = backend == "jsbsim"
    env_overrides = {
        "backend": backend,
        "use_jsbsim": use_jsbsim,
        "strict_backend": use_jsbsim,
    }
    for key in (
        "aircraft_model",
        "origin",
        "sim_freq",
        "decision_freq",
        "high_level_dt",
        "low_level_dt",
        "action_repeat",
        "max_high_level_steps",
        "min_altitude_m",
        "max_altitude_m",
        "max_range_m",
        "xy_limit_m",
    ):
        for src in (base_config.get("env", {}), task_config.get("env", {})):
            if key in src:
                env_overrides[key] = src[key]
    config["env"] = _deep_update(config.get("env", {}), env_overrides)

    # Preserve guidance gains from checkpoint but ensure mode is available.
    guidance_overrides = {"mode": "los_rate"}
    for src in (base_config.get("guidance", {}), task_config.get("guidance", {})):
        if "mode" in src:
            guidance_overrides["mode"] = src["mode"]
    config["guidance"] = _deep_update(config.get("guidance", {}), guidance_overrides)

    # Preserve limits from comparison configs (physical limits are part of the task).
    limits_overrides = base_config.get("limits", {})
    limits_overrides.update(task_config.get("limits", {}))
    if limits_overrides:
        config["limits"] = _deep_update(config.get("limits", {}), limits_overrides)

    # Detect APIC-style checkpoints (PPO outputs PID gain deltas, not VPP offsets).
    ckpt_ll = ckpt_config.get("low_level_controller", {})
    if isinstance(ckpt_ll, str):
        ckpt_ll = {}
    action_dim = ckpt_config.get("policy", {}).get("action_dim")
    is_apic = bool(ckpt_ll.get("apic", {}).get("enabled", False)) or \
              (int(action_dim) == 6 if action_dim is not None else False)

    # Ensure low-level controller type is inherited from the comparison base/task
    # configs. This guarantees PPO-FixedPID is evaluated with Enhanced PID if the
    # base config specifies it, even if the checkpoint was trained without it.
    # For APIC checkpoints, preserve the checkpoint's low-level/apic configuration.
    ll_override = None
    for src in (task_config, base_config):
        if "low_level_controller" in src:
            ll_override = src["low_level_controller"]
            break
    if is_apic:
        if ll_override is not None:
            # Merge base type but keep checkpoint's apic block.
            merged_ll = copy.deepcopy(ll_override)
            if isinstance(merged_ll, dict):
                merged_ll["apic"] = ckpt_ll.get("apic", {"enabled": True})
            config["low_level_controller"] = merged_ll
    else:
        if ll_override is not None:
            config["low_level_controller"] = copy.deepcopy(ll_override)

    # Ensure VPP is enabled for PPO controllers.
    if "virtual_point" not in config:
        config["virtual_point"] = {}
    config["virtual_point"]["enabled"] = True
    # APIC uses zero_offset/direct_track; legacy PPO uses normal VPP mode.
    config["virtual_point"]["mode"] = "zero_offset" if is_apic else "normal"
    if is_apic:
        if "guidance" not in config:
            config["guidance"] = {}
        config["guidance"]["direct_track_mode"] = True

    # Merge task/base virtual_point overrides (e.g. dynamics_aware for sustained turn)
    # while preserving the policy action_dim from the checkpoint.
    vp_overrides = {}
    for src in (base_config.get("virtual_point", {}), task_config.get("virtual_point", {})):
        for k, v in src.items():
            vp_overrides[k] = v
    if vp_overrides:
        config["virtual_point"] = _deep_update(config.get("virtual_point", {}), vp_overrides)

    # Task block overrides everything.
    if "task" in task_config:
        config["task"] = copy.deepcopy(task_config["task"])

    return config


def _build_pid_eval_config(
    base_config: dict,
    task_config: dict,
    ll_type: str,
    action_dim: int = 4,
    backend_override: Optional[str] = None,
) -> dict:
    """Build an evaluation config for a fixed-gain PID controller."""
    config = copy.deepcopy(base_config)
    config = _deep_update(config, task_config)

    # Overlay backend from task/base config, with optional CLI override.
    backend = backend_override or task_config.get("backend") or base_config.get("backend") or "jsbsim"
    backend = str(backend).lower()
    config["backend"] = backend
    if "env" not in config:
        config["env"] = {}
    use_jsbsim = backend == "jsbsim"
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = use_jsbsim
    config["env"]["strict_backend"] = use_jsbsim

    # Ensure zero-offset VPP so the guidance chain tracks the reference target.
    if "virtual_point" not in config:
        config["virtual_point"] = {}
    config["virtual_point"]["enabled"] = True
    config["virtual_point"]["mode"] = "zero_offset"
    config["virtual_point"]["action_dim"] = action_dim

    # Set low-level controller type.
    # Baseline PID is upgraded from a simple feedforward filter to a real PID
    # using the enhanced controller architecture but with advanced protections
    # disabled, so the comparison is against a genuine low-level PID.
    if ll_type == "baseline":
        ll_cfg = config.get("low_level_controller", {})
        if isinstance(ll_cfg, str):
            ll_cfg = {}
        ll_cfg["type"] = "enhanced"
        ll_cfg.setdefault("use_pid", True)
        ll_cfg.setdefault("use_rudder", False)
        ll_cfg.setdefault("use_aoa_protection", False)
        ll_cfg.setdefault("use_beta_suppression", False)
        config["low_level_controller"] = ll_cfg
    else:
        if isinstance(config.get("low_level_controller"), dict):
            config["low_level_controller"]["type"] = ll_type
        else:
            config["low_level_controller"] = {"type": ll_type}

    # Policy block is irrelevant for PID but kept for config consistency.
    if "policy" not in config:
        config["policy"] = {}
    config["policy"]["action_dim"] = action_dim

    return config


def _build_adapter(
    controller: str,
    config: dict,
    checkpoint_ppo_pid: Optional[str],
    checkpoint_ppo: Optional[str],
    checkpoint_apic_pid: Optional[str] = None,
    device: str = "cpu",
) -> ControllerAdapter:
    """Build a controller adapter for the named controller."""
    if controller == "ppo_pid":
        path = checkpoint_ppo_pid
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"PPO+PID checkpoint required but missing: {checkpoint_ppo_pid}"
            )
        env = _build_env(config)
        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 4))
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(path)
        env.close()
        return PPOAdapter(agent, name="ppo_pid")

    if controller == "apic_pid":
        path = checkpoint_apic_pid
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"APIC-PID checkpoint required but missing: {checkpoint_apic_pid}"
            )
        env = _build_env(config)
        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 6))
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(path)
        env.close()
        return PPOAdapter(agent, name="apic_pid")

    if controller == "ppo":
        path = checkpoint_ppo
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"PPO checkpoint required but missing: {checkpoint_ppo}"
            )
        env = _build_env(config)
        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(config.get("policy", {}).get("action_dim", 3))
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
        agent.load(path)
        env.close()
        return PPOAdapter(agent, name="ppo")

    if controller == "enhanced_pid":
        return ZeroOffsetAdapter(name="enhanced_pid", action_dim=4)
    if controller == "robust_pid":
        return ZeroOffsetAdapter(name="robust_pid", action_dim=4)
    if controller == "baseline_pid":
        return ZeroOffsetAdapter(name="baseline_pid", action_dim=4)
    if controller == "gain_scheduled_pid":
        return ZeroOffsetAdapter(name="gain_scheduled_pid", action_dim=4)

    raise ValueError(f"Unknown controller: {controller}")


# ---------------------------------------------------------------------------
# Single evaluation worker
# ---------------------------------------------------------------------------


def _run_episode(
    env,
    adapter: ControllerAdapter,
    seed: int,
    run_id: str,
    task: str,
    controller: str,
    episode: int,
    config: dict,
    config_sha256: str,
    git_commit: str,
    save_full: bool = True,
    scenario: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one episode and return the final episode JSON."""
    backend = config.get("backend", "jsbsim")
    strict_backend = config.get("env", {}).get(
        "strict_backend", backend == "jsbsim"
    )
    recorder = EpisodeRecorder(
        run_id=run_id,
        task=task,
        controller=controller,
        seed=seed,
        episode=episode,
        config=config,
        config_sha256=config_sha256,
        git_commit=git_commit,
        backend=backend,
        strict_backend=bool(strict_backend),
        save_full=save_full,
    )

    reset_kwargs = {"seed": seed}
    if scenario is not None:
        reset_kwargs["scenario"] = scenario
    obs = env.reset(**reset_kwargs)
    adapter.reset()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    info = {}
    dt = env.env_config.get("high_level_dt", 0.2)

    while not (terminated or truncated):
        action = adapter.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        own_state = info.get("own_state", {})
        target_state = info.get("target_state", {})
        recorder.record_step(steps, steps * dt, own_state, target_state, info, reward)

        if steps >= env.max_steps:
            truncated = True
            break

    # Final info may be empty if the loop never ran; fall back to env state.
    if not info:
        own_state, target_state = env._get_current_states()
        info = {
            "own_state": own_state,
            "target_state": target_state,
            "reason": "timeout",
            "is_success": False,
        }

    own_state = info.get("own_state", {})
    total_time_s = steps * dt
    termination_reason = info.get("reason", "timeout")
    success = bool(info.get("is_success", False))

    return recorder.finalize(
        steps=steps,
        total_time_s=total_time_s,
        total_reward=total_reward,
        termination_reason=termination_reason,
        success=success,
        final_position_m=own_state.get("position_m", own_state.get("position_neu")),
        final_speed_mps=float(own_state.get("speed_mps", 250.0)),
        final_altitude_m=float(own_state.get("altitude_m", 5000.0)),
    )


def _evaluate_worker(args: tuple) -> List[Dict[str, Any]]:
    """Worker that runs all episodes for one (task, controller, seed) triplet."""
    (
        run_id,
        task,
        controller,
        seed,
        n_episodes,
        base_config,
        task_config,
        checkpoint_ppo_pid,
        checkpoint_ppo,
        checkpoint_apic_pid,
        save_full,
        device,
        backend_override,
    ) = args

    # Build config and adapter in the subprocess to isolate JSBSim state.
    if controller in ("ppo_pid", "apic_pid", "ppo"):
        path = {
            "ppo_pid": checkpoint_ppo_pid,
            "apic_pid": checkpoint_apic_pid,
            "ppo": checkpoint_ppo,
        }[controller]
        config = _build_ppo_eval_config(path, base_config, task_config, backend_override=backend_override)
    else:
        ll_type = {
            "enhanced_pid": "enhanced_pid",
            "robust_pid": "robust_pid",
            "baseline_pid": "baseline_pid",
            "gain_scheduled_pid": "gain_scheduled_pid",
        }[controller]
        config = _build_pid_eval_config(base_config, task_config, ll_type, backend_override=backend_override)

    env = _build_env(config)
    adapter = _build_adapter(controller, config, checkpoint_ppo_pid, checkpoint_ppo, checkpoint_apic_pid, device=device)

    config_sha256 = _config_sha256(config)
    git_commit = _get_git_commit()
    records = []
    for ep in range(n_episodes):
        ep_seed = seed * 10000 + ep
        ep_json = _run_episode(
            env,
            adapter,
            ep_seed,
            run_id,
            task,
            controller,
            ep,
            config,
            config_sha256,
            git_commit,
            save_full=save_full,
        )
        records.append(ep_json)

    env.close()
    return records


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="Flight-control comparison runner")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    parser.add_argument(
        "--controllers",
        type=str,
        nargs="+",
        default=["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"],
        help="Controllers to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["multi_waypoint", "sustained_turn"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20)),
        help="Seeds to evaluate (paired across controllers)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Episodes per seed",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--checkpoint-ppo-pid",
        type=str,
        default=None,
        help="Checkpoint path for PPO+PID-Hybrid",
    )
    parser.add_argument(
        "--checkpoint-ppo",
        type=str,
        default=None,
        help="Checkpoint path for PPO-FixedPID",
    )
    parser.add_argument(
        "--checkpoint-apic-pid",
        type=str,
        default=None,
        help="Checkpoint path for APIC-PID",
    )
    parser.add_argument(
        "--config-base",
        type=str,
        default="config/experiment/compare_flight_control_base.yaml",
        help="Base comparison config",
    )
    parser.add_argument(
        "--config-mw",
        type=str,
        default="config/experiment/task_multi_waypoint.yaml",
        help="Multi-waypoint task config",
    )
    parser.add_argument(
        "--config-st",
        type=str,
        default="config/experiment/task_sustained_turn.yaml",
        help="Sustained-turn task config",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/flight_control_compare",
        help="Root output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for PPO inference",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override simulation backend for all evaluation configs",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable full trajectory saving to reduce disk usage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs/paths and exit without running simulations",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="formal",
        choices=["smoke", "formal"],
        help="Run status for manifest and report (smoke = exploratory, formal = final)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_dir = Path(args.output_root) / args.run_id
    raw_dir = run_dir / "raw"
    manifest_dir = run_dir / "manifests"
    log_dir = run_dir / "logs"
    for d in (run_dir, raw_dir, manifest_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "runner.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Loading configs")
    base_config = load_experiment_config(args.config_base)
    mw_config = load_experiment_config(args.config_mw)
    st_config = load_experiment_config(args.config_st)

    task_configs = {
        "multi_waypoint": mw_config,
        "sustained_turn": st_config,
    }

    git_commit = _get_git_commit()
    logger.info(f"Git commit: {git_commit}")

    # Validate controllers
    unknown = [c for c in args.controllers if c not in CONTROLLER_ALIASES]
    if unknown:
        raise ValueError(f"Unknown controllers: {unknown}")

    # Validate checkpoints
    if "ppo_pid" in args.controllers and not args.checkpoint_ppo_pid:
        raise ValueError("--checkpoint-ppo-pid required when evaluating ppo_pid")
    if "apic_pid" in args.controllers and not args.checkpoint_apic_pid:
        raise ValueError("--checkpoint-apic-pid required when evaluating apic_pid")
    if "ppo" in args.controllers and not args.checkpoint_ppo:
        raise ValueError("--checkpoint-ppo required when evaluating ppo")

    if args.dry_run:
        logger.info("DRY RUN: validating configs and checkpoints")
        for controller in args.controllers:
            for task in args.tasks:
                if controller in ("ppo_pid", "apic_pid", "ppo"):
                    path = {
                        "ppo_pid": args.checkpoint_ppo_pid,
                        "apic_pid": args.checkpoint_apic_pid,
                        "ppo": args.checkpoint_ppo,
                    }[controller]
                    cfg = _build_ppo_eval_config(path, base_config, task_configs[task], backend_override=args.backend)
                else:
                    ll_type = {
                        "enhanced_pid": "enhanced_pid",
                        "robust_pid": "robust_pid",
                        "baseline_pid": "baseline_pid",
                        "gain_scheduled_pid": "gain_scheduled_pid",
                    }[controller]
                    cfg = _build_pid_eval_config(base_config, task_configs[task], ll_type, backend_override=args.backend)
                logger.info(f"  {controller}/{task}: config ok (sha256={_config_sha256(cfg)})")
        logger.info("DRY RUN complete")
        return

    # Build work list
    work_items = []
    for task in args.tasks:
        for controller in args.controllers:
            for seed in args.seeds:
                work_items.append(
                    (
                        args.run_id,
                        task,
                        controller,
                        seed,
                        args.n_episodes,
                        base_config,
                        task_configs[task],
                        args.checkpoint_ppo_pid,
                        args.checkpoint_ppo,
                        args.checkpoint_apic_pid,
                        not args.no_trajectory,
                        args.device,
                        args.backend,
                    )
                )

    logger.info(f"Total work items: {len(work_items)} ({len(args.controllers)} controllers x {len(args.tasks)} tasks x {len(args.seeds)} seeds x {args.n_episodes} episodes)")

    # Run workers
    all_records = []
    start_time = time.time()
    max_workers = min(args.jobs, len(work_items))

    if max_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_evaluate_worker, item): item for item in work_items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                    logger.info(f"Finished {item[1]}/{item[2]}/seed{item[3]} ({len(records)} episodes)")
                except Exception as exc:
                    logger.exception(f"Failed {item[1]}/{item[2]}/seed{item[3]}: {exc}")
    else:
        for item in work_items:
            try:
                records = _evaluate_worker(item)
                all_records.extend(records)
                logger.info(f"Finished {item[1]}/{item[2]}/seed{item[3]} ({len(records)} episodes)")
            except Exception as exc:
                logger.exception(f"Failed {item[1]}/{item[2]}/seed{item[3]}: {exc}")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s; {len(all_records)} episodes saved")

    # Save episode JSONs and aggregate JSON via the independent recorder layer.
    run_recorder = RunRecorder(run_dir)
    for rec in all_records:
        run_recorder.add(rec)
    run_recorder.write()
    logger.info(f"Wrote aggregate JSON with {len(all_records)} episodes to {run_recorder.aggregate_dir / 'episode_records.json'}")

    # Build and save per-controller/task merged config snapshots.
    config_snapshots = {}
    for controller in args.controllers:
        for task in args.tasks:
            if controller in ("ppo_pid", "apic_pid", "ppo"):
                path = {
                    "ppo_pid": args.checkpoint_ppo_pid,
                    "apic_pid": args.checkpoint_apic_pid,
                    "ppo": args.checkpoint_ppo,
                }[controller]
                cfg = _build_ppo_eval_config(path, base_config, task_configs[task], backend_override=args.backend)
            else:
                ll_type = {
                    "enhanced_pid": "enhanced_pid",
                    "robust_pid": "robust_pid",
                    "baseline_pid": "baseline_pid",
                    "gain_scheduled_pid": "gain_scheduled_pid",
                }[controller]
                cfg = _build_pid_eval_config(base_config, task_configs[task], ll_type, backend_override=args.backend)
            snapshot_path = manifest_dir / f"config_{controller}_{task}.yaml"
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            config_snapshots[f"{controller}/{task}"] = {
                "path": str(snapshot_path),
                "sha256": _config_sha256(cfg),
            }
    logger.info(f"Saved {len(config_snapshots)} merged config snapshots to {manifest_dir}")

    # Write manifest
    effective_backend = args.backend or base_config.get("backend", "jsbsim")
    manifest = {
        "run_id": args.run_id,
        "status": args.status,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "elapsed_seconds": elapsed,
        "git_commit": git_commit,
        "backend": effective_backend,
        "strict_backend": effective_backend == "jsbsim",
        "controllers": args.controllers,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "n_episodes": args.n_episodes,
        "checkpoint_ppo_pid": args.checkpoint_ppo_pid,
        "checkpoint_ppo": args.checkpoint_ppo,
        "checkpoint_apic_pid": args.checkpoint_apic_pid,
        "config_base": args.config_base,
        "config_base_sha256": _sha256_file(args.config_base),
        "config_mw": args.config_mw,
        "config_mw_sha256": _sha256_file(args.config_mw),
        "config_st": args.config_st,
        "config_st_sha256": _sha256_file(args.config_st),
        "config_snapshots": config_snapshots,
        "total_episodes": len(all_records),
        "episodes_per_controller_task": {
            f"{rec['task']}/{rec['controller']}": sum(
                1 for r in all_records if r["task"] == rec["task"] and r["controller"] == rec["controller"]
            )
            for rec in all_records
        },
    }
    manifest_path = manifest_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
