#!/usr/bin/env python3
"""
Curriculum + Adversarial Air-Combat Training Orchestrator.

Integrates three existing components:
  1. `scripts/train_curriculum_ppo.py`  -> stage-1 pursuer vs fixed maneuver target
  2. `scripts/train_adversarial_target.py` -> train RL evader for stage 2/3
  3. `scripts/train_adversarial_pursuer.py` -> fine-tune pursuer vs the evader

Stage design:
  Stage 1 (easy): pursuer vs maneuver-library bandit (difficulty=easy)
  Stage 2 (medium): pursuer vs RL target trained at medium difficulty
  Stage 3 (hard): pursuer vs RL target trained at hard difficulty

Each stage transition is gated by eval success rate >= ``stage_gate_sr``.

Usage:
    # Full three-stage pipeline
    python scripts/train_curriculum_adversarial.py

    # Smoke test (minimal timesteps, fast validation)
    python scripts/train_curriculum_adversarial.py --smoke

    # Custom output directory / seed
    python scripts/train_curriculum_adversarial.py --output-dir outputs/curriculum_adv --seed 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on path when running the script directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse existing training functions
from scripts.train_curriculum_ppo import train_ppo_curriculum
from scripts.train_adversarial_target import train_target
from scripts.train_adversarial_pursuer import train_pursuer
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.swanlab_logger import SwanLabLogger


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STAGE1_CONFIG = "config/experiment/close_range_curriculum_stage1.yaml"
DEFAULT_STAGE2_TARGET_CONFIG = "config/adversarial/train_target_pilot.yaml"
DEFAULT_STAGE3_TARGET_CONFIG = "config/adversarial/train_target_pilot.yaml"
DEFAULT_STAGE2_PURSuer_CONFIG = "config/adversarial/train_pursuer_v2.yaml"
DEFAULT_STAGE3_PURSuer_CONFIG = "config/adversarial/train_pursuer_v2.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load and merge experiment configuration with includes."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def _load_and_override(config_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Load config and apply shallow overrides."""
    config = load_experiment_config(config_path)
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = value
    return config


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _policy_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Extract the policy architecture stored in a PPO checkpoint."""
    import torch
    if not os.path.exists(checkpoint_path):
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception as exc:
        logger.warning("Failed to read policy config from %s: %s", checkpoint_path, exc)
        return None


def evaluate_pursuer_checkpoint(
    pursuer_ckpt: str,
    target_ckpt: str,
    config: Dict[str, Any],
    num_episodes: int = 30,
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a pursuer checkpoint against a frozen target checkpoint.

    Returns a dict with at least ``success_rate`` and other diagnostics.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    # Import here to avoid heavy JSBSim startup unless needed
    import torch
    from uav_vpp_guidance.agents.ppo_agent import PPOAgent
    from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
    from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
    from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler

    env = AdversarialJSBSimEnv(config)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

    # Load pursuer, matching architecture to the saved checkpoint
    pursuer_cfg = dict(config)
    ckpt_policy = _policy_config_from_checkpoint(pursuer_ckpt)
    if ckpt_policy:
        pursuer_cfg["policy"] = ckpt_policy
    pursuer_agent = PPOAgent(
        obs_dim=16, action_dim=3, config=pursuer_cfg,
        device=config.get("ppo", {}).get("device", "cpu"),
    )
    pursuer_agent.load(pursuer_ckpt)
    pursuer_agent.network.eval()

    # Load target, matching architecture to the saved checkpoint
    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
    target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
    ckpt_target_policy = _policy_config_from_checkpoint(target_ckpt)
    if ckpt_target_policy:
        target_cfg["policy"] = ckpt_target_policy
    target_agent = AdversarialTargetAgent(config=target_cfg)
    target_agent.load(target_ckpt)
    target_agent.eval()

    episodes: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            for ep in range(num_episodes // max(1, len(seeds))):
                ep_seed = seed * 10000 + ep
                scenario = sampler.sample() if sampler else None
                p_obs, t_obs = env.reset(scenario=scenario, seed=ep_seed)
                for _ in range(env.max_steps):
                    p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
                    t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])
                    p_obs, t_obs, _, _, terminated, truncated, info = env.step(p_action, t_action)
                    if terminated or truncated:
                        reason = info.get("termination", {}).get("reason", "unknown")
                        episodes.append({"success": reason == "success"})
                        break
    finally:
        env.close()

    sr = sum(1 for e in episodes if e["success"]) / max(1, len(episodes))
    return {"num_episodes": len(episodes), "success_rate": sr}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage1_train_pursuer_vs_bandit(
    output_dir: str,
    config_path: str,
    smoke: bool,
    seed: int,
    swanlab_logger=None,
) -> str:
    """Stage 1: train pursuer against fixed easy maneuver-library bandit."""
    print("\n" + "=" * 70)
    print("STAGE 1: Pursuer vs fixed maneuver-library bandit (easy)")
    print("=" * 70)

    config = load_experiment_config(config_path)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = "curriculum_adversarial_stage1"

    stage_dir = os.path.join(output_dir, "stage1")
    os.makedirs(stage_dir, exist_ok=True)

    ckpt_dir = train_ppo_curriculum(config, stage_dir, smoke=smoke, swanlab_logger=swanlab_logger)
    best_ckpt = os.path.join(ckpt_dir, "checkpoints", "best.pt")
    print(f"[Stage 1] Best pursuer checkpoint: {best_ckpt}")
    return best_ckpt


def stage2_or_3_train_target_and_pursuer(
    output_dir: str,
    stage: int,
    target_config_path: str,
    pursuer_config_path: str,
    prev_pursuer_ckpt: str,
    smoke: bool,
    seed: int,
    target_difficulty: str,
    swanlab_logger=None,
) -> str:
    """Train an RL target and fine-tune the pursuer against it."""
    print("\n" + "=" * 70)
    print(f"STAGE {stage}: Pursuer vs RL target ({target_difficulty})")
    print("=" * 70)

    # ---- Train target ----
    target_cfg = _load_and_override(
        target_config_path,
        {
            "experiment": {"name": f"curriculum_adversarial_stage{stage}_target", "seed": seed},
            "pursuer_checkpoint": prev_pursuer_ckpt,
        },
    )
    # Make the target harder by adjusting maneuver difficulty / noise if requested
    if "env" not in target_cfg:
        target_cfg["env"] = {}
    if "bandit" not in target_cfg["env"]:
        target_cfg["env"]["bandit"] = {}
    target_cfg["env"]["bandit"]["difficulty"] = target_difficulty
    if target_difficulty == "hard":
        target_cfg["env"]["bandit"].setdefault("selector", {})
        target_cfg["env"]["bandit"]["selector"]["reaction_time_s"] = 1.5
        target_cfg["env"]["bandit"]["selector"]["selector_noise"] = 0.25
        target_cfg["env"]["bandit"]["selector"]["param_perturb_scale"] = 0.25

    target_dir = os.path.join(output_dir, f"stage{stage}", "target")
    os.makedirs(target_dir, exist_ok=True)
    train_target(target_cfg, target_dir, smoke=smoke, swanlab_logger=swanlab_logger, log_prefix=f"s{stage}/target/")
    target_ckpt = os.path.join(target_dir, "checkpoints", "best.pt")
    print(f"[Stage {stage}] Best target checkpoint: {target_ckpt}")

    # ---- Fine-tune pursuer ----
    pursuer_cfg = _load_and_override(
        pursuer_config_path,
        {
            "experiment": {"name": f"curriculum_adversarial_stage{stage}_pursuer", "seed": seed},
            "pursuer_checkpoint": prev_pursuer_ckpt,
            "target_checkpoint": target_ckpt,
        },
    )
    pursuer_dir = os.path.join(output_dir, f"stage{stage}", "pursuer")
    os.makedirs(pursuer_dir, exist_ok=True)
    train_pursuer(pursuer_cfg, pursuer_dir, smoke=smoke, swanlab_logger=swanlab_logger, log_prefix=f"s{stage}/pursuer/")
    best_ckpt = os.path.join(pursuer_dir, "checkpoints", "best.pt")
    last_ckpt = os.path.join(pursuer_dir, "checkpoints", "last.pt")
    if os.path.exists(best_ckpt):
        pursuer_ckpt = best_ckpt
        print(f"[Stage {stage}] Best pursuer checkpoint: {pursuer_ckpt}")
    elif os.path.exists(last_ckpt):
        pursuer_ckpt = last_ckpt
        print(f"[Stage {stage}] Best not found, using last pursuer checkpoint: {pursuer_ckpt}")
    else:
        raise FileNotFoundError(f"No pursuer checkpoint found in {pursuer_dir}/checkpoints")
    return pursuer_ckpt


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_curriculum_adversarial(
    output_dir: str,
    smoke: bool,
    seed: int,
    stage_gate_sr: float,
    stage_configs: Dict[str, str],
    swanlab_logger=None,
) -> Dict[str, Any]:
    """Run the full curriculum + adversarial pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "smoke": smoke,
        "stage_gate_sr": stage_gate_sr,
        "stages": {},
    }

    # Stage 1: fixed bandit
    stage1_ckpt = stage1_train_pursuer_vs_bandit(
        output_dir,
        stage_configs["stage1_pursuer"],
        smoke,
        seed,
    )
    manifest["stages"]["stage1"] = {"pursuer_ckpt": stage1_ckpt}
    if swanlab_logger is not None:
        swanlab_logger.log({"gate/stage1_pursuer_ready": 1.0}, step=0)

    # Stage 2: medium RL target
    stage2_ckpt = stage2_or_3_train_target_and_pursuer(
        output_dir,
        stage=2,
        target_config_path=stage_configs["stage2_target"],
        pursuer_config_path=stage_configs["stage2_pursuer"],
        prev_pursuer_ckpt=stage1_ckpt,
        smoke=smoke,
        seed=seed,
        target_difficulty="medium",
        swanlab_logger=swanlab_logger,
    )
    manifest["stages"]["stage2"] = {"pursuer_ckpt": stage2_ckpt}

    # Gate check (only meaningful if target checkpoint exists)
    # In smoke mode the target may not train enough, so we skip the gate.
    if not smoke:
        stage2_target_dir = os.path.join(output_dir, "stage2", "target")
        target_ckpt = os.path.join(stage2_target_dir, "checkpoints", "best.pt")
        eval_cfg = load_experiment_config(stage_configs["stage2_pursuer"])
        eval_cfg["target_checkpoint"] = target_ckpt
        eval_cfg["pursuer_checkpoint"] = stage2_ckpt
        eval_res = evaluate_pursuer_checkpoint(stage2_ckpt, target_ckpt, eval_cfg)
        manifest["stages"]["stage2"]["eval"] = eval_res
        print(f"[Stage 2 gate] SR={eval_res['success_rate']:.2%} (threshold={stage_gate_sr:.0%})")
        if swanlab_logger is not None:
            swanlab_logger.log({
                "gate/stage2_success_rate": eval_res["success_rate"],
                "gate/stage2_passed": float(eval_res["success_rate"] >= stage_gate_sr),
            }, step=0)
        if eval_res["success_rate"] < stage_gate_sr:
            print("[Stage 2 gate] FAILED. Stopping pipeline.")
            manifest["status"] = "stage2_gate_failed"
            return manifest

    # Stage 3: hard RL target
    stage3_ckpt = stage2_or_3_train_target_and_pursuer(
        output_dir,
        stage=3,
        target_config_path=stage_configs["stage3_target"],
        pursuer_config_path=stage_configs["stage3_pursuer"],
        prev_pursuer_ckpt=stage2_ckpt,
        smoke=smoke,
        seed=seed,
        target_difficulty="hard",
        swanlab_logger=swanlab_logger,
    )
    manifest["stages"]["stage3"] = {"pursuer_ckpt": stage3_ckpt}

    if not smoke:
        stage3_target_dir = os.path.join(output_dir, "stage3", "target")
        target_ckpt = os.path.join(stage3_target_dir, "checkpoints", "best.pt")
        eval_cfg = load_experiment_config(stage_configs["stage3_pursuer"])
        eval_cfg["target_checkpoint"] = target_ckpt
        eval_cfg["pursuer_checkpoint"] = stage3_ckpt
        eval_res = evaluate_pursuer_checkpoint(stage3_ckpt, target_ckpt, eval_cfg)
        manifest["stages"]["stage3"]["eval"] = eval_res
        print(f"[Stage 3 gate] SR={eval_res['success_rate']:.2%} (threshold={stage_gate_sr:.0%})")
        if swanlab_logger is not None:
            swanlab_logger.log({
                "gate/stage3_success_rate": eval_res["success_rate"],
                "gate/stage3_passed": float(eval_res["success_rate"] >= stage_gate_sr),
            }, step=0)

    manifest["status"] = "completed"
    manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")

    if swanlab_logger is not None:
        final_stage3_sr = manifest.get("stages", {}).get("stage3", {}).get("eval", {}).get("success_rate")
        swanlab_logger.log({
            "final/stage1_pursuer_ready": 1.0,
            "final/stage3_success_rate": final_stage3_sr if final_stage3_sr is not None else 0.0,
            "final/status_completed": 1.0 if manifest["status"] == "completed" else 0.0,
        }, step=0)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_no_vpp_config(config_path: str) -> bool:
    """Heuristic: config filename contains 'no_vpp' or content sets zero_offset."""
    if "no_vpp" in os.path.basename(config_path).lower():
        return True
    try:
        cfg = load_yaml_config(config_path)
        mode = cfg.get("virtual_point", {}).get("mode", "")
        return mode == "zero_offset"
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-stage curriculum + adversarial air-combat training."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/adversarial_curriculum",
        help="Root output directory for all stages.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run minimal smoke test (fast validation).",
    )
    parser.add_argument(
        "--stage-gate-sr", type=float, default=None,
        help="Minimum eval success rate to advance to the next stage. "
             "Defaults: vpp=0.15, no_vpp=0.10, otherwise=0.50.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["vpp", "no_vpp"], default=None,
        help="Convenience flag: auto-select Stage 1 / Stage 2/3 pursuer configs.",
    )
    parser.add_argument(
        "--stage1-pursuer-config", type=str, default=DEFAULT_STAGE1_CONFIG,
        help="Stage 1 pursuer curriculum config.",
    )
    parser.add_argument(
        "--stage2-target-config", type=str, default=DEFAULT_STAGE2_TARGET_CONFIG,
    )
    parser.add_argument(
        "--stage2-pursuer-config", type=str, default=DEFAULT_STAGE2_PURSuer_CONFIG,
    )
    parser.add_argument(
        "--stage3-target-config", type=str, default=DEFAULT_STAGE3_TARGET_CONFIG,
    )
    parser.add_argument(
        "--stage3-pursuer-config", type=str, default=DEFAULT_STAGE3_PURSuer_CONFIG,
    )
    parser.add_argument(
        "--use-swanlab", action="store_true", help="Enable SwanLab logging"
    )
    parser.add_argument(
        "--swanlab-project", type=str, default=None, help="SwanLab project name"
    )
    parser.add_argument(
        "--swanlab-exp", type=str, default=None, help="SwanLab experiment name"
    )
    args = parser.parse_args()

    # Auto-select configs based on mode
    if args.mode == "vpp":
        args.stage1_pursuer_config = DEFAULT_STAGE1_CONFIG
        args.stage2_pursuer_config = "config/adversarial/train_pursuer_v2_pilot_aggressive.yaml"
        args.stage3_pursuer_config = "config/adversarial/train_pursuer_v2_pilot_aggressive_stage3.yaml"
        if args.stage_gate_sr is None:
            args.stage_gate_sr = 0.15
    elif args.mode == "no_vpp":
        args.stage1_pursuer_config = "config/experiment/close_range_curriculum_stage1_no_vpp.yaml"
        args.stage2_pursuer_config = "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml"
        args.stage3_pursuer_config = "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml"
        if args.stage_gate_sr is None:
            args.stage_gate_sr = 0.10
    else:
        if args.stage_gate_sr is None:
            args.stage_gate_sr = 0.50

    # Safety check: Stage 1 no_vpp should pair with no_vpp Stage 2/3 pursuer configs
    stage1_no_vpp = _is_no_vpp_config(args.stage1_pursuer_config)
    for stage_key, path in [
        ("stage2_pursuer", args.stage2_pursuer_config),
        ("stage3_pursuer", args.stage3_pursuer_config),
    ]:
        stage_no_vpp = _is_no_vpp_config(path)
        if stage1_no_vpp and not stage_no_vpp:
            raise ValueError(
                f"Stage 1 config is No-VPP but {stage_key}={path!r} does not appear to be No-VPP. "
                "Use --mode no_vpp or ensure pursuer configs are zero-offset."
            )
        if not stage1_no_vpp and stage_no_vpp:
            raise ValueError(
                f"Stage 1 config is VPP but {stage_key}={path!r} appears to be No-VPP. "
                "Use --mode vpp or ensure pursuer configs use VPP offsets."
            )

    stage_configs = {
        "stage1_pursuer": args.stage1_pursuer_config,
        "stage2_target": args.stage2_target_config,
        "stage2_pursuer": args.stage2_pursuer_config,
        "stage3_target": args.stage3_target_config,
        "stage3_pursuer": args.stage3_pursuer_config,
    }

    swanlab_logger = None
    if args.use_swanlab:
        swanlab_logger = SwanLabLogger(
            project=args.swanlab_project,
            experiment=args.swanlab_exp,
            config={
                "seed": args.seed,
                "mode": args.mode,
                "stage_gate_sr": args.stage_gate_sr,
                "smoke": args.smoke,
                "stage_configs": stage_configs,
            },
            enabled=True,
        )

    try:
        run_curriculum_adversarial(
            output_dir=args.output_dir,
            smoke=args.smoke,
            seed=args.seed,
            stage_gate_sr=args.stage_gate_sr,
            stage_configs=stage_configs,
            swanlab_logger=swanlab_logger,
        )
    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()


if __name__ == "__main__":
    main()
