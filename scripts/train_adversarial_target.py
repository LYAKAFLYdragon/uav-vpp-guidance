#!/usr/bin/env python3
"""
Phase 1: Train Adversarial Target Agent against a Frozen Pre-trained Pursuer.

The pursuer uses a pre-trained PPO policy (VPP → LOS guidance pipeline)
with deterministic actions. The target (evader) learns from scratch to
maximize survival time and escape distance.

Usage:
    # Smoke test (fast)
    python scripts/train_adversarial_target.py \\
        --config config/adversarial/train_target.yaml --smoke

    # Full training (500K steps)
    python scripts/train_adversarial_target.py \\
        --config config/adversarial/train_target.yaml

    # Resume from checkpoint
    python scripts/train_adversarial_target.py \\
        --config config/adversarial/train_target.yaml --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root for direct script execution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_adversarial_target")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load and merge experiment configuration with includes."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(
                os.path.dirname(config_path), "..", os.path.basename(inc_path)
            )
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def _policy_config_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Extract the policy architecture stored in a PPO checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception as exc:
        logger.warning("Failed to read policy config from %s: %s", checkpoint_path, exc)
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_target(
    env: AdversarialJSBSimEnv,
    pursuer_agent: PPOAgent,
    target_agent: AdversarialTargetAgent,
    config: Dict[str, Any],
    num_episodes: int = 10,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the target agent against the frozen pursuer.

    Returns metrics including survival rate, mean escape duration,
    and mean final range.
    """
    if seeds is None:
        seeds = [0]

    episodes: List[Dict[str, Any]] = []
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))
    for seed in seeds:
        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            scenario = sampler.sample() if sampler else None
            p_obs, t_obs = env.reset(scenario=scenario, seed=ep_seed)

            ep_return: float = 0.0
            ep_length: int = 0
            min_range: float = float("inf")
            final_range: float = 0.0
            reason: str = "timeout"

            for step in range(env.max_steps):
                # Pursuer: deterministic VPP action
                p_obs_vec = p_obs["observation_vector"]
                p_action = pursuer_agent.get_deterministic_action(p_obs_vec)

                # Target: stochastic action (during training) or deterministic (eval)
                t_obs_vec = t_obs["observation_vector"]
                t_action = target_agent.get_deterministic_action(t_obs_vec)

                (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = env.step(
                    p_action, t_action
                )

                ep_return += t_rew
                ep_length += 1

                rel = t_obs.get("relative_state", {})
                range_m = float(rel.get("range_m", 0.0))
                min_range = min(min_range, range_m)
                final_range = range_m

                if terminated or truncated:
                    reason = info.get("termination", {}).get("reason", "unknown")
                    break

            episodes.append({
                "seed": seed,
                "episode": ep,
                "return": ep_return,
                "length": ep_length,
                "min_range_m": min_range,
                "final_range_m": final_range,
                "reason": reason,
                "survived": reason in ("timeout", "out_of_bounds"),
                "captured": reason == "success",
                "crashed": reason == "crash",
            })

    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    survived = sum(1 for e in episodes if e["survived"])
    captured = sum(1 for e in episodes if e["captured"])
    crashed = sum(1 for e in episodes if e["crashed"])
    timeouts = sum(1 for e in episodes if e["reason"] == "timeout")
    oobs = sum(1 for e in episodes if e["reason"] == "out_of_bounds")

    return {
        "num_episodes": len(episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "survival_rate": survived / max(1, len(episodes)),
        "capture_rate": captured / max(1, len(episodes)),
        "crash_rate": crashed / max(1, len(episodes)),
        "timeout_rate": timeouts / max(1, len(episodes)),
        "oob_rate": oobs / max(1, len(episodes)),
        "mean_final_range_m": float(np.mean([e["final_range_m"] for e in episodes])),
        "mean_min_range_m": float(np.mean([e["min_range_m"] for e in episodes])),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_target(
    config: Dict[str, Any],
    output_dir: str,
    smoke: bool = False,
    resume: bool = False,
    swanlab_logger=None,
    log_prefix: str = "target/",
) -> None:
    """
    Train the adversarial target agent against a frozen pre-trained pursuer.

    Args:
        config: Full experiment configuration.
        output_dir: Output directory for checkpoints and logs.
        smoke: If True, run a minimal smoke test.
        resume: If True, resume from the last checkpoint.
    """
    # ---- Setup ----
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    training_cfg = config.get("training", {})
    total_steps = 5000 if smoke else int(training_cfg.get("total_timesteps", 500000))
    eval_interval = int(training_cfg.get("eval_interval", 10000))
    save_interval = int(training_cfg.get("save_interval", 50000))

    # ---- Create environment ----
    logger.info("Creating adversarial environment...")
    env = AdversarialJSBSimEnv(config)

    csv_file = None
    try:
        # ---- Create scenario sampler ----
        sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

        # ---- Load frozen pursuer agent ----
        pursuer_ckpt = config.get("pursuer_checkpoint",
                                  "outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt")
        logger.info("Loading pursuer from: %s", pursuer_ckpt)
        # Match the PPOAgent architecture to the saved checkpoint so the
        # load succeeds regardless of the generic ppo.yaml policy block.
        ckpt_policy = _policy_config_from_checkpoint(pursuer_ckpt)
        if ckpt_policy:
            config["policy"] = ckpt_policy
        pursuer_agent = PPOAgent(
            obs_dim=16,
            action_dim=3,
            config=config,
            device=config.get("ppo", {}).get("device", "cpu"),
        )
        if os.path.exists(pursuer_ckpt):
            pursuer_agent.load(pursuer_ckpt)
        else:
            logger.warning("Pursuer checkpoint not found at %s. Using random policy.", pursuer_ckpt)
        pursuer_agent.network.eval()
        for param in pursuer_agent.network.parameters():
            param.requires_grad = False

        # ---- Create target agent ----
        target_cfg = config.get("target_ppo", {})
        # Merge with policy config
        full_target_cfg = dict(config)
        full_target_cfg["ppo"] = target_cfg
        full_target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
        device = target_cfg.get("device", "cpu")

        logger.info("Creating target agent...")
        target_agent = AdversarialTargetAgent(config=full_target_cfg, device=device)

        # Resume if requested
        start_step = 0
        best_eval_return = -float("inf")
        resume_path = os.path.join(ckpt_dir, "last.pt")
        if resume and os.path.exists(resume_path):
            logger.info("Resuming from %s", resume_path)
            target_agent.load(resume_path)
            # Try to load training state
            state_path = os.path.join(ckpt_dir, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                start_step = state.get("step", 0)
                best_eval_return = state.get("best_eval_return", -float("inf"))

        # ---- CSV logging ----
        log_path = os.path.join(log_dir, "target_training_log.csv")
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_fraction", "explained_variance", "eval_return",
            "eval_survival_rate", "eval_capture_rate", "eval_crash_rate",
            "eval_timeout_rate", "eval_oob_rate", "wall_time_s",
        ])

        # ---- Training loop ----
        rollout_steps = int(target_cfg.get("rollout_steps", 1024))
        scenario = sampler.sample() if sampler else None
        p_obs, t_obs = env.reset(scenario=scenario, seed=config["experiment"]["seed"])
        episode_return: float = 0.0
        episode_length: int = 0
        global_step: int = start_step
        last_eval_step: int = start_step
        t_start = time.time()

        logger.info("Starting training: total_steps=%d rollout_steps=%d", total_steps, rollout_steps)

        while global_step < total_steps:
            # Collect rollout
            for _ in range(rollout_steps):
                # Pursuer: deterministic action
                p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])

                # Target: stochastic action for exploration
                t_obs_vec = t_obs["observation_vector"]
                t_action, t_log_prob, t_value = target_agent.select_action(
                    t_obs_vec, deterministic=False, store=False
                )

                (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = env.step(
                    p_action, t_action
                )

                episode_return += t_rew
                episode_length += 1

                # Store transition
                target_agent.store_transition(
                    obs=t_obs_vec,
                    action=t_action,
                    log_prob=t_log_prob,
                    reward=t_rew,
                    done=terminated or truncated,
                    value=t_value,
                )

                if terminated or truncated:
                    scenario = sampler.sample() if sampler else None
                    p_obs, t_obs = env.reset(scenario=scenario)
                    episode_return = 0.0
                    episode_length = 0

            global_step += rollout_steps

            # PPO update
            next_t_obs_vec = t_obs["observation_vector"] if t_obs is not None else None
            update_stats = target_agent.update(next_t_obs_vec)

            # Periodic evaluation (eval_interval may not align with rollout_steps)
            eval_stats = {}
            if global_step - last_eval_step >= eval_interval or global_step >= total_steps:
                last_eval_step = global_step
                logger.info("Evaluating at step %d...", global_step)
                eval_stats = evaluate_target(
                    env, pursuer_agent, target_agent, config,
                    num_episodes=3 if smoke else 10,
                    seeds=[0],
                )
                eval_return = eval_stats.get("mean_return", 0.0)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    target_agent.save(best_path)
                    logger.info("New best model: return=%.2f", best_eval_return)

            # Logging
            wall_time = time.time() - t_start
            csv_writer.writerow([
                global_step,
                update_stats.get("policy_loss", 0.0),
                update_stats.get("value_loss", 0.0),
                update_stats.get("entropy", 0.0),
                update_stats.get("approx_kl", 0.0),
                update_stats.get("clip_fraction", 0.0),
                update_stats.get("explained_variance", 0.0),
                eval_stats.get("mean_return", 0.0),
                eval_stats.get("survival_rate", 0.0),
                eval_stats.get("capture_rate", 0.0),
                eval_stats.get("crash_rate", 0.0),
                eval_stats.get("timeout_rate", 0.0),
                eval_stats.get("oob_rate", 0.0),
                wall_time,
            ])
            csv_file.flush()

            # SwanLab logging
            if swanlab_logger is not None:
                swanlab_log = {
                    f"{log_prefix}policy_loss": update_stats.get("policy_loss"),
                    f"{log_prefix}value_loss": update_stats.get("value_loss"),
                    f"{log_prefix}entropy": update_stats.get("entropy"),
                    f"{log_prefix}approx_kl": update_stats.get("approx_kl"),
                    f"{log_prefix}clip_fraction": update_stats.get("clip_fraction"),
                    f"{log_prefix}explained_variance": update_stats.get("explained_variance"),
                    f"{log_prefix}wall_time_s": wall_time,
                }
                if eval_stats:
                    swanlab_log.update({
                        f"{log_prefix}eval_return": eval_stats.get("mean_return"),
                        f"{log_prefix}eval_survival_rate": eval_stats.get("survival_rate"),
                        f"{log_prefix}eval_capture_rate": eval_stats.get("capture_rate"),
                        f"{log_prefix}eval_crash_rate": eval_stats.get("crash_rate"),
                        f"{log_prefix}eval_timeout_rate": eval_stats.get("timeout_rate"),
                        f"{log_prefix}eval_oob_rate": eval_stats.get("oob_rate"),
                        f"{log_prefix}eval_mean_final_range_m": eval_stats.get("mean_final_range_m"),
                        f"{log_prefix}eval_mean_min_range_m": eval_stats.get("mean_min_range_m"),
                    })
                swanlab_logger.log({k: v for k, v in swanlab_log.items() if v is not None}, step=global_step)

            logger.info(
                "Step %d/%d | loss=%.4f ent=%.4f kl=%.4f | "
                "eval_ret=%.2f surv=%.2f capt=%.2f crash=%.2f oob=%.2f timeout=%.2f | wall=%.0fs",
                global_step, total_steps,
                update_stats.get("policy_loss", 0.0),
                update_stats.get("entropy", 0.0),
                update_stats.get("approx_kl", 0.0),
                eval_stats.get("mean_return", 0.0),
                eval_stats.get("survival_rate", 0.0),
                eval_stats.get("capture_rate", 0.0),
                eval_stats.get("crash_rate", 0.0),
                eval_stats.get("oob_rate", 0.0),
                eval_stats.get("timeout_rate", 0.0),
                wall_time,
            )

            # Periodic checkpoint
            if global_step % save_interval == 0:
                last_path = os.path.join(ckpt_dir, "last.pt")
                target_agent.save(last_path)
                # Save training state
                with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                    json.dump({
                        "step": global_step,
                        "best_eval_return": best_eval_return,
                    }, f)

        # ---- Final save ----
        final_path = os.path.join(ckpt_dir, "last.pt")
        target_agent.save(final_path)
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"step": global_step, "best_eval_return": best_eval_return}, f)

        logger.info("Training complete. Total steps: %d, wall time: %.0fs", global_step, time.time() - t_start)
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train adversarial target agent against frozen pursuer."
    )
    parser.add_argument(
        "--config", type=str,
        default="config/adversarial/train_target.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run a minimal smoke test.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device override (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint.",
    )
    parser.add_argument(
        "--pursuer-checkpoint", type=str, default=None,
        help="Override pursuer checkpoint path.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device is not None:
        if "target_ppo" not in config:
            config["target_ppo"] = {}
        config["target_ppo"]["device"] = args.device
    if args.pursuer_checkpoint is not None:
        config["pursuer_checkpoint"] = args.pursuer_checkpoint

    # Set global random seed
    seed = config.get("experiment", {}).get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    output_dir = args.output_dir or config.get("output_dir",
                                                "outputs/adversarial/training/target_agent_ppo")

    train_target(config, output_dir, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
