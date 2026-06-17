#!/usr/bin/env python3
"""
Phase 2: Adversarial Fine-Tuning of Pursuer against Trained Target.

Loads a pre-trained target agent (frozen) and a pre-trained pursuer agent,
then fine-tunes the pursuer to improve capture performance against the
intelligent evader.

Usage:
    # Smoke test
    python scripts/train_adversarial_pursuer.py \\
        --config config/adversarial/train_pursuer.yaml --smoke

    # Full fine-tuning (300K steps)
    python scripts/train_adversarial_pursuer.py \\
        --config config/adversarial/train_pursuer.yaml

    # Resume
    python scripts/train_adversarial_pursuer.py \\
        --config config/adversarial/train_pursuer.yaml --resume
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
logger = logging.getLogger("train_adversarial_pursuer")


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


def evaluate_pursuer(
    env: AdversarialJSBSimEnv,
    pursuer_agent: PPOAgent,
    target_agent: AdversarialTargetAgent,
    config: Dict[str, Any],
    num_episodes: int = 10,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate pursuer against frozen target agent."""
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
            reason: str = "timeout"

            for step in range(env.max_steps):
                p_action = pursuer_agent.get_deterministic_action(p_obs["observation_vector"])
                t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])

                (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = env.step(
                    p_action, t_action
                )
                ep_return += p_rew
                ep_length += 1

                rel = p_obs.get("relative_state", {})
                range_m = float(rel.get("range_m", 0.0))
                min_range = min(min_range, range_m)

                if terminated or truncated:
                    reason = info.get("termination", {}).get("reason", "unknown")
                    break

            episodes.append({
                "seed": seed,
                "episode": ep,
                "return": ep_return,
                "length": ep_length,
                "min_range_m": min_range,
                "reason": reason,
                "is_success": reason == "success",
            })

    returns = [e["return"] for e in episodes]
    success_count = sum(1 for e in episodes if e["is_success"])

    return {
        "num_episodes": len(episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "capture_rate": success_count / max(1, len(episodes)),
        "mean_min_range_m": float(np.mean([e["min_range_m"] for e in episodes])),
    }


def train_pursuer(
    config: Dict[str, Any],
    output_dir: str,
    smoke: bool = False,
    resume: bool = False,
) -> None:
    """Fine-tune pursuer against frozen adversarial target."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    training_cfg = config.get("training", {})
    total_steps = 5000 if smoke else int(training_cfg.get("total_timesteps", 300000))
    eval_interval = int(training_cfg.get("eval_interval", 10000))
    save_interval = int(training_cfg.get("save_interval", 50000))

    # ---- Create environment ----
    logger.info("Creating adversarial environment...")
    env = AdversarialJSBSimEnv(config)

    csv_file = None
    try:
        # ---- Create scenario sampler ----
        sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

        # ---- Load FROZEN target agent ----
        target_ckpt = config.get("target_checkpoint",
                                 "outputs/adversarial/training/target_agent_ppo/checkpoints/best.pt")
        target_cfg = dict(config)
        target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
        target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))

        logger.info("Loading target agent from: %s", target_ckpt)
        target_agent = AdversarialTargetAgent(config=target_cfg)
        if os.path.exists(target_ckpt):
            target_agent.load(target_ckpt)
        else:
            logger.warning("Target checkpoint not found! Using random target policy.")
        target_agent.eval()

        # ---- Load pre-trained pursuer ----
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
            logger.warning("Pursuer checkpoint not found! Starting from scratch.")

        # Resume
        start_step = 0
        best_capture_rate = 0.0
        resume_path = os.path.join(ckpt_dir, "last.pt")
        if resume and os.path.exists(resume_path):
            pursuer_agent.load(resume_path)
            state_path = os.path.join(ckpt_dir, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                start_step = state.get("step", 0)
                best_capture_rate = state.get("best_capture_rate", 0.0)

        # ---- CSV logging ----
        log_path = os.path.join(log_dir, "adversarial_training_log.csv")
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "policy_loss", "value_loss", "entropy", "approx_kl",
            "capture_rate", "mean_return", "wall_time_s",
        ])

        # ---- Training loop ----
        rollout_steps = int(config.get("ppo", {}).get("rollout_steps", 1024))
        scenario = sampler.sample() if sampler else None
        p_obs, t_obs = env.reset(scenario=scenario, seed=config["experiment"]["seed"])
        global_step = start_step
        t_start = time.time()

        logger.info("Starting adversarial fine-tuning: total=%d rollout=%d", total_steps, rollout_steps)

        while global_step < total_steps:
            for _ in range(rollout_steps):
                # Pursuer: stochastic action for training
                p_obs_vec = p_obs["observation_vector"]
                p_action, p_log_prob, p_value = pursuer_agent.select_action(
                    p_obs_vec, deterministic=False, store=False
                )

                # Target: deterministic (frozen)
                t_action = target_agent.get_deterministic_action(t_obs["observation_vector"])

                (p_obs, t_obs, p_rew, t_rew, terminated, truncated, info) = env.step(
                    p_action, t_action
                )

                pursuer_agent.store_transition(
                    obs=p_obs_vec,
                    action=p_action,
                    log_prob=p_log_prob,
                    reward=p_rew,
                    done=terminated or truncated,
                    value=p_value,
                )

                if terminated or truncated:
                    scenario = sampler.sample() if sampler else None
                    p_obs, t_obs = env.reset(scenario=scenario)

            global_step += rollout_steps

            # PPO update
            next_obs = p_obs["observation_vector"]
            update_stats = pursuer_agent.update(next_obs)

            # Evaluation
            eval_stats = {}
            if global_step % eval_interval == 0 or global_step >= total_steps:
                eval_stats = evaluate_pursuer(
                    env, pursuer_agent, target_agent, config,
                    num_episodes=3 if smoke else 10,
                    seeds=[0],
                )
                cr = eval_stats.get("capture_rate", 0.0)
                if cr > best_capture_rate:
                    best_capture_rate = cr
                    pursuer_agent.save(os.path.join(ckpt_dir, "best.pt"))
                    logger.info("New best model: capture_rate=%.2f", cr)

            wall_time = time.time() - t_start
            csv_writer.writerow([
                global_step,
                update_stats.get("policy_loss", 0.0),
                update_stats.get("value_loss", 0.0),
                update_stats.get("entropy", 0.0),
                update_stats.get("approx_kl", 0.0),
                eval_stats.get("capture_rate", 0.0),
                eval_stats.get("mean_return", 0.0),
                wall_time,
            ])
            csv_file.flush()

            logger.info(
                "Step %d/%d | loss=%.4f ent=%.4f | capt=%.2f ret=%.2f | wall=%.0fs",
                global_step, total_steps,
                update_stats.get("policy_loss", 0.0),
                update_stats.get("entropy", 0.0),
                eval_stats.get("capture_rate", 0.0),
                eval_stats.get("mean_return", 0.0),
                wall_time,
            )

            if global_step % save_interval == 0:
                pursuer_agent.save(os.path.join(ckpt_dir, "last.pt"))
                with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
                    json.dump({"step": global_step, "best_capture_rate": best_capture_rate}, f)

        # Final save
        pursuer_agent.save(os.path.join(ckpt_dir, "last.pt"))
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"step": global_step, "best_capture_rate": best_capture_rate}, f)

        logger.info("Training complete. Steps=%d wall=%.0fs", global_step, time.time() - t_start)
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune pursuer against adversarial target.")
    parser.add_argument("--config", type=str, default="config/adversarial/train_pursuer.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-checkpoint", type=str, default=None)
    parser.add_argument("--pursuer-checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device is not None:
        config.setdefault("ppo", {})["device"] = args.device
    if args.target_checkpoint is not None:
        config["target_checkpoint"] = args.target_checkpoint
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
                                                "outputs/adversarial/training/pursuer_adversarial")
    train_pursuer(config, output_dir, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
