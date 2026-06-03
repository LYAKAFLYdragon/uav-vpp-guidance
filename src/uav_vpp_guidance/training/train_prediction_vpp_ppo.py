"""
Stage 6A: Classical CV/CA Prediction VPP PPO Training.

Trains a PPO policy to output virtual pursuit point offsets Δp
with optional Constant Velocity / Constant Acceleration trajectory prediction.

Usage:
    # Smoke test with CV
    python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
        --config config/experiment/train_vpp_ppo_cv.yaml --smoke

    # Smoke test with CA
    python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
        --config config/experiment/train_vpp_ppo_ca.yaml --smoke
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def load_experiment_config(config_path):
    """Load and merge experiment configuration with includes."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def sample_scenario(config, rng):
    """Sample a random scenario from config."""
    scenarios = config.get("scenarios", {})
    if not scenarios:
        return None
    name = rng.choice(list(scenarios.keys()))
    return scenarios[name]


def run_evaluation(env, agent, config, num_episodes=10, seeds=None, save_trajectories=False, output_dir=None):
    """
    Evaluate a trained policy.

    Returns:
        dict: Aggregated evaluation metrics.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    all_episodes = []
    for seed in seeds:
        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenario = sample_scenario(config, rng)
            obs = env.reset(scenario=scenario, seed=ep_seed)

            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"
            trajectory = []

            for step in range(env.max_steps):
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                min_range = min(min_range, range_m)
                final_range = range_m
                final_ata = ata_deg

                if save_trajectories and output_dir is not None:
                    trajectory.append({
                        "step": step,
                        "time": step * env.env_config.get("high_level_dt", 0.2),
                        "range_m": range_m,
                        "ata_deg": ata_deg,
                        "reward": reward,
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_z": float(action[2]),
                    })

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            ep_result = {
                "seed": seed,
                "episode": ep,
                "return": ep_reward,
                "length": ep_length,
                "min_range_m": min_range,
                "final_range_m": final_range,
                "final_ata_deg": final_ata,
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
            }
            all_episodes.append(ep_result)

            if save_trajectories and output_dir is not None and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories", "eval")
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"eval_seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

    returns = [e["return"] for e in all_episodes]
    lengths = [e["length"] for e in all_episodes]
    success_count = sum(1 for e in all_episodes if e["is_success"])
    crash_count = sum(1 for e in all_episodes if e["is_crash"])
    oob_count = sum(1 for e in all_episodes if e["is_out_of_bounds"])
    timeout_count = sum(1 for e in all_episodes if e["is_timeout"])
    final_ranges = [e["final_range_m"] for e in all_episodes]
    final_atas = [e["final_ata_deg"] for e in all_episodes]

    return {
        "num_episodes": len(all_episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": success_count / max(1, len(all_episodes)),
        "crash_rate": crash_count / max(1, len(all_episodes)),
        "out_of_bounds_rate": oob_count / max(1, len(all_episodes)),
        "timeout_rate": timeout_count / max(1, len(all_episodes)),
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else 0.0,
        "mean_final_ata_deg": float(np.mean(final_atas)) if final_atas else 0.0,
    }


def train_ppo(config, output_dir, smoke=False):
    """
    Main PPO training loop.

    Args:
        config (dict): Full experiment configuration.
        output_dir (str): Output directory for logs and checkpoints.
        smoke (bool): If True, run a minimal smoke test.
    """
    # Create output directories
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # Save config snapshot
    import yaml
    config_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Hyperparameters
    ppo_cfg = config.get("ppo", {})
    total_timesteps = int(ppo_cfg.get("total_timesteps", 200000))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
    eval_interval = int(config.get("evaluation", {}).get("eval_interval", 10000))
    save_interval = int(config.get("checkpoint", {}).get("save_interval", 10000))
    save_best = bool(config.get("checkpoint", {}).get("save_best", True))
    save_last = bool(config.get("checkpoint", {}).get("save_last", True))

    if smoke:
        total_timesteps = 512
        rollout_steps = 128
        eval_interval = 256
        save_interval = 256
        print("[SMOKE] Running smoke mode with reduced settings:")
        print(f"  total_timesteps={total_timesteps}, rollout_steps={rollout_steps}")

    # Environment
    env = CloseRangeTrackingEnv(config)
    backend = env._backend
    print(f"Backend: {backend}")

    # Get observation and action dimensions
    sample_obs = env.reset(seed=0)
    obs_vec = sample_obs["observation_vector"]
    obs_dim = int(obs_vec.shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Agent
    device = ppo_cfg.get("device", "cpu")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    print(f"Network parameters: {agent.network.count_parameters()}")

    # Training state
    global_step = 0
    episode_count = 0
    best_eval_return = -float("inf")

    # CSV loggers
    episode_log_path = os.path.join(log_dir, "episode_train_log.csv")
    update_log_path = os.path.join(log_dir, "update_train_log.csv")
    eval_log_path = os.path.join(log_dir, "eval_log.csv")

    episode_fieldnames = [
        "step", "episode", "episode_return", "episode_length",
        "success", "score_win", "crash", "out_of_bounds", "timeout",
        "mean_range", "final_range", "final_ata",
        "prediction_valid_rate", "fallback_rate", "predictor_init_failed_count",
    ]
    update_fieldnames = [
        "step", "update_num", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_fraction", "explained_variance", "learning_rate",
    ]
    eval_fieldnames = [
        "step", "num_episodes", "mean_return", "std_return",
        "success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        "mean_final_range_m", "mean_final_ata_deg",
    ]

    with open(episode_log_path, "w", newline="", encoding="utf-8") as f_ep:
        ep_writer = csv.DictWriter(f_ep, fieldnames=episode_fieldnames)
        ep_writer.writeheader()

        with open(update_log_path, "w", newline="", encoding="utf-8") as f_up:
            up_writer = csv.DictWriter(f_up, fieldnames=update_fieldnames)
            up_writer.writeheader()

            with open(eval_log_path, "w", newline="", encoding="utf-8") as f_eval:
                eval_writer = csv.DictWriter(f_eval, fieldnames=eval_fieldnames)
                eval_writer.writeheader()

                # Main training loop
                rng = np.random.default_rng(config.get("experiment", {}).get("seed", 0))
                obs = env.reset(seed=rng.integers(0, 1000000))

                episode_return = 0.0
                episode_length = 0
                episode_ranges = []
                episode_success = False
                episode_crash = False
                episode_oob = False
                episode_timeout = False
                episode_score_win = False
                episode_pred_valid_steps = 0
                episode_fallback_steps = 0
                episode_predictor_init_failed_steps = 0

                start_time = time.time()
                update_num = 0

                while global_step < total_timesteps:
                    # Collect rollout
                    for step in range(rollout_steps):
                        obs_vec = obs["observation_vector"]
                        action, log_prob, value = agent.select_action(obs_vec, deterministic=False, store=False)
                        agent.store_transition(obs_vec, action, log_prob, 0.0, False, value)

                        obs, reward, terminated, truncated, info = env.step(action)
                        global_step += 1
                        episode_return += reward
                        episode_length += 1

                        rel_state = obs.get("relative_state", {})
                        range_m = rel_state.get("range_m", 0.0)
                        episode_ranges.append(range_m)

                        # Predictor observability
                        if info.get("prediction_enabled", False):
                            if info.get("prediction_valid", False):
                                episode_pred_valid_steps += 1
                            if info.get("fallback", False) or info.get("prediction_fallback_reason") is not None:
                                episode_fallback_steps += 1
                            if info.get("predictor_init_failed", False):
                                episode_predictor_init_failed_steps += 1

                        # Update last stored transition with actual reward and done
                        agent.buffer.rewards[agent.buffer.ptr - 1] = float(reward)
                        agent.buffer.dones[agent.buffer.ptr - 1] = float(terminated or truncated)

                        if terminated or truncated:
                            # Episode ended
                            episode_count += 1
                            reason = info.get("reason", "unknown")
                            episode_success = reason == "success"
                            episode_crash = reason == "crash"
                            episode_oob = reason == "out_of_bounds"
                            episode_timeout = reason == "timeout"

                            # Compute score win
                            ego_score = info.get("ego_score", 0.0)
                            target_score = info.get("target_score", 0.0)
                            episode_score_win = ego_score > target_score

                            final_range = range_m
                            final_ata = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                            mean_range = float(np.mean(episode_ranges)) if episode_ranges else 0.0

                            ep_len = max(1, episode_length)
                            prediction_valid_rate = episode_pred_valid_steps / ep_len
                            fallback_rate = episode_fallback_steps / ep_len

                            # Log episode stats immediately
                            ep_row = {
                                "step": global_step,
                                "episode": episode_count,
                                "episode_return": episode_return,
                                "episode_length": episode_length,
                                "success": int(episode_success),
                                "score_win": int(episode_score_win),
                                "crash": int(episode_crash),
                                "out_of_bounds": int(episode_oob),
                                "timeout": int(episode_timeout),
                                "mean_range": mean_range,
                                "final_range": final_range,
                                "final_ata": final_ata,
                                "prediction_valid_rate": round(prediction_valid_rate, 4),
                                "fallback_rate": round(fallback_rate, 4),
                                "predictor_init_failed_count": episode_predictor_init_failed_steps,
                            }
                            ep_writer.writerow(ep_row)
                            f_ep.flush()

                            # Reset episode stats
                            episode_return = 0.0
                            episode_length = 0
                            episode_ranges = []
                            episode_pred_valid_steps = 0
                            episode_fallback_steps = 0
                            episode_predictor_init_failed_steps = 0

                            # Reset environment
                            scenario = sample_scenario(config, rng)
                            obs = env.reset(scenario=scenario, seed=rng.integers(0, 1000000))

                            # Check if buffer is full after this step
                            if agent.buffer.full:
                                break

                        if global_step >= total_timesteps:
                            break

                    # PPO update when buffer is full or training ended
                    if agent.buffer.full or (global_step >= total_timesteps and len(agent.buffer) > 0):
                        next_obs_vec = obs["observation_vector"]
                        update_stats = agent.update(next_obs=next_obs_vec)
                        update_num += 1

                        up_row = {
                            "step": global_step,
                            "update_num": update_num,
                            "policy_loss": update_stats.get("policy_loss", ""),
                            "value_loss": update_stats.get("value_loss", ""),
                            "entropy": update_stats.get("entropy", ""),
                            "approx_kl": update_stats.get("approx_kl", ""),
                            "clip_fraction": update_stats.get("clip_fraction", ""),
                            "explained_variance": update_stats.get("explained_variance", ""),
                            "learning_rate": update_stats.get("learning_rate", ""),
                        }
                        up_writer.writerow(up_row)
                        f_up.flush()

                        print(
                            f"Step {global_step}/{total_timesteps} | "
                            f"Ep {episode_count} | "
                            f"Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                            f"Value Loss: {update_stats.get('value_loss', 0):.4f} | "
                            f"Entropy: {update_stats.get('entropy', 0):.4f} | "
                            f"Explained Var: {update_stats.get('explained_variance', 0):.4f}"
                        )

                    # Evaluation
                    if eval_interval > 0 and global_step % eval_interval == 0 and global_step > 0:
                        print(f"\n--- Evaluation at step {global_step} ---")
                        eval_cfg = config.get("evaluation", {})
                        eval_metrics = run_evaluation(
                            env, agent, config,
                            num_episodes=eval_cfg.get("eval_episodes", 10),
                            seeds=eval_cfg.get("seeds", [0, 1, 2]),
                            save_trajectories=eval_cfg.get("save_trajectories", False),
                            output_dir=output_dir,
                        )
                        eval_row = {
                            "step": global_step,
                            "num_episodes": eval_metrics["num_episodes"],
                            "mean_return": eval_metrics["mean_return"],
                            "std_return": eval_metrics["std_return"],
                            "success_rate": eval_metrics["success_rate"],
                            "crash_rate": eval_metrics["crash_rate"],
                            "out_of_bounds_rate": eval_metrics["out_of_bounds_rate"],
                            "timeout_rate": eval_metrics["timeout_rate"],
                            "mean_final_range_m": eval_metrics["mean_final_range_m"],
                            "mean_final_ata_deg": eval_metrics["mean_final_ata_deg"],
                        }
                        eval_writer.writerow(eval_row)
                        f_eval.flush()

                        print(
                            f"Eval Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f} | "
                            f"Success: {eval_metrics['success_rate']:.2%} | "
                            f"Crash: {eval_metrics['crash_rate']:.2%} | "
                            f"OOB: {eval_metrics['out_of_bounds_rate']:.2%}"
                        )

                        # Save best checkpoint
                        if save_best and eval_metrics["mean_return"] > best_eval_return:
                            best_eval_return = eval_metrics["mean_return"]
                            best_path = os.path.join(checkpoint_dir, "best.pt")
                            agent.save(best_path)
                            print(f"  -> Saved best checkpoint (return={best_eval_return:.2f})")

                    # Periodic checkpoint save
                    if save_interval > 0 and global_step % save_interval == 0 and global_step > 0:
                        step_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
                        agent.save(step_path)

                # Save last checkpoint
                if save_last:
                    last_path = os.path.join(checkpoint_dir, "last.pt")
                    agent.save(last_path)
                    print(f"\nSaved last checkpoint to {last_path}")

                elapsed = time.time() - start_time
                print(f"\nTraining complete! Total steps: {global_step}, Episodes: {episode_count}, Time: {elapsed:.1f}s")

    env.close()

    # Smoke summary
    if smoke:
        tp_cfg = config.get("trajectory_prediction", {})
        # Aggregate predictor health from episode log if available
        pred_valid_rates = []
        fallback_rates = []
        init_failed_count = 0
        if os.path.exists(episode_log_path):
            try:
                with open(episode_log_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("prediction_valid_rate"):
                            pred_valid_rates.append(float(row["prediction_valid_rate"]))
                        if row.get("fallback_rate"):
                            fallback_rates.append(float(row["fallback_rate"]))
                        if row.get("predictor_init_failed_count"):
                            init_failed_count += int(row["predictor_init_failed_count"])
            except Exception:
                pass

        smoke_summary = {
            "smoke": True,
            "total_timesteps": global_step,
            "episodes": episode_count,
            "elapsed_seconds": elapsed,
            "backend": backend,
            "checkpoint_dir": checkpoint_dir,
            "episode_train_log": episode_log_path,
            "update_train_log": update_log_path,
            "eval_log": eval_log_path,
            "predictor_type": tp_cfg.get("predictor_type", "none"),
            "prediction_enabled": tp_cfg.get("enabled", False),
            "prediction_valid_rate": float(np.mean(pred_valid_rates)) if pred_valid_rates else None,
            "fallback_rate": float(np.mean(fallback_rates)) if fallback_rates else None,
            "predictor_init_failed": init_failed_count > 0,
        }
        smoke_path = os.path.join(log_dir, "smoke_summary.json")
        with open(smoke_path, "w", encoding="utf-8") as f:
            json.dump(smoke_summary, f, indent=2, ensure_ascii=False)
        print(f"Smoke summary saved to {smoke_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train CV/CA Prediction VPP PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (minimal training)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    set_seed(seed)

    exp_name = config.get("experiment", {}).get("name", "vpp_ppo_prediction")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "experiments",
            exp_name,
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {seed}")

    tp_enabled = config.get("trajectory_prediction", {}).get("enabled", False)
    predictor_type = config.get("trajectory_prediction", {}).get("predictor_type", "none")
    anchor_mode = config.get("virtual_point", {}).get("anchor_mode", "current_target")
    print(f"Anchor mode: {anchor_mode}")
    print(f"Trajectory prediction: {'enabled' if tp_enabled else 'disabled'} ({predictor_type})")

    train_ppo(config, output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
