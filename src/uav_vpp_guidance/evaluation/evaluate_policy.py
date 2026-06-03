"""
Policy evaluation runner.

Loads a trained PPO checkpoint and evaluates it on specified scenarios.
Supports both simple and JSBSim backends.

Usage:
    python -m uav_vpp_guidance.evaluation.evaluate_policy \
        --config config/experiment/train_no_prediction_vpp_ppo.yaml \
        --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
        --backend simple \
        --episodes 10 \
        --seeds 0 1 2 \
        --save-trajectories

    python -m uav_vpp_guidance.evaluation.evaluate_policy \
        --config config/experiment/train_no_prediction_vpp_ppo.yaml \
        --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt \
        --backend jsbsim \
        --episodes 2 \
        --seeds 0 \
        --save-trajectories
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

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


def evaluate_policy(env, agent, config, num_episodes=10, seeds=None, save_trajectories=False, output_dir=None):
    """
    Evaluate a trained policy across multiple seeds and episodes.

    Returns:
        dict: Aggregated metrics and per-episode results.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    all_episodes = []
    per_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        seed_episodes = []

        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenario = sample_scenario(config, rng)
            obs = env.reset(scenario=scenario, seed=ep_seed)

            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            min_ata_deg = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"
            trajectory = []
            # Predictor health counters
            ep_pred_valid_steps = 0
            ep_fallback_steps = 0
            ep_warmup_fallback_steps = 0
            ep_runtime_fallback_steps = 0
            ep_post_warmup_fallback_steps = 0
            ep_predictor_init_failed_steps = 0
            ep_prediction_errors = []
            ep_prediction_error_count = 0

            for step in range(env.max_steps):
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                aa_deg = float(np.rad2deg(rel_state.get("aa_rad", 0.0)))
                min_range = min(min_range, range_m)
                min_ata_deg = min(min_ata_deg, ata_deg)
                final_range = range_m
                final_ata = ata_deg

                if save_trajectories and output_dir is not None:
                    own_s = info.get("own_state", {})
                    target_s = info.get("target_state", {})
                    own_pos = own_s.get("position_m", own_s.get("position_neu", np.full(3, np.nan)))
                    target_pos = target_s.get("position_m", target_s.get("position_neu", np.full(3, np.nan)))

                    trajectory.append({
                        "step": step,
                        "time": step * env.env_config.get("high_level_dt", 0.2),
                        "backend": env._backend,
                        "ego_x": float(own_pos[0]) if len(own_pos) > 0 else np.nan,
                        "ego_y": float(own_pos[1]) if len(own_pos) > 1 else np.nan,
                        "ego_z": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
                        "target_x": float(target_pos[0]) if len(target_pos) > 0 else np.nan,
                        "target_y": float(target_pos[1]) if len(target_pos) > 1 else np.nan,
                        "target_z": float(target_pos[2]) if len(target_pos) > 2 else np.nan,
                        "range_m": range_m,
                        "ata_deg": ata_deg,
                        "aspect_deg": aa_deg,
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_z": float(action[2]),
                        "reward": reward,
                        "done": terminated or truncated,
                        "termination_reason": info.get("reason", ""),
                    })

                # Collect predictor health per step
                if info.get("prediction_enabled", False):
                    if info.get("prediction_valid", False):
                        ep_pred_valid_steps += 1
                    if info.get("fallback", False) or info.get("prediction_fallback_reason") is not None:
                        ep_fallback_steps += 1
                        phase = info.get("prediction_fallback_phase")
                        if phase == "warmup":
                            ep_warmup_fallback_steps += 1
                        elif phase == "runtime_failure":
                            ep_runtime_fallback_steps += 1
                        if phase != "warmup":
                            ep_post_warmup_fallback_steps += 1
                    if info.get("predictor_init_failed", False):
                        ep_predictor_init_failed_steps += 1
                    pred_err = info.get("prediction_error_m", np.nan)
                    if np.isfinite(pred_err):
                        ep_prediction_errors.append(float(pred_err))
                        ep_prediction_error_count += 1

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            ep_len = max(1, ep_length)
            ep_result = {
                "episode": ep,
                "seed": seed,
                "return": ep_reward,
                "length": ep_length,
                "min_range_m": min_range,
                "min_ata_deg": min_ata_deg,
                "final_range_m": final_range,
                "final_ata_deg": final_ata,
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
                "prediction_valid_rate": ep_pred_valid_steps / ep_len,
                "fallback_rate": ep_fallback_steps / ep_len,
                "post_warmup_fallback_rate": ep_post_warmup_fallback_steps / ep_len,
                "warmup_fallback_rate": ep_warmup_fallback_steps / ep_len,
                "runtime_fallback_rate": ep_runtime_fallback_steps / ep_len,
                "predictor_init_failed_count": ep_predictor_init_failed_steps,
                "mean_prediction_error_m": float(np.mean(ep_prediction_errors)) if ep_prediction_errors else np.nan,
                "median_prediction_error_m": float(np.median(ep_prediction_errors)) if ep_prediction_errors else np.nan,
                "prediction_error_count": ep_prediction_error_count,
            }
            seed_episodes.append(ep_result)
            all_episodes.append(ep_result)

            if save_trajectories and output_dir is not None and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

        per_seed_results[f"seed_{seed}"] = seed_episodes

    returns = [e["return"] for e in all_episodes]
    lengths = [e["length"] for e in all_episodes]
    final_ranges = [e["final_range_m"] for e in all_episodes]
    final_atas = [e["final_ata_deg"] for e in all_episodes]
    min_ranges = [e["min_range_m"] for e in all_episodes]

    def _safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else np.nan

    metrics = {
        "num_episodes": len(all_episodes),
        "num_seeds": len(seeds),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "success_rate": sum(1 for e in all_episodes if e["is_success"]) / max(1, len(all_episodes)),
        "crash_rate": sum(1 for e in all_episodes if e["is_crash"]) / max(1, len(all_episodes)),
        "out_of_bounds_rate": sum(1 for e in all_episodes if e["is_out_of_bounds"]) / max(1, len(all_episodes)),
        "timeout_rate": sum(1 for e in all_episodes if e["is_timeout"]) / max(1, len(all_episodes)),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else 0.0,
        "mean_final_ata_deg": float(np.mean(final_atas)) if final_atas else 0.0,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else 0.0,
        "prediction_valid_rate": _safe_mean([e["prediction_valid_rate"] for e in all_episodes]),
        "fallback_rate": _safe_mean([e["fallback_rate"] for e in all_episodes]),
        "post_warmup_fallback_rate": _safe_mean([e["post_warmup_fallback_rate"] for e in all_episodes]),
        "warmup_fallback_rate": _safe_mean([e["warmup_fallback_rate"] for e in all_episodes]),
        "runtime_fallback_rate": _safe_mean([e["runtime_fallback_rate"] for e in all_episodes]),
        "predictor_init_failed_count": sum(e["predictor_init_failed_count"] for e in all_episodes),
        "mean_prediction_error_m": _safe_mean([e["mean_prediction_error_m"] for e in all_episodes]),
        "median_prediction_error_m": _safe_mean([e["median_prediction_error_m"] for e in all_episodes]),
        "prediction_error_count": sum(e["prediction_error_count"] for e in all_episodes),
        "episodes": all_episodes,
        "per_seed": per_seed_results,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--save-trajectories", action="store_true", help="Save per-episode trajectory CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    # Override backend
    config["backend"] = args.backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = args.backend
    config["env"]["use_jsbsim"] = (args.backend == "jsbsim")

    # Ensure trajectory prediction is disabled
    if config.get("trajectory_prediction", {}).get("enabled", False):
        print("WARNING: trajectory_prediction.enabled is True! Forcing to False.")
        config["trajectory_prediction"]["enabled"] = False

    exp_name = config.get("experiment", {}).get("name", "no_prediction_vpp_ppo")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            "outputs", "tables", exp_name, args.backend
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Backend: {args.backend}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes} x {len(args.seeds)} seeds")

    # Environment
    env = CloseRangeTrackingEnv(config)
    print(f"Environment backend: {env._backend}")

    # Get dimensions
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))

    # Agent
    device = config.get("ppo", {}).get("device", "cpu")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Evaluate
    metrics = evaluate_policy(
        env, agent, config,
        num_episodes=args.episodes,
        seeds=args.seeds,
        save_trajectories=args.save_trajectories,
        output_dir=output_dir,
    )

    env.close()

    # Save metrics
    json_path = os.path.join(output_dir, "policy_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics JSON saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "policy_metrics.csv")
    scalar_keys = [
        "num_episodes", "num_seeds", "mean_return", "std_return",
        "success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        "mean_length", "mean_final_range_m", "mean_final_ata_deg", "mean_min_range_m",
        "prediction_valid_rate", "fallback_rate", "post_warmup_fallback_rate",
        "warmup_fallback_rate", "runtime_fallback_rate",
        "predictor_init_failed_count", "mean_prediction_error_m",
        "median_prediction_error_m", "prediction_error_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerow({k: metrics.get(k, "") for k in scalar_keys})
    print(f"Metrics CSV saved to: {csv_path}")

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"  Episodes: {metrics['num_episodes']}, Seeds: {metrics['num_seeds']}")
    print(f"  Mean return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Success rate: {metrics['success_rate']:.3f}")
    print(f"  Crash rate: {metrics['crash_rate']:.3f}")
    print(f"  OOB rate: {metrics['out_of_bounds_rate']:.3f}")
    print(f"  Timeout rate: {metrics['timeout_rate']:.3f}")
    print(f"  Mean final range: {metrics['mean_final_range_m']:.2f} m")
    print(f"  Mean final ATA: {metrics['mean_final_ata_deg']:.2f} deg")


if __name__ == "__main__":
    main()
