"""
ExpertVPPPolicy Evaluation Script

Evaluates the rule-driven ExpertVPP baseline across scenarios.

Usage:
    python -m uav_vpp_guidance.evaluation.evaluate_expert_vpp \
        --config config/experiment/expert_vpp_baseline.yaml \
        --backend simple \
        --episodes 10 --seeds 0 1 2 --save-trajectories

    python -m uav_vpp_guidance.evaluation.evaluate_expert_vpp \
        --config config/experiment/expert_vpp_baseline.yaml \
        --backend jsbsim \
        --episodes 2 --seeds 0
"""

import argparse
import csv
import json
import os
import numpy as np
from collections import Counter
from typing import List, Dict, Any

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.expert_system import ExpertVPPPolicy


def evaluate_expert_vpp(
    env: CloseRangeTrackingEnv,
    policy: ExpertVPPPolicy,
    num_episodes: int = 10,
    seeds: List[int] = None,
    save_trajectories: bool = False,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Evaluate ExpertVPPPolicy across multiple seeds and episodes.

    Returns:
        dict: Aggregated metrics and per-episode results.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    all_episodes = []
    per_seed_results = {}
    tactical_state_counts = Counter()
    intent_counts = Counter()
    rule_counts = Counter()
    fallback_count = 0
    unsafe_count = 0

    for seed in seeds:
        set_seed(seed)
        seed_episodes = []

        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenarios = env.config.get("scenarios", {})
            scenario = None
            if scenarios:
                scenario_name = rng.choice(list(scenarios.keys()))
                scenario = scenarios[scenario_name]

            obs = env.reset(scenario=scenario, seed=ep_seed)
            policy.reset_history()

            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            min_ata_deg = float("inf")
            min_aspect_deg = float("inf")
            final_range = 0.0
            final_ata = 0.0
            final_aspect = 0.0
            reason = "timeout"
            trajectory = []

            for step in range(env.max_steps):
                rel_state = obs.get("relative_state", {})
                own_state = obs.get("own_state", {})
                target_state = obs.get("target_state", {})
                action = policy.get_action(own_state, target_state, rel_state)
                diag = policy.get_last_diagnostics()

                # Accumulate expert diagnostics
                tactical_state_counts[diag.get("expert_tactical_state", "UNKNOWN")] += 1
                intent_counts[diag.get("expert_maneuver_intent", "UNKNOWN")] += 1
                rule_counts[diag.get("expert_rule_id", "UNKNOWN")] += 1
                if diag.get("expert_tactical_state") == "UNSAFE":
                    unsafe_count += 1
                if diag.get("expert_fallback_reason") is not None:
                    fallback_count += 1

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                aspect_deg = float(np.rad2deg(rel_state.get("aa_rad", 0.0)))
                min_range = min(min_range, range_m)
                min_ata_deg = min(min_ata_deg, ata_deg)
                min_aspect_deg = min(min_aspect_deg, aspect_deg)
                final_range = range_m
                final_ata = ata_deg
                final_aspect = aspect_deg

                if save_trajectories and output_dir is not None:
                    own_s = info.get("own_state", {})
                    target_s = info.get("target_state", {})
                    own_pos = own_s.get("position_m", own_s.get("position_neu", np.full(3, np.nan)))
                    target_pos = target_s.get("position_m", target_s.get("position_neu", np.full(3, np.nan)))

                    traj_entry = {
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
                        "aspect_deg": aspect_deg,
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_z": float(action[2]),
                        "reward": reward,
                        "done": terminated or truncated,
                        "termination_reason": info.get("reason", ""),
                    }
                    # Add expert diagnostics
                    traj_entry.update(diag)
                    trajectory.append(traj_entry)

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            ep_result = {
                "episode": ep,
                "seed": seed,
                "return": ep_reward,
                "length": ep_length,
                "min_range_m": min_range,
                "min_ata_deg": min_ata_deg,
                "min_aspect_deg": min_aspect_deg,
                "final_range_m": final_range,
                "final_ata_deg": final_ata,
                "final_aspect_deg": final_aspect,
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
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
    final_aspects = [e["final_aspect_deg"] for e in all_episodes]
    min_ranges = [e["min_range_m"] for e in all_episodes]
    min_atas = [e["min_ata_deg"] for e in all_episodes]
    min_aspects = [e["min_aspect_deg"] for e in all_episodes]

    total_steps = sum(lengths)
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
        "mean_final_aspect_deg": float(np.mean(final_aspects)) if final_aspects else 0.0,
        "mean_min_range_m": float(np.mean(min_ranges)) if min_ranges else 0.0,
        "mean_min_ata_deg": float(np.mean(min_atas)) if min_atas else 0.0,
        "mean_min_aspect_deg": float(np.mean(min_aspects)) if min_aspects else 0.0,
        "episodes": all_episodes,
        "per_seed": per_seed_results,
        # Expert-specific metrics
        "tactical_state_distribution": dict(tactical_state_counts),
        "maneuver_intent_distribution": dict(intent_counts),
        "rule_usage_count": dict(rule_counts),
        "fallback_rate": fallback_count / max(1, total_steps),
        "unsafe_rate": unsafe_count / max(1, total_steps),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ExpertVPPPolicy")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--save-trajectories", action="store_true", help="Save per-episode trajectory CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    includes = config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(args.config), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    config = merge_config(merged, config)

    # Override backend
    config["backend"] = args.backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = args.backend
    config["env"]["use_jsbsim"] = (args.backend == "jsbsim")

    # Ensure no prediction
    if config.get("trajectory_prediction", {}).get("enabled", False):
        print("WARNING: trajectory_prediction.enabled is True! Forcing to False.")
        config["trajectory_prediction"]["enabled"] = False

    exp_name = config.get("experiment", {}).get("name", "expert_vpp_baseline")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("outputs", "tables", exp_name, args.backend)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Backend: {args.backend}")
    print(f"Episodes: {args.episodes} x {len(args.seeds)} seeds")

    env = CloseRangeTrackingEnv(config)
    print(f"Environment backend: {env._backend}")

    expert_config = config.get("expert_vpp", {})
    policy = ExpertVPPPolicy(expert_config)
    print("ExpertVPPPolicy initialized")

    metrics = evaluate_expert_vpp(
        env=env,
        policy=policy,
        num_episodes=args.episodes,
        seeds=args.seeds,
        save_trajectories=args.save_trajectories,
        output_dir=output_dir,
    )

    env.close()

    # Save metrics JSON
    json_path = os.path.join(output_dir, "expert_vpp_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics JSON saved to: {json_path}")

    # Save metrics CSV
    csv_path = os.path.join(output_dir, "expert_vpp_metrics.csv")
    scalar_keys = [
        "num_episodes", "num_seeds", "mean_return", "std_return",
        "success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        "mean_length", "mean_final_range_m", "mean_final_ata_deg", "mean_min_range_m",
        "mean_final_aspect_deg", "mean_min_aspect_deg",
        "fallback_rate", "unsafe_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerow({k: metrics.get(k, "") for k in scalar_keys})
    print(f"Metrics CSV saved to: {csv_path}")

    # Print summary
    print("\n=== ExpertVPP Evaluation Summary ===")
    print(f"  Episodes: {metrics['num_episodes']}, Seeds: {metrics['num_seeds']}")
    print(f"  Mean return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Success rate: {metrics['success_rate']:.3f}")
    print(f"  Crash rate: {metrics['crash_rate']:.3f}")
    print(f"  OOB rate: {metrics['out_of_bounds_rate']:.3f}")
    print(f"  Timeout rate: {metrics['timeout_rate']:.3f}")
    print(f"  Mean final range: {metrics['mean_final_range_m']:.2f} m")
    print(f"  Mean final ATA: {metrics['mean_final_ata_deg']:.2f} deg")
    print(f"  Mean final aspect: {metrics['mean_final_aspect_deg']:.2f} deg")
    print(f"  Tactical state distribution: {metrics['tactical_state_distribution']}")
    print(f"  Maneuver intent distribution: {metrics['maneuver_intent_distribution']}")
    print(f"  Fallback rate: {metrics['fallback_rate']:.4f}")
    print(f"  Unsafe rate: {metrics['unsafe_rate']:.4f}")


if __name__ == "__main__":
    main()
