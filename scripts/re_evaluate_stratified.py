#!/usr/bin/env python3
"""Scenario-stratified re-evaluation of JSBSim checkpoints.

Fixes the ceiling effect of the original eval by forcing equal samples per
scenario and reporting per-scenario success rates.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.seed import set_seed


def load_config_snapshot(snapshot_path):
    with open(snapshot_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(env, agent, scenario, seed, max_steps, save_trajectory=False):
    """Run a single deterministic episode."""
    set_seed(seed)
    obs = env.reset(scenario=scenario, seed=seed)
    ep_reward = 0.0
    ep_length = 0
    min_range = float("inf")
    final_range = 0.0
    reason = "timeout"
    trajectory = []

    for step in range(max_steps):
        obs_vec = obs["observation_vector"]
        action = agent.get_deterministic_action(obs_vec)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1

        rel_state = obs.get("relative_state", {})
        range_m = rel_state.get("range_m", 0.0)
        min_range = min(min_range, range_m)
        final_range = range_m
        reason = info.get("reason", "timeout")

        if save_trajectory:
            trajectory.append({
                "step": step,
                "range_m": range_m,
                "reward": reward,
                "reason": reason,
            })

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "return": ep_reward,
        "length": ep_length,
        "min_range_m": min_range,
        "final_range_m": final_range,
        "reason": reason,
        "is_success": reason == "success",
        "is_crash": reason == "crash",
        "is_timeout": reason == "timeout",
        "is_out_of_bounds": reason == "out_of_bounds",
        "trajectory": trajectory,
    }


def evaluate_checkpoint(
    config,
    checkpoint_path,
    scenarios,
    n_seeds,
    episodes_per_scenario,
    save_trajectories=False,
    device=None,
):
    """Evaluate one checkpoint with scenario-stratified sampling."""
    if device is not None:
        if "ppo" not in config:
            config["ppo"] = {}
        config["ppo"]["device"] = device

    env = CloseRangeTrackingEnv(config)
    max_steps = env.max_steps

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))

    agent = PPOAgent(
        obs_dim=obs_dim, action_dim=action_dim, config=config, device=device or "cpu"
    )
    agent.load(checkpoint_path)

    results = {}
    all_episodes = []

    for scenario_name, scenario in scenarios.items():
        scenario_episodes = []
        for eval_seed in range(n_seeds):
            for ep in range(episodes_per_scenario):
                ep_seed = eval_seed * 100000 + ep
                ep_result = run_episode(
                    env,
                    agent,
                    scenario,
                    ep_seed,
                    max_steps,
                    save_trajectory=save_trajectories,
                )
                ep_result["scenario"] = scenario_name
                ep_result["eval_seed"] = eval_seed
                ep_result["episode"] = ep
                scenario_episodes.append(ep_result)
                all_episodes.append(ep_result)

        success_count = sum(1 for e in scenario_episodes if e["is_success"])
        total = len(scenario_episodes)
        results[scenario_name] = {
            "num_episodes": total,
            "success_count": success_count,
            "success_rate": success_count / total if total > 0 else 0.0,
            "crash_rate": sum(1 for e in scenario_episodes if e["is_crash"]) / total,
            "out_of_bounds_rate": sum(
                1 for e in scenario_episodes if e["is_out_of_bounds"]
            )
            / total,
            "timeout_rate": sum(1 for e in scenario_episodes if e["is_timeout"]) / total,
            "mean_final_range_m": float(
                np.mean([e["final_range_m"] for e in scenario_episodes])
            ),
            "mean_length": float(np.mean([e["length"] for e in scenario_episodes])),
        }

    overall_success = sum(1 for e in all_episodes if e["is_success"])
    results["overall"] = {
        "num_episodes": len(all_episodes),
        "success_count": overall_success,
        "success_rate": overall_success / len(all_episodes) if all_episodes else 0.0,
        "mean_final_range_m": float(
            np.mean([e["final_range_m"] for e in all_episodes])
        ),
        "mean_length": float(np.mean([e["length"] for e in all_episodes])),
    }

    env.close()
    return results, all_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint path (can be given multiple times)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/jsbsim_re_eval_stratified"
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["favorable", "neutral", "challenging", "disadvantage"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--target-mode", default=None,
                        help="Override env.target_mode, e.g. sinusoidal, sinusoidal_weaving")
    parser.add_argument("--weaving-amplitude-g", type=float, default=None)
    parser.add_argument("--weaving-frequency-rad-s", type=float, default=None)
    args = parser.parse_args()

    checkpoints = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        checkpoints.extend(matches)
    checkpoints = sorted(set(checkpoints))
    print(f"Checkpoints to evaluate: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  {cp}")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "checkpoints": checkpoints,
        "n_seeds": args.n_seeds,
        "episodes_per_scenario": args.episodes_per_scenario,
        "scenarios": args.scenarios,
        "device": args.device,
        "results": {},
    }

    for checkpoint_path in checkpoints:
        cp_path = Path(checkpoint_path)
        seed_label = cp_path.parts[-3]
        print(f"\n{'=' * 60}")
        print(f"Evaluating {seed_label}")
        print(f"{'=' * 60}")

        config_snapshot_path = cp_path.parent.parent / "config_snapshot.yaml"
        if not config_snapshot_path.exists():
            print(
                f"  WARNING: config_snapshot.yaml not found at {config_snapshot_path}, skipping"
            )
            continue
        config = load_config_snapshot(config_snapshot_path)

        config["backend"] = "jsbsim"
        config["env"]["backend"] = "jsbsim"
        config["env"]["use_jsbsim"] = True

        scenarios = {
            s: config["scenarios"][s]
            for s in args.scenarios
            if s in config.get("scenarios", {})
        }
        if len(scenarios) != len(args.scenarios):
            missing = set(args.scenarios) - set(scenarios.keys())
            print(f"  WARNING: scenarios not found in config: {missing}")

        start = time.time()
        results, all_episodes = evaluate_checkpoint(
            config,
            checkpoint_path,
            scenarios,
            args.n_seeds,
            args.episodes_per_scenario,
            save_trajectories=args.save_trajectories,
            device=args.device,
        )
        elapsed = time.time() - start

        out_dir = Path(args.output_dir) / seed_label
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stratified_eval.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(out_dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "scenario",
                "eval_seed",
                "episode",
                "seed",
                "return",
                "length",
                "min_range_m",
                "final_range_m",
                "reason",
                "is_success",
                "is_crash",
                "is_timeout",
                "is_out_of_bounds",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in all_episodes:
                writer.writerow({k: ep.get(k, "") for k in fieldnames})

        summary["results"][seed_label] = results

        print(f"  Completed in {elapsed / 60:.1f} minutes")
        for s in args.scenarios:
            if s in results:
                r = results[s]
                print(
                    f"  {s:12s}: {r['success_count']:3d}/{r['num_episodes']:3d} = {r['success_rate']:.3f}"
                )
        print(
            f"  overall     : {results['overall']['success_count']:3d}/{results['overall']['num_episodes']:3d} = {results['overall']['success_rate']:.3f}"
        )

    with open(Path(args.output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
