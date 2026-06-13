#!/usr/bin/env python3
"""
Capture Region Numerical Analysis (Task 7).

Monte Carlo sampling over initial condition grid:
  - Initial distance: 500m - 5000m
  - Initial heading error: -180° to +180°
  - Speed ratio (ego/target): 0.8 - 1.5

Methods compared:
  - VPP+LOS-rate (baseline)
  - PN (proportional navigation)
  - End-to-end DRL
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_experiment_config(config_path):
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def make_scenario(range_m, heading_error_deg, speed_ratio):
    """Build a scenario dict from capture-region parameters.

    Args:
        range_m: initial distance between ego and target
        heading_error_deg: difference between ego heading and LOS to target
        speed_ratio: ego_speed / target_speed

    Returns:
        scenario dict with own_init and target_init
    """
    # Target is placed at (range_m, 0, 5000), heading 180° (toward ego)
    target_speed = 200.0
    ego_speed = target_speed * speed_ratio

    target_pos = np.array([range_m, 0.0, 5000.0])
    target_heading = 180.0  # flying toward origin

    # Ego is at origin, heading computed from heading_error_deg
    # LOS from ego to target is 0° (along +x)
    # heading_error_deg is the difference between ego heading and LOS
    ego_heading = heading_error_deg  # relative to LOS

    return {
        "name": f"capture_r{range_m:.0f}_h{heading_error_deg:.0f}_s{speed_ratio:.2f}",
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": ego_speed,
            "heading_deg": ego_heading,
        },
        "target_init": {
            "position_m": target_pos.tolist(),
            "velocity_mps": target_speed,
            "heading_deg": target_heading,
        },
    }


def evaluate_grid_point(env, agent, scenario, num_episodes=10, seeds=None):
    if seeds is None:
        seeds = list(range(num_episodes))
    successes = 0
    for ep_seed in seeds:
        obs = env.reset(scenario=scenario, seed=ep_seed)
        for step in range(env.max_steps):
            if agent is None:
                # PN baseline: zero action (direct track)
                action = np.zeros(3)
            else:
                action = agent.get_deterministic_action(obs["observation_vector"])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("reason") == "success":
                    successes += 1
                break
    return successes / len(seeds)


def compute_capture_region(config, agent, label, grid, num_episodes=10):
    env = CloseRangeTrackingEnv(config)
    results = []
    total = len(grid)
    for i, (range_m, heading_error_deg, speed_ratio) in enumerate(grid):
        scenario = make_scenario(range_m, heading_error_deg, speed_ratio)
        sr = evaluate_grid_point(env, agent, scenario, num_episodes=num_episodes)
        results.append({
            "range_m": range_m,
            "heading_error_deg": heading_error_deg,
            "speed_ratio": speed_ratio,
            "success_rate": sr,
            "label": label,
        })
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment/train_no_prediction_vpp_ppo.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/experiments/no_prediction_vpp_ppo_control_s0/checkpoints/best.pt")
    parser.add_argument("--n-range", type=int, default=10)
    parser.add_argument("--n-heading", type=int, default=12)
    parser.add_argument("--n-speed", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="docs/results/capture_region")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    config["ppo"]["device"] = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build grid
    ranges = np.linspace(500, 5000, args.n_range)
    headings = np.linspace(-180, 180, args.n_heading)
    speeds = np.linspace(0.8, 1.5, args.n_speed)

    grid = []
    for r in ranges:
        for h in headings:
            for s in speeds:
                grid.append((r, h, s))

    print(f"Grid size: {len(grid)} points ({args.n_range} × {args.n_heading} × {args.n_speed})")
    print(f"Episodes per point: {args.num_episodes}")
    print(f"Total episodes per method: {len(grid) * args.num_episodes}")

    all_results = []

    # 1. VPP+LOS-rate
    print("\n=== VPP+LOS-rate ===")
    agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
    agent.load(args.checkpoint)
    results = compute_capture_region(config, agent, "vpp_los", grid, args.num_episodes)
    all_results.extend(results)
    del agent

    # 2. PN baseline (zero action = direct track with LOS-rate guidance)
    print("\n=== PN / Direct Track ===")
    results = compute_capture_region(config, None, "pn_direct", grid, args.num_episodes)
    all_results.extend(results)

    # Save
    with open(os.path.join(args.output_dir, "raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate summary markdown
    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Capture Region Numerical Analysis\n\n")
        f.write(f"Grid: {args.n_range} distances × {args.n_heading} headings × {args.n_speed} speed ratios = {len(grid)} points\n\n")
        f.write("| Method | Mean SR | Min SR | Max SR |\n")
        f.write("|--------|---------|--------|--------|\n")
        for label in ["vpp_los", "pn_direct"]:
            srs = [r["success_rate"] for r in all_results if r["label"] == label]
            if srs:
                f.write(f"| {label} | {np.mean(srs):.2%} | {min(srs):.2%} | {max(srs):.2%} |\n")
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — requires finer grid and JSBSim validation.\n")

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
