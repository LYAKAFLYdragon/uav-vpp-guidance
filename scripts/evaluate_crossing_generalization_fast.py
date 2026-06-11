#!/usr/bin/env python3
"""
Fast Crossing Geometry Generalization Test (reduced grid for speed).

Reduced from 175 to ~54 scenarios: 2 ranges × 3 angles × 3 speeds × 3 offsets.
Each scenario: 3 episodes.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from itertools import product

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


def make_crossing_scenario(range_m, crossing_angle_deg, speed_ratio, lateral_offset_m):
    target_speed = 200.0
    ego_speed = target_speed * speed_ratio

    if lateral_offset_m == 0:
        target_x = range_m
        target_y = 0.0
    else:
        target_y = lateral_offset_m
        target_x = np.sqrt(max(0, range_m**2 - target_y**2))

    return {
        "name": f"cross_r{range_m:.0f}_a{crossing_angle_deg:.0f}_s{speed_ratio:.1f}_y{lateral_offset_m:.0f}",
        "own_init": {
            "position_m": [0.0, 0.0, 5000.0],
            "velocity_mps": ego_speed,
            "heading_deg": 0.0,
        },
        "target_init": {
            "position_m": [float(target_x), float(target_y), 5000.0],
            "velocity_mps": target_speed,
            "heading_deg": float(crossing_angle_deg),
        },
    }


def evaluate_on_scenarios(env, agent, scenarios, num_episodes=3):
    results = {}
    for scen_name, scenario in scenarios.items():
        successes = 0
        for ep in range(num_episodes):
            obs = env.reset(scenario=scenario, seed=ep)
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if info.get("reason") == "success":
                        successes += 1
                    break
        results[scen_name] = successes / num_episodes
    return results


def main():
    baseline_cfg = load_experiment_config("config/experiment/train_no_prediction_vpp_ppo.yaml")
    constrained_cfg = load_experiment_config("config/experiment/stage6f5_feasible_geometry_constrained.yaml")
    baseline_cfg["ppo"]["device"] = "cpu"
    constrained_cfg["ppo"]["device"] = "cpu"

    # Reduced grid
    ranges = [1000, 3000]
    angles = [90, 180, 270]
    speed_ratios = [0.8, 1.0, 1.2]
    lateral_offsets = [-2000, 0, 2000]

    scenarios = {}
    for r, a, s, y in product(ranges, angles, speed_ratios, lateral_offsets):
        scen = make_crossing_scenario(r, a, s, y)
        scenarios[scen["name"]] = scen

    print(f"Total crossing scenarios: {len(scenarios)}")

    all_results = []

    for method_name, cfg, ckpt_template in [
        ("baseline", baseline_cfg, "outputs/experiments/no_prediction_vpp_ppo_control_s{seed}/checkpoints/best.pt"),
        ("constrained", constrained_cfg, "outputs/experiments/crossing_constrained_s{seed}/checkpoints/best.pt"),
    ]:
        print(f"\n=== {method_name.upper()} ===")
        env = CloseRangeTrackingEnv(cfg)
        for seed in [0, 1, 2]:
            ckpt = ckpt_template.format(seed=seed)
            if not os.path.exists(ckpt):
                print(f"  SKIP: {ckpt}")
                continue
            agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
            agent.load(ckpt)
            results = evaluate_on_scenarios(env, agent, scenarios, num_episodes=3)
            mean_sr = np.mean(list(results.values()))
            min_sr = np.min(list(results.values()))
            max_sr = np.max(list(results.values()))
            print(f"  Seed {seed}: mean={mean_sr:.2%}, min={min_sr:.2%}, max={max_sr:.2%}")
            all_results.append({
                "method": method_name,
                "seed": seed,
                "mean_sr": mean_sr,
                "min_sr": min_sr,
                "max_sr": max_sr,
                "per_scenario": results,
            })
            del agent
        env.close()

    print("\n" + "=" * 60)
    print("CROSSING GENERALIZATION SUMMARY")
    print("=" * 60)
    for method in ["baseline", "constrained"]:
        mean_srs = [r["mean_sr"] for r in all_results if r["method"] == method]
        min_srs = [r["min_sr"] for r in all_results if r["method"] == method]
        if mean_srs:
            print(f"{method:12s}: mean={np.mean(mean_srs):.2%} ± {np.std(mean_srs):.2%}, "
                  f"worst-case={np.mean(min_srs):.2%} ± {np.std(min_srs):.2%}")

    os.makedirs("docs/results/crossing_generalization", exist_ok=True)
    with open("docs/results/crossing_generalization/summary_fast.md", "w") as f:
        f.write("# Crossing Geometry Generalization (Fast, Reduced Grid)\n\n")
        f.write(f"Total scenarios: {len(scenarios)} (2 ranges × 3 angles × 3 speeds × 3 offsets)\n")
        f.write("3 episodes per scenario.\n\n")
        f.write("| Method | Mean SR | Worst-case SR | n seeds |\n")
        f.write("|--------|---------|---------------|---------|\n")
        for method in ["baseline", "constrained"]:
            mean_srs = [r["mean_sr"] for r in all_results if r["method"] == method]
            min_srs = [r["min_sr"] for r in all_results if r["method"] == method]
            if mean_srs:
                f.write(f"| {method} | {np.mean(mean_srs):.2%} ± {np.std(mean_srs):.2%} | "
                       f"{np.mean(min_srs):.2%} ± {np.std(min_srs):.2%} | {len(mean_srs)} |\n")
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 3 seeds, 3 episodes per scenario, 54 scenarios.\n")
        f.write("Requires full grid and 10 seeds for `paper_safe`.\n")

    with open("docs/results/crossing_generalization/raw_results_fast.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to docs/results/crossing_generalization/")


if __name__ == "__main__":
    main()
