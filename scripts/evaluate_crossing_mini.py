#!/usr/bin/env python3
"""Minimal crossing generalization: 8 key variants, 3 episodes each."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_cfg(path):
    c = load_yaml_config(path)
    incs = c.pop("includes", [])
    m = {}
    for i in incs:
        p = os.path.join(os.path.dirname(path), i)
        if os.path.exists(p):
            m = merge_config(m, load_yaml_config(p))
    return merge_config(m, c)


def make_scenario(range_m, lateral_m, target_heading, speed_ratio=1.0):
    target_speed = 210.0
    ego_speed = target_speed * speed_ratio
    target_x = np.sqrt(max(0, range_m**2 - lateral_m**2))
    return {
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": float(ego_speed), "heading_deg": 0.0},
        "target_init": {"position_m": [float(target_x), float(lateral_m), 5200.0], "velocity_mps": target_speed, "heading_deg": float(target_heading)},
    }


def evaluate(env, agent, scenario, n_eps=3):
    succ = 0
    for ep in range(n_eps):
        obs = env.reset(scenario=scenario, seed=ep)
        for step in range(env.max_steps):
            action = agent.get_deterministic_action(obs["observation_vector"])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("reason") == "success":
                    succ += 1
                break
    return succ / n_eps


def main():
    baseline_cfg = load_cfg("config/experiment/train_no_prediction_vpp_ppo.yaml")
    constrained_cfg = load_cfg("config/experiment/stage6f5_feasible_geometry_constrained.yaml")
    baseline_cfg["ppo"]["device"] = "cpu"
    constrained_cfg["ppo"]["device"] = "cpu"

    # Key variants around the known crossing geometry
    variants = [
        ("near", 1000, 1000, 225),
        ("near", 1000, -1000, 135),
        ("std", 1500, 1500, 225),
        ("std", 1500, -1500, 135),
        ("far", 2500, 2000, 225),
        ("far", 2500, -2000, 135),
        ("fast_target", 1500, 1500, 225, 0.8),  # ego slower
        ("slow_target", 1500, 1500, 225, 1.2),  # ego faster
    ]

    scenarios = {}
    for name, r, y, h, *sr in variants:
        s = sr[0] if sr else 1.0
        key = f"{name}_r{r}_y{y}_h{h}_s{s}"
        scenarios[key] = make_scenario(r, y, h, s)

    print(f"Evaluating {len(scenarios)} crossing variants, 3 episodes each")

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
            results = {}
            for key, scen in scenarios.items():
                sr = evaluate(env, agent, scen, n_eps=3)
                results[key] = sr
            mean_sr = np.mean(list(results.values()))
            min_sr = np.min(list(results.values()))
            print(f"  Seed {seed}: mean={mean_sr:.2%}, min={min_sr:.2%}")
            all_results.append({"method": method_name, "seed": seed, "mean": mean_sr, "min": min_sr, "detail": results})
            del agent
        env.close()

    print("\n" + "=" * 50)
    print("CROSSING MINI-GENERALIZATION")
    print("=" * 50)
    for method in ["baseline", "constrained"]:
        means = [r["mean"] for r in all_results if r["method"] == method]
        mins = [r["min"] for r in all_results if r["method"] == method]
        if means:
            print(f"{method:12s}: mean={np.mean(means):.2%} ± {np.std(means):.2%}, worst={np.mean(mins):.2%}")

    os.makedirs("docs/results/crossing_generalization", exist_ok=True)
    with open("docs/results/crossing_generalization/mini_summary.md", "w") as f:
        f.write("# Crossing Mini-Generalization (8 variants)\n\n")
        f.write("| Method | Mean SR | Worst-case | n seeds |\n")
        f.write("|--------|---------|------------|---------|\n")
        for method in ["baseline", "constrained"]:
            means = [r["mean"] for r in all_results if r["method"] == method]
            mins = [r["min"] for r in all_results if r["method"] == method]
            if means:
                f.write(f"| {method} | {np.mean(means):.2%} ± {np.std(means):.2%} | {np.mean(mins):.2%} | {len(means)} |\n")
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 3 seeds, 3 episodes, 8 variants.\n")
    print("Saved to docs/results/crossing_generalization/mini_summary.md")


if __name__ == "__main__":
    main()
