#!/usr/bin/env python3
"""Fast 6-seed crossing evaluation (crossing_left/right only)."""
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


def evaluate(env, agent, scenario, n_eps=5):
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

    crossing_scenarios = {
        "crossing_left": {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [1500.0, 1500.0, 5200.0], "velocity_mps": 210.0, "heading_deg": 225.0},
        },
        "crossing_right": {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [1500.0, -1500.0, 5200.0], "velocity_mps": 210.0, "heading_deg": 135.0},
        },
    }

    all_results = []
    for method_name, cfg, ckpt_template in [
        ("baseline", baseline_cfg, "outputs/experiments/baseline_10seed_s{seed}/checkpoints/best.pt"),
        ("constrained", constrained_cfg, "outputs/experiments/constrained_10seed_s{seed}/checkpoints/best.pt"),
    ]:
        print(f"\n=== {method_name.upper()} ===")
        env = CloseRangeTrackingEnv(cfg)
        for seed in range(6):
            ckpt = ckpt_template.format(seed=seed)
            if not os.path.exists(ckpt):
                print(f"  SKIP: {ckpt}")
                continue
            agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
            agent.load(ckpt)
            for scen_name, scen in crossing_scenarios.items():
                sr = evaluate(env, agent, scen, n_eps=5)
                print(f"  s{seed} {scen_name}: {sr:.0%}")
                all_results.append({
                    "method": method_name,
                    "seed": seed,
                    "scenario": scen_name,
                    "sr": sr,
                })
            del agent
        env.close()

    print("\n" + "=" * 60)
    print("6-SEED CROSSING SUMMARY")
    print("=" * 60)
    for method in ["baseline", "constrained"]:
        for scen in ["crossing_left", "crossing_right"]:
            srs = [r["sr"] for r in all_results if r["method"] == method and r["scenario"] == scen]
            if srs:
                print(f"{method:12s} {scen:15s}: {np.mean(srs):.2%} ± {np.std(srs):.2%} (n={len(srs)})")

    os.makedirs("docs/results/10seed_evaluation", exist_ok=True)
    with open("docs/results/10seed_evaluation/summary_6seed_crossing.md", "w") as f:
        f.write("# 6-Seed Crossing Evaluation\n\n")
        f.write("| Method | Scenario | Mean SR | Std | n seeds |\n")
        f.write("|--------|----------|---------|-----|---------|\n")
        for method in ["baseline", "constrained"]:
            for scen in ["crossing_left", "crossing_right"]:
                srs = [r["sr"] for r in all_results if r["method"] == method and r["scenario"] == scen]
                if srs:
                    f.write(f"| {method} | {scen} | {np.mean(srs):.2%} | {np.std(srs):.2%} | {len(srs)} |\n")
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 6 seeds, 5 episodes per scenario.\n")

    with open("docs/results/10seed_evaluation/raw_6seed_crossing.json", "w") as f:
        import json
        json.dump(all_results, f, indent=2)
    print("\nSaved to docs/results/10seed_evaluation/")


if __name__ == "__main__":
    main()
