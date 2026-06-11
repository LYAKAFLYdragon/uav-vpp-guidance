#!/usr/bin/env python3
"""Evaluate all 10-seed models once training completes."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
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


def evaluate_checkpoint(ckpt_path, config, scenario_name, num_episodes=20, seeds=None):
    if seeds is None:
        seeds = list(range(num_episodes))
    env = CloseRangeTrackingEnv(config)
    agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
    agent.load(ckpt_path)
    scenarios = config.get("scenarios", {})
    scenario = scenarios.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Scenario {scenario_name} not found")
    
    successes = 0
    for ep_seed in seeds:
        obs = env.reset(scenario=scenario, seed=ep_seed)
        for step in range(env.max_steps):
            action = agent.get_deterministic_action(obs["observation_vector"])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("reason") == "success":
                    successes += 1
                break
    env.close()
    return successes / len(seeds)


def main():
    configs = {
        "baseline": load_experiment_config("config/experiment/train_no_prediction_vpp_ppo.yaml"),
        "constrained": load_experiment_config("config/experiment/stage6f5_feasible_geometry_constrained.yaml"),
    }
    for cfg in configs.values():
        cfg["ppo"]["device"] = "cpu"
    
    # Also test on true crossing scenarios
    crossing_scenarios = {
        "crossing_left": {
            "name": "crossing_left",
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [1500.0, 1500.0, 5200.0], "velocity_mps": 210.0, "heading_deg": 225.0},
        },
        "crossing_right": {
            "name": "crossing_right",
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [1500.0, -1500.0, 5200.0], "velocity_mps": 210.0, "heading_deg": 135.0},
        },
    }
    
    all_results = []
    
    for method_name, cfg in configs.items():
        print(f"\n=== {method_name.upper()} ===")
        for seed in range(10):
            ckpt = f"outputs/experiments/{method_name}_10seed_s{seed}/checkpoints/best.pt"
            if not os.path.exists(ckpt):
                print(f"  SKIP: {ckpt}")
                continue
            for scenario_name in ["favorable", "neutral", "disadvantage", "challenging"]:
                sr = evaluate_checkpoint(ckpt, cfg, scenario_name, num_episodes=20, seeds=list(range(20)))
                print(f"  s{seed} {scenario_name}: {sr:.2%}")
                all_results.append({"method": method_name, "seed": seed, "scenario": scenario_name, "sr": sr})
            
            # True crossing scenarios
            env = CloseRangeTrackingEnv(cfg)
            agent = PPOAgent(obs_dim=16, action_dim=3, config=cfg, device="cpu")
            agent.load(ckpt)
            for scen_name, scenario in crossing_scenarios.items():
                successes = 0
                for ep in range(20):
                    obs = env.reset(scenario=scenario, seed=ep)
                    for step in range(env.max_steps):
                        action = agent.get_deterministic_action(obs["observation_vector"])
                        obs, reward, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            if info.get("reason") == "success":
                                successes += 1
                            break
                sr = successes / 20
                print(f"  s{seed} {scen_name}: {sr:.2%}")
                all_results.append({"method": method_name, "seed": seed, "scenario": scen_name, "sr": sr})
            env.close()
            del agent
    
    # Aggregate
    print("\n" + "=" * 60)
    print("10-SEED SUMMARY")
    print("=" * 60)
    for method in ["baseline", "constrained"]:
        for scenario in ["favorable", "neutral", "disadvantage", "challenging", "crossing_left", "crossing_right"]:
            srs = [r["sr"] for r in all_results if r["method"] == method and r["scenario"] == scenario]
            if srs:
                print(f"{method:12s} {scenario:15s}: {np.mean(srs):.2%} ± {np.std(srs):.2%} (n={len(srs)})")
    
    # Save
    os.makedirs("docs/results/10seed_evaluation", exist_ok=True)
    with open("docs/results/10seed_evaluation/summary.md", "w") as f:
        f.write("# 10-Seed Evaluation Summary\n\n")
        f.write("| Method | Scenario | Mean SR | Std SR | n |\n")
        f.write("|--------|----------|---------|--------|---|\n")
        for method in ["baseline", "constrained"]:
            for scenario in ["favorable", "neutral", "disadvantage", "challenging", "crossing_left", "crossing_right"]:
                srs = [r["sr"] for r in all_results if r["method"] == method and r["scenario"] == scenario]
                if srs:
                    f.write(f"| {method} | {scenario} | {np.mean(srs):.2%} | {np.std(srs):.2%} | {len(srs)} |\n")
        f.write("\n## Evidence Grade\n")
        f.write("`paper_safe` — 10 seeds, 20 episodes per scenario.\n")
    
    with open("docs/results/10seed_evaluation/raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to docs/results/10seed_evaluation/")


if __name__ == "__main__":
    main()
