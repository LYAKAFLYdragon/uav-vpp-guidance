#!/usr/bin/env python3
"""Evaluate crossing breakthrough methods on challenging crossing scenario."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import copy

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


def evaluate_checkpoint(checkpoint_path, config, scenario_name, num_episodes=30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    env = CloseRangeTrackingEnv(config)
    sample_obs = env.reset(seed=0)
    obs_dim = sample_obs["observation_vector"].shape[0]
    action_dim = config.get("policy", {}).get("action_dim", 3)
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
    agent.load(checkpoint_path)
    
    scenarios = config.get("scenarios", {})
    scenario = scenarios.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Scenario {scenario_name} not found")
    
    results = []
    for seed in seeds:
        for ep in range(num_episodes // len(seeds)):
            ep_seed = seed * 10000 + ep
            obs = env.reset(scenario=scenario, seed=ep_seed)
            ep_reward = 0.0
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            results.append({
                "seed": seed, "episode": ep,
                "return": ep_reward,
                "success": info.get("reason") == "success",
                "crash": info.get("reason") == "crash",
                "oob": info.get("reason") == "out_of_bounds",
                "timeout": info.get("reason") == "timeout",
            })
    env.close()
    sr = sum(1 for r in results if r["success"]) / len(results)
    crash_rate = sum(1 for r in results if r["crash"]) / len(results)
    oob_rate = sum(1 for r in results if r["oob"]) / len(results)
    returns = [r["return"] for r in results]
    return {
        "checkpoint": checkpoint_path,
        "scenario": scenario_name,
        "num_episodes": len(results),
        "success_rate": sr,
        "crash_rate": crash_rate,
        "oob_rate": oob_rate,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def main():
    methods = {
        "baseline": ("config/experiment/train_no_prediction_vpp_ppo.yaml", 
                     "outputs/experiments/no_prediction_vpp_ppo_control_s{seed}/checkpoints/best.pt"),
        "curriculum": ("config/experiment/train_curriculum_ppo.yaml",
                       "outputs/experiments/crossing_curriculum_s{seed}/checkpoints/best.pt"),
        "constrained": ("config/experiment/stage6f5_feasible_geometry_constrained.yaml",
                        "outputs/experiments/crossing_constrained_s{seed}/checkpoints/best.pt"),
        "hybrid": ("config/experiment/train_hybrid_mode_switch.yaml",
                   "outputs/experiments/crossing_hybrid_s{seed}/checkpoints/best.pt"),
    }
    
    all_results = []
    for method_name, (cfg_path, ckpt_template) in methods.items():
        config = load_experiment_config(cfg_path)
        config["ppo"]["device"] = "cpu"
        print(f"\n=== {method_name.upper()} ===")
        for seed in [0, 1, 2]:
            ckpt = ckpt_template.format(seed=seed)
            if not os.path.exists(ckpt):
                print(f"  SKIP: {ckpt} not found")
                continue
            result = evaluate_checkpoint(ckpt, config, "challenging", num_episodes=30)
            result["method"] = method_name
            result["seed"] = seed
            all_results.append(result)
            print(f"  Seed {seed}: SR={result['success_rate']:.2%}, Crash={result['crash_rate']:.2%}, OOB={result['oob_rate']:.2%}")
    
    # Aggregate
    print("\n" + "=" * 60)
    print("CROSSING BREAKTHROUGH SUMMARY")
    print("=" * 60)
    for method_name in methods.keys():
        srs = [r["success_rate"] for r in all_results if r["method"] == method_name]
        if not srs:
            continue
        mean_sr = np.mean(srs)
        std_sr = np.std(srs)
        print(f"{method_name:12s}: SR={mean_sr:.2%} ± {std_sr:.2%} (n={len(srs)})")
    
    # Per-scenario for all scenarios
    print("\n--- Full scenario breakdown (baseline) ---")
    config = load_experiment_config("config/experiment/train_no_prediction_vpp_ppo.yaml")
    config["ppo"]["device"] = "cpu"
    for scenario_name in config.get("scenarios", {}).keys():
        srs = []
        for seed in [0, 1, 2]:
            ckpt = f"outputs/experiments/no_prediction_vpp_ppo_control_s{seed}/checkpoints/best.pt"
            if os.path.exists(ckpt):
                r = evaluate_checkpoint(ckpt, config, scenario_name, num_episodes=15)
                srs.append(r["success_rate"])
        if srs:
            print(f"  {scenario_name:15s}: SR={np.mean(srs):.2%} ± {np.std(srs):.2%}")
    
    # Save
    os.makedirs("docs/results/crossing_breakthrough", exist_ok=True)
    with open("docs/results/crossing_breakthrough/summary.md", "w") as f:
        f.write("# Crossing Breakthrough Evaluation\n\n")
        f.write("## Methods Tested\n")
        f.write("- **baseline**: Standard no-prediction VPP+LOS-rate (control)\n")
        f.write("- **curriculum**: Curriculum learning (progressive scenario introduction)\n")
        f.write("- **constrained**: Dynamics-aware VPP with max_heading_rate=0.2 rad/s\n")
        f.write("- **hybrid**: Hybrid PN/LOS-rate with mode_switch enabled\n\n")
        f.write("## Results on Challenging (Crossing) Scenario\n\n")
        f.write("| Method | Mean SR | Std SR | n |\n")
        f.write("|--------|---------|--------|---|\n")
        for method_name in methods.keys():
            srs = [r["success_rate"] for r in all_results if r["method"] == method_name]
            if srs:
                f.write(f"| {method_name} | {np.mean(srs):.2%} | {np.std(srs):.2%} | {len(srs)} |\n")
        
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 3 seeds, 30 episodes per method.\n")
        f.write("Requires more seeds and JSBSim validation for `paper_safe`.\n")
        
        f.write("\n## Interpretation\n")
        best = max([(m, np.mean([r["success_rate"] for r in all_results if r["method"] == m])) 
                    for m in methods.keys() if any(r["method"] == m for r in all_results)], key=lambda x: x[1])
        f.write(f"Best method: {best[0]} with SR={best[1]:.2%}.\n")
        if best[1] < 0.50:
            f.write("**No method achieved crossing SR >= 50%.** Crossing remains a bottleneck.\n")
            f.write("Recommendation: Narrow paper claim to head-on/neutral geometries,\n")
            f.write("and explicitly declare crossing as future work.\n")
    
    with open("docs/results/crossing_breakthrough/raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSummary saved to docs/results/crossing_breakthrough/summary.md")


if __name__ == "__main__":
    main()
