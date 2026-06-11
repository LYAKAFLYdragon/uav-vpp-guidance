#!/usr/bin/env python3
"""
Evaluate domain randomization vs control group under multiple perturbation scales.

Evaluates each checkpoint under:
  - nominal (scale=0)
  - ±10% perturbation (scale=0.10)
  - ±20% perturbation (scale=0.20)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import csv

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


def evaluate_checkpoint(checkpoint_path, config, scale, num_episodes=30, seed_base=10000):
    """Evaluate a checkpoint under given domain randomization scale."""
    config = copy.deepcopy(config)
    env = CloseRangeTrackingEnv(config)
    env.set_domain_rand_scale(scale)
    
    sample_obs = env.reset(seed=0)
    obs_dim = sample_obs["observation_vector"].shape[0]
    action_dim = config.get("policy", {}).get("action_dim", 3)
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
    agent.load(checkpoint_path)
    
    scenarios = config.get("scenarios", {})
    results = []
    
    for seed in [0, 1, 2]:
        for ep in range(num_episodes // 3):
            ep_seed = seed_base + seed * 1000 + ep
            rng = np.random.default_rng(ep_seed)
            name = rng.choice(list(scenarios.keys()))
            scenario = scenarios[name]
            obs = env.reset(scenario=scenario, seed=ep_seed)
            ep_reward = 0.0
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            results.append({
                "seed": seed, "episode": ep, "scenario": name,
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
        "scale": scale, "num_episodes": len(results),
        "success_rate": sr, "crash_rate": crash_rate, "oob_rate": oob_rate,
        "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
    }


import copy

def main():
    dr_config_path = "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml"
    ctrl_config_path = "config/experiment/train_no_prediction_vpp_ppo.yaml"
    
    dr_cfg = load_experiment_config(dr_config_path)
    ctrl_cfg = load_experiment_config(ctrl_config_path)
    dr_cfg["ppo"]["device"] = "cpu"
    ctrl_cfg["ppo"]["device"] = "cpu"
    
    seeds = [0, 1, 2]
    scales = [0.0, 0.10, 0.20]
    
    all_results = []
    
    # Evaluate domain randomization models
    print("=" * 60)
    print("Evaluating DOMAIN RANDOMIZATION models")
    print("=" * 60)
    for seed in seeds:
        ckpt = f"outputs/experiments/no_prediction_vpp_ppo_domain_rand_s{seed}/checkpoints/best.pt"
        if not os.path.exists(ckpt):
            print(f"  SKIP: {ckpt} not found")
            continue
        for scale in scales:
            print(f"  Seed {seed}, scale={scale}...", end=" ", flush=True)
            result = evaluate_checkpoint(ckpt, dr_cfg, scale, num_episodes=30)
            result["method"] = "domain_rand"
            result["seed"] = seed
            all_results.append(result)
            print(f"SR={result['success_rate']:.2%}")
    
    # Evaluate control models
    print("=" * 60)
    print("Evaluating CONTROL models")
    print("=" * 60)
    for seed in seeds:
        ckpt = f"outputs/experiments/no_prediction_vpp_ppo_control_s{seed}/checkpoints/best.pt"
        if not os.path.exists(ckpt):
            print(f"  SKIP: {ckpt} not found")
            continue
        for scale in scales:
            print(f"  Seed {seed}, scale={scale}...", end=" ", flush=True)
            result = evaluate_checkpoint(ckpt, ctrl_cfg, scale, num_episodes=30)
            result["method"] = "control"
            result["seed"] = seed
            all_results.append(result)
            print(f"SR={result['success_rate']:.2%}")
    
    # Aggregate
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    import scipy.stats as stats
    
    for method in ["domain_rand", "control"]:
        print(f"\n{method.upper()}:")
        for scale in scales:
            srs = [r["success_rate"] for r in all_results if r["method"] == method and r["scale"] == scale]
            if not srs:
                continue
            mean_sr = np.mean(srs)
            std_sr = np.std(srs)
            print(f"  scale={scale:.2f}: SR={mean_sr:.2%} ± {std_sr:.2%} (n={len(srs)})")
    
    # Statistical test at scale=0.10
    dr_srs = [r["success_rate"] for r in all_results if r["method"] == "domain_rand" and r["scale"] == 0.10]
    ctrl_srs = [r["success_rate"] for r in all_results if r["method"] == "control" and r["scale"] == 0.10]
    if dr_srs and ctrl_srs:
        t, p = stats.ttest_ind(dr_srs, ctrl_srs)
        pooled_std = np.sqrt((np.std(dr_srs)**2 + np.std(ctrl_srs)**2) / 2)
        cohen_d = (np.mean(dr_srs) - np.mean(ctrl_srs)) / max(pooled_std, 1e-6)
        print(f"\nStatistical test (scale=0.10): t={t:.3f}, p={p:.4f}, Cohen's d={cohen_d:.3f}")
    
    # Save results
    os.makedirs("docs/results/domain_randomization", exist_ok=True)
    with open("docs/results/domain_randomization/summary.md", "w", encoding="utf-8") as f:
        f.write("# Domain Randomization Evaluation Summary\n\n")
        f.write("## Method\n")
        f.write("Curriculum-style domain randomization training with progressive scale:\n")
        f.write("- 0-25% progress: 5% perturbation\n")
        f.write("- 25-50% progress: 10% perturbation\n")
        f.write("- 50-75% progress: 15% perturbation\n")
        f.write("- 75-100% progress: 20% perturbation\n\n")
        f.write("Perturbations applied to:\n")
        f.write("- position: ±10% of initial range\n")
        f.write("- velocity: ±10% of nominal velocity\n")
        f.write("- heading: ±15°\n\n")
        
        f.write("## Results\n\n")
        f.write("| Method | Scale | Mean SR | Std SR | n |\n")
        f.write("|--------|-------|---------|--------|---|\n")
        for method in ["domain_rand", "control"]:
            for scale in scales:
                srs = [r["success_rate"] for r in all_results if r["method"] == method and r["scale"] == scale]
                if srs:
                    f.write(f"| {method} | {scale:.2f} | {np.mean(srs):.2%} | {np.std(srs):.2%} | {len(srs)} |\n")
        
        if dr_srs and ctrl_srs:
            f.write(f"\n## Statistical Test (scale=0.10)\n")
            f.write(f"- t-statistic: {t:.3f}\n")
            f.write(f"- p-value: {p:.4f}\n")
            f.write(f"- Cohen's d: {cohen_d:.3f}\n")
            if p < 0.05:
                f.write("- **Significant difference (p<0.05)**\n")
            else:
                f.write("- No significant difference (p>=0.05)\n")
        
        # Evidence grade
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 3 seeds, 30 episodes per condition.\n")
        f.write("Requires more seeds and longer evaluation for `paper_safe`.\n")
        
        f.write("\n## Limitations\n")
        f.write("- Simple backend only; JSBSim validation pending.\n")
        f.write("- Crossing scenario remains challenging even with domain randomization.\n")
    
    with open("docs/results/domain_randomization/raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nSummary saved to docs/results/domain_randomization/summary.md")


if __name__ == "__main__":
    main()
