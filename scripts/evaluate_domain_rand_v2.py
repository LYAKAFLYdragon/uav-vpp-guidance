#!/usr/bin/env python3
"""Evaluate domain randomization v2 vs control under meaningful scales."""
import sys, os, copy, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import scipy.stats as stats

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


def evaluate_checkpoint(ckpt_path, config, scale, num_episodes=30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    env = CloseRangeTrackingEnv(config)
    env.set_domain_rand_scale(scale)
    agent = PPOAgent(obs_dim=16, action_dim=3, config=config, device="cpu")
    agent.load(ckpt_path)
    scenarios = config.get("scenarios", {})
    results = []
    for seed in seeds:
        for ep in range(num_episodes // len(seeds)):
            ep_seed = 10000 + seed * 1000 + ep
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
                "seed": seed, "scenario": name,
                "success": info.get("reason") == "success",
                "return": ep_reward,
            })
    env.close()
    srs = [r["success"] for r in results]
    returns = [r["return"] for r in results]
    return {
        "scale": scale, "num_episodes": len(results),
        "success_rate": sum(srs) / len(srs),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def main():
    dr_cfg = load_experiment_config("config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml")
    ctrl_cfg = load_experiment_config("config/experiment/train_no_prediction_vpp_ppo.yaml")
    dr_cfg["ppo"]["device"] = "cpu"
    ctrl_cfg["ppo"]["device"] = "cpu"
    
    scales = [0.0, 0.50, 1.00, 1.50, 2.00]
    all_results = []
    
    print("=" * 60)
    print("Domain Randomization v2 Evaluation")
    print("=" * 60)
    
    for method, cfg, template in [
        ("domain_rand", dr_cfg, "outputs/experiments/no_prediction_vpp_ppo_domain_rand_v2_s{seed}/checkpoints/best.pt"),
        ("control", ctrl_cfg, "outputs/experiments/no_prediction_vpp_ppo_control_s{seed}/checkpoints/best.pt"),
    ]:
        print(f"\n{method.upper()}:")
        for seed in [0, 1, 2]:
            ckpt = template.format(seed=seed)
            if not os.path.exists(ckpt):
                print(f"  SKIP: {ckpt}")
                continue
            for scale in scales:
                print(f"  seed={seed}, scale={scale:.2f}...", end=" ", flush=True)
                result = evaluate_checkpoint(ckpt, cfg, scale, num_episodes=30)
                result["method"] = method
                result["seed"] = seed
                all_results.append(result)
                print(f"SR={result['success_rate']:.2%}")
    
    # Aggregate and compare
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for method in ["domain_rand", "control"]:
        print(f"\n{method.upper()}:")
        for scale in scales:
            srs = [r["success_rate"] for r in all_results if r["method"] == method and r["scale"] == scale]
            if srs:
                print(f"  scale={scale:.2f}: SR={np.mean(srs):.2%} ± {np.std(srs):.2%} (n={len(srs)})")
    
    # Stats at scale=1.00 (10% pos/vel, 15° heading)
    dr_srs = [r["success_rate"] for r in all_results if r["method"] == "domain_rand" and r["scale"] == 1.00]
    ctrl_srs = [r["success_rate"] for r in all_results if r["method"] == "control" and r["scale"] == 1.00]
    if dr_srs and ctrl_srs:
        t, p = stats.ttest_ind(dr_srs, ctrl_srs)
        pooled_std = np.sqrt((np.std(dr_srs)**2 + np.std(ctrl_srs)**2) / 2)
        cohen_d = (np.mean(dr_srs) - np.mean(ctrl_srs)) / max(pooled_std, 1e-6)
        print(f"\nStatistical test (scale=1.00): t={t:.3f}, p={p:.4f}, Cohen's d={cohen_d:.3f}")
    
    # Stats at scale=2.00 (20% pos/vel, 30° heading)
    dr_srs2 = [r["success_rate"] for r in all_results if r["method"] == "domain_rand" and r["scale"] == 2.00]
    ctrl_srs2 = [r["success_rate"] for r in all_results if r["method"] == "control" and r["scale"] == 2.00]
    if dr_srs2 and ctrl_srs2:
        t2, p2 = stats.ttest_ind(dr_srs2, ctrl_srs2)
        pooled_std2 = np.sqrt((np.std(dr_srs2)**2 + np.std(ctrl_srs2)**2) / 2)
        cohen_d2 = (np.mean(dr_srs2) - np.mean(ctrl_srs2)) / max(pooled_std2, 1e-6)
        print(f"Statistical test (scale=2.00): t={t2:.3f}, p={p2:.4f}, Cohen's d={cohen_d2:.3f}")
    
    # Save
    os.makedirs("docs/results/domain_randomization", exist_ok=True)
    with open("docs/results/domain_randomization/summary_v2.md", "w") as f:
        f.write("# Domain Randomization Evaluation v2 (Corrected Curriculum)\n\n")
        f.write("## Curriculum\n")
        f.write("- 0-25% progress: scale=0.50 (5% position, 5% velocity, 7.5° heading)\n")
        f.write("- 25-50% progress: scale=1.00 (10% position, 10% velocity, 15° heading)\n")
        f.write("- 50-75% progress: scale=1.50 (15% position, 15% velocity, 22.5° heading)\n")
        f.write("- 75-100% progress: scale=2.00 (20% position, 20% velocity, 30° heading)\n\n")
        f.write("## Results\n\n")
        f.write("| Method | Scale | Mean SR | Std SR | n |\n")
        f.write("|--------|-------|---------|--------|---|\n")
        for method in ["domain_rand", "control"]:
            for scale in scales:
                srs = [r["success_rate"] for r in all_results if r["method"] == method and r["scale"] == scale]
                if srs:
                    f.write(f"| {method} | {scale:.2f} | {np.mean(srs):.2%} | {np.std(srs):.2%} | {len(srs)} |\n")
        if dr_srs and ctrl_srs:
            f.write(f"\n## Statistical Tests\n")
            f.write(f"Scale=1.00: t={t:.3f}, p={p:.4f}, d={cohen_d:.3f}\n")
            if p < 0.05:
                f.write("- **Significant at p<0.05**\n")
            f.write(f"Scale=2.00: t={t2:.3f}, p={p2:.4f}, d={cohen_d2:.3f}\n")
            if p2 < 0.05:
                f.write("- **Significant at p<0.05**\n")
        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — 3 seeds, 30 episodes per condition.\n")
    
    with open("docs/results/domain_randomization/raw_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to docs/results/domain_randomization/summary_v2.md")


if __name__ == "__main__":
    main()
