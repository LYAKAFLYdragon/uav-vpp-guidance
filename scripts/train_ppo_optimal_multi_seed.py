"""
Multi-seed full-budget PPO training with optimal hyperparameters.

Optimal config from parameter sensitivity sweep:
    lr=3e-4, clip=0.2, gae=0.9

Runs 3 seeds (0, 1, 2) with 200K timesteps each.
Outputs aggregated results to docs/results/optimal_ppo_multi_seed/.

Usage:
    python scripts/train_ppo_optimal_multi_seed.py \
        --seeds 0 1 2 \
        --timesteps 200000 \
        --output-dir outputs/optimal_ppo_multi_seed
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import List, Dict


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


# Optimal hyperparameters from sensitivity sweep
OPTIMAL_PPO = {
    "learning_rate": 3.0e-4,
    "clip_coef": 0.2,
    "gae_lambda": 0.9,
    "device": "cpu",
}


def build_config(base_config_path: str, overrides: dict) -> dict:
    """Load base config and apply optimal overrides."""
    base = load_yaml_config(base_config_path)
    includes = base.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(base_config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    merged = merge_config(merged, base)

    # Apply PPO overrides
    if "ppo" not in merged:
        merged["ppo"] = {}
    merged["ppo"].update(overrides)

    return merged


def run_training(seed: int, timesteps: int, output_dir: str, config_path: str) -> Dict:
    """Run a single training seed."""
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Build config with overrides
    config = build_config(config_path, {
        **OPTIMAL_PPO,
        "total_timesteps": timesteps,
    })
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"optimal_ppo_seed_{seed}"

    # Save merged config
    config_out = os.path.join(seed_dir, "config.yaml")
    import yaml
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Run training
    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", config_out,
        "--output-dir", seed_dir,
    ]

    print(f"[Seed {seed}] Starting training...")
    print(f"  Command: {' '.join(cmd)}")
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    success = result.returncode == 0
    if not success:
        print(f"[Seed {seed}] FAILED after {elapsed:.1f}s")
        print(f"  stderr: {result.stderr[-500:]}")
    else:
        print(f"[Seed {seed}] SUCCESS after {elapsed:.1f}s")

    return {
        "seed": seed,
        "success": success,
        "elapsed_s": elapsed,
        "output_dir": seed_dir,
        "stderr_tail": result.stderr[-500:] if not success else "",
    }


def aggregate_results(output_dir: str, results: List[Dict]) -> Dict:
    """Aggregate multi-seed results."""
    seeds = []
    success_rates = []
    best_returns = []

    for r in results:
        if not r["success"]:
            continue
        seed_dir = r["output_dir"]
        # Look for eval summary
        summary_path = os.path.join(seed_dir, "eval_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            seeds.append(r["seed"])
            success_rates.append(summary.get("success_rate", 0.0))
            best_returns.append(summary.get("best_return", float("-inf")))

    if not seeds:
        return {"status": "all_failed", "n_successful": 0}

    return {
        "status": "completed",
        "n_successful": len(seeds),
        "seeds": seeds,
        "success_rates": success_rates,
        "best_returns": best_returns,
        "mean_success_rate": sum(success_rates) / len(success_rates),
        "std_success_rate": (
            (sum((x - sum(success_rates)/len(success_rates))**2 for x in success_rates) / len(success_rates)) ** 0.5
            if len(success_rates) > 1 else 0.0
        ),
        "mean_best_return": sum(best_returns) / len(best_returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed optimal PPO training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds to run")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total timesteps per seed")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/optimal_ppo_multi_seed",
                        help="Output directory")
    parser.add_argument("--config", type=str,
                        default="config/experiment/train_no_prediction_vpp_ppo.yaml",
                        help="Base config path")
    parser.add_argument("--parallel", action="store_true",
                        help="Run seeds in parallel (not recommended for CPU-only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run training for each seed
    results = []
    for seed in args.seeds:
        result = run_training(seed, args.timesteps, args.output_dir, args.config)
        results.append(result)

    # Aggregate
    agg = aggregate_results(args.output_dir, results)

    # Save aggregate
    agg_path = os.path.join(args.output_dir, "aggregate_results.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate results saved to: {agg_path}")

    # Summary
    if agg["status"] == "completed":
        print(f"\n=== Multi-Seed Summary ===")
        print(f"Successful seeds: {agg['n_successful']}/{len(args.seeds)}")
        print(f"Mean success rate: {agg['mean_success_rate']:.1%}")
        print(f"Std success rate: {agg['std_success_rate']:.1%}")
        for s, sr, br in zip(agg["seeds"], agg["success_rates"], agg["best_returns"]):
            print(f"  Seed {s}: SR={sr:.1%}, BestReturn={br:.1f}")
    else:
        print("\nAll seeds failed. Check individual logs.")
        for r in results:
            if not r["success"]:
                print(f"\nSeed {r['seed']} stderr tail:\n{r['stderr_tail']}")

    # Write summary markdown
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Optimal PPO Multi-Seed Training Summary\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Config**: lr={OPTIMAL_PPO['learning_rate']}, "
                f"clip={OPTIMAL_PPO['clip_coef']}, gae={OPTIMAL_PPO['gae_lambda']}\n")
        f.write(f"**Timesteps**: {args.timesteps:,}\n")
        f.write(f"**Seeds**: {args.seeds}\n\n")
        if agg["status"] == "completed":
            f.write(f"## Results\n\n")
            f.write(f"| Seed | Success Rate | Best Return |\n")
            f.write(f"|------|-------------|-------------|\n")
            for s, sr, br in zip(agg["seeds"], agg["success_rates"], agg["best_returns"]):
                f.write(f"| {s} | {sr:.1%} | {br:.1f} |\n")
            f.write(f"\n**Mean SR**: {agg['mean_success_rate']:.1%} "
                    f"(±{agg['std_success_rate']:.1%})\n")
        else:
            f.write("All seeds failed.\n")
    print(f"Summary markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
