#!/usr/bin/env python3
"""Multi-seed end-to-end PPO training launcher.

Sequentially trains end-to-end PPO policies with multiple random seeds and
produces a summary manifest with the final evaluation success rate per seed.

Usage:
    python scripts/run_end_to_end_ppo_multi_seed.py \
        --config config/experiment/train_end_to_end_ppo.yaml \
        --seeds 0 1 2 \
        --device cuda

The user is expected to run this on a cloud host with a GPU.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_vpp_guidance.training.train_end_to_end_ppo import (
    train_ppo,
    load_experiment_config,
)


def get_final_eval_metrics(eval_log_path: Path) -> dict:
    """Read the last row of eval_log.csv and return key metrics."""
    if not eval_log_path.exists():
        return {}
    with open(eval_log_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    keys = [
        "step",
        "num_episodes",
        "mean_return",
        "std_return",
        "success_rate",
        "crash_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "mean_final_range_m",
        "mean_final_ata_deg",
    ]
    return {k: float(last[k]) if k in last and last[k] != "" else None for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed end-to-end PPO training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/train_end_to_end_ppo.yaml",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds to train (default: 0 1 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute device",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override simulation backend",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory override",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode for quick verification",
    )
    args = parser.parse_args()

    base_config = load_experiment_config(args.config)

    if args.device is not None:
        base_config.setdefault("ppo", {})["device"] = args.device
    if args.backend is not None:
        base_config["backend"] = args.backend
        base_config.setdefault("env", {})["backend"] = args.backend
        base_config["env"]["use_jsbsim"] = args.backend == "jsbsim"

    exp_name = base_config.get("experiment", {}).get("name", "end_to_end_ppo")
    output_root = args.output_root or base_config.get("experiment", {}).get("output_root", "outputs")

    seed_summaries = []
    for seed in args.seeds:
        config = json.loads(json.dumps(base_config))  # deep copy
        config["experiment"]["seed"] = seed
        output_dir = Path(output_root) / "experiments" / f"{exp_name}_s{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training seed {seed} -> {output_dir}")
        print(f"{'='*60}")

        train_ppo(config, str(output_dir), smoke=args.smoke)

        eval_log = output_dir / "logs" / "eval_log.csv"
        final_metrics = get_final_eval_metrics(eval_log)
        seed_summaries.append(
            {
                "seed": seed,
                "output_dir": str(output_dir),
                "eval_log": str(eval_log),
                "final_metrics": final_metrics,
            }
        )

    # Aggregate summary
    success_rates = [
        s["final_metrics"].get("success_rate")
        for s in seed_summaries
        if s["final_metrics"].get("success_rate") is not None
    ]
    summary = {
        "config": args.config,
        "seeds": args.seeds,
        "smoke": args.smoke,
        "seed_summaries": seed_summaries,
        "mean_final_success_rate": sum(success_rates) / len(success_rates) if success_rates else None,
    }

    summary_path = Path(output_root) / "experiments" / f"{exp_name}_multi_seed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMulti-seed summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
