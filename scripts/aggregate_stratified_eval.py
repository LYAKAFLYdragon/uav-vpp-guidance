#!/usr/bin/env python3
"""Aggregate scenario-stratified re-evaluation results across all checkpoints."""
import argparse
import glob
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/jsbsim_re_eval_stratified")
    parser.add_argument("--output", default="outputs/jsbsim_re_eval_stratified/aggregate_summary.json")
    args = parser.parse_args()

    results = {}
    checkpoint_results = {}

    summary_files = glob.glob(f"{args.input_dir}/batch_*/summary.json")
    print(f"Found {len(summary_files)} summary files")

    for summary_path in sorted(summary_files):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ckpt, ckpt_results in data["results"].items():
            m = re.match(r"(.+)_s\d+$", ckpt)
            method = m.group(1) if m else ckpt
            if method not in results:
                results[method] = {}
                checkpoint_results[method] = {}
            checkpoint_results[method][ckpt] = {}
            for scenario, vals in ckpt_results.items():
                if scenario not in results[method]:
                    results[method][scenario] = {"success": 0, "total": 0}
                results[method][scenario]["success"] += vals["success_count"]
                results[method][scenario]["total"] += vals["num_episodes"]
                checkpoint_results[method][ckpt][scenario] = {
                    "success_rate": vals["success_count"] / vals["num_episodes"],
                    "success_count": vals["success_count"],
                    "num_episodes": vals["num_episodes"],
                }

    aggregate = {"by_method": results, "per_checkpoint": checkpoint_results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("\nStratified re-evaluation aggregate results:")
    print("=" * 70)
    for method in sorted(results.keys()):
        print(f"\n{method}:")
        for scenario in ["favorable", "neutral", "challenging", "disadvantage", "overall"]:
            if scenario in results[method]:
                s = results[method][scenario]["success"]
                t = results[method][scenario]["total"]
                print(f"  {scenario:12s}: {s:4d}/{t:4d} = {s/t:.3f}")

    print(f"\nAggregate summary saved to {args.output}")


if __name__ == "__main__":
    main()
