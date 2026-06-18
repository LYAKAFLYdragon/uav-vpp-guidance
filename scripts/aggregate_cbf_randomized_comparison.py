"""Aggregate randomized CBF evaluation results across methods.

Example:
    python scripts/aggregate_cbf_randomized_comparison.py \
        --methods baseline pointmass envelope jsbsim_fd \
        --output outputs/cbf_adversarial/randomized_comparison.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import mannwhitneyu


METHOD_DIRS = {
    "baseline": "outputs/cbf_adversarial/randomized/baseline",
    "pointmass": "outputs/cbf_adversarial/randomized/cbf500",
    "envelope": "outputs/cbf_adversarial/randomized_envelope/cbf500",
    "jsbsim_fd": "outputs/cbf_adversarial/randomized_jsbsim_fd/cbf500",
    "jsbsim_fd_envelope": "outputs/cbf_adversarial/randomized_jsbsim_fd_envelope/cbf500",
}


def load_eval(path: str) -> Dict:
    with open(os.path.join(path, "eval_results.json"), "r") as f:
        return json.load(f)


def summarize(method: str, data: Dict) -> Dict:
    eps = data["raw_episodes"]
    n = len(eps)
    min_ranges = [ep["min_range_m"] for ep in eps]
    return {
        "n": n,
        "success_rate": sum(ep.get("captured", False) for ep in eps) / n,
        "crash_rate": sum(ep.get("crash", False) for ep in eps) / n,
        "oob_rate": sum(ep.get("out_of_bounds", False) for ep in eps) / n,
        "mean_min_range_m": float(np.mean(min_ranges)),
        "std_min_range_m": float(np.std(min_ranges, ddof=1)),
        "median_min_range_m": float(np.median(min_ranges)),
        "cbf_active_rate": data["overall"].get("cbf_intervention_rate", 0.0),
        "mean_solve_time_ms": data["overall"].get("mean_cbf_solve_time_ms", 0.0),
        "max_solve_time_ms": data["overall"].get("max_cbf_solve_time_ms", 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHOD_DIRS.keys()))
    parser.add_argument("--output", default="outputs/cbf_adversarial/randomized_comparison.json")
    args = parser.parse_args()

    results: Dict[str, Dict] = {}
    min_ranges: Dict[str, List[float]] = {}

    for m in args.methods:
        path = METHOD_DIRS.get(m)
        if path is None or not os.path.isdir(path):
            print(f"Skipping missing method: {m}")
            continue
        data = load_eval(path)
        results[m] = summarize(m, data)
        min_ranges[m] = [ep["min_range_m"] for ep in data["raw_episodes"]]

    # Pairwise Mann-Whitney U on min_range.
    stats = {}
    methods = list(results.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            if m1 in min_ranges and m2 in min_ranges:
                u, p = mannwhitneyu(min_ranges[m1], min_ranges[m2], alternative="two-sided")
                stats[f"{m1}_vs_{m2}"] = {"U": float(u), "p": float(p)}

    output = {"methods": results, "mann_whitney": stats}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print table.
    print(f"{'Method':<12} {'Succ':>5} {'Crash':>6} {'OOB':>6} {'MeanR':>8} {'StdR':>8} {'CBF%':>6} {'T_ms':>7}")
    print("-" * 70)
    for m, r in results.items():
        print(
            f"{m:<12} {r['success_rate']*100:>5.1f} {r['crash_rate']*100:>6.1f} "
            f"{r['oob_rate']*100:>6.1f} {r['mean_min_range_m']:>8.1f} "
            f"{r['std_min_range_m']:>8.1f} {r['cbf_active_rate']*100:>6.1f} "
            f"{r['mean_solve_time_ms']:>7.2f}"
        )

    print("\nMann-Whitney U p-values (min_range):")
    for key, val in stats.items():
        print(f"  {key}: p={val['p']:.4f}")


if __name__ == "__main__":
    main()
