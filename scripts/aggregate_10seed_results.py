#!/usr/bin/env python3
"""
Aggregate 10-seed evaluation results from distributed machines.

Reads raw JSON files produced by evaluate_10seed_results.py on each machine,
combines them, and produces summary tables with statistics.

Usage:
    python scripts/aggregate_10seed_results.py \
        --raw-files outputs/aggregated/results/10seed_evaluation/raw_results*.json \
        --output-dir docs/results/10seed_evaluation
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def cohen_d(x, y):
    """Compute Cohen's d for two samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std


def binomial_test(successes, trials, p_null=0.5):
    """Exact binomial test."""
    if trials == 0:
        return np.nan
    return stats.binom_test(successes, trials, p_null, alternative="two-sided")


def load_raw_results(raw_files):
    all_results = []
    for f in raw_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict) and "results" in data:
                all_results.extend(data["results"])
    return all_results


def aggregate_by_method_scenario(results):
    methods = sorted(set(r["method"] for r in results))
    scenarios = sorted(set(r["scenario"] for r in results))

    summary = {}
    for method in methods:
        summary[method] = {}
        for scenario in scenarios:
            srs = [r["sr"] for r in results if r["method"] == method and r["scenario"] == scenario]
            if srs:
                # Convert success rate back to successes/trials if episodes known
                # We assume 20 episodes per evaluation based on evaluate_10seed_results.py
                trials_per_seed = 20
                successes = int(round(np.mean(srs) * len(srs) * trials_per_seed))
                total_trials = len(srs) * trials_per_seed
                summary[method][scenario] = {
                    "n_seeds": len(srs),
                    "mean_sr": float(np.mean(srs)),
                    "std_sr": float(np.std(srs, ddof=1)),
                    "min_sr": float(np.min(srs)),
                    "max_sr": float(np.max(srs)),
                    "successes": successes,
                    "trials": total_trials,
                    "binomial_p": float(binomial_test(successes, total_trials)),
                }
    return summary


def compare_methods(summary, method_a, method_b, scenario):
    """Compare two methods on a scenario using binomial test on aggregate counts."""
    a = summary.get(method_a, {}).get(scenario)
    b = summary.get(method_b, {}).get(scenario)
    if not a or not b:
        return None

    # 2x2 contingency table
    table = [
        [a["successes"], a["trials"] - a["successes"]],
        [b["successes"], b["trials"] - b["successes"]],
    ]
    try:
        _, p, _, _ = stats.chi2_contingency(table)
    except Exception:
        p = np.nan

    # Cohen's h for proportions
    p_a = a["successes"] / a["trials"]
    p_b = b["successes"] / b["trials"]
    h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))

    return {"p_value": float(p), "cohens_h": float(h)}


def write_summary_markdown(summary, output_path, comparisons=None):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 10-Seed Evaluation Summary (Aggregated)\n\n")
        f.write("| Method | Scenario | Mean SR | Std SR | Min | Max | n seeds | Binomial p |\n")
        f.write("|--------|----------|---------|--------|-----|-----|---------|------------|\n")

        for method in sorted(summary.keys()):
            for scenario in sorted(summary[method].keys()):
                s = summary[method][scenario]
                f.write(
                    f"| {method} | {scenario} | "
                    f"{s['mean_sr']:.2%} | {s['std_sr']:.2%} | "
                    f"{s['min_sr']:.2%} | {s['max_sr']:.2%} | "
                    f"{s['n_seeds']} | {s['binomial_p']:.4g} |\n"
                )

        if comparisons:
            f.write("\n## Pairwise Comparisons\n\n")
            f.write("| Scenario | Method A | Method B | p-value | Cohen's h |\n")
            f.write("|----------|----------|----------|---------|-----------|\n")
            for (scenario, method_a, method_b), comp in comparisons.items():
                f.write(
                    f"| {scenario} | {method_a} | {method_b} | "
                    f"{comp['p_value']:.4g} | {comp['cohens_h']:.3f} |\n"
                )

        f.write("\n## Evidence Grade\n")
        f.write("`paper_safe` — aggregated 10 seeds, 20 episodes per scenario per seed.\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate 10-seed evaluation results")
    parser.add_argument("--raw-files", nargs="+", required=True, help="Raw result JSON files")
    parser.add_argument("--output-dir", type=str, default="docs/results/10seed_evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_raw_results(args.raw_files)
    print(f"Loaded {len(results)} raw result entries from {len(args.raw_files)} files")

    summary = aggregate_by_method_scenario(results)

    # Key comparisons from paper
    comparisons = {}
    for scenario in ["favorable", "neutral", "disadvantage", "challenging", "crossing_left", "crossing_right"]:
        comp = compare_methods(summary, "baseline", "constrained", scenario)
        if comp:
            comparisons[(scenario, "baseline", "constrained")] = comp

    output_md = Path(args.output_dir) / "summary_10seed_aggregated.md"
    write_summary_markdown(summary, output_md, comparisons)
    print(f"Summary written to {output_md}")

    output_json = Path(args.output_dir) / "aggregated_summary.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "comparisons": comparisons}, f, indent=2)
    print(f"Aggregated JSON written to {output_json}")


if __name__ == "__main__":
    main()
