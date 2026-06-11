#!/usr/bin/env python3
"""
Aggregate crossing generalization results from multiple machines/seeds.

Reads raw JSON files produced by evaluate_crossing_generalization.py and
produces summary statistics and heatmaps.

Usage:
    python scripts/aggregate_crossing_generalization.py \
        --raw-files outputs/aggregated/results/crossing_generalization/raw_results*.json \
        --output-dir docs/results/crossing_generalization
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def aggregate(results):
    """Aggregate per-scenario success rates across seeds/methods."""
    methods = sorted(set(r["method"] for r in results if "method" in r))
    scenarios = sorted(set(r["scenario"] for r in results if "scenario" in r))

    summary = {}
    for method in methods:
        summary[method] = {}
        for scenario in scenarios:
            srs = [r["sr"] for r in results if r.get("method") == method and r.get("scenario") == scenario]
            if srs:
                summary[method][scenario] = {
                    "n": len(srs),
                    "mean_sr": float(np.mean(srs)),
                    "std_sr": float(np.std(srs, ddof=1)),
                    "min_sr": float(np.min(srs)),
                    "max_sr": float(np.max(srs)),
                }
    return summary


def parse_scenario_name(name):
    """Parse crossing_r{r}_a{a}_s{s}_y{y} into components."""
    parts = {}
    try:
        tokens = name.split("_")
        for tok in tokens[1:]:
            key = tok[0]
            val = float(tok[1:])
            parts[key] = val
    except Exception:
        pass
    return parts


def plot_heatmaps(summary, output_dir):
    """Plot heatmaps of success rate vs range and lateral offset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, scen_data in summary.items():
        # Extract grid
        ranges = sorted(set(parse_scenario_name(s).get("r", 0) for s in scen_data))
        offsets = sorted(set(parse_scenario_name(s).get("y", 0) for s in scen_data))
        if len(ranges) < 2 or len(offsets) < 2:
            continue

        grid = np.zeros((len(ranges), len(offsets)))
        for i, r in enumerate(ranges):
            for j, y in enumerate(offsets):
                key = f"cross_r{r:.0f}_a180.0_s1.0_y{y:.0f}"
                if key in scen_data:
                    grid[i, j] = scen_data[key]["mean_sr"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels([f"{y:.0f}" for y in offsets])
        ax.set_yticks(range(len(ranges)))
        ax.set_yticklabels([f"{r:.0f}" for r in ranges])
        ax.set_xlabel("Lateral offset (m)")
        ax.set_ylabel("Range (m)")
        ax.set_title(f"Crossing Generalization: {method}")
        plt.colorbar(im, ax=ax, label="Success rate")
        plt.tight_layout()
        plt.savefig(output_dir / f"crossing_generalization_{method}.png", dpi=150)
        plt.close()


def write_summary_markdown(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Crossing Generalization Summary (Aggregated)\n\n")
        f.write("| Method | Mean SR | Std SR | Worst-case | n scenarios |\n")
        f.write("|--------|---------|--------|------------|-------------|\n")

        for method in sorted(summary.keys()):
            means = [v["mean_sr"] for v in summary[method].values()]
            mins = [v["min_sr"] for v in summary[method].values()]
            f.write(
                f"| {method} | {np.mean(means):.2%} | {np.std(means, ddof=1):.2%} | "
                f"{np.mean(mins):.2%} | {len(means)} |\n"
            )

        f.write("\n## Per-Scenario Breakdown\n\n")
        f.write("| Method | Scenario | Mean SR | Std SR | n seeds |\n")
        f.write("|--------|----------|---------|--------|---------|\n")
        for method in sorted(summary.keys()):
            for scenario in sorted(summary[method].keys()):
                s = summary[method][scenario]
                f.write(
                    f"| {method} | {scenario} | {s['mean_sr']:.2%} | "
                    f"{s['std_sr']:.2%} | {s['n']} |\n"
                )

        f.write("\n## Evidence Grade\n")
        f.write("`preliminary` — aggregated seeds, 5 episodes per scenario.\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate crossing generalization results")
    parser.add_argument("--raw-files", nargs="+", required=True, help="Raw result JSON files")
    parser.add_argument("--output-dir", type=str, default="docs/results/crossing_generalization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_raw_results(args.raw_files)
    print(f"Loaded {len(results)} raw result entries from {len(args.raw_files)} files")

    summary = aggregate(results)

    write_summary_markdown(summary, Path(args.output_dir) / "full_grid_summary.md")
    plot_heatmaps(summary, args.output_dir)

    with open(Path(args.output_dir) / "aggregated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {args.output_dir}")


if __name__ == "__main__":
    main()
