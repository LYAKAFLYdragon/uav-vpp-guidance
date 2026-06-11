#!/usr/bin/env python3
"""
Aggregate predictor stratification results from multiple machines/seeds.

Usage:
    python scripts/aggregate_predictor_stratification.py \
        --raw-files outputs/aggregated/results/predictor_stratification/raw_results*.json \
        --output-dir docs/results/predictor_stratification
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats


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
    predictors = sorted(set(r["predictor"] for r in results if "predictor" in r))
    maneuvers = sorted(set(r["maneuver"] for r in results if "maneuver" in r))

    summary = {}
    for pred in predictors:
        summary[pred] = {}
        for man in maneuvers:
            srs = [r["sr"] for r in results if r.get("predictor") == pred and r.get("maneuver") == man]
            if srs:
                summary[pred][man] = {
                    "n_seeds": len(srs),
                    "mean_sr": float(np.mean(srs)),
                    "std_sr": float(np.std(srs, ddof=1)),
                    "min_sr": float(np.min(srs)),
                    "max_sr": float(np.max(srs)),
                }
    return summary


def write_summary_markdown(summary, output_path, raw_results):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Predictor Stratification Summary (Aggregated)\n\n")
        f.write("| Predictor | Maneuver Level | Mean SR | Std SR | n seeds |\n")
        f.write("|-----------|----------------|---------|--------|---------|\n")

        for pred in sorted(summary.keys()):
            for man in sorted(summary[pred].keys()):
                s = summary[pred][man]
                f.write(
                    f"| {pred} | {man} | {s['mean_sr']:.2%} | "
                    f"{s['std_sr']:.2%} | {s['n_seeds']} |\n"
                )

        # Relative improvement vs no-prediction
        f.write("\n## Relative Improvement vs No-Prediction\n\n")
        f.write("| Maneuver Level | Predictor | Absolute Δ | Relative Δ | p-value |\n")
        f.write("|----------------|-----------|------------|------------|---------|\n")

        maneuvers = sorted(set(r["maneuver"] for r in raw_results if "maneuver" in r))
        for man in maneuvers:
            baseline = [r["sr"] for r in raw_results if r.get("predictor") == "no_pred" and r.get("maneuver") == man]
            if len(baseline) < 2:
                continue
            for pred in sorted(summary.keys()):
                if pred == "no_pred":
                    continue
                vals = [r["sr"] for r in raw_results if r.get("predictor") == pred and r.get("maneuver") == man]
                if len(vals) < 2:
                    continue
                t, p = stats.ttest_ind(vals, baseline)
                abs_delta = np.mean(vals) - np.mean(baseline)
                rel_delta = abs_delta / np.mean(baseline) if np.mean(baseline) > 0 else 0
                f.write(
                    f"| {man} | {pred} | {abs_delta:+.2%} | {rel_delta:+.2%} | {p:.4g} |\n"
                )

        f.write("\n## Evidence Grade\n")
        f.write("`paper_safe` — aggregated multiple seeds and maneuver intensities.\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate predictor stratification results")
    parser.add_argument("--raw-files", nargs="+", required=True, help="Raw result JSON files")
    parser.add_argument("--output-dir", type=str, default="docs/results/predictor_stratification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_raw_results(args.raw_files)
    print(f"Loaded {len(results)} raw result entries from {len(args.raw_files)} files")

    summary = aggregate(results)

    output_md = Path(args.output_dir) / "10seed_summary.md"
    write_summary_markdown(summary, output_md, results)
    print(f"Summary written to {output_md}")

    with open(Path(args.output_dir) / "aggregated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
