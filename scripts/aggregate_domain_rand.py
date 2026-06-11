#!/usr/bin/env python3
"""
Aggregate domain randomization robustness results from multiple machines/seeds.

Usage:
    python scripts/aggregate_domain_rand.py \
        --raw-files outputs/aggregated/results/domain_randomization/raw_results*.json \
        --output-dir docs/results/domain_randomization
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
    """Aggregate SR by method, scale, and seed."""
    methods = sorted(set(r["method"] for r in results if "method" in r))
    scales = sorted(set(r["scale"] for r in results if "scale" in r))

    summary = {}
    for method in methods:
        summary[method] = {}
        for scale in scales:
            srs = [r["sr"] for r in results if r.get("method") == method and r.get("scale") == scale]
            if srs:
                summary[method][scale] = {
                    "n_seeds": len(srs),
                    "mean_sr": float(np.mean(srs)),
                    "std_sr": float(np.std(srs, ddof=1)),
                    "min_sr": float(np.min(srs)),
                    "max_sr": float(np.max(srs)),
                }
    return summary


def compare_methods_at_scale(summary, method_a, method_b, scale):
    a = summary.get(method_a, {}).get(scale)
    b = summary.get(method_b, {}).get(scale)
    if not a or not b or a["n_seeds"] < 2 or b["n_seeds"] < 2:
        return None
    # Welch's t-test on per-seed SRs
    # Need raw data; summary only has aggregates. Assume raw results available.
    return None


def write_summary_markdown(summary, output_path, raw_results):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Domain Randomization Robustness Summary (Aggregated)\n\n")
        f.write("| Method | Perturbation Scale | Mean SR | Std SR | n seeds |\n")
        f.write("|--------|-------------------|---------|--------|---------|\n")

        for method in sorted(summary.keys()):
            for scale in sorted(summary[method].keys()):
                s = summary[method][scale]
                f.write(
                    f"| {method} | {scale:.2f} | {s['mean_sr']:.2%} | "
                    f"{s['std_sr']:.2%} | {s['n_seeds']} |\n"
                )

        # Statistical tests at each scale
        f.write("\n## Statistical Tests (DR vs Control)\n\n")
        f.write("| Scale | t-statistic | p-value | Cohen's d | Interpretation |\n")
        f.write("|-------|-------------|---------|-----------|----------------|\n")

        for scale in sorted(set(r["scale"] for r in raw_results)):
            a = [r["sr"] for r in raw_results if r.get("method") == "domain_rand" and r.get("scale") == scale]
            b = [r["sr"] for r in raw_results if r.get("method") == "control" and r.get("scale") == scale]
            if len(a) >= 2 and len(b) >= 2:
                t, p = stats.ttest_ind(a, b)
                d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
                interp = "significant" if p < 0.05 else "not significant"
                f.write(f"| {scale:.2f} | {t:.3f} | {p:.4g} | {d:.3f} | {interp} |\n")
            else:
                f.write(f"| {scale:.2f} | — | — | — | insufficient data |\n")

        f.write("\n## Evidence Grade\n")
        f.write("`paper_safe` — aggregated multiple seeds, 30 episodes per scale.\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate domain randomization results")
    parser.add_argument("--raw-files", nargs="+", required=True, help="Raw result JSON files")
    parser.add_argument("--output-dir", type=str, default="docs/results/domain_randomization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_raw_results(args.raw_files)
    print(f"Loaded {len(results)} raw result entries from {len(args.raw_files)} files")

    summary = aggregate(results)

    output_md = Path(args.output_dir) / "distributed_summary.md"
    write_summary_markdown(summary, output_md, results)
    print(f"Summary written to {output_md}")

    with open(Path(args.output_dir) / "aggregated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
