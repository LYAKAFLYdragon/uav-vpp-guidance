#!/usr/bin/env python3
"""
Aggregate Stage 6B multi-seed evaluation results.

Reads prediction_metrics.json from outputs/stage6b/eval_seed{0,1,2}/
and produces:
  - docs/results/stage6b/prediction_metrics.csv
  - docs/results/stage6b/summary.md
  - docs/results/stage6b/cross_seed_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_metrics(path: Path) -> list:
    """Load prediction_metrics.json if it exists."""
    p = path / "prediction_metrics.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_method(metrics_list: list) -> dict:
    """Aggregate metrics across training seeds."""
    if not metrics_list:
        return {}

    def _mean_std(key):
        vals = [float(m.get(key, np.nan)) for m in metrics_list if key in m]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    sr_mean, sr_std = _mean_std("instant_success_rate")
    ret_mean, ret_std = _mean_std("mean_return")
    fr_mean, fr_std = _mean_std("failure_rate")
    cr_mean, cr_std = _mean_std("crash_rate")
    oob_mean, oob_std = _mean_std("out_of_bounds_rate")
    to_mean, to_std = _mean_std("timeout_rate")
    mr_mean, mr_std = _mean_std("mean_final_range_m")
    ma_mean, ma_std = _mean_std("mean_final_ata_deg")

    return {
        "num_training_seeds": len(metrics_list),
        "instant_success_rate_mean": sr_mean,
        "instant_success_rate_std": sr_std,
        "mean_return_mean": ret_mean,
        "mean_return_std": ret_std,
        "failure_rate_mean": fr_mean,
        "failure_rate_std": fr_std,
        "crash_rate_mean": cr_mean,
        "crash_rate_std": cr_std,
        "out_of_bounds_rate_mean": oob_mean,
        "out_of_bounds_rate_std": oob_std,
        "timeout_rate_mean": to_mean,
        "timeout_rate_std": to_std,
        "mean_final_range_m_mean": mr_mean,
        "mean_final_range_m_std": mr_std,
        "mean_final_ata_deg_mean": ma_mean,
        "mean_final_ata_deg_std": ma_std,
    }


def build_summary_table(methods: dict) -> str:
    """Build Markdown summary table."""
    lines = [
        "# Stage 6B: No-Prediction vs CV vs CA — Cross-Seed Summary",
        "",
        f"**Git commit**: `{get_git_hash()}`  ",
        f"**Aggregated seeds**: 0, 1, 2  ",
        "",
        "| Method | Success Rate | Return | Crash Rate | OOB Rate | Timeout Rate |",
        "|--------|-------------|--------|-----------|----------|--------------|",
    ]
    for method_name, agg in methods.items():
        sr = agg.get("instant_success_rate_mean", np.nan)
        sr_s = agg.get("instant_success_rate_std", 0.0)
        ret = agg.get("mean_return_mean", np.nan)
        ret_s = agg.get("mean_return_std", 0.0)
        cr = agg.get("crash_rate_mean", np.nan)
        cr_s = agg.get("crash_rate_std", 0.0)
        oob = agg.get("out_of_bounds_rate_mean", np.nan)
        oob_s = agg.get("out_of_bounds_rate_std", 0.0)
        to = agg.get("timeout_rate_mean", np.nan)
        to_s = agg.get("timeout_rate_std", 0.0)
        lines.append(
            f"| {method_name} | "
            f"{sr:.1%} ± {sr_s:.1%} | "
            f"{ret:.1f} ± {ret_s:.1f} | "
            f"{cr:.1%} ± {cr_s:.1%} | "
            f"{oob:.1%} ± {oob_s:.1%} | "
            f"{to:.1%} ± {to_s:.1%} |"
        )
    lines.append("")
    lines.append("## Per-Scenario Breakdown (mean across 3 training seeds)")
    lines.append("")
    # TODO: add per-scenario breakdown if needed
    lines.append("*See raw_episodes.csv for episode-level data.*")
    lines.append("")
    return "\n".join(lines)


def get_git_hash() -> str:
    import subprocess
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage 6B results")
    parser.add_argument("--input-root", type=str, default="outputs/stage6b")
    parser.add_argument("--output-dir", type=str, default="docs/results/stage6b")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics from each training seed
    all_raw = []
    methods_by_seed = {}
    for seed in range(3):
        seed_dir = input_root / f"eval_seed{seed}"
        metrics = load_metrics(seed_dir)
        for m in metrics:
            m["training_seed"] = seed
            all_raw.append(m)
            method = m.get("method", "unknown")
            methods_by_seed.setdefault(method, []).append(m)

    if not all_raw:
        print("ERROR: No metrics found. Have the evaluations finished?")
        sys.exit(1)

    # Aggregate per method
    cross_summary = {}
    for method_name, metrics_list in methods_by_seed.items():
        cross_summary[method_name] = aggregate_method(metrics_list)

    # Save cross-seed summary JSON
    cross_path = output_dir / "cross_seed_summary.json"
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump({
            "git_commit": get_git_hash(),
            "methods": cross_summary,
        }, f, indent=2, default=str)
    print(f"Saved: {cross_path}")

    # Save raw combined CSV
    df = pd.DataFrame(all_raw)
    csv_path = output_dir / "prediction_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save Markdown summary
    summary_md = build_summary_table(cross_summary)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\nStage 6B aggregation complete.")


if __name__ == "__main__":
    main()
