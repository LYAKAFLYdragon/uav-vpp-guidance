#!/usr/bin/env python3
"""
Aggregate Stage 6F ablation results into paper-ready summary tables.

Supports two-level aggregation:
  1. Episodes → per-training-seed statistics
  2. Per-training-seed → cross-seed mean/std/95% CI

Reads:
  - <input>/experiment_plan.json (top-level plan for manifest validation)
  - <input>/train_seed<N>/prediction_metrics.json (per-training-seed comparison results)

Outputs:
  - summary.csv
  - summary_latex.tex
  - summary_markdown.md
  - cross_seed_summary.json

Usage:
    python scripts/aggregate_stage6f_results.py \
        --input outputs/tables/stage6f_full_ablation \
        --output outputs/tables/stage6f
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRICS_SCHEMA_VERSION = "6f.2"

# Columns that should be aggregated at episode-level then averaged across training seeds
NUMERIC_AGG_COLS = [
    "instant_success_rate",
    "mean_return",
    "mean_final_range_m",
    "mean_final_ata_deg",
    "prediction_valid_rate",
    "prediction_fallback_rate",
    "runtime_fallback_rate",
    "post_warmup_fallback_rate",
    "mean_env_prediction_error_m",
    "median_env_prediction_error_m",
    "mean_offline_aligned_error_m",
    "median_offline_aligned_error_m",
]

# Count columns that should be summed at episode-level, then averaged across seeds
COUNT_COLS = [
    "unknown_fallback_phase_count",
    "missing_fallback_phase_count",
    "configured_current_target_fallback_count",
    "predictor_init_failed_count",
]

CROSS_SEED_COLS = [
    "method",
    "num_training_seeds",
    "num_episodes_per_training_seed",
    "num_evaluation_seeds",
    "num_scenarios",
    "episodes_per_scenario",
    "scenario_balance_ok",
    "success_rate_mean",
    "success_rate_std",
    "mean_return_mean",
    "mean_return_std",
    "mean_final_range_m_mean",
    "mean_final_range_m_std",
    "mean_final_ata_deg_mean",
    "mean_final_ata_deg_std",
    "prediction_valid_rate_mean",
    "prediction_valid_rate_std",
    "fallback_rate_mean",
    "fallback_rate_std",
    "runtime_fallback_rate_mean",
    "runtime_fallback_rate_std",
    "post_warmup_fallback_rate_mean",
    "post_warmup_fallback_rate_std",
    "mean_env_prediction_error_m_mean",
    "mean_env_prediction_error_m_std",
    "median_env_prediction_error_m_mean",
    "median_env_prediction_error_m_std",
    "mean_offline_aligned_error_m_mean",
    "mean_offline_aligned_error_m_std",
    "unknown_fallback_phase_count_mean",
    "missing_fallback_phase_count_mean",
    "configured_current_target_fallback_count_mean",
    "invalid_for_paper",
]


def _safe_float(val):
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _safe_mean(vals):
    clean = [v for v in vals if np.isfinite(v)]
    return float(np.mean(clean)) if clean else np.nan


def _safe_std(vals):
    clean = [v for v in vals if np.isfinite(v)]
    return float(np.std(clean, ddof=0)) if clean else np.nan


def _safe_ci95(vals):
    """Return 95% CI half-width assuming normal distribution."""
    clean = [v for v in vals if np.isfinite(v)]
    if len(clean) < 2:
        return np.nan
    std = np.std(clean, ddof=1)
    n = len(clean)
    return float(1.96 * std / math.sqrt(n))


def _check_manifest(plan_data: dict):
    """Validate experiment plan manifest fields."""
    warnings = []
    if plan_data.get("metrics_schema_version") != METRICS_SCHEMA_VERSION:
        warnings.append(
            f"Schema version mismatch: plan={plan_data.get('metrics_schema_version')} "
            f"expected={METRICS_SCHEMA_VERSION}"
        )
    if plan_data.get("formal") is not True:
        warnings.append("formal=False in experiment plan")
    if plan_data.get("allow_random_policy") is not False:
        warnings.append("allow_random_policy=True in experiment plan")
    return warnings


def _extract_per_seed_episodes(method_metrics: dict) -> list:
    """Extract flat episode list from method metrics dict."""
    episodes = method_metrics.get("raw_episodes", [])
    if episodes:
        return episodes
    per_seed = method_metrics.get("per_seed", {})
    return [ep for seed_eps in per_seed.values() for ep in seed_eps]


def aggregate_episodes_to_training_seed(episodes: list) -> dict:
    """Aggregate a list of episodes into a single training-seed row."""
    if not episodes:
        return {}
    returns = [_safe_float(e.get("return")) for e in episodes]
    success_flags = [bool(e.get("is_success", False)) for e in episodes]
    row = {
        "num_episodes": len(episodes),
        "instant_success_rate": _safe_mean(success_flags),
        "mean_return": _safe_mean(returns),
        "mean_final_range_m": _safe_mean([_safe_float(e.get("final_range_m")) for e in episodes]),
        "mean_final_ata_deg": _safe_mean([_safe_float(e.get("final_ata_deg")) for e in episodes]),
        "prediction_valid_rate": _safe_mean([_safe_float(e.get("prediction_valid_rate")) for e in episodes]),
        "prediction_fallback_rate": _safe_mean([_safe_float(e.get("prediction_fallback_rate")) for e in episodes]),
        "runtime_fallback_rate": _safe_mean([_safe_float(e.get("runtime_fallback_rate")) for e in episodes]),
        "post_warmup_fallback_rate": _safe_mean([_safe_float(e.get("post_warmup_fallback_rate")) for e in episodes]),
        "mean_env_prediction_error_m": _safe_mean([_safe_float(e.get("mean_env_prediction_error_m")) for e in episodes]),
        "median_env_prediction_error_m": _safe_mean([_safe_float(e.get("median_env_prediction_error_m")) for e in episodes]),
        "mean_offline_aligned_error_m": _safe_mean([_safe_float(e.get("mean_offline_aligned_error_m")) for e in episodes]),
        "median_offline_aligned_error_m": _safe_mean([_safe_float(e.get("median_offline_aligned_error_m")) for e in episodes]),
        "unknown_fallback_phase_count": sum(int(e.get("unknown_fallback_phase_count", 0) or 0) for e in episodes),
        "missing_fallback_phase_count": sum(int(e.get("missing_fallback_phase_count", 0) or 0) for e in episodes),
        "configured_current_target_fallback_count": sum(int(e.get("configured_current_target_fallback_count", 0) or 0) for e in episodes),
        "predictor_init_failed_count": sum(int(e.get("predictor_init_failed_count", 0) or 0) for e in episodes),
    }
    return row


def aggregate_training_seeds_to_cross_seed(rows: list, method_metadata: dict) -> dict:
    """Aggregate per-training-seed rows into cross-seed statistics."""
    if not rows:
        return {}

    def col_vals(col):
        return [r[col] for r in rows if col in r]

    invalid = bool(method_metadata.get("allow_random_policy", False))
    if not method_metadata.get("loaded_policy_checkpoint_path"):
        invalid = True

    episodes_per_seed = [r.get("num_episodes", 0) for r in rows]
    scenario_balance_ok = all(
        r.get("scenario_balance_ok") for r in rows
    ) if "scenario_balance_ok" in rows[0] else None

    result = {
        "method": method_metadata.get("method_name", method_metadata.get("method", "unknown")),
        "num_training_seeds": len(rows),
        "num_episodes_per_training_seed": int(np.mean(episodes_per_seed)) if episodes_per_seed else 0,
        "num_evaluation_seeds": len(method_metadata.get("evaluation_seeds", [])),
        "num_scenarios": len(method_metadata.get("per_scenario", {})) if method_metadata.get("per_scenario") else 0,
        "episodes_per_scenario": method_metadata.get("episodes_per_scenario"),
        "scenario_balance_ok": scenario_balance_ok,
        "invalid_for_paper": invalid,
    }

    for col in NUMERIC_AGG_COLS:
        vals = col_vals(col)
        result[f"{col}_mean"] = _safe_mean(vals)
        result[f"{col}_std"] = _safe_std(vals)
        result[f"{col}_ci95"] = _safe_ci95(vals)

    for col in COUNT_COLS:
        vals = col_vals(col)
        result[f"{col}_mean"] = _safe_mean(vals)
        result[f"{col}_std"] = _safe_std(vals)

    return result


def discover_training_seed_dirs(root: Path) -> list:
    """Discover all train_seed*/prediction_metrics.json files."""
    dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("train_seed"):
            metrics_json = p / "prediction_metrics.json"
            if metrics_json.exists():
                dirs.append((p, metrics_json))
    return sorted(dirs, key=lambda x: x[0].name)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage 6F results")
    parser.add_argument("--input", type=str, required=True,
                        help="Root directory containing train_seed*/ subdirs and experiment_plan.json")
    parser.add_argument("--output", type=str, default="outputs/tables/stage6f",
                        help="Output directory for summary tables")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Validate experiment plan
    plan_path = input_root / "experiment_plan.json"
    plan_data = {}
    if plan_path.exists():
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        manifest_warnings = _check_manifest(plan_data)
        if manifest_warnings:
            print("WARNING: Manifest validation issues:")
            for w in manifest_warnings:
                print(f"  - {w}")
        else:
            print("Experiment plan manifest OK.")
    else:
        print(f"WARNING: Experiment plan not found: {plan_path}")

    # Discover training seed directories
    seed_dirs = discover_training_seed_dirs(input_root)
    if not seed_dirs:
        print(f"ERROR: No train_seed*/prediction_metrics.json found in {input_root}")
        sys.exit(1)

    print(f"Found {len(seed_dirs)} training seed directories:")
    for p, _ in seed_dirs:
        print(f"  - {p.name}")

    # Collect per-method, per-training-seed data
    # Structure: {method_name: [{training_seed, per_seed_row, metadata}]}
    method_data = {}

    for seed_dir, metrics_json in seed_dirs:
        training_seed_str = seed_dir.name.replace("train_seed", "")
        try:
            training_seed = int(training_seed_str)
        except ValueError:
            training_seed = training_seed_str

        with open(metrics_json, "r", encoding="utf-8") as f:
            all_methods = json.load(f)

        for method_metrics in all_methods:
            method_name = method_metrics.get("method_name", method_metrics.get("method", "unknown"))
            episodes = _extract_per_seed_episodes(method_metrics)
            per_seed_row = aggregate_episodes_to_training_seed(episodes)

            # Attach scenario info from metadata
            per_seed_row["training_seed"] = training_seed
            per_seed_row["scenario_balance_ok"] = method_metrics.get("scenario_balance_ok")
            per_seed_row["episodes_per_scenario"] = method_metrics.get("episodes_per_scenario")

            method_data.setdefault(method_name, []).append({
                "training_seed": training_seed,
                "row": per_seed_row,
                "metadata": method_metrics,
                "num_episodes": len(episodes),
            })

    # Two-level aggregation: episodes → training seed → cross-seed
    cross_seed_rows = []
    per_seed_flat = []  # For debugging / intermediate CSV

    for method_name, entries in sorted(method_data.items()):
        rows = [e["row"] for e in entries]
        metadata = entries[0]["metadata"]
        cross_row = aggregate_training_seeds_to_cross_seed(rows, metadata)
        cross_seed_rows.append(cross_row)

        for e in entries:
            e["row"]["method"] = method_name
            per_seed_flat.append(e["row"])

    # ---- Output: per-training-seed CSV ----
    per_seed_df = pd.DataFrame(per_seed_flat)
    per_seed_csv = output_dir / "per_training_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False, float_format="%.6f")
    print(f"Saved per-training-seed CSV: {per_seed_csv}")

    # ---- Output: cross-seed summary CSV ----
    cross_df = pd.DataFrame(cross_seed_rows)
    # Reorder columns to preferred order
    ordered_cols = []
    for c in CROSS_SEED_COLS:
        if c in cross_df.columns:
            ordered_cols.append(c)
    for c in cross_df.columns:
        if c not in ordered_cols:
            ordered_cols.append(c)
    cross_df = cross_df[ordered_cols]

    cross_csv = output_dir / "summary.csv"
    cross_df.to_csv(cross_csv, index=False, float_format="%.6f")
    print(f"Saved cross-seed CSV summary: {cross_csv}")

    # ---- Output: JSON with full detail ----
    cross_json = output_dir / "cross_seed_summary.json"
    with open(cross_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_plan": plan_data,
            "schema_version": METRICS_SCHEMA_VERSION,
            "methods": cross_seed_rows,
            "per_training_seed": {method_name: [e["row"] for e in entries]
                                   for method_name, entries in method_data.items()},
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved cross-seed JSON: {cross_json}")

    # ---- Output: LaTeX ----
    tex_path = output_dir / "summary_latex.tex"
    # Select display columns (drop count cols and CI for brevity)
    display_cols = [c for c in ordered_cols
                    if not c.endswith("_ci95")
                    and c not in ("missing_fallback_phase_count_mean",
                                   "configured_current_target_fallback_count_mean",
                                   "predictor_init_failed_count_mean")]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Stage 6F Ablation Summary (Cross-Training-Seed Mean\\(Std))}\n")
        f.write("\\label{tab:stage6f_summary}\n")
        f.write("\\begin{tabular}{" + "l" + "r" * (len(display_cols) - 1) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(display_cols) + " \\\n")
        f.write("\\hline\n")
        for _, row in cross_df.iterrows():
            cells = []
            for col in display_cols:
                val = row[col]
                if col == "method":
                    cells.append(str(val))
                elif col == "invalid_for_paper":
                    cells.append("Yes" if val else "No")
                elif col == "scenario_balance_ok":
                    cells.append("Yes" if val else ("No" if val is not None else "N/A"))
                elif pd.isna(val):
                    cells.append("-")
                elif isinstance(val, bool):
                    cells.append("Yes" if val else "No")
                else:
                    cells.append(f"{val:.4f}")
            f.write(" & ".join(cells) + " \\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX summary: {tex_path}")

    # ---- Output: Markdown ----
    md_path = output_dir / "summary_markdown.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage 6F Ablation Summary (Cross-Training-Seed)\n\n")
        f.write(f"**Schema Version**: {METRICS_SCHEMA_VERSION}\n\n")
        if plan_data:
            f.write(f"**Experiment Plan**: `{plan_data.get('git_commit', 'unknown')}` on `{plan_data.get('branch', 'unknown')}`\n\n")
        f.write("| " + " | ".join(ordered_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(ordered_cols)) + "|\n")
        for _, row in cross_df.iterrows():
            cells = []
            for col in ordered_cols:
                val = row[col]
                if col == "method":
                    cells.append(str(val))
                elif col == "invalid_for_paper":
                    cells.append("Yes" if val else "No")
                elif col == "scenario_balance_ok":
                    cells.append("Yes" if val else ("No" if val is not None else "N/A"))
                elif pd.isna(val):
                    cells.append("-")
                elif isinstance(val, bool):
                    cells.append("Yes" if val else "No")
                else:
                    cells.append(f"{val:.4f}")
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")
        if cross_df["invalid_for_paper"].any():
            invalid_methods = cross_df.loc[cross_df["invalid_for_paper"], "method"].tolist()
            f.write(f"**Warning**: The following methods are marked invalid for paper: {invalid_methods}\n")
    print(f"Saved Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
