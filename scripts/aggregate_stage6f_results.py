#!/usr/bin/env python3
"""
Aggregate Stage 6F ablation results into paper-ready summary tables.

Reads comparison evaluation JSON/CSV and outputs:
    - summary.csv
    - summary_latex.tex
    - summary_markdown.md

Usage:
    python scripts/aggregate_stage6f_results.py \
        --input outputs/tables/stage6f_full_ablation \
        --output outputs/tables/stage6f
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


UNIFIED_COLUMNS = [
    "method",
    "num_seeds",
    "num_episodes",
    "success_rate_mean",
    "success_rate_std",
    "mean_return_mean",
    "mean_return_std",
    "mean_final_range_m_mean",
    "mean_final_ata_deg_mean",
    "prediction_valid_rate_mean",
    "fallback_rate_mean",
    "runtime_fallback_rate_mean",
    "post_warmup_fallback_rate_mean",
    "mean_env_prediction_error_m_mean",
    "median_env_prediction_error_m_mean",
    "mean_offline_aligned_error_m_mean",
    "median_offline_aligned_error_m_mean",
    "unknown_fallback_phase_count",
    "missing_fallback_phase_count",
    "configured_current_target_fallback_count",
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
    return float(np.std(clean)) if clean else np.nan


def aggregate_method(method_metrics: dict) -> dict:
    """Aggregate metrics for a single method from comparison JSON entry."""
    episodes = method_metrics.get("raw_episodes", [])
    if not episodes:
        episodes = []
        for seed_eps in method_metrics.get("per_seed", {}).values():
            episodes.extend(seed_eps)

    invalid = bool(method_metrics.get("allow_random_policy", False))
    if not method_metrics.get("loaded_policy_checkpoint_path"):
        invalid = True

    returns = [_safe_float(e.get("return")) for e in episodes]
    success_flags = [bool(e.get("is_success", False)) for e in episodes]
    final_ranges = [_safe_float(e.get("final_range_m")) for e in episodes]
    final_atas = [_safe_float(e.get("final_ata_deg")) for e in episodes]

    row = {
        "method": method_metrics.get("method_name", method_metrics.get("method", "unknown")),
        "num_seeds": len(method_metrics.get("per_seed", {})),
        "num_episodes": len(episodes),
        "success_rate_mean": _safe_mean(success_flags),
        "success_rate_std": _safe_std(success_flags),
        "mean_return_mean": _safe_mean(returns),
        "mean_return_std": _safe_std(returns),
        "mean_final_range_m_mean": _safe_mean(final_ranges),
        "mean_final_ata_deg_mean": _safe_mean(final_atas),
        "prediction_valid_rate_mean": _safe_float(method_metrics.get("mean_prediction_valid_rate")),
        "fallback_rate_mean": _safe_float(method_metrics.get("mean_prediction_fallback_rate")),
        "runtime_fallback_rate_mean": _safe_float(method_metrics.get("mean_runtime_fallback_rate")),
        "post_warmup_fallback_rate_mean": _safe_float(method_metrics.get("mean_post_warmup_fallback_rate")),
        "mean_env_prediction_error_m_mean": _safe_float(method_metrics.get("mean_env_prediction_error_m")),
        "median_env_prediction_error_m_mean": _safe_float(method_metrics.get("median_env_prediction_error_m")),
        "mean_offline_aligned_error_m_mean": _safe_float(method_metrics.get("mean_offline_aligned_error_m")),
        "median_offline_aligned_error_m_mean": _safe_float(method_metrics.get("median_offline_aligned_error_m")),
        "unknown_fallback_phase_count": int(method_metrics.get("unknown_fallback_phase_count", 0)),
        "missing_fallback_phase_count": int(method_metrics.get("missing_fallback_phase_count", 0)),
        "configured_current_target_fallback_count": int(method_metrics.get("configured_current_target_fallback_count", 0)),
        "invalid_for_paper": invalid,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage 6F results")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing prediction_metrics.json")
    parser.add_argument("--output", type=str, default="outputs/tables/stage6f",
                        help="Output directory for summary tables")
    args = parser.parse_args()

    json_path = Path(args.input) / "prediction_metrics.json"
    if not json_path.exists():
        print(f"ERROR: Metrics JSON not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        all_methods = json.load(f)

    rows = [aggregate_method(m) for m in all_methods]
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)

    os.makedirs(args.output, exist_ok=True)

    csv_path = Path(args.output) / "summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV summary: {csv_path}")

    tex_path = Path(args.output) / "summary_latex.tex"
    display_cols = [c for c in UNIFIED_COLUMNS if c not in ("missing_fallback_phase_count", "configured_current_target_fallback_count")]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Stage 6F Ablation Summary}\n")
        f.write("\\label{tab:stage6f_summary}\n")
        f.write("\\begin{tabular}{" + "l" + "r" * (len(display_cols) - 1) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(display_cols) + " \\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            cells = []
            for col in display_cols:
                val = row[col]
                if col == "method":
                    cells.append(str(val))
                elif col == "invalid_for_paper":
                    cells.append("Yes" if val else "No")
                elif pd.isna(val):
                    cells.append("-")
                else:
                    cells.append(f"{val:.4f}")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX summary: {tex_path}")

    md_path = Path(args.output) / "summary_markdown.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Stage 6F Ablation Summary\n\n")
        # Simple markdown table
        f.write("| " + " | ".join(UNIFIED_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(UNIFIED_COLUMNS)) + "|\n")
        for _, row in df.iterrows():
            cells = []
            for col in UNIFIED_COLUMNS:
                val = row[col]
                if col == "method":
                    cells.append(str(val))
                elif col == "invalid_for_paper":
                    cells.append("Yes" if val else "No")
                elif pd.isna(val):
                    cells.append("-")
                else:
                    cells.append(f"{val:.4f}")
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")
        if df["invalid_for_paper"].any():
            invalid_methods = df.loc[df["invalid_for_paper"], "method"].tolist()
            f.write(f"**Warning**: The following methods are marked invalid for paper: {invalid_methods}\n")
    print(f"Saved Markdown summary: {md_path}")


if __name__ == "__main__":
    import sys
    main()
