#!/usr/bin/env python3
"""
Stage 6F.5 Results Analysis Script.

Analyzes re-ablation results from feasible_geometry and maneuvering_target suites.
Produces:
  - overall summary
  - per-scenario summary
  - CV vs CA delta table
  - neural vs classical predictor table
  - target acceleration vs prediction error correlation (maneuvering target only)

Usage:
    python scripts/analyze_stage6f5_results.py \
        --input outputs/tables/stage6f5_feasible_geometry \
        --output outputs/tables/stage6f5_feasible_geometry/analysis

    python scripts/analyze_stage6f5_results.py \
        --input outputs/tables/stage6f5_maneuvering_target \
        --output outputs/tables/stage6f5_maneuvering_target/analysis
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

METRICS_SCHEMA_VERSION = "6f.2"


def load_cross_seed_summary(path: Path) -> dict:
    cross_json = path / "cross_seed_summary.json"
    if cross_json.exists():
        with open(cross_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def discover_training_seeds(raw_root: Path) -> list:
    if not raw_root.exists():
        return []
    seeds = []
    for d in raw_root.iterdir():
        if d.is_dir() and d.name.startswith("train_seed"):
            if (d / "prediction_metrics.json").exists():
                try:
                    seeds.append(int(d.name.replace("train_seed", "")))
                except ValueError:
                    pass
    return sorted(seeds)


def load_raw_prediction_metrics(raw_root: Path, training_seed: int) -> list:
    p = raw_root / f"train_seed{training_seed}" / "prediction_metrics.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_episode_df(raw_root: Path, seeds: list) -> pd.DataFrame:
    rows = []
    for ts in seeds:
        data = load_raw_prediction_metrics(raw_root, ts)
        for m in data:
            method = m.get("method_name", m.get("method", "unknown"))
            for ep in m.get("raw_episodes", []):
                rows.append({
                    "method": method,
                    "training_seed": ts,
                    "scenario": ep.get("scenario", "unknown"),
                    "return": float(ep.get("return", np.nan)) if ep.get("return") is not None else np.nan,
                    "is_success": bool(ep.get("is_success", False)),
                    "reason": ep.get("reason", "unknown"),
                    "mean_env_prediction_error_m": float(ep.get("mean_env_prediction_error_m", np.nan)) if ep.get("mean_env_prediction_error_m") is not None else np.nan,
                    "mean_offline_aligned_error_m": float(ep.get("mean_offline_aligned_error_m", np.nan)) if ep.get("mean_offline_aligned_error_m") is not None else np.nan,
                    "prediction_valid_rate": float(ep.get("prediction_valid_rate", np.nan)) if ep.get("prediction_valid_rate") is not None else np.nan,
                    "runtime_fallback_rate": float(ep.get("runtime_fallback_rate", np.nan)) if ep.get("runtime_fallback_rate") is not None else np.nan,
                })
    return pd.DataFrame(rows)


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(df["method"].unique()):
        mdf = df[df["method"] == method]
        rows.append({
            "method": method,
            "n_episodes": len(mdf),
            "success_rate": mdf["is_success"].mean(),
            "mean_return": mdf["return"].mean(),
            "std_return": mdf["return"].std(ddof=1),
            "mean_env_error": mdf["mean_env_prediction_error_m"].mean(),
            "mean_offline_error": mdf["mean_offline_aligned_error_m"].mean(),
        })
    return pd.DataFrame(rows)


def build_per_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(df["method"].unique()):
        for scenario in sorted(df["scenario"].unique()):
            sdf = df[(df["method"] == method) & (df["scenario"] == scenario)]
            if len(sdf) == 0:
                continue
            rows.append({
                "method": method,
                "scenario": scenario,
                "n_episodes": len(sdf),
                "success_rate": sdf["is_success"].mean(),
                "mean_return": sdf["return"].mean(),
                "std_return": sdf["return"].std(ddof=1),
                "mean_env_error": sdf["mean_env_prediction_error_m"].mean(),
            })
    return pd.DataFrame(rows)


def build_cv_ca_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CV vs CA differences per scenario."""
    cv_df = df[df["method"] == "cv_prediction"]
    ca_df = df[df["method"] == "ca_prediction"]
    if cv_df.empty or ca_df.empty:
        return pd.DataFrame()

    rows = []
    for scenario in sorted(df["scenario"].unique()):
        cv_sc = cv_df[cv_df["scenario"] == scenario]
        ca_sc = ca_df[ca_df["scenario"] == scenario]
        if len(cv_sc) == 0 or len(ca_sc) == 0:
            continue
        rows.append({
            "scenario": scenario,
            "cv_success_rate": cv_sc["is_success"].mean(),
            "ca_success_rate": ca_sc["is_success"].mean(),
            "delta_success_rate": ca_sc["is_success"].mean() - cv_sc["is_success"].mean(),
            "cv_mean_return": cv_sc["return"].mean(),
            "ca_mean_return": ca_sc["return"].mean(),
            "delta_mean_return": ca_sc["return"].mean() - cv_sc["return"].mean(),
            "cv_env_error": cv_sc["mean_env_prediction_error_m"].mean(),
            "ca_env_error": ca_sc["mean_env_prediction_error_m"].mean(),
            "delta_env_error": ca_sc["mean_env_prediction_error_m"].mean() - cv_sc["mean_env_prediction_error_m"].mean(),
        })
    return pd.DataFrame(rows)


def build_neural_vs_classical(df: pd.DataFrame) -> pd.DataFrame:
    """Compare neural (LSTM/GRU) vs classical (CV/CA/no_prediction)."""
    classical = ["no_prediction", "cv_prediction", "ca_prediction"]
    neural = ["lstm_frozen", "gru_frozen"]

    rows = []
    for scenario in sorted(df["scenario"].unique()):
        c_df = df[(df["method"].isin(classical)) & (df["scenario"] == scenario)]
        n_df = df[(df["method"].isin(neural)) & (df["scenario"] == scenario)]
        if len(c_df) == 0 or len(n_df) == 0:
            continue
        rows.append({
            "scenario": scenario,
            "classical_success_rate": c_df["is_success"].mean(),
            "neural_success_rate": n_df["is_success"].mean(),
            "delta_success_rate": n_df["is_success"].mean() - c_df["is_success"].mean(),
            "classical_mean_return": c_df["return"].mean(),
            "neural_mean_return": n_df["return"].mean(),
            "delta_mean_return": n_df["return"].mean() - c_df["return"].mean(),
        })
    return pd.DataFrame(rows)


def build_target_accel_correlation(df: pd.DataFrame, experiment_plan: dict) -> pd.DataFrame:
    """For maneuvering target, correlate target acceleration RMS with prediction error."""
    # target_acceleration_rms is in scenario metadata, not episode data
    # We approximate by grouping scenario-level metadata with episode-level errors
    scenarios_meta = experiment_plan.get("scenarios", {})
    if not scenarios_meta:
        return pd.DataFrame()

    rows = []
    for method in sorted(df["method"].unique()):
        for scenario in sorted(df["scenario"].unique()):
            sdf = df[(df["method"] == method) & (df["scenario"] == scenario)]
            sc_meta = scenarios_meta.get(scenario, {}).get("metadata", {})
            target_accel_rms = sc_meta.get("target_acceleration_rms")
            if len(sdf) == 0 or target_accel_rms is None:
                continue
            rows.append({
                "method": method,
                "scenario": scenario,
                "target_acceleration_rms": target_accel_rms,
                "mean_env_error": sdf["mean_env_prediction_error_m"].mean(),
                "mean_offline_error": sdf["mean_offline_aligned_error_m"].mean(),
                "success_rate": sdf["is_success"].mean(),
                "mean_return": sdf["return"].mean(),
            })
    return pd.DataFrame(rows)


def render_analysis_md(
    overall: pd.DataFrame,
    per_scenario: pd.DataFrame,
    cv_ca_delta: pd.DataFrame,
    neural_classical: pd.DataFrame,
    accel_corr: pd.DataFrame,
    experiment_plan: dict,
) -> str:
    lines = []
    lines.append("# Stage 6F.5 Results Analysis")
    lines.append("")
    suite = experiment_plan.get("suite", "unknown")
    lines.append(f"**Suite**: {suite}")
    lines.append(f"**Schema Version**: {METRICS_SCHEMA_VERSION}")
    lines.append("")

    # Overall summary
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Method | Episodes | Success Rate | Mean Return | Env Error (m) |")
    lines.append("|--------|---------:|-------------:|------------:|--------------:|")
    for _, row in overall.iterrows():
        env_err = f"{row['mean_env_error']:.1f}" if not np.isnan(row['mean_env_error']) else "N/A"
        lines.append(
            f"| {row['method']} | {row['n_episodes']} | {row['success_rate']:.2%} | "
            f"{row['mean_return']:.1f} | {env_err} |"
        )
    lines.append("")

    # Per-scenario
    lines.append("## Per-Scenario Summary")
    lines.append("")
    lines.append("| Method | Scenario | Episodes | Success Rate | Mean Return |")
    lines.append("|--------|----------|---------:|-------------:|------------:|")
    for _, row in per_scenario.iterrows():
        lines.append(
            f"| {row['method']} | {row['scenario']} | {row['n_episodes']} | "
            f"{row['success_rate']:.2%} | {row['mean_return']:.1f} |"
        )
    lines.append("")

    # CV vs CA
    if not cv_ca_delta.empty:
        lines.append("## CV vs CA Delta")
        lines.append("")
        lines.append("| Scenario | CV SR | CA SR | Delta SR | CV Return | CA Return | Delta Return |")
        lines.append("|----------|------:|------:|---------:|----------:|----------:|-------------:|")
        for _, row in cv_ca_delta.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['cv_success_rate']:.2%} | {row['ca_success_rate']:.2%} | "
                f"{row['delta_success_rate']:+.2%} | {row['cv_mean_return']:.1f} | {row['ca_mean_return']:.1f} | "
                f"{row['delta_mean_return']:+.1f} |"
            )
        lines.append("")

    # Neural vs classical
    if not neural_classical.empty:
        lines.append("## Neural vs Classical Predictors")
        lines.append("")
        lines.append("| Scenario | Classical SR | Neural SR | Delta SR | Classical Return | Neural Return | Delta Return |")
        lines.append("|----------|-------------:|----------:|---------:|-----------------:|--------------:|-------------:|")
        for _, row in neural_classical.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['classical_success_rate']:.2%} | {row['neural_success_rate']:.2%} | "
                f"{row['delta_success_rate']:+.2%} | {row['classical_mean_return']:.1f} | {row['neural_mean_return']:.1f} | "
                f"{row['delta_mean_return']:+.1f} |"
            )
        lines.append("")

    # Acceleration correlation
    if not accel_corr.empty:
        lines.append("## Target Acceleration vs Prediction Error")
        lines.append("")
        lines.append("| Method | Scenario | Target Accel RMS | Env Error | Offline Error | Success Rate |")
        lines.append("|--------|----------|-----------------:|----------:|--------------:|-------------:|")
        for _, row in accel_corr.iterrows():
            off_err = f"{row['mean_offline_error']:.1f}" if not np.isnan(row['mean_offline_error']) else "N/A"
            lines.append(
                f"| {row['method']} | {row['scenario']} | {row['target_acceleration_rms']:.1f} | "
                f"{row['mean_env_error']:.1f} | {off_err} | {row['success_rate']:.2%} |"
            )
        lines.append("")

        # Compute correlation
        for method in sorted(accel_corr["method"].unique()):
            mdf = accel_corr[accel_corr["method"] == method]
            if len(mdf) >= 2:
                corr = mdf["target_acceleration_rms"].corr(mdf["mean_env_error"])
                lines.append(f"- **{method}**: correlation(target_accel, env_error) = {corr:.3f}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 6F.5 Results Analysis")
    parser.add_argument("--input", type=str, required=True,
                        help="Root directory containing train_seed*/ subdirs and experiment_plan.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for analysis artifacts")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Load experiment plan
    plan_path = input_root / "experiment_plan.json"
    experiment_plan = {}
    if plan_path.exists():
        with open(plan_path, "r", encoding="utf-8") as f:
            experiment_plan = json.load(f)

    seeds = discover_training_seeds(input_root)
    if not seeds:
        print(f"WARNING: No train_seed*/prediction_metrics.json found in {input_root}")
        # Create empty dataframes for dry-run compatibility
        episode_df = pd.DataFrame(columns=["method", "scenario", "is_success", "return", "mean_env_prediction_error_m", "mean_offline_aligned_error_m"])
    else:
        episode_df = build_episode_df(input_root, seeds)
        print(f"Loaded {len(episode_df)} episodes from {len(seeds)} training seeds")

    overall = build_overall_summary(episode_df)
    per_scenario = build_per_scenario_summary(episode_df)
    cv_ca_delta = build_cv_ca_delta(episode_df)
    neural_classical = build_neural_vs_classical(episode_df)
    accel_corr = build_target_accel_correlation(episode_df, experiment_plan)

    # Save CSVs
    overall.to_csv(output_dir / "overall_summary.csv", index=False, float_format="%.6f")
    per_scenario.to_csv(output_dir / "per_scenario_summary.csv", index=False, float_format="%.6f")
    if not cv_ca_delta.empty:
        cv_ca_delta.to_csv(output_dir / "cv_ca_delta.csv", index=False, float_format="%.6f")
    if not neural_classical.empty:
        neural_classical.to_csv(output_dir / "neural_vs_classical.csv", index=False, float_format="%.6f")
    if not accel_corr.empty:
        accel_corr.to_csv(output_dir / "target_accel_correlation.csv", index=False, float_format="%.6f")

    # Save Markdown report
    md = render_analysis_md(overall, per_scenario, cv_ca_delta, neural_classical, accel_corr, experiment_plan)
    with open(output_dir / "analysis_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved analysis report: {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
