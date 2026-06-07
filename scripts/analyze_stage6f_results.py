#!/usr/bin/env python3
"""
Stage 6F.4 Results Diagnosis and Audit Report Generator.

Reads aggregated Stage 6F results and produces:
  - stage6f_diagnosis.md          (overall ranking, per-scenario, seed stability,
                                   prediction health, CV/CA failure diagnosis)
  - stage6f_method_summary.csv    (method-level cross-seed stats)
  - stage6f_scenario_summary.csv  (method × scenario breakdown)
  - stage6f_seed_summary.csv      (method × training_seed stability)
  - stage6f_failure_cases.csv     (worst N episodes per method)

Usage:
    python scripts/analyze_stage6f_results.py \
        --input outputs/tables/stage6f \
        --raw outputs/tables/stage6f_full_ablation \
        --output outputs/tables/stage6f_diagnosis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_SCHEMA_VERSION = "6f.2"


def _safe_float(val):
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load_cross_seed_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_prediction_metrics(raw_root: Path, training_seed: int) -> list:
    """Load prediction_metrics.json for a given training seed."""
    p = raw_root / f"train_seed{training_seed}" / "prediction_metrics.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_method_summary(cross_data: dict) -> pd.DataFrame:
    """Build method_summary.csv from cross_seed_summary."""
    rows = []
    for m in cross_data.get("methods", []):
        rows.append({
            "method": m["method"],
            "num_training_seeds": m.get("num_training_seeds", 0),
            "num_episodes_per_training_seed": m.get("num_episodes_per_training_seed", 0),
            "scenario_balance_ok": m.get("scenario_balance_ok"),
            "invalid_for_paper": m.get("invalid_for_paper", False),
            "success_rate_mean": m.get("instant_success_rate_mean", np.nan),
            "success_rate_std": m.get("instant_success_rate_std", np.nan),
            "success_rate_ci95": m.get("instant_success_rate_ci95", np.nan),
            "mean_return_mean": m.get("mean_return_mean", np.nan),
            "mean_return_std": m.get("mean_return_std", np.nan),
            "mean_final_range_m_mean": m.get("mean_final_range_m_mean", np.nan),
            "mean_final_ata_deg_mean": m.get("mean_final_ata_deg_mean", np.nan),
            "prediction_valid_rate_mean": m.get("prediction_valid_rate_mean", np.nan),
            "runtime_fallback_rate_mean": m.get("runtime_fallback_rate_mean", np.nan),
            "post_warmup_fallback_rate_mean": m.get("post_warmup_fallback_rate_mean", np.nan),
            "mean_env_prediction_error_m_mean": m.get("mean_env_prediction_error_m_mean", np.nan),
            "mean_offline_aligned_error_m_mean": m.get("mean_offline_aligned_error_m_mean", np.nan),
            "unknown_fallback_phase_count_mean": m.get("unknown_fallback_phase_count_mean", np.nan),
            "missing_fallback_phase_count_mean": m.get("missing_fallback_phase_count_mean", np.nan),
            "configured_current_target_fallback_count_mean": m.get("configured_current_target_fallback_count_mean", np.nan),
            "predictor_init_failed_count_mean": m.get("predictor_init_failed_count_mean", np.nan),
        })
    return pd.DataFrame(rows)


def build_scenario_summary(cross_data: dict, raw_root: Path) -> pd.DataFrame:
    """Build scenario_summary.csv by aggregating episodes per scenario."""
    rows = []
    per_training_seed = cross_data.get("per_training_seed", {})
    if not raw_root.exists():
        return pd.DataFrame(rows)
    # Discover training seeds from raw_root
    seed_dirs = sorted([
        int(d.name.replace("train_seed", ""))
        for d in raw_root.iterdir()
        if d.is_dir() and d.name.startswith("train_seed")
        and (d / "prediction_metrics.json").exists()
    ])

    for method_name in sorted(per_training_seed.keys()):
        # Gather all episodes for this method across training seeds
        all_episodes = []
        for ts in seed_dirs:
            raw_methods = load_raw_prediction_metrics(raw_root, ts)
            for m in raw_methods:
                if m.get("method_name") == method_name or m.get("method") == method_name:
                    eps = m.get("raw_episodes", [])
                    all_episodes.extend(eps)

        if not all_episodes:
            continue

        # Group by scenario
        scenarios = {}
        for ep in all_episodes:
            sc = ep.get("scenario", "unknown")
            scenarios.setdefault(sc, []).append(ep)

        for sc, eps in sorted(scenarios.items()):
            success_flags = [bool(e.get("is_success", False)) for e in eps]
            returns = [_safe_float(e.get("return")) for e in eps]
            rows.append({
                "method": method_name,
                "scenario": sc,
                "num_episodes": len(eps),
                "success_rate": np.mean(success_flags) if success_flags else np.nan,
                "mean_return": np.mean(returns) if returns else np.nan,
                "std_return": np.std(returns, ddof=0) if len(returns) > 1 else np.nan,
                "mean_final_range_m": np.mean([_safe_float(e.get("final_range_m")) for e in eps]),
                "mean_final_ata_deg": np.mean([_safe_float(e.get("final_ata_deg")) for e in eps]),
            })
    return pd.DataFrame(rows)


def build_seed_summary(cross_data: dict) -> pd.DataFrame:
    """Build seed_summary.csv from per_training_seed data."""
    rows = []
    per_training_seed = cross_data.get("per_training_seed", {})
    for method_name, seeds in sorted(per_training_seed.items()):
        success_rates = []
        for s in seeds:
            sr = s.get("instant_success_rate", np.nan)
            success_rates.append(sr)
            rows.append({
                "method": method_name,
                "training_seed": s.get("training_seed", -1),
                "num_episodes": s.get("num_episodes", 0),
                "success_rate": sr,
                "mean_return": s.get("mean_return", np.nan),
                "mean_final_range_m": s.get("mean_final_range_m", np.nan),
                "mean_final_ata_deg": s.get("mean_final_ata_deg", np.nan),
                "prediction_valid_rate": s.get("prediction_valid_rate", np.nan),
                "runtime_fallback_rate": s.get("runtime_fallback_rate", np.nan),
            })
        # Add outlier flag if any seed is >2 std from mean
        if len(success_rates) >= 3:
            mean_sr = np.mean(success_rates)
            std_sr = np.std(success_rates, ddof=1)
            if std_sr > 0:
                for r in rows[-len(success_rates):]:
                    r["seed_outlier"] = abs(r["success_rate"] - mean_sr) > 2 * std_sr
            else:
                for r in rows[-len(success_rates):]:
                    r["seed_outlier"] = False
        else:
            for r in rows[-len(seeds):]:
                r["seed_outlier"] = False
    return pd.DataFrame(rows)


def build_failure_cases(cross_data: dict, raw_root: Path, top_n: int = 5) -> pd.DataFrame:
    """Extract worst N episodes per method by return."""
    rows = []
    per_training_seed = cross_data.get("per_training_seed", {})
    if not raw_root.exists():
        return pd.DataFrame(rows)
    seed_dirs = sorted([
        int(d.name.replace("train_seed", ""))
        for d in raw_root.iterdir()
        if d.is_dir() and d.name.startswith("train_seed")
        and (d / "prediction_metrics.json").exists()
    ])

    for method_name in sorted(per_training_seed.keys()):
        all_episodes = []
        for ts in seed_dirs:
            raw_methods = load_raw_prediction_metrics(raw_root, ts)
            for m in raw_methods:
                if m.get("method_name") == method_name or m.get("method") == method_name:
                    eps = m.get("raw_episodes", [])
                    for ep in eps:
                        ep["_training_seed"] = ts
                    all_episodes.extend(eps)

        if not all_episodes:
            continue
        # Sort by return ascending (worst first)
        all_episodes.sort(key=lambda e: _safe_float(e.get("return", float("inf"))))
        for ep in all_episodes[:top_n]:
            rows.append({
                "method": method_name,
                "training_seed": ep.get("_training_seed", -1),
                "evaluation_seed": ep.get("evaluation_seed", -1),
                "episode_seed": ep.get("episode_seed", -1),
                "scenario": ep.get("scenario", "unknown"),
                "return": _safe_float(ep.get("return")),
                "length": ep.get("length", 0),
                "final_range_m": _safe_float(ep.get("final_range_m")),
                "final_ata_deg": _safe_float(ep.get("final_ata_deg")),
                "reason": ep.get("reason", "unknown"),
                "is_success": ep.get("is_success", False),
                "is_crash": ep.get("is_crash", False),
                "is_timeout": ep.get("is_timeout", False),
                "is_out_of_bounds": ep.get("is_out_of_bounds", False),
            })
    return pd.DataFrame(rows)


def compute_cv_ca_diagnosis(method_summary: pd.DataFrame) -> list:
    """Check if CV/CA methods underperform no_prediction baseline."""
    diagnosis = []
    baseline = method_summary[method_summary["method"] == "no_prediction"]
    if baseline.empty:
        return diagnosis
    baseline_sr = baseline.iloc[0]["success_rate_mean"]

    for _, row in method_summary.iterrows():
        method = row["method"]
        if method in ("cv_prediction", "ca_prediction"):
            sr = row["success_rate_mean"]
            if not np.isnan(sr) and not np.isnan(baseline_sr) and sr < baseline_sr:
                diagnosis.append({
                    "method": method,
                    "success_rate": sr,
                    "baseline_success_rate": baseline_sr,
                    "delta": sr - baseline_sr,
                    "cv_ca_underperform_baseline": True,
                    "note": f"{method} success rate ({sr:.2%}) is below no_prediction baseline ({baseline_sr:.2%})",
                })
            else:
                diagnosis.append({
                    "method": method,
                    "success_rate": sr,
                    "baseline_success_rate": baseline_sr,
                    "delta": sr - baseline_sr,
                    "cv_ca_underperform_baseline": False,
                    "note": "",
                })
    return diagnosis


def render_diagnosis_md(
    method_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    failure_cases: pd.DataFrame,
    cv_ca_diag: list,
    cross_data: dict,
) -> str:
    """Render the full diagnosis markdown report."""
    lines = []
    lines.append("# Stage 6F Results Diagnosis Report")
    lines.append("")
    plan = cross_data.get("experiment_plan", {})
    lines.append(f"**Schema Version**: {METRICS_SCHEMA_VERSION}")
    lines.append(f"**Git Commit**: `{plan.get('git_commit', 'unknown')}`")
    lines.append(f"**Branch**: `{plan.get('branch', 'unknown')}`")
    lines.append(f"**Timestamp**: {plan.get('timestamp', 'unknown')}")
    lines.append("")

    # A. Overall Method Ranking
    lines.append("## A. Overall Method Ranking")
    lines.append("")
    lines.append("| Rank | Method | Success Rate | Mean Return | Final Range (m) | Final ATA (deg) | Valid? |")
    lines.append("|------|--------|-------------:|------------:|----------------:|----------------:|--------|")
    sorted_methods = method_summary.sort_values("success_rate_mean", ascending=False).reset_index(drop=True)
    for i, row in sorted_methods.iterrows():
        valid_str = "Yes" if not row["invalid_for_paper"] else "**NO**"
        lines.append(
            f"| {i+1} | {row['method']} | "
            f"{row['success_rate_mean']:.2%} ± {row['success_rate_std']:.2%} | "
            f"{row['mean_return_mean']:.1f} ± {row['mean_return_std']:.1f} | "
            f"{row['mean_final_range_m_mean']:.1f} | "
            f"{row['mean_final_ata_deg_mean']:.1f} | {valid_str} |"
        )
    lines.append("")

    # B. Per-Scenario Breakdown
    lines.append("## B. Per-Scenario Breakdown")
    lines.append("")
    if not scenario_summary.empty:
        scenarios = sorted(scenario_summary["scenario"].unique())
        for sc in scenarios:
            lines.append(f"### Scenario: {sc}")
            lines.append("")
            lines.append("| Method | Episodes | Success Rate | Mean Return | Final Range (m) |")
            lines.append("|--------|---------:|-------------:|------------:|----------------:|")
            sc_df = scenario_summary[scenario_summary["scenario"] == sc].sort_values("success_rate", ascending=False)
            for _, row in sc_df.iterrows():
                lines.append(
                    f"| {row['method']} | {row['num_episodes']} | "
                    f"{row['success_rate']:.2%} | {row['mean_return']:.1f} | {row['mean_final_range_m']:.1f} |"
                )
            lines.append("")
    else:
        lines.append("_No scenario-level data available._")
        lines.append("")

    # C. Per-Training-Seed Stability
    lines.append("## C. Per-Training-Seed Stability")
    lines.append("")
    if not seed_summary.empty:
        methods = sorted(seed_summary["method"].unique())
        for method in methods:
            lines.append(f"### {method}")
            lines.append("")
            lines.append("| Seed | Episodes | Success Rate | Mean Return | Outlier? |")
            lines.append("|------|---------:|-------------:|------------:|:--------:|")
            mdf = seed_summary[seed_summary["method"] == method].sort_values("training_seed")
            for _, row in mdf.iterrows():
                outlier = "**YES**" if row.get("seed_outlier", False) else "No"
                lines.append(
                    f"| {row['training_seed']} | {row['num_episodes']} | "
                    f"{row['success_rate']:.2%} | {row['mean_return']:.1f} | {outlier} |"
                )
            # Compute seed stability stats
            srs = mdf["success_rate"].dropna().values
            if len(srs) >= 2:
                lines.append("")
                lines.append(f"- **Mean success rate across seeds**: {np.mean(srs):.2%}")
                lines.append(f"- **Std across seeds**: {np.std(srs, ddof=1):.2%}")
                lines.append(f"- **Range (max - min)**: {np.max(srs) - np.min(srs):.2%}")
            lines.append("")
    else:
        lines.append("_No seed-level data available._")
        lines.append("")

    # D. Prediction Health Diagnosis
    lines.append("## D. Prediction Health Diagnosis")
    lines.append("")
    for _, row in sorted_methods.iterrows():
        method = row["method"]
        pvr = row["prediction_valid_rate_mean"]
        rfr = row["runtime_fallback_rate_mean"]
        pwfr = row["post_warmup_fallback_rate_mean"]
        env_err = row["mean_env_prediction_error_m_mean"]
        offline_err = row["mean_offline_aligned_error_m_mean"]

        notes = []
        if not np.isnan(pvr):
            if pvr < 0.5:
                notes.append(f"prediction_valid_rate very low ({pvr:.2%})")
            elif pvr < 0.9:
                notes.append(f"prediction_valid_rate moderate ({pvr:.2%})")
        if not np.isnan(rfr) and rfr > 0.1:
            notes.append(f"runtime_fallback_rate high ({rfr:.2%})")
        if not np.isnan(pwfr) and pwfr > 0.05:
            notes.append(f"post_warmup_fallback_rate elevated ({pwfr:.2%})")
        if not np.isnan(env_err) and env_err > 100:
            notes.append(f"env prediction error large ({env_err:.1f} m)")
        if not np.isnan(offline_err) and offline_err > 1000:
            notes.append(f"offline aligned error large ({offline_err:.1f} m)")

        if notes:
            lines.append(f"- **{method}**: {'; '.join(notes)}")
        else:
            lines.append(f"- **{method}**: prediction pipeline healthy")
    lines.append("")

    # E. CV/CA Failure Diagnosis
    lines.append("## E. CV/CA Failure Diagnosis")
    lines.append("")
    if cv_ca_diag:
        underperformers = [d for d in cv_ca_diag if d["cv_ca_underperform_baseline"]]
        if underperformers:
            lines.append("**WARNING**: The following CV/CA methods underperform the `no_prediction` baseline:")
            lines.append("")
            for d in underperformers:
                lines.append(f"- {d['note']}")
            lines.append("")
            lines.append("### Possible Causes")
            lines.append("- The neural predictor (CV/CA) is not aligned with the actual target dynamics.")
            lines.append("- Prediction errors accumulate and mislead the policy into poor engagement geometry.")
            lines.append("- The frozen PPO policy was trained with a different prediction backend;")
            lines.append("  CV/CA checkpoint mismatch may cause degraded performance.")
            lines.append("- Check `mean_offline_aligned_error_m` vs `mean_env_prediction_error_m` —")
            lines.append("  large discrepancies indicate predictor-to-environment misalignment.")
        else:
            lines.append("All CV/CA methods meet or exceed the `no_prediction` baseline.")
    else:
        lines.append("_No_prediction baseline not found; skipping CV/CA diagnosis._")
    lines.append("")

    # F. Failure Case Extraction
    lines.append("## F. Failure Case Extraction (Worst 5 Episodes per Method)")
    lines.append("")
    if not failure_cases.empty:
        methods = sorted(failure_cases["method"].unique())
        for method in methods:
            lines.append(f"### {method}")
            lines.append("")
            lines.append("| Rank | Scenario | Return | Final Range (m) | Final ATA (deg) | Reason |")
            lines.append("|------|----------|--------:|----------------:|----------------:|--------|")
            mdf = failure_cases[failure_cases["method"] == method].reset_index(drop=True)
            for i, row in mdf.iterrows():
                lines.append(
                    f"| {i+1} | {row['scenario']} | {row['return']:.1f} | "
                    f"{row['final_range_m']:.1f} | {row['final_ata_deg']:.1f} | {row['reason']} |"
                )
            lines.append("")
    else:
        lines.append("_No failure case data available._")
    lines.append("")

    # G. Audit Summary
    lines.append("## G. Audit Summary")
    lines.append("")
    invalid = sorted_methods[sorted_methods["invalid_for_paper"] == True]
    if not invalid.empty:
        lines.append(f"**Invalid methods**: {', '.join(invalid['method'].tolist())}")
    else:
        lines.append("**All methods valid for paper.**")

    sb_ok = sorted_methods[sorted_methods["scenario_balance_ok"] == True]
    sb_bad = sorted_methods[sorted_methods["scenario_balance_ok"] != True]
    if not sb_bad.empty:
        lines.append(f"**Scenario balance issues**: {', '.join(sb_bad['method'].tolist())}")
    else:
        lines.append("**Scenario balance OK for all methods.**")

    outlier_seeds = seed_summary[seed_summary.get("seed_outlier", pd.Series([False]*len(seed_summary))) == True]
    if not outlier_seeds.empty:
        lines.append(f"**Outlier seeds detected**: {len(outlier_seeds)} seed(s)")
        for _, row in outlier_seeds.iterrows():
            lines.append(f"  - {row['method']} seed {row['training_seed']} (success rate {row['success_rate']:.2%})")
    else:
        lines.append("**No outlier seeds detected.**")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 6F Results Diagnosis")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing cross_seed_summary.json and summary.csv")
    parser.add_argument("--raw", type=str, required=True,
                        help="Root directory with train_seed*/prediction_metrics.json raw data")
    parser.add_argument("--output", type=str, default="outputs/tables/stage6f_diagnosis",
                        help="Output directory for diagnosis artifacts")
    parser.add_argument("--top-n-failures", type=int, default=5,
                        help="Number of worst episodes to extract per method")
    args = parser.parse_args()

    input_dir = Path(args.input)
    raw_root = Path(args.raw)
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    cross_json_path = input_dir / "cross_seed_summary.json"
    if not cross_json_path.exists():
        print(f"ERROR: {cross_json_path} not found")
        sys.exit(1)

    cross_data = load_cross_seed_summary(cross_json_path)

    # Build tables
    method_summary = build_method_summary(cross_data)
    scenario_summary = build_scenario_summary(cross_data, raw_root)
    seed_summary = build_seed_summary(cross_data)
    failure_cases = build_failure_cases(cross_data, raw_root, top_n=args.top_n_failures)
    cv_ca_diag = compute_cv_ca_diagnosis(method_summary)

    # Save CSVs
    method_csv = output_dir / "stage6f_method_summary.csv"
    method_summary.to_csv(method_csv, index=False, float_format="%.6f")
    print(f"Saved method summary: {method_csv}")

    scenario_csv = output_dir / "stage6f_scenario_summary.csv"
    scenario_summary.to_csv(scenario_csv, index=False, float_format="%.6f")
    print(f"Saved scenario summary: {scenario_csv}")

    seed_csv = output_dir / "stage6f_seed_summary.csv"
    seed_summary.to_csv(seed_csv, index=False, float_format="%.6f")
    print(f"Saved seed summary: {seed_csv}")

    failure_csv = output_dir / "stage6f_failure_cases.csv"
    failure_cases.to_csv(failure_csv, index=False, float_format="%.6f")
    print(f"Saved failure cases: {failure_csv}")

    # Render and save diagnosis markdown
    md = render_diagnosis_md(
        method_summary, scenario_summary, seed_summary, failure_cases, cv_ca_diag, cross_data
    )
    md_path = output_dir / "stage6f_diagnosis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved diagnosis report: {md_path}")


if __name__ == "__main__":
    main()
