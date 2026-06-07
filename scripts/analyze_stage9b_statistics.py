#!/usr/bin/env python3
"""Stage 9B.1: Statistical analysis of official paper-safe benchmark.

Reads raw_episodes.csv from a Stage 9B benchmark run and produces:
- statistics_summary.md
- statistics_tables/*.md
- statistics.json
- figures/*.png
- failure_root_cause.csv
- analysis_manifest.json
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASELINE_METHOD = "no_prediction"
METHOD_ORDER = ["no_prediction", "cv_prediction", "ca_prediction", "gain_only"]
REGRESSION_SCENARIOS = ["regression_neutral", "regression_challenging", "regression_crossing_left", "regression_crossing_right"]
CANDIDATE_SCENARIOS = ["candidate_head_on_close", "candidate_head_on_far", "candidate_crossing_close", "candidate_head_on_medium"]

# ---------------------------------------------------------------------------
# Helpers: CI and effect sizes
# ---------------------------------------------------------------------------
def wilson_ci(successes, n, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=95, seed=42):
    """Percentile bootstrap CI for a statistic."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"))
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot.append(statistic(sample))
    boot = np.sort(boot)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return (float(np.percentile(boot, lo)), float(np.percentile(boot, hi)))


def cohens_h(p1, p2):
    """Cohen's h for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def cohens_d_paired(diff):
    """Cohen's d for paired differences."""
    diff = np.asarray(diff)
    if len(diff) == 0 or diff.std(ddof=1) == 0:
        return 0.0 if diff.mean() == 0 else float("inf")
    return float(diff.mean() / diff.std(ddof=1))


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------
def classify_failure(row: pd.Series) -> dict:
    """Classify a single failed episode.

    Returns dict with root_cause and supporting_telemetry string.
    """
    causes = []
    telemetry = []

    # Priority 1: explicit termination reason
    if row.get("is_timeout", False):
        causes.append("timeout")
        telemetry.append("reason=timeout")
    if row.get("is_crash", False):
        causes.append("crash")
        telemetry.append("reason=crash")
    if row.get("is_out_of_bounds", False):
        causes.append("out_of_bounds")
        telemetry.append("reason=out_of_bounds")

    # Priority 2: miss heuristic (not caught by above)
    final_range = row.get("final_range_m", float("nan"))
    if not causes and not math.isnan(final_range) and final_range > 200:
        causes.append("miss")
        telemetry.append(f"final_range_m={final_range:.1f}>200")

    # Priority 3: candidate geometry
    scenario = row.get("scenario", "")
    if scenario.startswith("candidate_"):
        causes.append("candidate_tail_chase")
        telemetry.append(f"scenario={scenario}")

    # Priority 4: prediction degradation (for methods with predictors)
    method = row.get("method", "")
    pred_err = row.get("mean_prediction_error_m", float("nan"))
    pred_fallback = row.get("prediction_fallback_rate", float("nan"))
    if method in ("cv_prediction", "ca_prediction"):
        if not math.isnan(pred_err) and pred_err > 50:
            causes.append("prediction_degradation")
            telemetry.append(f"mean_prediction_error_m={pred_err:.1f}>50")
        elif not math.isnan(pred_fallback) and pred_fallback > 0.1:
            causes.append("prediction_degradation")
            telemetry.append(f"prediction_fallback_rate={pred_fallback:.3f}>0.1")

    # Priority 5: unstable command
    nz_mod = row.get("nz_cmd_modification_rate", float("nan"))
    nz_sat = row.get("nz_cmd_saturation_rate", float("nan"))
    if not math.isnan(nz_mod) and nz_mod > 0.9:
        causes.append("unstable_command")
        telemetry.append(f"nz_cmd_modification_rate={nz_mod:.3f}>0.9")
    elif not math.isnan(nz_sat) and nz_sat > 0.5:
        causes.append("unstable_command")
        telemetry.append(f"nz_cmd_saturation_rate={nz_sat:.3f}>0.5")

    # Priority 6: low energy
    energy = row.get("energy_proxy", float("nan"))
    if not math.isnan(energy) and energy < 5000:
        causes.append("low_energy_geometry")
        telemetry.append(f"energy_proxy={energy:.1f}<5000")

    if not causes:
        causes.append("unclassified")
        telemetry.append("no heuristic matched")

    return {
        "root_cause": " + ".join(causes),
        "supporting_telemetry": "; ".join(telemetry),
    }


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------
def compute_method_stats(df: pd.DataFrame, method: str) -> dict:
    sub = df[df["method"] == method]
    n = len(sub)
    successes = int(sub["is_success"].sum())
    sr = successes / n if n > 0 else 0.0
    sr_lo, sr_hi = wilson_ci(successes, n)

    def _boot(col):
        data = sub[col].dropna().values
        if len(data) == 0:
            return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        lo, hi = bootstrap_ci(data)
        return {"mean": float(np.mean(data)), "ci_lo": lo, "ci_hi": hi}

    return {
        "n_episodes": n,
        "success_rate": sr,
        "success_rate_ci": (sr_lo, sr_hi),
        "return": _boot("return"),
        "length": _boot("length"),
        "min_range_m": _boot("min_range_m"),
        "nz_cmd_mean": _boot("nz_cmd_mean"),
        "roll_rate_cmd_mean": _boot("roll_rate_cmd_mean"),
        "throttle_cmd_mean": _boot("throttle_cmd_mean"),
        "energy_proxy": _boot("energy_proxy"),
        "prediction_fallback_rate": _boot("prediction_fallback_rate"),
        "mean_prediction_error_m": _boot("mean_prediction_error_m"),
    }


def compute_paired_comparison(df: pd.DataFrame, baseline: str, method: str, metric: str) -> dict:
    """Paired comparison by (scenario, seed)."""
    base = df[df["method"] == baseline][["scenario", "seed", metric]].dropna()
    comp = df[df["method"] == method][["scenario", "seed", metric]].dropna()
    merged = pd.merge(base, comp, on=["scenario", "seed"], suffixes=("_base", "_comp"))
    if len(merged) == 0:
        return {"n_pairs": 0, "mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "effect_size": float("nan")}

    diff = merged[f"{metric}_comp"].values - merged[f"{metric}_base"].values
    lo, hi = bootstrap_ci(diff)
    # Wilcoxon signed-rank for paired non-parametric test
    try:
        w_stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        p_value = 1.0
    d = cohens_d_paired(diff)
    return {
        "n_pairs": len(merged),
        "mean_diff": float(np.mean(diff)),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": float(p_value),
        "effect_size": d,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def plot_success_rate_ci(stats_dict, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in METHOD_ORDER if m in stats_dict]
    srs = [stats_dict[m]["success_rate"] for m in methods]
    cis = [stats_dict[m]["success_rate_ci"] for m in methods]
    lows = [sr - ci[0] for sr, ci in zip(srs, cis)]
    highs = [ci[1] - sr for sr, ci in zip(srs, cis)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    ax.bar(x, srs, yerr=[lows, highs], capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success Rate with 95% Wilson CI")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_metric_ci(stats_dict, metric_key, ylabel, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in METHOD_ORDER if m in stats_dict]
    means = [stats_dict[m][metric_key]["mean"] for m in methods]
    lows = [means[i] - stats_dict[m][metric_key]["ci_lo"] for i, m in enumerate(methods)]
    highs = [stats_dict[m][metric_key]["ci_hi"] - means[i] for i, m in enumerate(methods)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=[lows, highs], capsize=5, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} with 95% Bootstrap CI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------
def generate_summary(stats_dict, paired_results, stratified, ranking, failure_df, output_dir, args, manifest):
    lines = [
        "# Stage 9B.1 Statistical Analysis Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Source**: {args.input}",
        f"**Git Commit**: {manifest.get('git_commit', 'unknown')}",
        f"**Benchmark Manifest**: {args.benchmark_manifest or 'not linked'}",
        "",
        "## Summary of Paper-Safe Claims",
        "",
    ]

    for method in METHOD_ORDER:
        if method not in stats_dict:
            continue
        s = stats_dict[method]
        sr = s["success_rate"]
        lo, hi = s["success_rate_ci"]
        lines.append(f"### {method}")
        lines.append(f"- Success Rate: {sr:.2%} (95% CI: {lo:.2%} – {hi:.2%}), n={s['n_episodes']}")
        lines.append(f"- Mean Return: {s['return']['mean']:.2f} (95% CI: {s['return']['ci_lo']:.2f} – {s['return']['ci_hi']:.2f})")
        lines.append(f"- Mean Min Range: {s['min_range_m']['mean']:.1f} m (95% CI: {s['min_range_m']['ci_lo']:.1f} – {s['min_range_m']['ci_hi']:.1f})")
        lines.append("")

    lines.extend([
        "## Pairwise Comparison vs Baseline (no_prediction)",
        "",
        "Paired by (scenario, seed). Effect size = Cohen's d for continuous metrics, Cohen's h for success rate.",
        "",
    ])

    for method in METHOD_ORDER:
        if method == BASELINE_METHOD or method not in paired_results:
            continue
        lines.append(f"### {method} vs {BASELINE_METHOD}")
        for metric, res in paired_results[method].items():
            sig = "*" if res["p_value"] < 0.05 else "ns"
            lines.append(
                f"- **{metric}**: mean_diff={res['mean_diff']:.3f}, "
                f"CI=[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}], "
                f"p={res['p_value']:.4f}{sig}, "
                f"d={res['effect_size']:.3f}, n_pairs={res['n_pairs']}"
            )
        lines.append("")

    lines.extend([
        "## Stratified Results",
        "",
        "### Regression Scenarios",
        "",
    ])
    for method in METHOD_ORDER:
        if method not in stratified["regression"]:
            continue
        s = stratified["regression"][method]
        lines.append(f"- {method}: SR={s['success_rate']:.2%} (n={s['n_episodes']})")
    lines.append("")
    lines.extend([
        "### Candidate Scenarios",
        "",
    ])
    for method in METHOD_ORDER:
        if method not in stratified["candidate"]:
            continue
        s = stratified["candidate"][method]
        lines.append(f"- {method}: SR={s['success_rate']:.2%} (n={s['n_episodes']})")
    lines.append("")

    lines.extend([
        "## Per-Scenario Ranking",
        "",
        "| Scenario | Rank | Method | Success Rate |",
        "|---|---|---|---|",
    ])
    for _, row in ranking.iterrows():
        lines.append(f"| {row['scenario']} | {row['rank']} | {row['method']} | {row['success_rate']:.2%} |")
    lines.append("")

    lines.extend([
        "## Failure Taxonomy",
        "",
        f"Total failures: {len(failure_df)}",
        "",
        "| Root Cause | Count |",
        "|---|---|",
    ])
    cause_counts = failure_df["root_cause"].value_counts().to_dict()
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cause} | {count} |")
    lines.append("")

    lines.extend([
        "## Interpretation Rules",
        "",
        "- A claim is **paper-safe** only if supported by the full matrix, statistical significance, and cross-seed consistency.",
        "- p-values alone are insufficient: effect size must be reported alongside.",
        "- Candidate scenario results are **preliminary** until replicated with extended seeds.",
        "",
    ])

    summary_path = output_dir / "statistics_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 9B.1 statistical analysis")
    parser.add_argument("--input", type=str, required=True, help="Path to raw_episodes.csv")
    parser.add_argument("--benchmark-manifest", type=str, default=None, help="Path to source benchmark run_manifest.json")
    parser.add_argument("--output-dir", type=str, default="outputs/stage9b_statistics")
    parser.add_argument("--baseline", type=str, default=BASELINE_METHOD)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "statistics_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    # Validate required columns
    required = {"method", "scenario", "seed", "is_success", "return", "length", "min_range_m",
                "nz_cmd_mean", "roll_rate_cmd_mean", "throttle_cmd_mean", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw_episodes.csv: {missing}")

    # ------------------------------------------------------------------
    # Per-method statistics
    # ------------------------------------------------------------------
    stats_dict = {}
    for method in df["method"].unique():
        stats_dict[method] = compute_method_stats(df, method)

    # ------------------------------------------------------------------
    # Paired comparison vs baseline
    # ------------------------------------------------------------------
    paired_metrics = ["return", "length", "min_range_m", "nz_cmd_mean"]
    paired_results = {}
    for method in df["method"].unique():
        if method == args.baseline:
            continue
        paired_results[method] = {}
        for metric in paired_metrics:
            paired_results[method][metric] = compute_paired_comparison(df, args.baseline, method, metric)

    # ------------------------------------------------------------------
    # Stratified analysis
    # ------------------------------------------------------------------
    stratified = {"regression": {}, "candidate": {}}
    for method in df["method"].unique():
        reg_df = df[(df["method"] == method) & (df["scenario"].isin(REGRESSION_SCENARIOS))]
        cand_df = df[(df["method"] == method) & (df["scenario"].isin(CANDIDATE_SCENARIOS))]
        stratified["regression"][method] = {
            "n_episodes": len(reg_df),
            "success_rate": reg_df["is_success"].mean() if len(reg_df) else 0.0,
        }
        stratified["candidate"][method] = {
            "n_episodes": len(cand_df),
            "success_rate": cand_df["is_success"].mean() if len(cand_df) else 0.0,
        }

    # ------------------------------------------------------------------
    # Per-scenario ranking
    # ------------------------------------------------------------------
    ranking = []
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        grp = sub.groupby("method")["is_success"].mean().reset_index()
        grp = grp.sort_values("is_success", ascending=False)
        grp["rank"] = range(1, len(grp) + 1)
        grp["scenario"] = scenario
        grp = grp.rename(columns={"is_success": "success_rate"})
        ranking.append(grp)
    if ranking:
        ranking = pd.concat(ranking, ignore_index=True)
        ranking = ranking[["scenario", "rank", "method", "success_rate"]]
    else:
        ranking = pd.DataFrame(columns=["scenario", "rank", "method", "success_rate"])

    # ------------------------------------------------------------------
    # Failure taxonomy
    # ------------------------------------------------------------------
    failures = df[df["is_success"] == False].copy().reset_index(drop=True)
    if not failures.empty:
        classified = failures.apply(classify_failure, axis=1, result_type="expand")
        failures = pd.concat([failures, classified], axis=1)
    else:
        failures["root_cause"] = []
        failures["supporting_telemetry"] = []

    failure_csv_path = output_dir / "failure_root_cause.csv"
    cause_counts = {}
    if not failures.empty:
        failures.to_csv(failure_csv_path, index=False)
        print(f"Saved: {failure_csv_path}")
        cause_counts = failures["root_cause"].value_counts().to_dict()
    else:
        failure_csv_path.write_text("No failures recorded.\n", encoding="utf-8")
        print(f"Saved: {failure_csv_path} (empty)")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_success_rate_ci(stats_dict, figures_dir / "stage9b_success_rate_ci.png")
    plot_metric_ci(stats_dict, "return", "Mean Return", figures_dir / "stage9b_metric_return_ci.png")
    plot_metric_ci(stats_dict, "min_range_m", "Min Range (m)", figures_dir / "stage9b_metric_range_ci.png")

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    # Method summary table
    table_rows = []
    for method in METHOD_ORDER:
        if method not in stats_dict:
            continue
        s = stats_dict[method]
        table_rows.append({
            "Method": method,
            "Success Rate": f"{s['success_rate']:.2%} [{s['success_rate_ci'][0]:.2%}, {s['success_rate_ci'][1]:.2%}]",
            "Mean Return": f"{s['return']['mean']:.2f} [{s['return']['ci_lo']:.2f}, {s['return']['ci_hi']:.2f}]",
            "Min Range (m)": f"{s['min_range_m']['mean']:.1f} [{s['min_range_m']['ci_lo']:.1f}, {s['min_range_m']['ci_hi']:.1f}]",
            "N": s["n_episodes"],
        })
    pd.DataFrame(table_rows).to_csv(tables_dir / "method_summary.csv", index=False)

    # Pairwise table
    pairwise_rows = []
    for method, metrics in paired_results.items():
        for metric, res in metrics.items():
            sig = "*" if res["p_value"] < 0.05 else "ns"
            pairwise_rows.append({
                "Comparison": f"{method} vs {args.baseline}",
                "Metric": metric,
                "Mean Diff": f"{res['mean_diff']:.3f}",
                "CI": f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]",
                "p-value": f"{res['p_value']:.4f}{sig}",
                "Effect Size (d)": f"{res['effect_size']:.3f}",
                "N Pairs": res["n_pairs"],
            })
    pd.DataFrame(pairwise_rows).to_csv(tables_dir / "pairwise_comparison.csv", index=False)

    # Stratified table
    strat_rows = []
    for suite_name, suite_data in stratified.items():
        for method, s in suite_data.items():
            strat_rows.append({
                "Suite": suite_name,
                "Method": method,
                "Success Rate": f"{s['success_rate']:.2%}",
                "N": s["n_episodes"],
            })
    pd.DataFrame(strat_rows).to_csv(tables_dir / "stratified_results.csv", index=False)

    # Ranking table
    ranking.to_csv(tables_dir / "per_scenario_ranking.csv", index=False)

    # ------------------------------------------------------------------
    # JSON statistics
    # ------------------------------------------------------------------
    stats_json = {
        "methods": stats_dict,
        "paired_comparisons": paired_results,
        "stratified": stratified,
        "failure_taxonomy": cause_counts if not failures.empty else {},
    }
    stats_json_path = output_dir / "statistics.json"
    stats_json_path.write_text(json.dumps(stats_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved: {stats_json_path}")

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    git_info = {"commit": "unknown", "dirty": False}
    try:
        git_info["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        git_info["dirty"] = len(subprocess.check_output(["git", "status", "--short"], text=True).strip()) > 0
    except Exception:
        pass

    bm_manifest = {}
    if args.benchmark_manifest and Path(args.benchmark_manifest).exists():
        bm_manifest = json.loads(Path(args.benchmark_manifest).read_text(encoding="utf-8"))

    manifest = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "source_raw_episodes": str(input_path),
        "source_benchmark_manifest": args.benchmark_manifest,
        "source_benchmark_manifest_hash": hashlib.sha256(
            json.dumps(bm_manifest, sort_keys=True, default=str).encode()
        ).hexdigest()[:16] if bm_manifest else None,
        "analysis_script_command": sys.argv,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {manifest_path}")

    generate_summary(stats_dict, paired_results, stratified, ranking, failures, output_dir, args, manifest)

    print("\nStage 9B.1 analysis complete.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
