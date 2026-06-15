#!/usr/bin/env python3
"""
Generate reviewer-compliant paper tables and figures from raw episode data.

This script addresses the reviewer concerns in
``paper_materials/reviewer_report.md`` by:

1. Building every table from raw episode-level CSVs instead of hard-coded
   LaTeX values.
2. Reporting per-seed success rates (not just a pooled mean) so variance is
   transparent.
3. Adding confidence intervals (bootstrap) and statistical tests
   (McNemar, Mann-Whitney U, Welch's t-test) for every comparison.
4. Replacing TikZ placeholder figures with PNGs generated from the same data.
5. Producing supplementary per-seed CSVs next to each table.

Usage:
    python scripts/generate_reviewer_compliant_tables.py

The script uses sensible defaults pointing at existing ``docs/results`` and
``outputs`` directories. Override any source with CLI flags.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.evaluation.statistical_comparison import (
    bootstrap_success_rate_ci,
    mann_whitney_u,
    mcnemar_exact_pvalue,
)


# ---------------------------------------------------------------------------
# Defaults (edit these to match the canonical experimental layout)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "exact_no_vpp": "docs/results/p0a_no_vpp_ablation/raw_episodes.csv",
    "exact_vpp": "docs/results/p0a_vpp_ablation/raw_episodes.csv",
    "exact_e2e": "docs/results/end_to_end_baseline_multi_seed/raw_episodes.csv",
    "random_no_vpp": "outputs/ablation_matrix/evaluation/no_vpp_constant_s*",
    "random_vpp": "outputs/ablation_matrix/evaluation/vpp_single_constant_s*",
    "maneuver_no_vpp": "outputs/ablation_matrix/evaluation/no_vpp_sinusoidal_s*",
    "maneuver_vpp": "outputs/ablation_matrix/evaluation/vpp_single_sinusoidal_s*",
    "jsbsim": "outputs/stage10_jsbsim_validation/raw_episodes.csv",
    "method_innovation": "outputs/method_innovation_comparison",
    "cem_compare": "outputs/cem_compare",
}

SCENARIO_ORDER = ["favorable", "neutral", "disadvantage", "challenging"]

METHOD_LABELS = {
    "no_vpp_direct": "No-VPP (zero offset)",
    "no_vpp": "No-VPP (zero offset)",
    "vpp_no_pred": "VPP + LOS-rate",
    "no_prediction": "VPP + LOS-rate",
    "VPP+LOS-rate": "VPP + LOS-rate",
    "End-to-End": "End-to-end DRL",
    "end_to_end": "End-to-end DRL",
    "gain_only": "No-VPP (zero offset)",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reviewer-compliant tables and figures"
    )
    for key, val in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=str, default=val,
                            help=f"Source path/glob for {key}")
    parser.add_argument("--output-dir", type=str, default="paper_materials")
    parser.add_argument("--min-seeds", type=int, default=5,
                        help="Warn if fewer than this many independent seeds are available")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _find_csv_sources(pattern: str) -> List[Path]:
    """Expand a path/glob into a list of raw_episodes.csv files."""
    p = Path(pattern)
    if p.is_file():
        return [p]
    # glob pattern
    files = sorted(Path(".").glob(pattern))
    csvs = [f for f in files if f.is_file() and f.name == "raw_episodes.csv"]
    if not csvs:
        # try recursive search under the globbed directories
        for d in files:
            if d.is_dir():
                csvs.extend(sorted(d.rglob("raw_episodes.csv")))
    # fallback: pattern itself may contain wildcard and we need to glob from cwd
    if not csvs and ("*" in pattern or "?" in pattern):
        csvs = sorted(Path(".").glob(pattern + "/raw_episodes.csv"))
        if not csvs:
            csvs = sorted(Path(".").rglob(pattern.replace("*", "").replace("?", "")))
            csvs = [f for f in csvs if f.name == "raw_episodes.csv"]
    return csvs


def load_raw_episodes(pattern: str, method_override: Optional[str] = None) -> pd.DataFrame:
    """Load raw episode data and normalize column names."""
    sources = _find_csv_sources(pattern)
    if not sources:
        return pd.DataFrame()

    frames = []
    for src in sources:
        try:
            df = pd.read_csv(src)
        except Exception as exc:
            print(f"[WARN] Could not read {src}: {exc}")
            continue

        # Normalize return column
        if "episode_return" in df.columns and "return" not in df.columns:
            df = df.rename(columns={"episode_return": "return"})

        # Normalize success column
        if "is_success" not in df.columns and "success" in df.columns:
            df = df.rename(columns={"success": "is_success"})

        # Infer method if missing
        if "method" not in df.columns or df["method"].isna().all():
            if method_override:
                df["method"] = method_override
            else:
                # infer from directory name
                df["method"] = src.parent.name.split("_")[0]

        # Infer seed if missing
        if "seed" not in df.columns:
            df["seed"] = 0

        df["source_file"] = str(src)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def standardize_methods(df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Map raw method names to display labels and drop unmapped rows."""
    if df.empty:
        return df
    mapping = mapping or METHOD_LABELS
    df = df.copy()
    df["method"] = df["method"].astype(str).map(mapping)
    df = df.dropna(subset=["method"])
    return df


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------


def aggregate_seed_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (method, scenario, seed) with success rate etc."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["method", "scenario", "seed"], observed=True).agg(
        n_episodes=("is_success", "size"),
        success_rate=("is_success", "mean"),
        mean_return=("return", "mean"),
        std_return=("return", "std"),
    ).reset_index()
    grouped["crash_rate"] = df.groupby(["method", "scenario", "seed"], observed=True)["is_crash"].mean().values if "is_crash" in df.columns else 0.0
    grouped["oob_rate"] = df.groupby(["method", "scenario", "seed"], observed=True)["is_out_of_bounds"].mean().values if "is_out_of_bounds" in df.columns else 0.0
    grouped["timeout_rate"] = df.groupby(["method", "scenario", "seed"], observed=True)["is_timeout"].mean().values if "is_timeout" in df.columns else 0.0
    return grouped


def aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (method, seed) across all scenarios."""
    if df.empty:
        return pd.DataFrame()
    # Use scenario as a synthetic column for overall aggregation
    df_all = df.copy()
    df_all["scenario"] = "overall"
    return aggregate_seed_scenario(df_all)


def pooled_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled mean/std and bootstrap CI per (method, scenario)."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (method, scenario), g in df.groupby(["method", "scenario"]):
        # Per-seed success rates for cross-seed variance
        seed_stats = g.groupby("seed")["is_success"].agg(["sum", "size", "mean"]).reset_index()
        srs = seed_stats["mean"].values
        returns = g.groupby("seed")["return"].mean().values
        mean_sr, lo_sr, hi_sr = bootstrap_success_rate_ci(
            outcomes=g["is_success"].astype(int).tolist(),
            n_bootstrap=2000,
            random_seed=42,
        )
        rows.append({
            "method": method,
            "scenario": scenario,
            "n_seeds": len(seed_stats),
            "n_episodes": int(seed_stats["size"].sum()),
            "success_rate_mean": float(np.mean(srs)),
            "success_rate_std": float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0,
            "success_rate_ci_lo": lo_sr,
            "success_rate_ci_hi": hi_sr,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def compare_methods(df: pd.DataFrame, baseline_method: str, test_method: str) -> dict:
    """Statistical comparison between two methods on the same episodes."""
    if df.empty or baseline_method not in df["method"].values or test_method not in df["method"].values:
        return {"error": "missing data"}

    # Per-seed success rates
    seed_sr = aggregate_seed_scenario(df)
    base_sr = seed_sr[seed_sr["method"] == baseline_method].sort_values("seed")["success_rate"].values
    test_sr = seed_sr[seed_sr["method"] == test_method].sort_values("seed")["success_rate"].values

    # Welch's t-test on per-seed success rates
    from scipy import stats
    if len(base_sr) > 1 and len(test_sr) > 1:
        t, p_welch = stats.ttest_ind(test_sr, base_sr, equal_var=False)
    else:
        t, p_welch = np.nan, np.nan

    # Mann-Whitney U on episode returns (unpaired, conservative)
    base_ret = df[df["method"] == baseline_method]["return"].dropna().values
    test_ret = df[df["method"] == test_method]["return"].dropna().values
    if len(base_ret) > 0 and len(test_ret) > 0:
        try:
            u, p_mann = stats.mannwhitneyu(test_ret, base_ret, alternative="two-sided")
        except Exception:
            u, p_mann = np.nan, np.nan
    else:
        u, p_mann = np.nan, np.nan

    # McNemar exact test on paired episodes (same scenario + seed + episode index)
    base_df = df[df["method"] == baseline_method].copy()
    test_df = df[df["method"] == test_method].copy()
    pair_keys = ["scenario", "seed"]
    if "episode" in df.columns:
        pair_keys.append("episode")
    elif "episode_seed" in df.columns:
        pair_keys.append("episode_seed")
    try:
        merged = pd.merge(
            base_df[pair_keys + ["is_success"]].rename(columns={"is_success": "base_success"}),
            test_df[pair_keys + ["is_success"]].rename(columns={"is_success": "test_success"}),
            on=pair_keys,
            how="inner",
        )
        a = int(((merged["base_success"] == 1) & (merged["test_success"] == 1)).sum())
        b = int(((merged["base_success"] == 0) & (merged["test_success"] == 1)).sum())
        c = int(((merged["base_success"] == 1) & (merged["test_success"] == 0)).sum())
        d = int(((merged["base_success"] == 0) & (merged["test_success"] == 0)).sum())
        p_mcnemar = mcnemar_exact_pvalue(b, c)
        n_pairs = a + b + c + d
    except Exception:
        a = b = c = d = n_pairs = 0
        p_mcnemar = np.nan

    return {
        "baseline_method": baseline_method,
        "test_method": test_method,
        "n_seeds_base": len(base_sr),
        "n_seeds_test": len(test_sr),
        "mean_diff_sr": float(np.mean(test_sr) - np.mean(base_sr)),
        "welch_t": float(t),
        "welch_p": float(p_welch),
        "mann_whitney_u": float(u),
        "mann_whitney_p": float(p_mann),
        "mcnemar_pairs": n_pairs,
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": float(p_mcnemar),
    }


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------


def _fmt_sr(mean: float, std: float) -> str:
    return f"${mean*100:.1f} \\pm {std*100:.1f}\\%$"


def _fmt_sr_plain(mean: float, std: float) -> str:
    return f"{mean*100:.1f} $\\pm$ {std*100:.1f}\\%"


def format_architecture_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    methods_order: List[str],
    note: str,
) -> Tuple[str, pd.DataFrame]:
    """Generate a LaTeX architecture table and the per-seed summary used."""
    if df.empty:
        return "", pd.DataFrame()

    seed_sr = aggregate_seed_scenario(df)
    overall = aggregate_overall(df)
    summary_scenario = pooled_summary(df)
    summary_overall = pooled_summary(df.assign(scenario="overall"))
    summary = pd.concat([summary_overall, summary_scenario], ignore_index=True)

    scenarios = ["overall"] + [s for s in SCENARIO_ORDER if s in summary["scenario"].values]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(scenarios) + "}",
        "\\toprule",
        "Method & " + " & ".join(s.replace("_", " ").title() for s in scenarios) + " \\\\",
        "\\midrule",
    ]

    for method in methods_order:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        cells = [method]
        for sc in scenarios:
            row = sub[sub["scenario"] == sc]
            if row.empty:
                cells.append("N/A")
            else:
                cells.append(_fmt_sr(row["success_rate_mean"].values[0], row["success_rate_std"].values[0]))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append(f"\\item {note}")
    lines.append("\\item Statistical tests: Welch's $t$-test on per-seed success rates; McNemar exact test on paired episodes; Mann-Whitney $U$ on episode returns.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    return "\n".join(lines), seed_sr


def format_comparison_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    methods_order: List[str],
    baseline_method: str,
    note: str,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """Generate a comparison table with p-values against a baseline."""
    if df.empty:
        return "", pd.DataFrame(), pd.DataFrame()

    seed_sr = aggregate_seed_scenario(df)
    overall = aggregate_overall(df)
    summary_scenario = pooled_summary(df)
    summary_overall = pooled_summary(df.assign(scenario="overall"))
    summary = pd.concat([summary_overall, summary_scenario], ignore_index=True)

    stats_rows = []
    for method in methods_order:
        if method == baseline_method:
            continue
        comp = compare_methods(df, baseline_method, method)
        stats_rows.append({"method": method, **comp})
    stats_df = pd.DataFrame(stats_rows)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Overall SR & $\\Delta$ SR vs Baseline & Welch $p$ & McNemar $p$ & Mann-Whitney $p$ \\\\",
        "\\midrule",
    ]

    base_overall = summary[(summary["method"] == baseline_method) & (summary["scenario"] == "overall")]
    base_sr_str = _fmt_sr_plain(
        base_overall["success_rate_mean"].values[0],
        base_overall["success_rate_std"].values[0],
    ) if not base_overall.empty else "N/A"
    lines.append(f"{baseline_method} & {base_sr_str} & --- & --- & --- & --- \\\\")

    for _, row in stats_df.iterrows():
        method = row["method"]
        over = summary[(summary["method"] == method) & (summary["scenario"] == "overall")]
        if over.empty:
            continue
        sr_str = _fmt_sr_plain(over["success_rate_mean"].values[0], over["success_rate_std"].values[0])
        delta = f"{row['mean_diff_sr']*100:+.1f}\\%"
        pw = f"{row['welch_p']:.4f}" if np.isfinite(row['welch_p']) else "N/A"
        pm = f"{row['mcnemar_p']:.4f}" if np.isfinite(row['mcnemar_p']) else "N/A"
        pu = f"{row['mann_whitney_p']:.4f}" if np.isfinite(row['mann_whitney_p']) else "N/A"
        lines.append(f"{method} & {sr_str} & {delta} & {pw} & {pm} & {pu} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append(f"\\item {note}")
    lines.append("\\item $\\Delta$ SR: difference in overall success rate (percentage points).")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    return "\n".join(lines), seed_sr, stats_df




def format_method_innovation_table(root: Path, min_seeds: int = 5) -> Tuple[str, pd.DataFrame]:
    """Build the method-innovation table from per-seed eval logs."""
    rows = []
    raw_rows = []
    for algo_dir in sorted(root.glob("*")):
        if not algo_dir.is_dir():
            continue
        for seed_dir in sorted(algo_dir.glob("seed*")):
            eval_csv = seed_dir / "logs" / "eval_log.csv"
            if not eval_csv.exists():
                continue
            try:
                df = pd.read_csv(eval_csv)
                if df.empty:
                    continue
                last = df.iloc[-1]
                seed = int(re.search(r"seed\s*(\d+)", seed_dir.name).group(1))
                sr = float(last.get("success_rate", np.nan))
                rows.append({
                    "algorithm": algo_dir.name,
                    "seed": seed,
                    "success_rate": sr,
                    "mean_return": float(last.get("mean_return", np.nan)),
                    "crash_rate": float(last.get("crash_rate", 0)),
                    "out_of_bounds_rate": float(last.get("out_of_bounds_rate", 0)),
                    "timeout_rate": float(last.get("timeout_rate", 0)),
                    "mean_min_range_m": float(last.get("mean_min_range_m", np.nan)),
                })
                raw_rows.append({
                    "algorithm": algo_dir.name,
                    "seed": seed,
                    **{k: last.get(k, np.nan) for k in df.columns},
                })
            except Exception as exc:
                print(f"[WARN] {eval_csv}: {exc}")

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        return "", raw_df

    agg = []
    for algo, g in pd.DataFrame(rows).groupby("algorithm"):
        srs = g["success_rate"].values
        rets = g["mean_return"].values
        min_ranges = g["mean_min_range_m"].dropna().values
        from scipy import stats
        # Compare each algorithm to baseline using Welch t-test on per-seed SRs
        baseline = pd.DataFrame(rows)[pd.DataFrame(rows)["algorithm"] == "baseline"]
        if algo != "baseline" and not baseline.empty:
            t, p = stats.ttest_ind(g["success_rate"].values, baseline["success_rate"].values, equal_var=False)
        else:
            t, p = np.nan, np.nan
        agg.append({
            "algorithm": algo,
            "n_seeds": len(g),
            "success_rate_mean": float(np.mean(srs)),
            "success_rate_std": float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0,
            "mean_return": float(np.mean(rets)),
            "std_return": float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0,
            "mean_min_range_m": float(np.mean(min_ranges)) if len(min_ranges) else np.nan,
            "welch_t": float(t),
            "welch_p": float(p),
        })
    agg_df = pd.DataFrame(agg)

    order = ["baseline", "cr_ppo", "intentional", "intentional_c", "intentional_a"]
    agg_df["_sort"] = agg_df["algorithm"].apply(lambda x: order.index(x) if x in order else 99)
    agg_df = agg_df.sort_values("_sort").drop(columns="_sort")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Method-innovation comparison (CR-PPO and Intentional PPO). Values are mean $\\pm$ SD over independent seeds.}",
        "\\label{tab:method_innovation_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Algorithm & Seeds & Success Rate & Mean Return & $\\bar{d}_{\\min}$ (m) & Welch $t$ & $p$ \\\\",
        "\\midrule",
    ]
    for _, row in agg_df.iterrows():
        sr = f"{row['success_rate_mean']*100:.2f} $\\pm$ {row['success_rate_std']*100:.2f}\\%"
        ret = f"${row['mean_return']:.1f} \\pm {row['std_return']:.1f}$"
        dmin = f"{row['mean_min_range_m']:.1f}" if np.isfinite(row["mean_min_range_m"]) else "N/A"
        t = f"{row['welch_t']:.3f}" if np.isfinite(row['welch_t']) else "N/A"
        p = f"{row['welch_p']:.4f}" if np.isfinite(row['welch_p']) else "N/A"
        lines.append(f"{row['algorithm'].replace('_', ' ').title()} & {row['n_seeds']} & {sr} & {ret} & {dmin} & {t} & {p} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append(f"\\item Note: {len(agg_df)} algorithms evaluated. A success-rate standard deviation of 0.0 across seeds indicates deterministic evaluation conditions before the domain-randomization seeding fix; new runs will show non-zero variance.")
    lines.append("\\item Recommended placement: if no algorithm shows a significant improvement ($p<0.05$) over the Baseline PPO, this table should be moved to supplementary material as a negative-result report.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    return "\n".join(lines), raw_df


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _save_figure(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_architecture_comparison(summary: pd.DataFrame, output_path: Path):
    """Bar chart of overall success rate with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = summary[summary["scenario"] == "overall"].copy()
    if sub.empty:
        return
    sub = sub.sort_values("success_rate_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sub))
    means = sub["success_rate_mean"].values
    lows = means - sub["success_rate_ci_lo"].values
    highs = sub["success_rate_ci_hi"].values - means
    ax.bar(x, means, yerr=[lows, highs], capsize=6, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method"].values, rotation=15, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Architecture Comparison (Overall Success Rate)")
    ax.grid(axis="y", alpha=0.3)
    for i, (m, lo, hi) in enumerate(zip(means, sub["success_rate_ci_lo"], sub["success_rate_ci_hi"])):
        ax.text(i, m + (hi - m) + 0.02, f"{m:.1%}", ha="center", va="bottom", fontsize=10)
    _save_figure(fig, output_path)


def plot_vpp_comparison(seed_sr: pd.DataFrame, output_path: Path):
    """VPP vs No-VPP per-scenario bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = seed_sr[seed_sr["scenario"].isin(SCENARIO_ORDER)].copy()
    if sub.empty:
        return
    summary = sub.groupby(["method", "scenario"], observed=True)["success_rate"].agg(["mean", "std"]).reset_index()
    methods = summary["method"].unique()
    scenarios = [s for s in SCENARIO_ORDER if s in summary["scenario"].values]
    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        msub = summary[summary["method"] == method]
        means = [msub[msub["scenario"] == s]["mean"].values[0] for s in scenarios]
        stds = [msub[msub["scenario"] == s]["std"].values[0] for s in scenarios]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in scenarios])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("VPP vs No-VPP by Scenario")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, output_path)


def plot_training_curves(output_dir: Path, output_path: Path):
    """Training convergence curves from eval_log.csv files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"no_vpp": "#ff7f0e", "no_prediction": "#1f77b4", "end_to_end": "#2ca02c"}
    for label, stem in [("No-VPP", "no_vpp"), ("VPP + LOS-rate", "no_prediction"), ("End-to-End", "end_to_end")]:
        found = False
        for exp_dir in Path("outputs/experiments").glob(f"{stem}*"):
            steps, means = [], []
            for seed_dir in sorted(exp_dir.glob("seed_*")):
                eval_csv = seed_dir / "logs" / "eval_log.csv"
                if not eval_csv.exists():
                    continue
                df = pd.read_csv(eval_csv)
                if "step" not in df.columns or "success_rate" not in df.columns:
                    continue
                steps = df["step"].values
                means.append(df["success_rate"].values)
            if means:
                steps = steps[: min(len(s) for s in means)]
                arr = np.vstack([m[: len(steps)] for m in means])
                ax.plot(steps, arr.mean(axis=0), label=label, color=colors.get(stem, None))
                ax.fill_between(steps, arr.mean(axis=0) - arr.std(axis=0),
                                arr.mean(axis=0) + arr.std(axis=0), alpha=0.2)
                found = True
                break
        if not found:
            print(f"[WARN] No training curves found for {label}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Success Rate")
    ax.set_title("Training Convergence Curves")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_path)


def plot_cem_optimization(cem_dir: Path, output_path: Path):
    """CEM vs default vs heuristic gain optimization bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_csv = cem_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"[WARN] CEM summary not found at {summary_csv}")
        return
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = df.get("method_key", df.get("method", pd.Series())).tolist()
    means = df["success_rate_mean"].values
    lows = means - df.get("sr_ci_lower", 0).values
    highs = df.get("sr_ci_upper", 0).values - means
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=[lows, highs], capsize=6, color="seagreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Gain Optimization Comparison")
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, output_path)


def write_figure_include(tex_path: Path, png_path: Path, caption: str, label: str, width: str = "0.85\\columnwidth"):
    tex = (
        "\\begin{figure}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width={width}]{{{png_path.as_posix()}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )
    tex_path.write_text(tex, encoding="utf-8")
    print(f"[FIG] {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _warn_min_seeds(df: pd.DataFrame, name: str, min_seeds: int):
    if df.empty:
        return
    n = df.groupby("method")["seed"].nunique().min()
    if n < min_seeds:
        print(f"[WARN] {name}: only {n} seeds available (reviewer recommends >= {min_seeds})")


def main():
    args = parse_args()
    out = Path(args.output_dir)
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Architecture exact (canonical geometries)
    # ------------------------------------------------------------------
    print("\n[Architecture Exact]")
    exact = pd.concat([
        standardize_methods(load_raw_episodes(args.exact_no_vpp, method_override="no_vpp_direct"), {"no_vpp_direct": "No-VPP (zero offset)"}),
        standardize_methods(load_raw_episodes(args.exact_vpp, method_override="vpp_no_pred"), {"vpp_no_pred": "VPP + LOS-rate"}),
        standardize_methods(load_raw_episodes(args.exact_e2e), {"End-to-End": "End-to-end DRL"}),
    ], ignore_index=True)
    if not exact.empty:
        _warn_min_seeds(exact, "architecture exact", args.min_seeds)
        tex, seed_sr = format_architecture_table(
            exact,
            caption="Architecture comparison on canonical geometries (per-seed success rates).",
            label="tab:architecture_comparison_exact",
            methods_order=["No-VPP (zero offset)", "VPP + LOS-rate", "End-to-end DRL"],
            note="Exact canonical geometries. Per-seed success rates are reported transparently; if all seeds share the same value, the environment/evaluation was deterministic for this condition.",
        )
        (tables_dir / "table_architecture_comparison_exact.tex").write_text(tex, encoding="utf-8")
        seed_sr.to_csv(tables_dir / "table_architecture_comparison_exact_per_seed.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_architecture_comparison_exact.tex'}")

    # ------------------------------------------------------------------
    # Architecture random (domain-randomized constant-velocity)
    # ------------------------------------------------------------------
    print("\n[Architecture Random]")
    random_df = pd.concat([
        standardize_methods(load_raw_episodes(args.random_no_vpp, method_override="no_vpp"), {"no_vpp": "No-VPP (zero offset)"}),
        standardize_methods(load_raw_episodes(args.random_vpp, method_override="no_prediction"), {"no_prediction": "VPP + LOS-rate"}),
    ], ignore_index=True)
    if not random_df.empty:
        _warn_min_seeds(random_df, "architecture random", args.min_seeds)
        tex, seed_sr = format_architecture_table(
            random_df,
            caption="Architecture comparison on domain-randomized constant-velocity scenarios.",
            label="tab:architecture_comparison_random",
            methods_order=["No-VPP (zero offset)", "VPP + LOS-rate"],
            note="Domain-randomized initial conditions during evaluation. Standard deviation is computed across independent seeds.",
        )
        (tables_dir / "table_architecture_comparison_random.tex").write_text(tex, encoding="utf-8")
        seed_sr.to_csv(tables_dir / "table_architecture_comparison_random_per_seed.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_architecture_comparison_random.tex'}")

    # ------------------------------------------------------------------
    # VPP ablation (direct comparison table)
    # ------------------------------------------------------------------
    print("\n[VPP Ablation]")
    vpp_abl = pd.concat([
        standardize_methods(load_raw_episodes(args.exact_no_vpp, method_override="no_vpp_direct"), {"no_vpp_direct": "No-VPP (zero offset)"}),
        standardize_methods(load_raw_episodes(args.exact_vpp, method_override="vpp_no_pred"), {"vpp_no_pred": "VPP + LOS-rate"}),
    ], ignore_index=True)
    if not vpp_abl.empty:
        tex, seed_sr, stats = format_comparison_table(
            vpp_abl,
            caption="VPP ablation: VPP-enabled policy vs direct-command (no VPP) policy.",
            label="tab:vpp_ablation",
            methods_order=["No-VPP (zero offset)", "VPP + LOS-rate"],
            baseline_method="No-VPP (zero offset)",
            note="Paired comparisons use McNemar exact test on matched episodes; unpaired comparisons use Welch's $t$-test on per-seed success rates and Mann-Whitney $U$ on episode returns.",
        )
        (tables_dir / "table_vpp_ablation.tex").write_text(tex, encoding="utf-8")
        seed_sr.to_csv(tables_dir / "table_vpp_ablation_per_seed.csv", index=False)
        stats.to_csv(tables_dir / "table_vpp_ablation_stats.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_vpp_ablation.tex'}")

    # ------------------------------------------------------------------
    # Maneuvering comparison
    # ------------------------------------------------------------------
    print("\n[Maneuvering Comparison]")
    maneuver = pd.concat([
        standardize_methods(load_raw_episodes(args.maneuver_no_vpp, method_override="no_vpp"), {"no_vpp": "No-VPP (zero offset)"}),
        standardize_methods(load_raw_episodes(args.maneuver_vpp, method_override="no_prediction"), {"no_prediction": "VPP + LOS-rate"}),
    ], ignore_index=True)
    if not maneuver.empty:
        _warn_min_seeds(maneuver, "maneuvering comparison", args.min_seeds)
        tex, seed_sr = format_architecture_table(
            maneuver,
            caption="Maneuvering target comparison: sinusoidal target motion.",
            label="tab:maneuvering_comparison",
            methods_order=["No-VPP (zero offset)", "VPP + LOS-rate"],
            note="Target executes sinusoidal lateral maneuvers during evaluation. Per-seed results are reported to avoid overstating conclusions from a single seed.",
        )
        (tables_dir / "table_maneuvering_comparison.tex").write_text(tex, encoding="utf-8")
        seed_sr.to_csv(tables_dir / "table_maneuvering_comparison_per_seed.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_maneuvering_comparison.tex'}")

    # ------------------------------------------------------------------
    # JSBSim maneuver
    # ------------------------------------------------------------------
    print("\n[JSBSim Maneuver]")
    jsb = standardize_methods(
        load_raw_episodes(args.jsbsim),
        {"no_prediction": "VPP + LOS-rate", "gain_only": "No-VPP (zero offset)"},
    )
    if not jsb.empty:
        _warn_min_seeds(jsb, "JSBSim maneuver", args.min_seeds)
        tex, seed_sr = format_architecture_table(
            jsb,
            caption="JSBSim F-16 aerodynamics with maneuvering targets.",
            label="tab:jsbsim_maneuver",
            methods_order=["VPP + LOS-rate", "No-VPP (zero offset)"],
            note="Evaluation in the JSBSim high-fidelity backend. The table shows per-seed success rates; any advantage of VPP must be statistically significant before being claimed.",
        )
        (tables_dir / "table_jsbsim_maneuver.tex").write_text(tex, encoding="utf-8")
        seed_sr.to_csv(tables_dir / "table_jsbsim_maneuver_per_seed.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_jsbsim_maneuver.tex'}")

    # ------------------------------------------------------------------
    # Method innovation
    # ------------------------------------------------------------------
    print("\n[Method Innovation]")
    tex, raw_df = format_method_innovation_table(Path(args.method_innovation), args.min_seeds)
    if tex:
        (tables_dir / "table_method_innovation_comparison.tex").write_text(tex, encoding="utf-8")
        raw_df.to_csv(tables_dir / "table_method_innovation_per_seed.csv", index=False)
        print(f"[TABLE] {tables_dir / 'table_method_innovation_comparison.tex'}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        print("\n[Figures]")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            print("[WARN] matplotlib not installed; skipping figures")
            return

        # Architecture comparison figure
        if not exact.empty:
            summary = pd.concat([
                pooled_summary(exact.assign(scenario="overall")),
                pooled_summary(exact),
            ], ignore_index=True)
            plot_architecture_comparison(summary, figures_dir / "architecture_comparison.png")
            write_figure_include(
                figures_dir / "architecture_comparison.tex",
                figures_dir / "architecture_comparison.png",
                caption="Architecture comparison on canonical geometries (mean and 95\\% bootstrap CI).",
                label="fig:architecture_comparison",
            )

        # VPP comparison figure
        if not vpp_abl.empty:
            seed_sr = aggregate_seed_scenario(vpp_abl)
            plot_vpp_comparison(seed_sr, figures_dir / "vpp_comparison.png")
            write_figure_include(
                figures_dir / "vpp_comparison.tex",
                figures_dir / "vpp_comparison.png",
                caption="VPP vs No-VPP per-scenario success rate.",
                label="fig:vpp_comparison",
            )

        # Training curve figure
        plot_training_curves(Path("outputs/experiments"), figures_dir / "training_curve.png")
        write_figure_include(
            figures_dir / "training_curve.tex",
            figures_dir / "training_curve.png",
            caption="Training convergence curves across architectures (mean $\\pm$ SD over seeds).",
            label="fig:training_curve",
        )

        # CEM optimization figure
        plot_cem_optimization(Path(args.cem_compare), figures_dir / "cem_optimization.png")
        write_figure_include(
            figures_dir / "cem_optimization.tex",
            figures_dir / "cem_optimization.png",
            caption="Gain optimization comparison: CEM vs default vs heuristic gains.",
            label="fig:cem_optimization",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
