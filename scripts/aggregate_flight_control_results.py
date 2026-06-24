#!/usr/bin/env python3
"""
Aggregate flight-control comparison results.

Loads raw episode JSONs, computes descriptive statistics with BCa 95% CIs,
performs paired significance tests (t-test or Wilcoxon with Holm correction),
and exports summary CSVs.

Usage:
    python scripts/aggregate_flight_control_results.py \
        --run-dir outputs/flight_control_compare/fc_compare_20260621_103000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def _holm_correction(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction without statsmodels."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    adj_sorted = np.empty(m, dtype=float)
    prev = 0.0
    for i, p in enumerate(sorted_p):
        adj = min(max(p * (m - i), prev), 1.0)
        adj_sorted[i] = adj
        prev = adj
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    reject = adj <= alpha
    return reject, adj


# Statistical families for paired testing.  Tests and Holm correction are
# applied *within* each family so that unrelated comparisons do not inflate
# the correction penalty.
COMPARISON_FAMILIES = {
    # System-level closed-loop comparison: both use Enhanced PID low-level;
    # the only difference is adaptive (PPO+PID-Hybrid) vs. fixed (PPO) guidance.
    "system_closed_loop": [("ppo_pid", "ppo")],
    # Low-level controller isolation: both use zero-offset VPP guidance;
    # the only difference is Enhanced vs. Baseline PID.
    "low_level_isolation": [("enhanced_pid", "baseline_pid")],
    # APIC low-level comparison: both track the same reference with zero-offset
    # guidance; the only difference is RL-tuned PID gains vs. fixed Enhanced PID.
    "apic_low_level": [("apic_pid", "enhanced_pid")],
}


METRICS_BY_TASK = {
    # Canonical metric names follow the task specification (sections 7.5 / 8.5).
    "multi_waypoint": [
        "completed_waypoints",
        "mean_track_error_m",
        "median_track_error_m",
        "max_track_error_m",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_nz_g",
        "max_nz_g",
        "saturation_ratio",
        "mean_aggressiveness",
        "mean_gain_scale",
    ],
    "sustained_turn": [
        "completed_orbits",
        "avg_turn_rate_deg_s",
        "std_turn_rate_deg_s",
        "avg_turn_radius_m",
        "std_turn_radius_m",
        "radius_std_m",
        "avg_speed_mps",
        "std_speed_mps",
        "mean_nz_g",
        "max_nz_g",
        "energy_loss_rate_mps2",
        "saturation_ratio",
        "mean_aggressiveness",
        "mean_gain_scale",
    ],
    "break_turn": [
        "mean_track_error_m",
        "std_track_error_m",
        "median_track_error_m",
        "max_track_error_m",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_nz_g",
        "max_nz_g",
        "energy_loss_rate_mps2",
        "saturation_ratio",
        "mean_aggressiveness",
        "mean_gain_scale",
    ],
    "crossing_threshold": [
        "mean_track_error_m",
        "std_track_error_m",
        "median_track_error_m",
        "max_track_error_m",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_nz_g",
        "max_nz_g",
        "energy_loss_rate_mps2",
        "saturation_ratio",
        "mean_aggressiveness",
        "mean_gain_scale",
    ],
}


def _load_episode_rows(run_dir: Path) -> pd.DataFrame:
    rows = []
    raw_dir = run_dir / "raw"
    # Backward-compatible aliases for older episode JSONs.
    _ALIASES = {
        "mean_track_error_m": ["mean_range_m"],
        "median_track_error_m": ["median_range_m"],
        "max_track_error_m": ["max_range_m"],
        "std_track_error_m": ["std_range_m"],
        "avg_turn_rate_deg_s": ["mean_turn_rate_deg_s"],
        "std_turn_rate_deg_s": ["std_turn_rate_deg_s"],
        "avg_turn_radius_m": ["mean_turn_radius_m"],
        "std_turn_radius_m": ["std_turn_radius_m"],
        "avg_speed_mps": ["mean_speed_mps"],
        "std_speed_mps": ["std_speed_mps"],
        "radius_std_m": ["radius_std_m"],
    }

    def _get_stat(stats_block: dict, metric: str):
        if metric in stats_block:
            return stats_block[metric]
        for alias in _ALIASES.get(metric, []):
            if alias in stats_block:
                return stats_block[alias]
        return np.nan

    for json_path in raw_dir.rglob("*.json"):
        if json_path.name.endswith("_aggregate.json"):
            continue
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        stats_block = obj.get("statistics", {})
        row = {
            "task": obj["task"],
            "controller": obj["controller"],
            "seed": obj["seed"],
            "episode": obj["episode"],
            "scenario": obj.get("scenario"),
            "success": obj.get("success", False),
            "termination_reason": obj.get("termination_reason", "unknown"),
            "total_time_s": obj.get("total_time_s", np.nan),
            "steps": obj.get("steps", np.nan),
        }
        for task, metrics in METRICS_BY_TASK.items():
            if obj["task"] == task:
                for m in metrics:
                    row[m] = _get_stat(stats_block, m)
                if task == "multi_waypoint":
                    row["completed_waypoints"] = obj.get("completed_waypoints", stats_block.get("completed_waypoints", np.nan))
                elif task == "sustained_turn":
                    row["completed_orbits"] = obj.get("completed_orbits", stats_block.get("completed_orbits", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def ci95_bca(x: np.ndarray, seed: int = 1234) -> tuple[float, float]:
    """BCa bootstrap 95% confidence interval for the mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    res = stats.bootstrap(
        (x,),
        np.mean,
        n_resamples=10000,
        confidence_level=0.95,
        method="BCa",
        random_state=seed,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _summarize_group(g: pd.DataFrame, metrics: list, top_level_metrics: list, row_base: dict) -> dict:
    """Compute mean/std/BCa-CI for a single group and return a row dict."""
    row = dict(row_base)
    row["n"] = int(len(g))
    for m in metrics:
        if m not in g.columns:
            continue
        vals = g[m].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        lo, hi = ci95_bca(vals)
        row[f"{m}_ci95_low"] = lo
        row[f"{m}_ci95_high"] = hi
    for m in top_level_metrics:
        if m not in g.columns:
            continue
        vals = g[m].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        lo, hi = ci95_bca(vals)
        row[f"{m}_ci95_low"] = lo
        row[f"{m}_ci95_high"] = hi
    return row


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std/BCa-CI for every task x controller x metric."""
    rows = []
    metrics = sorted({m for ms in METRICS_BY_TASK.values() for m in ms})
    top_level_metrics = ["total_time_s"]
    for (task, controller), g in df.groupby(["task", "controller"], sort=False):
        rows.append(_summarize_group(g, metrics, top_level_metrics, {"task": task, "controller": controller}))
    return pd.DataFrame(rows)


def summarize_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std/BCa-CI for every task x scenario x controller x metric."""
    rows = []
    metrics = sorted({m for ms in METRICS_BY_TASK.values() for m in ms})
    top_level_metrics = ["total_time_s"]
    df = df.copy()
    df["scenario"] = df["scenario"].fillna("__default__")
    for (task, scenario, controller), g in df.groupby(["task", "scenario", "controller"], sort=False):
        rows.append(
            _summarize_group(
                g,
                metrics,
                top_level_metrics,
                {"task": task, "scenario": scenario, "controller": controller},
            )
        )
    return pd.DataFrame(rows)


def _compute_diff(df: pd.DataFrame, task: str, metric: str, a: str, b: str) -> pd.DataFrame:
    """Return paired differences for one controller pair and task/metric."""
    da = df[(df.task == task) & (df.controller == a)][["seed", "episode", metric]].rename(
        columns={metric: "a"}
    )
    db = df[(df.task == task) & (df.controller == b)][["seed", "episode", metric]].rename(
        columns={metric: "b"}
    )
    return da.merge(db, on=["seed", "episode"], how="inner").dropna()


def _is_degenerate(diff: pd.DataFrame, metric: str) -> tuple[bool, str]:
    """
    Detect degenerate comparisons that cannot support meaningful inference.

    Returns (is_degenerate, test_name).
    """
    if len(diff) < 3:
        return True, "not_applicable_insufficient_pairs"

    a_vals = diff["a"].to_numpy(dtype=float)
    b_vals = diff["b"].to_numpy(dtype=float)
    deltas = a_vals - b_vals

    # All-zero metrics (e.g. saturation_ratio when no actuator saturation occurs).
    if np.allclose(a_vals, 0.0) and np.allclose(b_vals, 0.0):
        return True, "not_applicable_all_zero"

    # No variation at all across either group.
    if np.std(a_vals, ddof=1) == 0.0 and np.std(b_vals, ddof=1) == 0.0:
        return True, "not_applicable_zero_variance"

    # All paired differences are exactly zero.
    if np.allclose(deltas, 0.0):
        return True, "not_applicable_zero_difference"

    return False, ""


def paired_test(df: pd.DataFrame, task: str, metric: str, family: str) -> pd.DataFrame:
    """Run paired tests for one task/metric within a single comparison family."""
    raw_pvalues = []
    merged_cache = []
    pairs = COMPARISON_FAMILIES[family]

    for a, b in pairs:
        d = _compute_diff(df, task, metric, a, b)
        diff = (d["a"] - d["b"]).to_numpy(dtype=float)

        degenerate, degenerate_name = _is_degenerate(d, metric)

        if degenerate:
            merged_cache.append({
                "task": task,
                "metric": metric,
                "comparison_family": family,
                "lhs": a,
                "rhs": b,
                "n_pairs": int(len(diff)),
                "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "shapiro_p": np.nan,
                "test": degenerate_name,
                "p_raw": np.nan,
                "p_adj_holm": np.nan,
                "effect_dz": np.nan,
                "significant": False,
            })
            continue

        shapiro_p = stats.shapiro(diff).pvalue

        if np.isfinite(shapiro_p) and shapiro_p >= 0.05:
            res = stats.ttest_rel(d["a"], d["b"])
            p = float(res.pvalue)
            test_name = "paired_t"
        else:
            method = "exact" if len(diff) <= 50 else "asymptotic"
            rounded_diff = np.round(diff, 8)
            try:
                res = stats.wilcoxon(rounded_diff, method=method)
                p = float(res.pvalue)
            except ValueError:
                p = 1.0
            test_name = f"wilcoxon_{method}"

        lo, hi = ci95_bca(diff)
        dz = (
            float(np.mean(diff) / np.std(diff, ddof=1))
            if len(diff) > 1 and np.std(diff, ddof=1) > 0 else np.nan
        )

        merged_cache.append({
            "task": task,
            "metric": metric,
            "comparison_family": family,
            "lhs": a,
            "rhs": b,
            "n_pairs": int(len(diff)),
            "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "shapiro_p": float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
            "test": test_name,
            "p_raw": p,
            "effect_dz": dz,
        })
        raw_pvalues.append(p)

    # Apply Holm correction only to the non-degenerate p-values within this family.
    if raw_pvalues:
        reject, p_adj = _holm_correction(raw_pvalues, alpha=0.05)
        # Walk through the list and attach corrected p-values/significance only
        # to rows that actually contributed a p-value.
        padj_iter = iter(p_adj)
        rej_iter = iter(reject)
        for row in merged_cache:
            if not np.isnan(row.get("p_raw", np.nan)):
                row["p_adj_holm"] = float(next(padj_iter))
                row["significant"] = bool(next(rej_iter))

    return pd.DataFrame(merged_cache)


def _read_run_id(run_dir: Path) -> str:
    """Read run_id from manifest if available, otherwise fall back to directory name."""
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    if manifest_path.exists():
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            return obj.get("run_id", run_dir.name)
        except Exception:
            pass
    return run_dir.name


def main():
    parser = argparse.ArgumentParser(description="Aggregate flight-control comparison results")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run root directory")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_id = _read_run_id(run_dir)
    aggregate_dir = run_dir / "aggregate"
    tables_dir = run_dir / "tables"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _load_episode_rows(run_dir)
    if df.empty:
        raise ValueError(f"No episode JSONs found under {run_dir / 'raw'}")

    df.to_csv(aggregate_dir / "all_episodes.csv", index=False)

    summary = summarize_metrics(df)
    summary.to_csv(aggregate_dir / "summary.csv", index=False)
    summary.to_json(aggregate_dir / "summary.json", orient="records", indent=2)

    by_scenario = summarize_by_scenario(df)
    by_scenario.to_csv(aggregate_dir / "summary_by_scenario.csv", index=False)
    by_scenario.to_json(aggregate_dir / "summary_by_scenario.json", orient="records", indent=2)

    def _agg_key(m: str, prefix: str) -> str:
        """Build aggregate JSON key: strip redundant 'mean_' when present."""
        if m.startswith("mean_"):
            base = m[len("mean_"):]
        else:
            base = m
        return f"{prefix}_{base}"

    # Per-task aggregate JSON files
    for (task, controller), g in df.groupby(["task", "controller"]):
        metrics = METRICS_BY_TASK.get(task, [])
        agg = {
            "run_id": run_id,
            "task": task,
            "controller": controller,
            "n_episodes": int(len(g)),
        }
        for m in metrics:
            vals = g[m].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            agg[_agg_key(m, "mean")] = float(np.mean(vals)) if len(vals) else np.nan
            agg[_agg_key(m, "std")] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            lo, hi = ci95_bca(vals)
            agg[_agg_key(m, "ci95")] = [lo, hi]
        out_path = aggregate_dir / f"{task}_{controller}_aggregate.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)

    # Paired tests, separated by comparison family and Holm-corrected within each.
    all_tests = []
    for task in df["task"].unique():
        metrics = METRICS_BY_TASK.get(task, [])
        for metric in metrics:
            if df[(df.task == task) & df[metric].notna()].empty:
                continue
            for family in COMPARISON_FAMILIES:
                test_df = paired_test(df, task, metric, family)
                all_tests.append(test_df)
                test_df.to_csv(tables_dir / f"pairwise_{family}_{task}_{metric}.csv", index=False)

    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(
            tables_dir / "pairwise_statistics.csv", index=False
        )

    print(f"Aggregated {len(df)} episodes")
    print(f"Summary: {aggregate_dir / 'summary.csv'}")
    print(f"Pairwise stats: {tables_dir / 'pairwise_statistics.csv'}")


if __name__ == "__main__":
    main()
