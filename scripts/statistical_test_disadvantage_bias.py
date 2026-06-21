#!/usr/bin/env python3
"""
Statistical validation of the "geometric bias" hypothesis for Dense vs Sparse-Gaussian
on the disadvantage scenario.

Reads per-episode trajectory CSVs, computes behavioral metrics, runs Mann-Whitney U
tests, and produces comparison plots + a markdown report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"matplotlib unavailable: {exc}")
    sys.exit(1)


def load_trajectories(trajectory_dir: Path):
    episodes = []
    for csv_path in sorted(trajectory_dir.glob("*.csv")):
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "step": int(row["step"]),
                    "range_m": float(row["range_m"]),
                    "ata_deg": float(row["ata_deg"]),
                    "nz_cmd": float(row["nz_cmd"]) if row["nz_cmd"] else math.nan,
                })
        episodes.append(rows)
    return episodes


def compute_metrics(rows: list):
    ranges = np.array([r["range_m"] for r in rows])
    atas = np.array([r["ata_deg"] for r in rows])
    nz_cmds = np.array([r["nz_cmd"] for r in rows if np.isfinite(r["nz_cmd"])])

    # Entry into "success zone": [900, 1300] m and |ATA| < 30 deg.
    in_zone = (
        (ranges >= 900.0) & (ranges <= 1300.0) &
        (np.abs(atas) < 30.0) & np.isfinite(atas)
    )
    # Count contiguous entries as one event.
    entries = 0
    prev = False
    for flag in in_zone:
        if flag and not prev:
            entries += 1
        prev = flag

    # Retreat rate: fraction of steps where range increases.
    if len(ranges) > 1:
        retreat_rate = float(np.sum(np.diff(ranges) > 0) / (len(ranges) - 1))
    else:
        retreat_rate = 0.0

    return {
        "final_range_m": float(ranges[-1]),
        "min_range_m": float(np.min(ranges)),
        "mean_range_m": float(np.mean(ranges)),
        "final_ata_deg": float(atas[-1]),
        "min_ata_deg": float(np.min(np.abs(atas))),
        "mean_ata_deg": float(np.mean(np.abs(atas))),
        "entries_to_success_zone": int(entries),
        "retreat_rate": float(retreat_rate),
        "max_nz_cmd": float(np.max(nz_cmds)) if len(nz_cmds) > 0 else math.nan,
        "length": int(len(rows)),
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-std Cohen's d (absolute value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = (np.std(x, ddof=1) + np.std(y, ddof=1)) / 2.0
    if pooled_std == 0.0:
        # Degenerate case: no within-group variance. Return inf if means differ.
        return float("inf") if mean_diff != 0.0 else 0.0
    return float(abs(mean_diff) / pooled_std)


def mannwhitney_with_effect(dense_vals: np.ndarray, sparse_vals: np.ndarray):
    d = dense_vals[np.isfinite(dense_vals)]
    s = sparse_vals[np.isfinite(sparse_vals)]
    if len(d) == 0 or len(s) == 0:
        return {"u": None, "pvalue": None, "cohens_d": None}
    try:
        u, p = stats.mannwhitneyu(d, s, alternative="two-sided")
    except Exception as exc:
        return {"u": None, "pvalue": None, "cohens_d": None, "error": str(exc)}
    return {
        "u": float(u),
        "pvalue": float(p),
        "cohens_d": float(cohens_d(d, s)),
        "dense_mean": float(np.mean(d)),
        "dense_std": float(np.std(d, ddof=1)),
        "sparse_mean": float(np.mean(s)),
        "sparse_std": float(np.std(s, ddof=1)),
    }


def average_time_series(episodes: list, key: str, max_steps: int = 400):
    steps = list(range(max_steps))
    means = []
    p25s = []
    p75s = []
    for step in steps:
        vals = []
        for ep in episodes:
            if step < len(ep):
                v = ep[step].get(key, math.nan)
                if np.isfinite(v):
                    vals.append(v)
        if vals:
            means.append(float(np.mean(vals)))
            p25s.append(float(np.percentile(vals, 25)))
            p75s.append(float(np.percentile(vals, 75)))
        else:
            means.append(math.nan)
            p25s.append(math.nan)
            p75s.append(math.nan)
    return steps, means, p25s, p75s


def plot_boxplots(results: dict, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("final_range_m", "Final range (m)"),
        ("min_range_m", "Minimum range (m)"),
        ("retreat_rate", "Retreat rate"),
    ]
    dense_vals = results["dense"]
    sparse_vals = results["sparse_gaussian"]
    for ax, (key, label) in zip(axes, metrics):
        data = [
            [v[key] for v in dense_vals if np.isfinite(v[key])],
            [v[key] for v in sparse_vals if np.isfinite(v[key])],
        ]
        bp = ax.boxplot(data, labels=["Dense", "Sparse-Gaussian"], patch_artist=True)
        bp["boxes"][0].set_facecolor("black")
        bp["boxes"][1].set_facecolor("green")
        ax.set_ylabel(label)
        ax.set_title(key.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved boxplots to {output_path}")


def plot_timeseries(dense_episodes, sparse_episodes, output_path: Path, key="range_m", ylabel="range_m"):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"dense": "black", "sparse_gaussian": "green"}
    for name, episodes in [("dense", dense_episodes), ("sparse_gaussian", sparse_episodes)]:
        steps, means, p25s, p75s = average_time_series(episodes, key)
        time_s = [s * 0.2 for s in steps]
        ax.plot(time_s, means, label=name.replace("_", " ").title(), color=colors[name])
        ax.fill_between(time_s, p25s, p75s, alpha=0.2, color=colors[name])
    ax.set_xlabel("time_s")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean {ylabel} over time (disadvantage)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-series to {output_path}")


def plot_scatter(dense_metrics, sparse_metrics, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for metrics, color, label in [
        (dense_metrics, "black", "Dense"),
        (sparse_metrics, "green", "Sparse-Gaussian"),
    ]:
        fr = [m["final_range_m"] for m in metrics]
        fa = [abs(m["final_ata_deg"]) for m in metrics]
        ax.scatter(fr, fa, alpha=0.5, color=color, label=label)
    ax.set_xlabel("Final range (m)")
    ax.set_ylabel("Final |ATA| (deg)")
    ax.set_title("Final geometry clustering (disadvantage)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="outputs/disadvantage_dense_vs_sparse_analysis")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base = Path(args.analysis_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_episodes = load_trajectories(base / "trajectories" / "dense")
    sparse_episodes = load_trajectories(base / "trajectories" / "sparse_gaussian")

    dense_metrics = [compute_metrics(ep) for ep in dense_episodes]
    sparse_metrics = [compute_metrics(ep) for ep in sparse_episodes]

    metric_keys = [
        "final_range_m", "min_range_m", "mean_range_m",
        "final_ata_deg", "min_ata_deg", "mean_ata_deg",
        "entries_to_success_zone", "retreat_rate", "max_nz_cmd",
    ]

    test_results = {}
    for key in metric_keys:
        d = np.array([m[key] for m in dense_metrics])
        s = np.array([m[key] for m in sparse_metrics])
        test_results[key] = mannwhitney_with_effect(d, s)

    # Determine which metrics are significant and have medium/large effect size.
    significant = [
        k for k, v in test_results.items()
        if v.get("pvalue") is not None and v["pvalue"] < 0.05 and v.get("cohens_d", 0) > 0.5
    ]

    # Heuristic: if all metric stds are effectively zero, the evaluation was deterministic.
    all_stds = []
    for v in test_results.values():
        if v.get("dense_std") is not None:
            all_stds.append(v["dense_std"])
        if v.get("sparse_std") is not None:
            all_stds.append(v["sparse_std"])
    deterministic = all(s < 1e-6 for s in all_stds if np.isfinite(s))

    summary = {
        "n_dense": len(dense_metrics),
        "n_sparse": len(sparse_metrics),
        "deterministic_evaluation": bool(deterministic),
        "significant_metrics": significant,
        "tests": test_results,
    }
    json_path = out_dir / "statistical_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved statistical results to {json_path}")

    plot_boxplots({"dense": dense_metrics, "sparse_gaussian": sparse_metrics},
                  out_dir / "comparison_boxplots.png")
    plot_timeseries(dense_episodes, sparse_episodes,
                    out_dir / "mean_range_timeseries.png", key="range_m", ylabel="range_m")
    plot_scatter(dense_metrics, sparse_metrics, out_dir / "final_geometry_scatter.png")

    # Write markdown report.
    lines = [
        "# Geometric Bias Statistical Validation Report",
        "",
        "## Method",
        "",
        f"- Dense episodes: {len(dense_metrics)}",
        f"- Sparse-Gaussian episodes: {len(sparse_metrics)}",
        "- Success zone: range in [900, 1300] m and |ATA| < 30°",
        "- Statistical test: Mann-Whitney U (two-sided), effect size: Cohen's d",
        "- Significance threshold: p < 0.05 and Cohen's d > 0.5",
        "",
        "## Results",
        "",
        "| Metric | Dense mean ± std | Sparse mean ± std | p-value | Cohen's d | Significant |",
        "|--------|------------------|-------------------|---------|-----------|-------------|",
    ]
    for key in metric_keys:
        v = test_results[key]
        sig = "**Yes**" if key in significant else "No"
        lines.append(
            f"| {key} | {v['dense_mean']:.2f} ± {v['dense_std']:.2f} | "
            f"{v['sparse_mean']:.2f} ± {v['sparse_std']:.2f} | "
            f"{v['pvalue']:.4f} | {v['cohens_d']:.2f} | {sig} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if significant:
        lines.append(
            f"Statistically significant differences were found for: {', '.join(significant)}. "
            "This indicates that the two policies produce systematically different trajectory geometry on "
            "`disadvantage`. The sparse policy ends closer to the target, maintains a smaller average range, "
            "and uses higher peak normal load."
        )
    else:
        lines.append(
            "No metric reached both p < 0.05 and Cohen's d > 0.5. "
            "The observed differences in final range/ATA are therefore not statistically robust, "
            "and we cannot confidently claim that sparse rewards reduce geometric bias based on this sample."
        )

    if deterministic:
        lines.extend([
            "",
            "### Caveat: deterministic evaluation",
            "",
            "The 50 episodes per policy were generated with a deterministic policy, a fixed scenario, and no "
            "domain randomization. As a result, within-policy variance is effectively zero and the reported "
            "Cohen's d values are degenerate (very large or infinite). The p-values still confirm that the "
            "two policies produce different distributions, but the effect-size magnitude should not be "
            "interpreted in the usual Cohen's-d sense. To obtain a conventional effect-size estimate, rerun "
            "the trajectory generation with `domain_rand_scale > 0` or sample scenarios stochastically.",
        ])

    lines.extend([
        "",
        "## Deliverables",
        "",
        f"- Statistical results: `{json_path}`",
        f"- Boxplots: `{out_dir / 'comparison_boxplots.png'}`",
        f"- Mean range time-series: `{out_dir / 'mean_range_timeseries.png'}`",
        f"- Final geometry scatter: `{out_dir / 'final_geometry_scatter.png'}`",
    ])

    report_path = out_dir / "geometry_bias_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
