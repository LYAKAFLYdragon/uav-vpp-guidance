#!/usr/bin/env python3
"""Compute cluster-robust standard errors for VPP vs No-VPP comparisons.

Reads outputs/fair_comparison/raw_episodes.csv, aggregates success rates per
scenario and training seed (cluster), and reports cluster-robust SEs for the
difference in success rates. This addresses the reviewer's concern that
episode-level Fisher tests ignore within-seed correlation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SCENARIOS = ["favorable", "neutral", "disadvantage", "challenging"]


def cluster_robust_diff(seed_rates_a, seed_rates_b):
    """Return mean difference, cluster-robust SE, t-stat, df, p-value.

    The clusters are paired (same seeds for both methods). The cluster-robust
    variance is the empirical variance of the paired differences across clusters
    divided by the number of clusters, with a finite-sample correction.
    """
    diffs = np.asarray(seed_rates_a) - np.asarray(seed_rates_b)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    # Finite-sample cluster-robust variance (unbiased estimator)
    variance = float(np.sum((diffs - mean_diff) ** 2) / (n * (n - 1)))
    se = float(np.sqrt(variance))
    t_stat = mean_diff / se if se > 0 else 0.0
    dof = n - 1
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof)) if se > 0 else 1.0
    return mean_diff, se, t_stat, dof, p_value


def compute(args):
    df = pd.read_csv(args.input)
    df = df[df["method"].isin(["vpp", "no_vpp"])]

    rows = []
    for scen in SCENARIOS:
        sub = df[df["scenario"] == scen]
        vpp = (
            sub[sub["method"] == "vpp"]
            .groupby("training_seed")["is_success"]
            .mean()
            .sort_index()
        )
        no_vpp = (
            sub[sub["method"] == "no_vpp"]
            .groupby("training_seed")["is_success"]
            .mean()
            .sort_index()
        )
        common = vpp.index.intersection(no_vpp.index)
        vpp_rates = vpp.loc[common].values
        no_vpp_rates = no_vpp.loc[common].values
        n_seeds = len(common)

        mean_diff, se, t_stat, dof, p_value = cluster_robust_diff(vpp_rates, no_vpp_rates)
        rows.append({
            "scenario": scen,
            "n_seeds": n_seeds,
            "vpp_sr_mean": float(np.mean(vpp_rates)),
            "no_vpp_sr_mean": float(np.mean(no_vpp_rates)),
            "diff": mean_diff,
            "cluster_robust_se": se,
            "t_stat": t_stat,
            "df": dof,
            "p_value": p_value,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[CSV] {out_path}")

    tex_path = Path("paper_materials/tables/table_clustered_se.tex")
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Cluster-robust standard errors for VPP vs No-VPP success-rate differences (seed as cluster).}",
        "\\label{tab:clustered_se}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Scenario & $\\Delta$SR & Cluster-robust SE & $t$ (df) & $p$-value \\\\",
        "\\midrule",
    ]
    for _, r in out_df.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        tex_lines.append(
            f"{r['scenario'].capitalize()} & {r['diff']:.3f} & {r['cluster_robust_se']:.3f} & "
            f"{r['t_stat']:.2f} ({int(r['df'])}) & {r['p_value']:.4f} {sig} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Cluster-robust SEs treat each training seed as an independent cluster, addressing within-seed episode correlation. Significance: *** $p<0.001$, ** $p<0.01$, * $p<0.05$.",
        "\\end{tablenotes}",
        "\\end{table}",
    ])
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"[TEX] {tex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/fair_comparison/raw_episodes.csv")
    parser.add_argument("--output", type=str, default="outputs/fair_comparison/clustered_se.csv")
    args = parser.parse_args()
    compute(args)


if __name__ == "__main__":
    main()
