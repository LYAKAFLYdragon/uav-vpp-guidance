#!/usr/bin/env python3
"""Generate LaTeX tables from evaluation results for paper update.

Usage:
    python scripts/generate_paper_tables.py

Reads evaluation outputs from docs/results/ and generates:
    - paper_materials/tables/table_vpp_ablation.tex
    - paper_materials/tables/table_bilevel_ablation.tex
    - paper_materials/tables/table_maneuver_comparison.tex
    - paper_materials/tables/table_cross_mode_summary.tex
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.evaluation.statistical_comparison import paired_t_test, cohens_d


def load_eval_csv(result_dir: Path) -> pd.DataFrame:
    """Load raw_episodes.csv from a benchmark output directory."""
    csv_path = result_dir / "raw_episodes.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute success rate, mean return, std return from evaluation dataframe."""
    if df.empty:
        return {"success_rate": 0.0, "mean_return": 0.0, "std_return": 0.0, "n": 0}
    success = df["is_success"].astype(float).mean()
    returns = df["episode_return"].astype(float)
    return {
        "success_rate": success,
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "n": len(returns),
    }


def compare_to_baseline(df_method: pd.DataFrame, df_baseline: pd.DataFrame) -> tuple:
    """Return p-value and Cohen's d comparing method to baseline on episode return."""
    if df_method.empty or df_baseline.empty:
        return float("nan"), float("nan")
    r_method = df_method["episode_return"].astype(float).values
    r_baseline = df_baseline["episode_return"].astype(float).values
    if len(r_method) != len(r_baseline):
        # truncate to minimum length for paired test
        n = min(len(r_method), len(r_baseline))
        r_method = r_method[:n]
        r_baseline = r_baseline[:n]
    p_val, _ = paired_t_test(r_method, r_baseline)
    d_val = cohens_d(r_method, r_baseline)
    return p_val, d_val


def format_table_vpp_ablation(vpp_df: pd.DataFrame, no_vpp_df: pd.DataFrame) -> str:
    vpp = compute_metrics(vpp_df)
    no_vpp = compute_metrics(no_vpp_df)
    p, d = compare_to_baseline(vpp_df, no_vpp_df)

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{VPP ablation: VPP-enabled policy vs direct-command (no VPP) policy under constant-velocity target motion.}\n"
        "\\label{tab:vpp_ablation}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Success Rate & Mean Return & $p$ vs No-VPP & Cohen's $d$ \\\\\n"
        "\\midrule\n"
        f"Direct Command (No VPP) & {no_vpp['success_rate']:.1%} & "
        f"${no_vpp['mean_return']:.2f} \\pm {no_vpp['std_return']:.2f}$ & --- & --- \\\\\n"
        f"VPP + No-Prediction & {vpp['success_rate']:.1%} & "
        f"${vpp['mean_return']:.2f} \\pm {vpp['std_return']:.2f}$ & "
        f"{p:.4f} & {d:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\small\n"
        "\\item Evaluated on $80$ episodes ($10$ seeds $\\times$ $8$ scenarios).\n"
        "\\item $p$: paired $t$-test on episode return vs No-VPP baseline.\n"
        "\\end{tablenotes}\n"
        "\\end{table}\n"
    )


def format_table_bilevel_ablation(single_df: pd.DataFrame, bilevel_df: pd.DataFrame) -> str:
    single = compute_metrics(single_df)
    bilevel = compute_metrics(bilevel_df)
    p, d = compare_to_baseline(bilevel_df, single_df)

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Bilevel ablation: single-layer PPO vs bilevel strategy-gain optimization under constant-velocity target motion.}\n"
        "\\label{tab:bilevel_ablation}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Success Rate & Mean Return & $p$ vs Single & Cohen's $d$ \\\\\n"
        "\\midrule\n"
        f"VPP + Single-Layer PPO & {single['success_rate']:.1%} & "
        f"${single['mean_return']:.2f} \\pm {single['std_return']:.2f}$ & --- & --- \\\\\n"
        f"VPP + Bilevel PPO & {bilevel['success_rate']:.1%} & "
        f"${bilevel['mean_return']:.2f} \\pm {bilevel['std_return']:.2f}$ & "
        f"{p:.4f} & {d:.3f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\small\n"
        "\\item Single-layer: PPO policy with fixed guidance gains.\n"
        "\\item Bilevel: PPO policy co-optimized with CEM gain search.\n"
        "\\item $p$: paired $t$-test on episode return vs single-layer baseline.\n"
        "\\end{tablenotes}\n"
        "\\end{table}\n"
    )


def format_table_maneuver_comparison(dfs: dict) -> str:
    methods = ["no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"]
    labels = {
        "no_prediction": "No-Prediction",
        "cv_prediction": "CV Prediction",
        "ca_prediction": "CA Prediction",
        "lstm_frozen": "LSTM (frozen)",
        "gru_frozen": "GRU (frozen)",
    }
    baseline_df = dfs.get("no_prediction", pd.DataFrame())

    rows = []
    for m in methods:
        df = dfs.get(m, pd.DataFrame())
        metrics = compute_metrics(df)
        if m == "no_prediction":
            rows.append(
                f"{labels[m]} & {metrics['success_rate']:.1%} & "
                f"${metrics['mean_return']:.2f} \\pm {metrics['std_return']:.2f}$ & --- & --- \\\\\n"
            )
        else:
            p, d = compare_to_baseline(df, baseline_df)
            rows.append(
                f"{labels[m]} & {metrics['success_rate']:.1%} & "
                f"${metrics['mean_return']:.2f} \\pm {metrics['std_return']:.2f}$ & "
                f"{p:.4f} & {d:.3f} \\\\\n"
            )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Predictor comparison under maneuvering target motion (sinusoidal lateral weave, $N = 80$ episodes per method).}\n"
        "\\label{tab:maneuver_comparison}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Method & Success Rate & Mean Return & $p$ vs No-Pred & Cohen's $d$ \\\\\n"
        "\\midrule\n"
        + "".join(rows) +
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\small\n"
        "\\item Evaluated on $80$ episodes ($10$ seeds $\\times$ $8$ scenarios) under sinusoidal target motion.\n"
        "\\item $p$: paired $t$-test on episode return vs No-Prediction baseline.\n"
        "\\end{tablenotes}\n"
        "\\end{table}\n"
    )


def format_table_cross_mode_summary(cv_dfs: dict, maneuver_dfs: dict) -> str:
    methods = ["no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen"]
    labels = {
        "no_prediction": "No-Prediction",
        "cv_prediction": "Parametric (CV/CA)",
        "ca_prediction": "Parametric (CV/CA)",
        "lstm_frozen": "Neural (LSTM)",
    }

    # Group parametric
    cv_no_pred = compute_metrics(cv_dfs.get("no_prediction", pd.DataFrame()))["success_rate"]
    cv_param = compute_metrics(cv_dfs.get("ca_prediction", cv_dfs.get("cv_prediction", pd.DataFrame())))["success_rate"]
    maneuver_no_pred = compute_metrics(maneuver_dfs.get("no_prediction", pd.DataFrame()))["success_rate"]
    maneuver_param = compute_metrics(maneuver_dfs.get("ca_prediction", maneuver_dfs.get("cv_prediction", pd.DataFrame())))["success_rate"]
    maneuver_lstm = compute_metrics(maneuver_dfs.get("lstm_frozen", pd.DataFrame()))["success_rate"]

    rows = [
        f"No-Prediction & {cv_no_pred:.1%} & {maneuver_no_pred:.1%} & {maneuver_no_pred - cv_no_pred:.0f} \\\\\n",
        f"Parametric (CV/CA) & {cv_param:.1%} & {maneuver_param:.1%} & {maneuver_param - cv_param:.0f} \\\\\n",
        f"Neural (LSTM) & --- & {maneuver_lstm:.1%} & --- \\\\\n",
    ]

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Cross-target-mode summary: predictor value depends on target kinematics.}\n"
        "\\label{tab:cross_mode_summary}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Method & Constant Velocity & Maneuvering & $\\Delta$ \\\\\n"
        "\\midrule\n"
        + "".join(rows) +
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\small\n"
        "\\item ---: Not trained/evaluated under this target mode.\n"
        "\\item $\\Delta$: absolute difference in success rate (percentage points).\n"
        "\\end{tablenotes}\n"
        "\\end{table}\n"
    )


def main():
    results_root = Path("docs/results")
    tables_dir = Path("paper_materials/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # P0-A
    p0a_vpp = load_eval_csv(results_root / "p0a_vpp_ablation")
    p0a_no_vpp = load_eval_csv(results_root / "p0a_no_vpp_ablation")
    if not p0a_vpp.empty and not p0a_no_vpp.empty:
        (tables_dir / "table_vpp_ablation.tex").write_text(
            format_table_vpp_ablation(p0a_vpp, p0a_no_vpp), encoding="utf-8"
        )
        print("Generated table_vpp_ablation.tex")
    else:
        print("Skipping table_vpp_ablation.tex (missing data)")

    # P0-B
    p0b_single = load_eval_csv(results_root / "stage6b_constant_velocity")
    p0b_bilevel = load_eval_csv(results_root / "p0b_bilevel_ablation")
    if not p0b_single.empty and not p0b_bilevel.empty:
        (tables_dir / "table_bilevel_ablation.tex").write_text(
            format_table_bilevel_ablation(p0b_single, p0b_bilevel), encoding="utf-8"
        )
        print("Generated table_bilevel_ablation.tex")
    else:
        print("Skipping table_bilevel_ablation.tex (missing data)")

    # P1-A / P1-B
    maneuver = {
        "no_prediction": load_eval_csv(results_root / "p1a_maneuver_target"),
        "cv_prediction": load_eval_csv(results_root / "p1a_maneuver_target"),
        "ca_prediction": load_eval_csv(results_root / "p1a_maneuver_target"),
        "lstm_frozen": load_eval_csv(results_root / "p1b_neural_maneuver"),
        "gru_frozen": load_eval_csv(results_root / "p1b_neural_maneuver"),
    }
    # run_paper_benchmark puts all methods in one CSV; we need to filter by method column
    for key in maneuver:
        if not maneuver[key].empty:
            maneuver[key] = maneuver[key][maneuver[key]["method"] == key]

    if any(not df.empty for df in maneuver.values()):
        (tables_dir / "table_maneuver_comparison.tex").write_text(
            format_table_maneuver_comparison(maneuver), encoding="utf-8"
        )
        print("Generated table_maneuver_comparison.tex")
    else:
        print("Skipping table_maneuver_comparison.tex (missing data)")

    # Cross-mode summary
    cv = {
        "no_prediction": load_eval_csv(results_root / "stage6b_constant_velocity"),
        "cv_prediction": load_eval_csv(results_root / "stage6b_constant_velocity"),
        "ca_prediction": load_eval_csv(results_root / "stage6b_constant_velocity"),
    }
    for key in cv:
        if not cv[key].empty:
            cv[key] = cv[key][cv[key]["method"] == key]

    if any(not df.empty for df in list(cv.values()) + list(maneuver.values())):
        (tables_dir / "table_cross_mode_summary.tex").write_text(
            format_table_cross_mode_summary(cv, maneuver), encoding="utf-8"
        )
        print("Generated table_cross_mode_summary.tex")
    else:
        print("Skipping table_cross_mode_summary.tex (missing data)")

    print("\nDone.")


if __name__ == "__main__":
    main()
