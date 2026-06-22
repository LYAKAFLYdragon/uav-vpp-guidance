#!/usr/bin/env python3
"""
Export Table 3 (flight-control comparison summary) as Markdown, CSV, and Excel.

Usage:
    python scripts/export_table3.py \
        --run-dir outputs/flight_control_compare/fc_compare_20260621_103000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


CONTROLLER_LABELS = {
    "ppo_pid": "PPO+PID",
    "ppo": "PPO",
    "apic_pid": "APIC-PID",
    "enhanced_pid": "Enhanced PID",
    "baseline_pid": "Baseline PID",
    "gain_scheduled_pid": "GainScheduled PID",
}


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No data_"
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.values.tolist()]
    col_widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    body_lines = [
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([header_line, separator] + body_lines)


def _fmt_mean_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _fmt_mean_ci(mean: float, ci_low: float, ci_high: float) -> str:
    if pd.isna(mean):
        return "N/A"
    if pd.isna(ci_low) or pd.isna(ci_high):
        return f"{mean:.2f}"
    return f"{mean:.2f} [{ci_low:.2f}, {ci_high:.2f}]"


def _load_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "aggregate" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def _build_table3a(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Only include controllers that actually appear in the summary.
    evaluated = set(summary[summary["task"] == "multi_waypoint"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "multi_waypoint") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed waypoints": _fmt_mean_std(row.get("completed_waypoints_mean"), row.get("completed_waypoints_std")),
                "Total time s": _fmt_mean_std(row.get("total_time_s_mean"), row.get("total_time_s_std")),
                "Mean track error m": _fmt_mean_std(row.get("mean_track_error_m_mean"), row.get("mean_track_error_m_std")),
                "Mean speed m/s": _fmt_mean_std(row.get("mean_speed_mps_mean"), row.get("mean_speed_mps_std")),
                "Mean NZ g": _fmt_mean_std(row.get("mean_nz_g_mean"), row.get("mean_nz_g_std")),
                "Mean aggressiveness": (
                    f"{row.get('mean_aggressiveness_mean'):.3f}"
                    if pd.notna(row.get("mean_aggressiveness_mean"))
                    else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_table3b(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluated = set(summary[summary["task"] == "sustained_turn"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "sustained_turn") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed orbits": _fmt_mean_std(row.get("completed_orbits_mean"), row.get("completed_orbits_std")),
                "Avg turn rate °/s": _fmt_mean_std(row.get("avg_turn_rate_deg_s_mean"), row.get("avg_turn_rate_deg_s_std")),
                "Avg turn radius m": _fmt_mean_std(row.get("avg_turn_radius_m_mean"), row.get("avg_turn_radius_m_std")),
                "Radius std m": _fmt_mean_std(row.get("radius_std_m_mean"), row.get("radius_std_m_std")),
                "Mean speed m/s": _fmt_mean_std(row.get("avg_speed_mps_mean"), row.get("avg_speed_mps_std")),
                "Mean NZ g": _fmt_mean_std(row.get("mean_nz_g_mean"), row.get("mean_nz_g_std")),
                "Max NZ g": _fmt_mean_std(row.get("max_nz_g_mean"), row.get("max_nz_g_std")),
                "Energy loss rate m/s²": _fmt_mean_std(row.get("energy_loss_rate_mps2_mean"), row.get("energy_loss_rate_mps2_std")),
                "Mean aggressiveness": (
                    f"{row.get('mean_aggressiveness_mean'):.3f}"
                    if pd.notna(row.get("mean_aggressiveness_mean"))
                    else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_table3c(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Build a simple overall ranking table."""
    rows = []
    mw_controllers = set(summary[summary["task"] == "multi_waypoint"]["controller"].unique())
    st_controllers = set(summary[summary["task"] == "sustained_turn"]["controller"].unique())
    evaluated = mw_controllers | st_controllers
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        mw = summary[(summary["task"] == "multi_waypoint") & (summary["controller"] == controller)]
        st = summary[(summary["task"] == "sustained_turn") & (summary["controller"] == controller)]
        mw_score = mw["completed_waypoints_mean"].values[0] if not mw.empty else -float("inf")
        st_score = st["completed_orbits_mean"].values[0] if not st.empty else -float("inf")
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Multi-waypoint score": float(mw_score) if pd.notna(mw_score) else -float("inf"),
                "Sustained-turn score": float(st_score) if pd.notna(st_score) else -float("inf"),
            }
        )
    df = pd.DataFrame(rows)
    df["Multi-waypoint rank"] = df["Multi-waypoint score"].rank(ascending=False, method="min").astype(int)
    df["Sustained-turn rank"] = df["Sustained-turn score"].rank(ascending=False, method="min").astype(int)
    df["Average rank"] = (df["Multi-waypoint rank"] + df["Sustained-turn rank"]) / 2.0
    df["Overall rank"] = df["Average rank"].rank(method="min").astype(int)

    # Simple recommendation based on overall rank.
    best_rank = df["Overall rank"].min()
    df["Recommendation"] = df["Overall rank"].apply(
        lambda r: "Recommended" if r == best_rank else "Keep under review"
    )

    # Drop helper numeric columns for display.
    display = df[
        ["Controller", "Overall rank", "Multi-waypoint rank", "Sustained-turn rank", "Recommendation"]
    ]
    return display


def _build_table3a_mean_only(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean-only version of Table 3A for the Excel table3_mean sheet."""
    rows = []
    evaluated = set(summary[summary["task"] == "multi_waypoint"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "multi_waypoint") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed waypoints": _fmt_value(row.get("completed_waypoints_mean")),
                "Total time s": _fmt_value(row.get("total_time_s_mean")),
                "Mean track error m": _fmt_value(row.get("mean_track_error_m_mean")),
                "Mean speed m/s": _fmt_value(row.get("mean_speed_mps_mean")),
                "Mean NZ g": _fmt_value(row.get("mean_nz_g_mean")),
                "Mean aggressiveness": (
                    f"{row.get('mean_aggressiveness_mean'):.3f}"
                    if pd.notna(row.get("mean_aggressiveness_mean")) else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_table3b_mean_only(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean-only version of Table 3B for the Excel table3_mean sheet."""
    rows = []
    evaluated = set(summary[summary["task"] == "sustained_turn"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "sustained_turn") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed orbits": _fmt_value(row.get("completed_orbits_mean")),
                "Avg turn rate °/s": _fmt_value(row.get("avg_turn_rate_deg_s_mean")),
                "Avg turn radius m": _fmt_value(row.get("avg_turn_radius_m_mean")),
                "Radius std m": _fmt_value(row.get("radius_std_m_mean")),
                "Mean speed m/s": _fmt_value(row.get("avg_speed_mps_mean")),
                "Mean NZ g": _fmt_value(row.get("mean_nz_g_mean")),
                "Max NZ g": _fmt_value(row.get("max_nz_g_mean")),
                "Energy loss rate m/s²": _fmt_value(row.get("energy_loss_rate_mps2_mean")),
                "Mean aggressiveness": (
                    f"{row.get('mean_aggressiveness_mean'):.3f}"
                    if pd.notna(row.get("mean_aggressiveness_mean")) else "N/A"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_table3a_std_ci(summary: pd.DataFrame) -> pd.DataFrame:
    """Std + CI95 version of Table 3A for the Excel table3_std_ci sheet."""
    rows = []
    evaluated = set(summary[summary["task"] == "multi_waypoint"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "multi_waypoint") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed waypoints": _fmt_mean_ci(
                    row.get("completed_waypoints_mean"), row.get("completed_waypoints_ci95_low"), row.get("completed_waypoints_ci95_high")
                ),
                "Total time s": _fmt_mean_ci(
                    row.get("total_time_s_mean"), row.get("total_time_s_ci95_low"), row.get("total_time_s_ci95_high")
                ),
                "Mean track error m": _fmt_mean_ci(
                    row.get("mean_track_error_m_mean"), row.get("mean_track_error_m_ci95_low"), row.get("mean_track_error_m_ci95_high")
                ),
                "Mean speed m/s": _fmt_mean_ci(
                    row.get("mean_speed_mps_mean"), row.get("mean_speed_mps_ci95_low"), row.get("mean_speed_mps_ci95_high")
                ),
                "Mean NZ g": _fmt_mean_ci(
                    row.get("mean_nz_g_mean"), row.get("mean_nz_g_ci95_low"), row.get("mean_nz_g_ci95_high")
                ),
                "Mean aggressiveness": "N/A",
            }
        )
    return pd.DataFrame(rows)


def _build_table3b_std_ci(summary: pd.DataFrame) -> pd.DataFrame:
    """Std + CI95 version of Table 3B for the Excel table3_std_ci sheet."""
    rows = []
    evaluated = set(summary[summary["task"] == "sustained_turn"]["controller"].unique())
    for controller in CONTROLLER_LABELS:
        if controller not in evaluated:
            continue
        sub = summary[(summary["task"] == "sustained_turn") & (summary["controller"] == controller)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Controller": CONTROLLER_LABELS[controller],
                "Completed orbits": _fmt_mean_ci(
                    row.get("completed_orbits_mean"), row.get("completed_orbits_ci95_low"), row.get("completed_orbits_ci95_high")
                ),
                "Avg turn rate °/s": _fmt_mean_ci(
                    row.get("avg_turn_rate_deg_s_mean"), row.get("avg_turn_rate_deg_s_ci95_low"), row.get("avg_turn_rate_deg_s_ci95_high")
                ),
                "Avg turn radius m": _fmt_mean_ci(
                    row.get("avg_turn_radius_m_mean"), row.get("avg_turn_radius_m_ci95_low"), row.get("avg_turn_radius_m_ci95_high")
                ),
                "Radius std m": _fmt_mean_ci(
                    row.get("radius_std_m_mean"), row.get("radius_std_m_ci95_low"), row.get("radius_std_m_ci95_high")
                ),
                "Mean speed m/s": _fmt_mean_ci(
                    row.get("avg_speed_mps_mean"), row.get("avg_speed_mps_ci95_low"), row.get("avg_speed_mps_ci95_high")
                ),
                "Mean NZ g": _fmt_mean_ci(
                    row.get("mean_nz_g_mean"), row.get("mean_nz_g_ci95_low"), row.get("mean_nz_g_ci95_high")
                ),
                "Max NZ g": _fmt_mean_ci(
                    row.get("max_nz_g_mean"), row.get("max_nz_g_ci95_low"), row.get("max_nz_g_ci95_high")
                ),
                "Energy loss rate m/s²": _fmt_mean_ci(
                    row.get("energy_loss_rate_mps2_mean"), row.get("energy_loss_rate_mps2_ci95_low"), row.get("energy_loss_rate_mps2_ci95_high")
                ),
                "Mean aggressiveness": "N/A",
            }
        )
    return pd.DataFrame(rows)


def _fmt_value(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def _combine_sectioned_sheets(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    table_c: pd.DataFrame,
    label_a: str,
    label_b: str,
    label_c: str,
) -> pd.DataFrame:
    """Combine three tables with a Section label column, padding missing columns."""
    a = table_a.copy()
    b = table_b.copy()
    c = table_c.copy()
    a.insert(0, "Section", label_a)
    b.insert(0, "Section", label_b)
    c.insert(0, "Section", label_c)
    all_cols = ["Section"] + sorted((set(a.columns) | set(b.columns) | set(c.columns)) - {"Section"})
    for df in (a, b, c):
        for col in all_cols:
            if col not in df.columns:
                df[col] = ""
    return pd.concat([a[all_cols], b[all_cols], c[all_cols]], ignore_index=True)


def _build_pairwise_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise is None or pairwise.empty:
        return pd.DataFrame()
    rows = []
    for _, row in pairwise.iterrows():
        rows.append(
            {
                "Task": row["task"],
                "Metric": row["metric"],
                "Comparison": f"{row['lhs']} vs {row['rhs']}",
                "n_pairs": int(row["n_pairs"]),
                "Mean diff": row["mean_diff"],
                "CI95 low": row["ci95_low"],
                "CI95 high": row["ci95_high"],
                "p_raw": row["p_raw"],
                "p_adj_holm": row["p_adj_holm"],
                "Significant": row["significant"],
                "Effect size dz": row["effect_dz"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export Table 3")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    tables_dir = args.run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(args.run_dir)
    pairwise_path = args.run_dir / "tables" / "pairwise_statistics.csv"
    pairwise = pd.read_csv(pairwise_path) if pairwise_path.exists() else None

    table3a = _build_table3a(summary)
    table3b = _build_table3b(summary)
    table3c = _build_table3c(summary, pairwise)

    # Markdown (manual formatter to avoid optional tabulate dependency)
    md_lines = [
        "# Table 3: Flight-Control Comparison Summary\n",
        "## Table 3A: Multi-Waypoint Tracking\n",
        _df_to_markdown(table3a),
        "\n## Table 3B: Sustained Turn\n",
        _df_to_markdown(table3b),
        "\n## Table 3C: Overall Ranking\n",
        _df_to_markdown(table3c),
    ]
    if pairwise is not None and not pairwise.empty:
        pw_summary = _build_pairwise_summary(pairwise)
        md_lines += [
            "\n## Pairwise Statistical Tests\n",
            _df_to_markdown(pw_summary),
        ]
    (tables_dir / "table3_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # CSV
    table3a.to_csv(tables_dir / "table3a_multi_waypoint.csv", index=False)
    table3b.to_csv(tables_dir / "table3b_sustained_turn.csv", index=False)
    table3c.to_csv(tables_dir / "table3c_overall.csv", index=False)

    # Combined Table 3 CSV with a section column for convenience.
    table3a_combined = table3a.copy()
    table3a_combined.insert(0, "Section", "Table 3A: Multi-Waypoint Tracking")
    table3b_combined = table3b.copy()
    table3b_combined.insert(0, "Section", "Table 3B: Sustained Turn")
    table3c_combined = table3c.copy()
    table3c_combined.insert(0, "Section", "Table 3C: Overall Ranking")
    # Align columns by taking the union and filling missing values with empty strings.
    all_cols = ["Section"] + sorted(
        (
            set(table3a_combined.columns)
            | set(table3b_combined.columns)
            | set(table3c_combined.columns)
        )
        - {"Section"}
    )
    for df in (table3a_combined, table3b_combined, table3c_combined):
        for col in all_cols:
            if col not in df.columns:
                df[col] = ""
    table3_summary = pd.concat(
        [table3a_combined[all_cols], table3b_combined[all_cols], table3c_combined[all_cols]],
        ignore_index=True,
    )
    table3_summary.to_csv(tables_dir / "table3_summary.csv", index=False)

    # Excel with multiple sheets (include spec-mandated table3_mean / table3_std_ci / pairwise_tests).
    table3a_mean = _build_table3a_mean_only(summary)
    table3b_mean = _build_table3b_mean_only(summary)
    table3a_std_ci = _build_table3a_std_ci(summary)
    table3b_std_ci = _build_table3b_std_ci(summary)

    excel_path = tables_dir / "table3_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        table3a.to_excel(writer, sheet_name="multi_waypoint", index=False)
        table3b.to_excel(writer, sheet_name="sustained_turn", index=False)
        table3c.to_excel(writer, sheet_name="overall", index=False)
        if pairwise is not None and not pairwise.empty:
            pw_summary = _build_pairwise_summary(pairwise)
            pw_summary.to_excel(writer, sheet_name="pairwise_tests", index=False)
        summary.to_excel(writer, sheet_name="summary_mean_std_ci", index=False)

        # Mean-only combined sheet.
        table3_mean = _combine_sectioned_sheets(
            table3a_mean, table3b_mean, table3c, "Table 3A: Multi-Waypoint", "Table 3B: Sustained Turn", "Table 3C: Overall"
        )
        table3_mean.to_excel(writer, sheet_name="table3_mean", index=False)

        # Std + CI95 combined sheet.
        table3_std_ci = _combine_sectioned_sheets(
            table3a_std_ci, table3b_std_ci, table3c, "Table 3A: Multi-Waypoint", "Table 3B: Sustained Turn", "Table 3C: Overall"
        )
        table3_std_ci.to_excel(writer, sheet_name="table3_std_ci", index=False)

    # Summary report
    report = _build_summary_report(summary, pairwise, table3c)
    (tables_dir / "summary_report.md").write_text(report, encoding="utf-8")

    print(f"Exported Table 3 and summary report to {tables_dir}")


def _build_summary_report(summary: pd.DataFrame, pairwise: pd.DataFrame, ranking: pd.DataFrame) -> str:
    """Generate a short analytical report from aggregated results."""
    lines = [
        "# Flight-Control Comparison: Summary Report\n",
        "## 1. Scope and Methods\n",
        "- Controllers: PPO+PID-Hybrid, PPO-FixedPID, Enhanced PID, Baseline PID.\n",
        "- Tasks: Multi-waypoint tracking (5 waypoints) and sustained turn (90 s).\n",
        "- Backend: JSBSim F-16 with `strict_backend=true`.\n",
        "- Seeds are paired across controllers for fair comparison.\n",
        "- Statistical tests: Shapiro-Wilk → paired t-test or Wilcoxon, with Holm correction applied separately within each comparison family (system-level: PPO+PID vs PPO; low-level isolation: Enhanced PID vs Baseline PID).\n",
        "\n## 2. Multi-Waypoint Task (Primary Metric: Completed Waypoints)\n",
    ]

    mw = summary[summary["task"] == "multi_waypoint"].set_index("controller")
    for controller in ["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"]:
        if controller in mw.index:
            row = mw.loc[controller]
            mean = row.get("completed_waypoints_mean")
            std = row.get("completed_waypoints_std")
            lines.append(f"- {CONTROLLER_LABELS[controller]}: {mean:.2f} ± {std:.2f} waypoints\n")

    lines.append("\n## 3. Sustained Turn Task (Primary Metric: Completed Orbits)\n")
    st = summary[summary["task"] == "sustained_turn"].set_index("controller")
    for controller in ["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"]:
        if controller in st.index:
            row = st.loc[controller]
            mean = row.get("completed_orbits_mean")
            std = row.get("completed_orbits_std")
            lines.append(f"- {CONTROLLER_LABELS[controller]}: {mean:.2f} ± {std:.2f} orbits\n")

    lines.append("\n## 4. Pairwise Conclusions (PPO+PID vs Others)\n")
    if pairwise is not None and not pairwise.empty:
        primary_metrics = [("multi_waypoint", "completed_waypoints"), ("sustained_turn", "completed_orbits")]
        for task, metric in primary_metrics:
            sub = pairwise[(pairwise["task"] == task) & (pairwise["metric"] == metric)]
            if sub.empty:
                continue
            lines.append(f"\n### {task} – {metric}\n")
            for _, row in sub.iterrows():
                comparison = f"{row['lhs']} vs {row['rhs']}"
                sig = "significant" if row["significant"] else "not significant"
                direction = "higher" if row["mean_diff"] > 0 else "lower"
                lines.append(
                    f"- {comparison}: mean_diff={row['mean_diff']:.3f} "
                    f"(CI95 [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]), "
                    f"p_adj={row['p_adj_holm']:.4f}, {sig}; PPO+PID is {direction}.\n"
                )
    else:
        lines.append("- No pairwise statistics available.\n")

    lines.append("\n## 5. Answers to the Two Core Questions\n")

    # Question 1: system-level PPO+PID vs PPO-FixedPID.
    lines.append("### Q1: Does PPO+PID-Hybrid significantly outperform PPO-FixedPID at the system level?\n")
    q1_mw = pairwise[
        (pairwise["task"] == "multi_waypoint") &
        (pairwise["metric"] == "completed_waypoints") &
        (pairwise["comparison_family"] == "system_closed_loop")
    ] if pairwise is not None else pd.DataFrame()
    q1_st = pairwise[
        (pairwise["task"] == "sustained_turn") &
        (pairwise["metric"] == "completed_orbits") &
        (pairwise["comparison_family"] == "system_closed_loop")
    ] if pairwise is not None else pd.DataFrame()
    if not q1_mw.empty:
        row = q1_mw.iloc[0]
        sig = "significant" if row["significant"] else "not significant"
        direction = "higher" if row["mean_diff"] > 0 else "lower"
        lines.append(
            f"- Multi-waypoint: PPO+PID is {direction} by {abs(row['mean_diff']):.2f} waypoints "
            f"(p_adj={row['p_adj_holm']:.4f}, {sig}); thus PPO+PID is **not** significantly better.\n"
        )
    if not q1_st.empty:
        row = q1_st.iloc[0]
        sig = "significant" if row["significant"] else "not significant"
        direction = "higher" if row["mean_diff"] > 0 else "lower"
        lines.append(
            f"- Sustained-turn: PPO+PID is {direction} by {abs(row['mean_diff']):.3f} orbits "
            f"(p_adj={row['p_adj_holm']:.4f}, {sig}); thus PPO+PID is **significantly worse**.\n"
        )
    # Build a dynamic conclusion for Q1 based on actual data.
    q1_conclusions = []
    if not q1_mw.empty and not q1_st.empty:
        mw_sig = q1_mw.iloc[0]["significant"]
        st_sig = q1_st.iloc[0]["significant"]
        mw_better = q1_mw.iloc[0]["mean_diff"] > 0
        st_better = q1_st.iloc[0]["mean_diff"] > 0
        if mw_sig and st_sig and mw_better and st_better:
            q1_conclusions.append("- **Conclusion**: Yes. PPO+PID-Hybrid is significantly better than PPO-FixedPID on both tasks.")
        elif not mw_sig and not st_sig:
            q1_conclusions.append("- **Conclusion**: No. PPO+PID-Hybrid is not significantly different from PPO-FixedPID on either task.")
        elif (mw_sig and not mw_better) or (st_sig and not st_better):
            q1_conclusions.append("- **Conclusion**: No. PPO+PID-Hybrid is significantly worse on at least one task.")
        else:
            q1_conclusions.append("- **Conclusion**: Partially. PPO+PID-Hybrid shows significant improvement on one task but not the other.")
    elif not q1_mw.empty or not q1_st.empty:
        q1_conclusions.append("- **Conclusion**: Insufficient data to draw a complete conclusion (only one task has pairwise statistics).")
    else:
        q1_conclusions.append("- **Conclusion**: No pairwise statistics available for PPO+PID vs PPO-FixedPID.")
    lines.extend([c + "\n" for c in q1_conclusions])

    # Question 2: low-level Enhanced PID vs Baseline PID.
    lines.append("\n### Q2: Does Enhanced PID significantly outperform Baseline PID in low-level control isolation?\n")
    q2_mw = pairwise[
        (pairwise["task"] == "multi_waypoint") &
        (pairwise["metric"] == "completed_waypoints") &
        (pairwise["comparison_family"] == "low_level_isolation")
    ] if pairwise is not None else pd.DataFrame()
    q2_st = pairwise[
        (pairwise["task"] == "sustained_turn") &
        (pairwise["metric"] == "completed_orbits") &
        (pairwise["comparison_family"] == "low_level_isolation")
    ] if pairwise is not None else pd.DataFrame()
    if not q2_mw.empty:
        row = q2_mw.iloc[0]
        sig = "significant" if row["significant"] else "not significant"
        direction = "higher" if row["mean_diff"] > 0 else "lower"
        lines.append(
            f"- Multi-waypoint: Enhanced PID is {direction} by {abs(row['mean_diff']):.2f} waypoints "
            f"(p_adj={row['p_adj_holm']:.4f}, {sig}).\n"
        )
    if not q2_st.empty:
        row = q2_st.iloc[0]
        sig = "significant" if row["significant"] else "not significant"
        direction = "higher" if row["mean_diff"] > 0 else "lower"
        lines.append(
            f"- Sustained-turn: Enhanced PID is {direction} by {abs(row['mean_diff']):.3f} orbits "
            f"(p_adj={row['p_adj_holm']:.4f}, {sig}).\n"
        )
    # Dynamic conclusion for Q2
    q2_conclusions = []
    if not q2_mw.empty and not q2_st.empty:
        mw_sig = q2_mw.iloc[0]["significant"]
        st_sig = q2_st.iloc[0]["significant"]
        mw_better = q2_mw.iloc[0]["mean_diff"] > 0
        st_better = q2_st.iloc[0]["mean_diff"] > 0
        if mw_sig and st_sig and mw_better and st_better:
            q2_conclusions.append("- **Conclusion**: Yes. Enhanced PID is significantly better than Baseline PID on both tasks.")
        elif not mw_sig and not st_sig:
            q2_conclusions.append("- **Conclusion**: No. Enhanced PID is not significantly different from Baseline PID on either task.")
        elif (mw_sig and not mw_better) or (st_sig and not st_better):
            q2_conclusions.append("- **Conclusion**: No. Enhanced PID is significantly worse on at least one task.")
        else:
            q2_conclusions.append("- **Conclusion**: Partially. Enhanced PID shows significant improvement on one task but not the other.")
    elif not q2_mw.empty or not q2_st.empty:
        q2_conclusions.append("- **Conclusion**: Insufficient data to draw a complete conclusion (only one task has pairwise statistics).")
    else:
        q2_conclusions.append("- **Conclusion**: No pairwise statistics available for Enhanced PID vs Baseline PID.")
    lines.extend([c + "\n" for c in q2_conclusions])

    lines.append("\n## 6. Overall Ranking\n")
    for _, row in ranking.iterrows():
        lines.append(
            f"- {row['Controller']}: overall rank {row['Overall rank']} "
            f"(MW rank {row['Multi-waypoint rank']}, ST rank {row['Sustained-turn rank']}) — {row['Recommendation']}\n"
        )

    lines.append("\n## 7. Scope Honesty and Limitations\n")
    lines.append("- Cross scenarios are explicitly scoped out of this deliverable.\n")
    lines.append("- GainScheduled PID is implemented as an optional controller but is not part of the main four-controller comparison.\n")
    lines.append("- All physical limits follow the current actuator mapping (7 g / 1.5 rad/s).\n")
    lines.append("- If PPO+PID does not outperform the baselines, the negative result is reported as-is.\n")

    return "".join(lines)


if __name__ == "__main__":
    main()
