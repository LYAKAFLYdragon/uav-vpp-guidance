#!/usr/bin/env python3
"""
Stage 6F.6 GRU-vs-LSTM Mechanism Audit.

Deep-dive analysis into why GRU outperforms LSTM in weaving_headon scenario.
Reads trajectory CSVs and episode-level metrics.

Usage:
    python scripts/analyze_gru_lstm_mechanism.py \
        --input outputs/tables/stage6f5_maneuvering_target \
        --scenario weaving_headon \
        --output outputs/tables/stage6f5_maneuvering_target/gru_lstm_mechanism
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

METRICS_SCHEMA_VERSION = "6f.2"
EXPERIMENT_SUITE_VERSION = "6f.6"

EXPECTED_FIELDS = {
    "episode": [
        "mean_env_prediction_error_m",
        "mean_offline_aligned_error_m",
        "mean_virtual_point_shift_m",
        "final_range_m",
        "final_ata_deg",
        "length",
        "return",
        "is_success",
        "reason",
    ],
    "trajectory": [
        "prediction_error_m",
        "virtual_point_shift_m",
        "nz_cmd",
        "roll_rate_cmd",
        "throttle_cmd",
        "range_m",
        "ata_deg",
        "ego_z",
    ],
}


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


def build_episode_df(raw_root: Path, seeds: list, scenario: str) -> pd.DataFrame:
    rows = []
    for ts in seeds:
        data = load_raw_prediction_metrics(raw_root, ts)
        for m in data:
            method = m.get("method_name", m.get("method", "unknown"))
            if method not in ("lstm_frozen", "gru_frozen"):
                continue
            for ep in m.get("raw_episodes", []):
                if ep.get("scenario") != scenario:
                    continue
                rows.append({
                    "method": method,
                    "training_seed": ts,
                    "evaluation_seed": ep.get("evaluation_seed", ep.get("seed", "unknown")),
                    "episode": ep.get("episode", ep.get("episode_seed", 0)),
                    "return": float(ep.get("return", np.nan)) if ep.get("return") is not None else np.nan,
                    "is_success": bool(ep.get("is_success", False)),
                    "reason": ep.get("reason", "unknown"),
                    "length": int(ep.get("length", 0)),
                    "mean_env_prediction_error_m": float(ep.get("mean_env_prediction_error_m", np.nan)) if ep.get("mean_env_prediction_error_m") is not None else np.nan,
                    "mean_offline_aligned_error_m": float(ep.get("mean_offline_aligned_error_m", np.nan)) if ep.get("mean_offline_aligned_error_m") is not None else np.nan,
                    "mean_virtual_point_shift_m": float(ep.get("mean_virtual_point_shift_m", np.nan)) if ep.get("mean_virtual_point_shift_m") is not None else np.nan,
                    "final_range_m": float(ep.get("final_range_m", np.nan)) if ep.get("final_range_m") is not None else np.nan,
                    "final_ata_deg": float(ep.get("final_ata_deg", np.nan)) if ep.get("final_ata_deg") is not None else np.nan,
                    "min_range_m": float(ep.get("min_range_m", np.nan)) if ep.get("min_range_m") is not None else np.nan,
                    "min_ata_deg": float(ep.get("min_ata_deg", np.nan)) if ep.get("min_ata_deg") is not None else np.nan,
                })
    return pd.DataFrame(rows)


def load_trajectory(raw_root: Path, training_seed: int, method: str, eval_seed: int, episode: int) -> pd.DataFrame:
    traj_dir = raw_root / f"train_seed{training_seed}" / "trajectories" / method
    candidates = [
        traj_dir / f"seed{eval_seed}_ep{episode}.csv",
        traj_dir / f"seed{eval_seed}_episode{episode}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return pd.read_csv(cand)
    return pd.DataFrame()


def check_missing_fields(episode_df: pd.DataFrame, trajectory_dfs: list) -> dict:
    missing = {}
    for field in EXPECTED_FIELDS["episode"]:
        if field not in episode_df.columns:
            missing.setdefault("episode", []).append(field)
    if trajectory_dfs:
        sample = trajectory_dfs[0]
        for field in EXPECTED_FIELDS["trajectory"]:
            if field not in sample.columns:
                missing.setdefault("trajectory", []).append(field)
    return missing


def analyze_prediction_error_timeseries(episode_df: pd.DataFrame, raw_root: Path, scenario: str) -> pd.DataFrame:
    rows = []
    for _, ep in episode_df.iterrows():
        traj = load_trajectory(raw_root, ep["training_seed"], ep["method"], ep["evaluation_seed"], ep["episode"])
        if traj.empty or "prediction_error_m" not in traj.columns:
            continue
        # Drop NaN prediction errors (warmup)
        valid = traj["prediction_error_m"].dropna()
        if len(valid) == 0:
            continue
        rows.append({
            "method": ep["method"],
            "training_seed": ep["training_seed"],
            "evaluation_seed": ep["evaluation_seed"],
            "episode": ep["episode"],
            "is_success": ep["is_success"],
            "mean_pred_error": valid.mean(),
            "max_pred_error": valid.max(),
            "final_pred_error": valid.iloc[-1],
            "early_pred_error": valid.iloc[:min(20, len(valid))].mean(),
            "late_pred_error": valid.iloc[-min(20, len(valid)):].mean(),
        })
    return pd.DataFrame(rows)


def analyze_vpp_shift_distribution(episode_df: pd.DataFrame, raw_root: Path, scenario: str) -> pd.DataFrame:
    rows = []
    for _, ep in episode_df.iterrows():
        traj = load_trajectory(raw_root, ep["training_seed"], ep["method"], ep["evaluation_seed"], ep["episode"])
        if traj.empty or "virtual_point_shift_m" not in traj.columns:
            continue
        shifts = traj["virtual_point_shift_m"].dropna()
        if len(shifts) == 0:
            continue
        # Decompose into longitudinal/lateral/vertical if available
        # Fallback: use shift magnitude statistics
        rows.append({
            "method": ep["method"],
            "training_seed": ep["training_seed"],
            "evaluation_seed": ep["evaluation_seed"],
            "episode": ep["episode"],
            "is_success": ep["is_success"],
            "mean_shift_m": shifts.mean(),
            "std_shift_m": shifts.std(ddof=1),
            "max_shift_m": shifts.max(),
            "final_shift_m": shifts.iloc[-1],
        })
    return pd.DataFrame(rows)


def analyze_action_distribution(episode_df: pd.DataFrame, raw_root: Path, scenario: str) -> pd.DataFrame:
    rows = []
    for _, ep in episode_df.iterrows():
        traj = load_trajectory(raw_root, ep["training_seed"], ep["method"], ep["evaluation_seed"], ep["episode"])
        if traj.empty:
            continue
        row = {
            "method": ep["method"],
            "training_seed": ep["training_seed"],
            "evaluation_seed": ep["evaluation_seed"],
            "episode": ep["episode"],
            "is_success": ep["is_success"],
        }
        for cmd in ["nz_cmd", "roll_rate_cmd", "throttle_cmd"]:
            if cmd in traj.columns:
                vals = traj[cmd].dropna()
                row[f"mean_{cmd}"] = vals.mean()
                row[f"std_{cmd}"] = vals.std(ddof=1)
                row[f"max_abs_{cmd}"] = vals.abs().max()
                row[f"saturation_rate_{cmd}"] = (vals.abs() >= (vals.abs().max() * 0.95)).mean()
            else:
                row[f"mean_{cmd}"] = np.nan
                row[f"std_{cmd}"] = np.nan
                row[f"max_abs_{cmd}"] = np.nan
                row[f"saturation_rate_{cmd}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_terminal_geometry(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["lstm_frozen", "gru_frozen"]:
        mdf = episode_df[episode_df["method"] == method]
        for outcome, label in [(True, "success"), (False, "failure")]:
            odf = mdf[mdf["is_success"] == outcome]
            if len(odf) == 0:
                continue
            rows.append({
                "method": method,
                "outcome": label,
                "n_episodes": len(odf),
                "mean_final_range_m": odf["final_range_m"].mean(),
                "std_final_range_m": odf["final_range_m"].std(ddof=1),
                "mean_final_ata_deg": odf["final_ata_deg"].mean(),
                "std_final_ata_deg": odf["final_ata_deg"].std(ddof=1),
                "mean_min_range_m": odf["min_range_m"].mean(),
                "mean_min_ata_deg": odf["min_ata_deg"].mean(),
            })
    return pd.DataFrame(rows)


def analyze_success_failure_contrast(episode_df: pd.DataFrame) -> pd.DataFrame:
    contrast = []
    for method in ["lstm_frozen", "gru_frozen"]:
        mdf = episode_df[episode_df["method"] == method]
        succ = mdf[mdf["is_success"] == True]
        fail = mdf[mdf["is_success"] == False]
        if len(succ) == 0 or len(fail) == 0:
            continue
        contrast.append({
            "method": method,
            "n_success": len(succ),
            "n_failure": len(fail),
            "success_mean_return": succ["return"].mean(),
            "failure_mean_return": fail["return"].mean(),
            "success_mean_env_error": succ["mean_env_prediction_error_m"].mean(),
            "failure_mean_env_error": fail["mean_env_prediction_error_m"].mean(),
            "success_mean_vpp_shift": succ["mean_virtual_point_shift_m"].mean(),
            "failure_mean_vpp_shift": fail["mean_virtual_point_shift_m"].mean(),
            "success_mean_length": succ["length"].mean(),
            "failure_mean_length": fail["length"].mean(),
        })
    return pd.DataFrame(contrast)


def render_mechanism_md(
    episode_df: pd.DataFrame,
    pred_error_df: pd.DataFrame,
    vpp_df: pd.DataFrame,
    action_df: pd.DataFrame,
    terminal_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    missing_fields: dict,
) -> str:
    lines = []
    lines.append("# GRU vs LSTM Mechanism Audit: weaving_headon")
    lines.append("")
    lines.append(f"**Metrics Schema Version**: {METRICS_SCHEMA_VERSION}")
    lines.append(f"**Experiment Suite Version**: {EXPERIMENT_SUITE_VERSION}")
    lines.append("")

    if missing_fields:
        lines.append("## Missing Fields Report")
        lines.append("")
        for category, fields in missing_fields.items():
            lines.append(f"**{category}**: {', '.join(fields)}")
        lines.append("")
        lines.append("### Telemetry Instrumentation Recommendations")
        lines.append("- Ensure trajectory CSVs include `prediction_error_m`, `virtual_point_shift_m`, `nz_cmd`, `roll_rate_cmd`, `throttle_cmd`.")
        lines.append("- Ensure episode JSONs include `mean_env_prediction_error_m`, `mean_offline_aligned_error_m`, `mean_virtual_point_shift_m`, `final_range_m`, `final_ata_deg`.")
        lines.append("")

    # Episode summary
    lines.append("## Episode Summary")
    lines.append("")
    lines.append("| Method | Episodes | Success Rate | Mean Return | Mean Env Error (m) | Mean VPP Shift (m) |")
    lines.append("|--------|---------:|-------------:|------------:|-------------------:|-------------------:|")
    for method in ["lstm_frozen", "gru_frozen"]:
        mdf = episode_df[episode_df["method"] == method]
        if len(mdf) == 0:
            continue
        lines.append(
            f"| {method} | {len(mdf)} | {mdf['is_success'].mean():.1%} | "
            f"{mdf['return'].mean():.1f} | {mdf['mean_env_prediction_error_m'].mean():.1f} | "
            f"{mdf['mean_virtual_point_shift_m'].mean():.1f} |"
        )
    lines.append("")

    # Prediction error
    if not pred_error_df.empty:
        lines.append("## Prediction Error Time Series Analysis")
        lines.append("")
        lines.append("| Method | Outcome | Mean Error | Max Error | Early Error | Late Error |")
        lines.append("|--------|---------|-----------:|----------:|------------:|-----------:|")
        for method in ["lstm_frozen", "gru_frozen"]:
            for outcome in [True, False]:
                odf = pred_error_df[(pred_error_df["method"] == method) & (pred_error_df["is_success"] == outcome)]
                if len(odf) == 0:
                    continue
                label = "success" if outcome else "failure"
                lines.append(
                    f"| {method} | {label} | {odf['mean_pred_error'].mean():.1f} | "
                    f"{odf['max_pred_error'].mean():.1f} | {odf['early_pred_error'].mean():.1f} | "
                    f"{odf['late_pred_error'].mean():.1f} |"
                )
        lines.append("")

    # VPP shift
    if not vpp_df.empty:
        lines.append("## VPP Shift Distribution")
        lines.append("")
        lines.append("| Method | Outcome | Mean Shift (m) | Std Shift | Max Shift | Final Shift |")
        lines.append("|--------|---------|---------------:|----------:|----------:|------------:|")
        for method in ["lstm_frozen", "gru_frozen"]:
            for outcome in [True, False]:
                odf = vpp_df[(vpp_df["method"] == method) & (vpp_df["is_success"] == outcome)]
                if len(odf) == 0:
                    continue
                label = "success" if outcome else "failure"
                lines.append(
                    f"| {method} | {label} | {odf['mean_shift_m'].mean():.1f} | "
                    f"{odf['std_shift_m'].mean():.1f} | {odf['max_shift_m'].mean():.1f} | "
                    f"{odf['final_shift_m'].mean():.1f} |"
                )
        lines.append("")

    # Action distribution
    if not action_df.empty:
        lines.append("## Action Distribution")
        lines.append("")
        lines.append("| Method | Outcome | Mean Nz | Std Nz | Sat Nz | Mean Roll | Std Roll | Sat Roll | Mean Thr | Std Thr |")
        lines.append("|--------|---------|--------:|-------:|-------:|----------:|---------:|---------:|---------:|--------:|")
        for method in ["lstm_frozen", "gru_frozen"]:
            for outcome in [True, False]:
                odf = action_df[(action_df["method"] == method) & (action_df["is_success"] == outcome)]
                if len(odf) == 0:
                    continue
                label = "success" if outcome else "failure"
                lines.append(
                    f"| {method} | {label} | {odf['mean_nz_cmd'].mean():.2f} | "
                    f"{odf['std_nz_cmd'].mean():.2f} | {odf['saturation_rate_nz_cmd'].mean():.1%} | "
                    f"{odf['mean_roll_rate_cmd'].mean():.2f} | {odf['std_roll_rate_cmd'].mean():.2f} | "
                    f"{odf['saturation_rate_roll_rate_cmd'].mean():.1%} | "
                    f"{odf['mean_throttle_cmd'].mean():.2f} | {odf['std_throttle_cmd'].mean():.2f} |"
                )
        lines.append("")

    # Terminal geometry
    if not terminal_df.empty:
        lines.append("## Terminal Geometry")
        lines.append("")
        lines.append("| Method | Outcome | N | Mean Final Range (m) | Std | Mean Final ATA (deg) | Std | Mean Min Range (m) |")
        lines.append("|--------|---------|---|---------------------:|----:|---------------------:|----:|-------------------:|")
        for _, row in terminal_df.iterrows():
            lines.append(
                f"| {row['method']} | {row['outcome']} | {row['n_episodes']} | "
                f"{row['mean_final_range_m']:.1f} | {row['std_final_range_m']:.1f} | "
                f"{row['mean_final_ata_deg']:.1f} | {row['std_final_ata_deg']:.1f} | "
                f"{row['mean_min_range_m']:.1f} |"
            )
        lines.append("")

    # Success vs failure contrast
    if not contrast_df.empty:
        lines.append("## Success vs Failure Contrast")
        lines.append("")
        lines.append("| Method | N Succ | N Fail | Succ Return | Fail Return | Succ Env Err | Fail Env Err | Succ VPP Shift | Fail VPP Shift | Succ Length | Fail Length |")
        lines.append("|--------|--------|--------|------------:|------------:|-------------:|-------------:|---------------:|---------------:|------------:|------------:|")
        for _, row in contrast_df.iterrows():
            lines.append(
                f"| {row['method']} | {row['n_success']} | {row['n_failure']} | "
                f"{row['success_mean_return']:.1f} | {row['failure_mean_return']:.1f} | "
                f"{row['success_mean_env_error']:.1f} | {row['failure_mean_env_error']:.1f} | "
                f"{row['success_mean_vpp_shift']:.1f} | {row['failure_mean_vpp_shift']:.1f} | "
                f"{row['success_mean_length']:.0f} | {row['failure_mean_length']:.0f} |"
            )
        lines.append("")

    # Mechanism interpretation
    lines.append("## Mechanism Interpretation")
    lines.append("")
    if not pred_error_df.empty:
        lstm_err = pred_error_df[pred_error_df["method"] == "lstm_frozen"]["mean_pred_error"].mean()
        gru_err = pred_error_df[pred_error_df["method"] == "gru_frozen"]["mean_pred_error"].mean()
        lines.append(f"- **Prediction error**: GRU mean={gru_err:.1f}m, LSTM mean={lstm_err:.1f}m")
    if not vpp_df.empty:
        lstm_shift = vpp_df[vpp_df["method"] == "lstm_frozen"]["mean_shift_m"].mean()
        gru_shift = vpp_df[vpp_df["method"] == "gru_frozen"]["mean_shift_m"].mean()
        lines.append(f"- **VPP shift**: GRU mean={gru_shift:.1f}m, LSTM mean={lstm_shift:.1f}m")
    if not action_df.empty:
        lstm_std_nz = action_df[action_df["method"] == "lstm_frozen"]["std_nz_cmd"].mean()
        gru_std_nz = action_df[action_df["method"] == "gru_frozen"]["std_nz_cmd"].mean()
        lines.append(f"- **Action smoothness (Nz std)**: GRU={gru_std_nz:.2f}, LSTM={lstm_std_nz:.2f}")
    lines.append("")
    lines.append("**Hypothesis**: GRU's gating mechanism allows faster adaptation to the sinusoidal target maneuver, resulting in lower prediction error and more conservative VPP shifts that maintain intercept geometry. LSTM's more rigid cell state may lag behind rapid heading changes, leading to larger late-phase errors and higher saturation rates.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="GRU vs LSTM Mechanism Audit")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="weaving_headon")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = discover_training_seeds(input_root)
    if not seeds:
        print(f"ERROR: No training seeds found in {input_root}")
        sys.exit(1)
    print(f"Discovered seeds: {seeds}")

    episode_df = build_episode_df(input_root, seeds, args.scenario)
    print(f"Loaded {len(episode_df)} episodes for {args.scenario}")

    # Check missing fields
    sample_trajs = []
    for _, ep in episode_df.head(5).iterrows():
        traj = load_trajectory(input_root, ep["training_seed"], ep["method"], ep["evaluation_seed"], ep["episode"])
        if not traj.empty:
            sample_trajs.append(traj)

    missing_fields = check_missing_fields(episode_df, sample_trajs)
    if missing_fields:
        print(f"WARNING: Missing fields detected: {missing_fields}")
        with open(output_dir / "missing_fields.md", "w", encoding="utf-8") as f:
            f.write("# Missing Fields Report\n\n")
            for category, fields in missing_fields.items():
                f.write(f"**{category}**: {', '.join(fields)}\n")
            f.write("\n## Recommendations\n")
            f.write("- Ensure trajectory CSVs include all required columns.\n")
            f.write("- Ensure episode JSONs include all required fields.\n")

    # Run analyses
    pred_error_df = analyze_prediction_error_timeseries(episode_df, input_root, args.scenario)
    vpp_df = analyze_vpp_shift_distribution(episode_df, input_root, args.scenario)
    action_df = analyze_action_distribution(episode_df, input_root, args.scenario)
    terminal_df = analyze_terminal_geometry(episode_df)
    contrast_df = analyze_success_failure_contrast(episode_df)

    # Save CSVs
    if not pred_error_df.empty:
        pred_error_df.to_csv(output_dir / "prediction_error_timeseries.csv", index=False, float_format="%.6f")
    if not vpp_df.empty:
        vpp_df.to_csv(output_dir / "vpp_shift_distribution.csv", index=False, float_format="%.6f")
    if not action_df.empty:
        action_df.to_csv(output_dir / "action_distribution.csv", index=False, float_format="%.6f")
    terminal_df.to_csv(output_dir / "terminal_geometry.csv", index=False, float_format="%.6f")
    contrast_df.to_csv(output_dir / "success_failure_contrast.csv", index=False, float_format="%.6f")

    # Save markdown report
    md = render_mechanism_md(episode_df, pred_error_df, vpp_df, action_df, terminal_df, contrast_df, missing_fields)
    with open(output_dir / "gru_lstm_mechanism.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Mechanism audit complete. Output: {output_dir}")
    print(f"  - gru_lstm_mechanism.md")
    if not pred_error_df.empty:
        print(f"  - prediction_error_timeseries.csv")
    if not vpp_df.empty:
        print(f"  - vpp_shift_distribution.csv")
    if not action_df.empty:
        print(f"  - action_distribution.csv")
    print(f"  - terminal_geometry.csv")
    print(f"  - success_failure_contrast.csv")


if __name__ == "__main__":
    main()
