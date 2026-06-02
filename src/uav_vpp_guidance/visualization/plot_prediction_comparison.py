"""
Plotting script for prediction comparison results.

Generates comparison figures for No-Prediction, CV-Prediction, and CA-Prediction.
Supports per-scenario breakdown side-by-side bar charts.

Usage:
    python -m uav_vpp_guidance.visualization.plot_prediction_comparison \
        --metrics outputs/tables/prediction_comparison/simple/prediction_metrics.csv \
        --metrics-dir outputs/tables/prediction_comparison/simple \
        --trajectories outputs/trajectories/prediction_comparison/simple \
        --output outputs/figures/prediction_comparison/simple
"""

import argparse
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics_csv(path: str) -> list:
    """Load aggregated metrics from CSV."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_metrics_json(path: str) -> list:
    """Load metrics JSON with per-scenario data."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory(path: str) -> list:
    """Load a trajectory CSV."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = np.nan
                elif v.lower() in ("true", "false"):
                    parsed[k] = v.lower() == "true"
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
        return rows


def find_first_trajectory(traj_root: str, method: str = "") -> str:
    """Find the first trajectory CSV under traj_root for a given method."""
    search_root = os.path.join(traj_root, method) if method else traj_root
    for root, dirs, files in os.walk(search_root):
        for fname in sorted(files):
            if fname.endswith(".csv"):
                return os.path.join(root, fname)
    return ""


def to_float(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# --- Color palette ---
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
METHOD_LABELS = {
    "no_prediction": "No Prediction",
    "cv_prediction": "CV Prediction",
    "ca_prediction": "CA Prediction",
}


def _get_methods(metrics):
    return [m.get("method", f"method_{i}") for i, m in enumerate(metrics)]


def _get_colors(n):
    return COLOR_PALETTE[:n]


# --- Overall comparison plots ---

def plot_success_rate_comparison(metrics: list, output_dir: str):
    if not metrics:
        return
    methods = _get_methods(metrics)
    success_rates = [to_float(m.get("success_rate", 0.0)) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = _get_colors(len(methods))
    bars = ax.bar([METHOD_LABELS.get(m, m) for m in methods], success_rates, color=colors, width=0.5)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate Comparison")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.2%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "comparison_success_rate.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_score_win_rate_comparison(metrics: list, output_dir: str):
    if not metrics:
        return
    methods = _get_methods(metrics)
    score_wins = [to_float(m.get("score_win_rate", 0.0)) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = _get_colors(len(methods))
    bars = ax.bar([METHOD_LABELS.get(m, m) for m in methods], score_wins, color=colors, width=0.5)
    ax.set_ylabel("Score Win Rate")
    ax.set_title("Score Win Rate Comparison")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.2%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "comparison_score_win_rate.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_final_range_comparison(metrics: list, output_dir: str):
    if not metrics:
        return
    methods = _get_methods(metrics)
    ranges = [to_float(m.get("mean_final_range_m", 0.0)) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = _get_colors(len(methods))
    bars = ax.bar([METHOD_LABELS.get(m, m) for m in methods], ranges, color=colors, width=0.5)
    ax.set_ylabel("Mean Final Range (m)")
    ax.set_title("Mean Final Range Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "comparison_final_range.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_final_ata_comparison(metrics: list, output_dir: str):
    if not metrics:
        return
    methods = _get_methods(metrics)
    atas = [to_float(m.get("mean_final_ata_deg", 0.0)) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = _get_colors(len(methods))
    bars = ax.bar([METHOD_LABELS.get(m, m) for m in methods], atas, color=colors, width=0.5)
    ax.set_ylabel("Mean Final ATA (deg)")
    ax.set_title("Mean Final ATA Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "comparison_final_ata.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_termination_distribution(metrics: list, output_dir: str):
    if not metrics:
        return
    methods = _get_methods(metrics)
    success = [to_float(m.get("success_rate", 0.0)) for m in metrics]
    crash = [to_float(m.get("crash_rate", 0.0)) for m in metrics]
    timeout = [to_float(m.get("timeout_rate", 0.0)) for m in metrics]
    oob = [to_float(m.get("out_of_bounds_rate", 0.0)) for m in metrics]

    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    ax.bar(x - 1.5 * width, success, width, label="Success", color="#2ca02c")
    ax.bar(x - 0.5 * width, crash, width, label="Crash", color="#d62728")
    ax.bar(x + 0.5 * width, timeout, width, label="Timeout", color="#ff7f0e")
    ax.bar(x + 1.5 * width, oob, width, label="OOB", color="#9467bd")
    ax.set_ylabel("Rate")
    ax.set_title("Termination Reason Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "termination_distribution_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# --- Per-scenario plots ---

def _collect_scenario_data(metrics_json, metric_key):
    """
    Build a dict: scenario_name -> {method_name -> value} for a given metric key.
    Returns (scenarios, methods, data_dict).
    """
    scenario_method_values = {}
    methods = set()
    for method_entry in metrics_json:
        method_name = method_entry.get("method", "unknown")
        methods.add(method_name)
        per_scenario = method_entry.get("per_scenario", {})
        for sc_name, sc_metrics in per_scenario.items():
            scenario_method_values.setdefault(sc_name, {})[method_name] = to_float(sc_metrics.get(metric_key, np.nan))
    all_scenarios = sorted(scenario_method_values.keys())
    all_methods = sorted(methods)
    return all_scenarios, all_methods, scenario_method_values


def plot_scenario_metric_bar(metrics_json: list, metric_key: str, ylabel: str, title: str,
                             output_dir: str, filename: str):
    """Side-by-side bar chart for a single metric across scenarios and methods."""
    if not metrics_json:
        return
    scenarios, methods, data = _collect_scenario_data(metrics_json, metric_key)
    if not scenarios:
        print(f"No scenario data for {metric_key}, skipping.")
        return

    n_scenarios = len(scenarios)
    n_methods = len(methods)
    x = np.arange(n_scenarios)
    width = 0.8 / n_methods
    colors = _get_colors(n_methods)

    width = max(2 + n_scenarios * 1.2, 5)
    fig, ax = plt.subplots(figsize=(width, 5))
    for i, method in enumerate(methods):
        values = [data[sc].get(method, np.nan) for sc in scenarios]
        offset = (i - (n_methods - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=METHOD_LABELS.get(method, method), color=colors[i])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if metric_key in ("success_rate", "score_win_rate"):
        ax.set_ylim(0, 1.1)
    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_scenario_comparisons(metrics_json: list, output_dir: str):
    """Generate all per-scenario comparison plots."""
    if not metrics_json:
        return
    plot_scenario_metric_bar(
        metrics_json, "success_rate", "Success Rate", "Success Rate by Scenario",
        output_dir, "scenario_success_rate.png"
    )
    plot_scenario_metric_bar(
        metrics_json, "score_win_rate", "Score Win Rate", "Score Win Rate by Scenario",
        output_dir, "scenario_score_win_rate.png"
    )
    plot_scenario_metric_bar(
        metrics_json, "mean_final_range_m", "Mean Final Range (m)", "Final Range by Scenario",
        output_dir, "scenario_final_range.png"
    )
    plot_scenario_metric_bar(
        metrics_json, "mean_final_ata_deg", "Mean Final ATA (deg)", "Final ATA by Scenario",
        output_dir, "scenario_final_ata.png"
    )
    plot_scenario_metric_bar(
        metrics_json, "mean_min_range_m", "Mean Min Range (m)", "Minimum Range by Scenario",
        output_dir, "scenario_min_range.png"
    )
    plot_scenario_metric_bar(
        metrics_json, "mean_prediction_error_m", "Prediction Error (m)", "Prediction Error by Scenario",
        output_dir, "scenario_prediction_error.png"
    )


# --- Trajectory-based plots ---

def plot_prediction_error_distribution(traj_root: str, output_dir: str):
    """Plot prediction error distribution from trajectories."""
    errors = []
    for root, dirs, files in os.walk(traj_root):
        for fname in sorted(files):
            if fname.endswith(".csv"):
                traj = load_trajectory(os.path.join(root, fname))
                for row in traj:
                    e = row.get("prediction_error_m", np.nan)
                    if np.isfinite(e):
                        errors.append(e)
    if not errors:
        print("No prediction error data found, skipping error distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(np.median(errors), color="darkred", linestyle="--", linewidth=2, label=f"Median: {np.median(errors):.1f} m")
    ax.axvline(np.percentile(errors, 90), color="darkorange", linestyle="--", linewidth=2, label=f"P90: {np.percentile(errors, 90):.1f} m")
    ax.set_xlabel("Prediction Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "prediction_error_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_2d_trajectory_comparison(traj_root: str, output_dir: str):
    """Plot 2D trajectories for the first episode of each method."""
    methods = ["no_prediction", "cv_prediction", "ca_prediction"]
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"no_prediction": "#1f77b4", "cv_prediction": "#ff7f0e", "ca_prediction": "#2ca02c"}
    labels = {"no_prediction": "No Prediction", "cv_prediction": "CV Prediction", "ca_prediction": "CA Prediction"}

    for method in methods:
        traj_path = find_first_trajectory(traj_root, method)
        if not traj_path:
            continue
        traj = load_trajectory(traj_path)
        if not traj:
            continue
        ego_x = [r["ego_x"] for r in traj]
        ego_y = [r["ego_y"] for r in traj]
        target_x = [r["target_x"] for r in traj]
        target_y = [r["target_y"] for r in traj]
        pred_x = [r.get("predicted_target_x", np.nan) for r in traj]
        pred_y = [r.get("predicted_target_y", np.nan) for r in traj]

        ax.plot(ego_x, ego_y, color=colors.get(method, "gray"), linewidth=1.5, linestyle="-",
                label=f"Ego ({labels.get(method, method)})")
        ax.plot(target_x, target_y, color=colors.get(method, "gray"), linewidth=1.0, linestyle="--", alpha=0.5)
        if not all(np.isnan(pred_x)):
            ax.plot(pred_x, pred_y, color=colors.get(method, "gray"), linewidth=1.0, linestyle=":", alpha=0.5)
        ax.scatter([ego_x[0]], [ego_y[0]], color=colors.get(method, "gray"), marker="o", s=40, zorder=5)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("2D Trajectory Comparison")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "trajectory_2d_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_range_ata_comparison(traj_root: str, output_dir: str):
    """Plot range and ATA over time for the first episode of each method."""
    methods = ["no_prediction", "cv_prediction", "ca_prediction"]
    colors = {"no_prediction": "#1f77b4", "cv_prediction": "#ff7f0e", "ca_prediction": "#2ca02c"}
    labels = {"no_prediction": "No Prediction", "cv_prediction": "CV Prediction", "ca_prediction": "CA Prediction"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for method in methods:
        traj_path = find_first_trajectory(traj_root, method)
        if not traj_path:
            continue
        traj = load_trajectory(traj_path)
        if not traj:
            continue
        time = [r["time"] for r in traj]
        range_m = [r["range_m"] for r in traj]
        ata_deg = [r["ata_deg"] for r in traj]

        axes[0].plot(time, range_m, color=colors.get(method, "gray"), linewidth=1.5, label=labels.get(method, method))
        axes[1].plot(time, ata_deg, color=colors.get(method, "gray"), linewidth=1.5, label=labels.get(method, method))

    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("Range over Time")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_ylabel("ATA (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("ATA over Time")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "range_ata_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot prediction comparison results")
    parser.add_argument("--metrics", type=str, required=True, help="Path to prediction_metrics.csv")
    parser.add_argument("--metrics-json", type=str, default="", help="Path to prediction_metrics.json (for per-scenario data)")
    parser.add_argument("--metrics-dir", type=str, default="", help="Directory containing per-scenario CSVs")
    parser.add_argument("--trajectories", type=str, default="", help="Path to trajectories root dir")
    parser.add_argument("--output", type=str, required=True, help="Output figures directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load overall metrics
    metrics = load_metrics_csv(args.metrics)
    print(f"Loaded {len(metrics)} method metrics from CSV")

    # Load JSON for per-scenario data
    metrics_json = []
    if args.metrics_json:
        metrics_json = load_metrics_json(args.metrics_json)
    elif args.metrics:
        # Try to infer JSON path
        inferred_json = args.metrics.replace(".csv", ".json")
        if os.path.exists(inferred_json):
            metrics_json = load_metrics_json(inferred_json)
            print(f"Loaded JSON metrics from {inferred_json}")

    # Aggregate comparison plots
    plot_success_rate_comparison(metrics, args.output)
    plot_score_win_rate_comparison(metrics, args.output)
    plot_final_range_comparison(metrics, args.output)
    plot_final_ata_comparison(metrics, args.output)
    plot_termination_distribution(metrics, args.output)

    # Per-scenario comparison plots
    if metrics_json:
        print("Generating per-scenario comparison plots...")
        plot_all_scenario_comparisons(metrics_json, args.output)
    else:
        print("No JSON metrics loaded, skipping per-scenario plots.")

    # Trajectory-based plots
    if args.trajectories and os.path.isdir(args.trajectories):
        plot_prediction_error_distribution(args.trajectories, args.output)
        plot_2d_trajectory_comparison(args.trajectories, args.output)
        plot_range_ata_comparison(args.trajectories, args.output)
    else:
        print("No trajectories directory provided or not found.")

    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
