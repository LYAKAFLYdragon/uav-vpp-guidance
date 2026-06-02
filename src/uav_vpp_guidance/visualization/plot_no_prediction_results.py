"""
Plotting script for No-Prediction VPP Baseline results.

Supports both SimplePointMass and JSBSim backends.

Usage:
    python -m uav_vpp_guidance.visualization.plot_no_prediction_results \
        --metrics outputs/tables/no_prediction_vpp/simple/scenario_metrics.csv \
        --trajectories outputs/trajectories/no_prediction_vpp/simple \
        --output outputs/figures/no_prediction_vpp/simple \
        --backend simple

    python -m uav_vpp_guidance.visualization.plot_no_prediction_results \
        --metrics outputs/tables/no_prediction_vpp/jsbsim/scenario_metrics.csv \
        --trajectories outputs/trajectories/no_prediction_vpp/jsbsim \
        --output outputs/figures/no_prediction_vpp/jsbsim \
        --backend jsbsim
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics_csv(path: str) -> list:
    """Load scenario metrics from CSV."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_scenario_bar_metrics(metrics: list, output_dir: str):
    """Plot 1: success_rate / score_win_rate bar chart."""
    if not metrics:
        return
    scenarios = [m.get("scenario", "") for m in metrics]
    success_rates = [float(m.get("success_rate", 0.0)) for m in metrics]
    score_win_rates = [float(m.get("score_win_rate", 0.0)) for m in metrics]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, success_rates, width, label="Success Rate", color="#2ca02c")
    ax.bar(x + width / 2, score_win_rates, width, label="Score Win Rate", color="#1f77b4")
    ax.set_ylabel("Rate")
    ax.set_title("Success Rate and Score Win Rate by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "bar_success_score_win.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scenario_failure_metrics(metrics: list, output_dir: str):
    """Plot 2: crash_rate / timeout_rate bar chart."""
    if not metrics:
        return
    scenarios = [m.get("scenario", "") for m in metrics]
    crash_rates = [float(m.get("crash_rate", 0.0)) for m in metrics]
    timeout_rates = [float(m.get("timeout_rate", 0.0)) for m in metrics]
    oob_rates = [float(m.get("out_of_bounds_rate", 0.0)) for m in metrics]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, crash_rates, width, label="Crash Rate", color="#d62728")
    ax.bar(x, timeout_rates, width, label="Timeout Rate", color="#ff7f0e")
    ax.bar(x + width, oob_rates, width, label="OOB Rate", color="#9467bd")
    ax.set_ylabel("Rate")
    ax.set_title("Failure Rates by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "bar_failure_rates.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_termination_distribution(metrics: list, output_dir: str):
    """Plot termination reason distribution across scenarios."""
    if not metrics:
        return
    scenarios = [m.get("scenario", "") for m in metrics]
    success = [float(m.get("success_rate", 0.0)) for m in metrics]
    crash = [float(m.get("crash_rate", 0.0)) for m in metrics]
    timeout = [float(m.get("timeout_rate", 0.0)) for m in metrics]
    oob = [float(m.get("out_of_bounds_rate", 0.0)) for m in metrics]

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, success, label="Success", color="#2ca02c")
    ax.bar(x, crash, bottom=success, label="Crash", color="#d62728")
    ax.bar(x, timeout, bottom=[s + c for s, c in zip(success, crash)], label="Timeout", color="#ff7f0e")
    ax.bar(x, oob, bottom=[s + c + t for s, c, t in zip(success, crash, timeout)], label="OOB", color="#9467bd")
    ax.set_ylabel("Rate")
    ax.set_title("Termination Reason Distribution by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "termination_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def find_first_trajectory(traj_root: str) -> str:
    """Find the first trajectory CSV under traj_root."""
    for root, dirs, files in os.walk(traj_root):
        for f in sorted(files):
            if f.endswith(".csv"):
                return os.path.join(root, f)
    return ""


def load_trajectory(path: str) -> list:
    """Load a trajectory CSV."""
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


def plot_2d_trajectory(traj: list, output_dir: str, label: str = "typical"):
    """Plot 3: 2D plane trajectory."""
    if not traj:
        return
    ego_x = [r["ego_x"] for r in traj]
    ego_y = [r["ego_y"] for r in traj]
    target_x = [r["target_x"] for r in traj]
    target_y = [r["target_y"] for r in traj]
    virtual_x = [r.get("virtual_x", np.nan) for r in traj]
    virtual_y = [r.get("virtual_y", np.nan) for r in traj]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ego_x, ego_y, "b-", label="Ego", linewidth=1.5)
    ax.plot(target_x, target_y, "r-", label="Target", linewidth=1.5)
    if not all(np.isnan(virtual_x)):
        ax.plot(virtual_x, virtual_y, "g--", label="Virtual Point", linewidth=1.0, alpha=0.6)
    ax.scatter([ego_x[0]], [ego_y[0]], c="blue", marker="o", s=60, zorder=5)
    ax.scatter([target_x[0]], [target_y[0]], c="red", marker="o", s=60, zorder=5)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"2D Plane Trajectory ({label})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"trajectory_2d_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_altitude(traj: list, output_dir: str, label: str = "typical"):
    """Plot altitude over time."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    ego_z = [r["ego_z"] for r in traj]
    target_z = [r["target_z"] for r in traj]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, ego_z, "b-", label="Ego Altitude", linewidth=1.5)
    ax.plot(time, target_z, "r-", label="Target Altitude", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Altitude over Time ({label})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"altitude_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_speed(traj: list, output_dir: str, label: str = "typical"):
    """Plot speed over time."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    ego_speed = [r.get("ego_speed", np.nan) for r in traj]
    target_speed = [r.get("target_speed", np.nan) for r in traj]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, ego_speed, "b-", label="Ego Speed", linewidth=1.5)
    ax.plot(time, target_speed, "r-", label="Target Speed", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Speed over Time ({label})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"speed_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_range_ata(traj: list, output_dir: str, label: str = "typical"):
    """Plot range and ATA over time."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    range_m = [r["range_m"] for r in traj]
    ata_deg = [r["ata_deg"] for r in traj]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = "#1f77b4"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Range (m)", color=color1)
    ax1.plot(time, range_m, color=color1, linewidth=1.5)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("ATA (deg)", color=color2)
    ax2.plot(time, ata_deg, color=color2, linewidth=1.5, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"range_ata_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scores(traj: list, output_dir: str, label: str = "typical"):
    """Plot ego/target scores over time."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    ego_scores = [r.get("ego_score", np.nan) for r in traj]
    target_scores = [r.get("target_score", np.nan) for r in traj]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, ego_scores, "b-", label="Ego Score", linewidth=1.5)
    ax.plot(time, target_scores, "r-", label="Target Score", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title(f"Scores over Time ({label})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"scores_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_commands(traj: list, output_dir: str, label: str = "typical"):
    """Plot guidance commands over time."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    nz = [r.get("nz_cmd", np.nan) for r in traj]
    roll_rate = [r.get("roll_rate_cmd", np.nan) for r in traj]
    throttle = [r.get("throttle_cmd", np.nan) for r in traj]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(time, nz, "b-", linewidth=1.5)
    axes[0].set_ylabel("Nz_cmd")
    axes[0].set_title("Normal Overload Command")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(time, roll_rate, "g-", linewidth=1.5)
    axes[1].set_ylabel("Roll Rate (rad/s)")
    axes[1].set_title("Roll Rate Command")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(time, throttle, "r-", linewidth=1.5)
    axes[2].set_ylabel("Throttle")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Throttle Command")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Command Profiles ({label})", y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"commands_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_actuators(traj: list, output_dir: str, label: str = "typical"):
    """Plot JSBSim actuator outputs (elevator, aileron, rudder, throttle_actual)."""
    if not traj:
        return
    time = [r["time"] for r in traj]
    elevator = [r.get("elevator_cmd", np.nan) for r in traj]
    aileron = [r.get("aileron_cmd", np.nan) for r in traj]
    rudder = [r.get("rudder_cmd", np.nan) for r in traj]
    throttle_actual = [r.get("throttle_actual", np.nan) for r in traj]

    # Skip if no actuator data present
    if all(np.isnan(elevator)) and all(np.isnan(aileron)):
        print("No actuator data found, skipping actuator plot.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(time, elevator, "b-", linewidth=1.5)
    axes[0].set_ylabel("Elevator")
    axes[0].set_title("Elevator Command")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(time, aileron, "g-", linewidth=1.5)
    axes[1].set_ylabel("Aileron")
    axes[1].set_title("Aileron Command")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(time, rudder, "m-", linewidth=1.5)
    axes[2].set_ylabel("Rudder")
    axes[2].set_title("Rudder Command")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].plot(time, throttle_actual, "r-", linewidth=1.5)
    axes[3].set_ylabel("Throttle")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Throttle Actual")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Actuator Outputs ({label})", y=1.01)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"actuators_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot No-Prediction VPP Baseline Results")
    parser.add_argument("--metrics", type=str, required=True, help="Path to scenario_metrics.csv")
    parser.add_argument("--trajectories", type=str, default="", help="Path to trajectories root dir")
    parser.add_argument("--output", type=str, required=True, help="Output figures directory")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Backend for plot labeling")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    metrics = load_metrics_csv(args.metrics)

    # Aggregate plots
    plot_scenario_bar_metrics(metrics, args.output)
    plot_scenario_failure_metrics(metrics, args.output)
    plot_termination_distribution(metrics, args.output)

    # Trajectory plots
    if args.trajectories and os.path.isdir(args.trajectories):
        traj_path = find_first_trajectory(args.trajectories)
        if traj_path:
            label = os.path.basename(os.path.dirname(os.path.dirname(traj_path)))
            traj = load_trajectory(traj_path)
            plot_2d_trajectory(traj, args.output, label=label)
            plot_range_ata(traj, args.output, label=label)
            plot_scores(traj, args.output, label=label)
            plot_commands(traj, args.output, label=label)
            plot_altitude(traj, args.output, label=label)
            plot_speed(traj, args.output, label=label)
            if args.backend == "jsbsim":
                plot_actuators(traj, args.output, label=label)
        else:
            print("No trajectory CSVs found.")
    else:
        print("No trajectories directory provided or not found.")

    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
