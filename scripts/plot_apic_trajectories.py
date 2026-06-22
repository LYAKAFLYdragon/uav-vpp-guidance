#!/usr/bin/env python3
"""
Plot representative 3D trajectories from the APIC-PID comparison runs.

Produces:
- multi_waypoint_trajectories_seed{N}.png
- sustained_turn_trajectories_seed{N}.png

Usage:
    python scripts/plot_apic_trajectories.py \
        --run-dir outputs/flight_control_compare/apic_pid_combined_200k \
        --seed 0 \
        --output-dir outputs/figures/apic_pid
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


CONTROLLERS = ["apic_pid", "enhanced_pid", "baseline_pid"]
LABELS = {
    "apic_pid": "APIC-PID",
    "enhanced_pid": "Enhanced PID",
    "baseline_pid": "Baseline PID",
}
COLORS = {
    "apic_pid": "#1f77b4",
    "enhanced_pid": "#ff7f0e",
    "baseline_pid": "#2ca02c",
}


def _load_episode(run_dir: Path, task: str, controller: str, seed: int, episode: int = 0):
    path = run_dir / "raw" / task / controller / f"seed_{seed:02d}" / f"episode_{episode:03d}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_3d(traj):
    own = np.array([p["own_pos_m"] for p in traj], dtype=float)
    tgt = np.array([p["target_pos_m"] for p in traj], dtype=float)
    return own, tgt


def _set_equal_3d(ax, positions):
    """Set equal aspect ratio for a 3D matplotlib axis."""
    xyz = np.vstack(positions)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    centers = (mins + maxs) / 2.0
    spans = (maxs - mins) / 2.0
    max_span = np.max(spans)
    if max_span <= 0:
        max_span = 1.0
    ax.set_xlim(centers[0] - max_span, centers[0] + max_span)
    ax.set_ylim(centers[1] - max_span, centers[1] + max_span)
    ax.set_zlim(centers[2] - max_span, centers[2] + max_span)


def plot_multi_waypoint(run_dir: Path, seed: int, output_dir: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    target_traj_plotted = False
    waypoints = None
    all_positions = []

    for controller in CONTROLLERS:
        ep = _load_episode(run_dir, "multi_waypoint", controller, seed)
        if ep is None or not ep.get("trajectory"):
            continue
        traj = ep["trajectory"]
        own, tgt = _extract_3d(traj)
        ax.plot(own[:, 0], own[:, 1], own[:, 2], color=COLORS[controller], label=LABELS[controller], linewidth=1.5)
        all_positions.append(own)
        if not target_traj_plotted:
            ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], "k--", alpha=0.5, linewidth=1.0, label="Target trajectory")
            all_positions.append(tgt)
            target_traj_plotted = True
        if waypoints is None and traj[0].get("waypoints"):
            waypoints = np.array([w["pos"] for w in traj[0]["waypoints"]], dtype=float)

    if waypoints is not None:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker="*", s=200, c="red", zorder=5, label="Waypoints")
        all_positions.append(waypoints)

    if all_positions:
        _set_equal_3d(ax, all_positions)

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Multi-Waypoint 3D Trajectories (seed {seed})")
    ax.legend(loc="best", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"multi_waypoint_trajectories_seed{seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sustained_turn(run_dir: Path, seed: int, output_dir: Path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    orbit_radius = 1200.0
    target_center = None
    all_positions = []

    for controller in CONTROLLERS:
        ep = _load_episode(run_dir, "sustained_turn", controller, seed)
        if ep is None or not ep.get("trajectory"):
            continue
        traj = ep["trajectory"]
        own, tgt = _extract_3d(traj)
        ax.plot(own[:, 0], own[:, 1], own[:, 2], color=COLORS[controller], label=LABELS[controller], linewidth=1.5)
        all_positions.append(own)
        if target_center is None:
            target_center = tgt[0]

    if target_center is not None:
        ax.scatter(target_center[0], target_center[1], target_center[2], marker="X", s=200, c="red", zorder=5, label="Orbit center")
        theta = np.linspace(0, 2 * np.pi, 200)
        circle_x = target_center[0] + orbit_radius * np.cos(theta)
        circle_y = target_center[1] + orbit_radius * np.sin(theta)
        circle_z = np.full_like(circle_x, target_center[2])
        ax.plot(circle_x, circle_y, circle_z, "k--", alpha=0.6, linewidth=1.0, label="Desired orbit (r=1200 m)")
        all_positions.append(np.column_stack([circle_x, circle_y, circle_z]))

    if all_positions:
        _set_equal_3d(ax, all_positions)

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Sustained Turn 3D Trajectories (seed {seed})")
    ax.legend(loc="best", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sustained_turn_trajectories_seed{seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot APIC-PID comparison 3D trajectories")
    parser.add_argument("--run-dir", type=str, required=True, help="Combined comparison run directory")
    parser.add_argument("--seed", type=int, default=0, help="Seed to visualize")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/apic_pid", help="Output figure directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    plot_multi_waypoint(run_dir, args.seed, output_dir)
    plot_sustained_turn(run_dir, args.seed, output_dir)


if __name__ == "__main__":
    main()
