#!/usr/bin/env python3
"""
Generate Figure 6 (multi-waypoint) and Figure 7 (sustained turn) for the
flight-control comparison paper.

Usage:
    python scripts/plot_flight_control_figures.py \
        --run-dir outputs/flight_control_compare/fc_compare_20260621_103000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


CONTROLLER_COLORS = {
    "ppo_pid": "C0",
    "ppo": "C1",
    "enhanced_pid": "C2",
    "baseline_pid": "C3",
    "gain_scheduled_pid": "C4",
}

CONTROLLER_LABELS = {
    "ppo_pid": "PPO+PID",
    "ppo": "PPO",
    "enhanced_pid": "Enhanced PID",
    "baseline_pid": "Baseline PID",
    "gain_scheduled_pid": "GainScheduled PID",
}

# Visualization caps (match physical caps in flight_control_metrics.py).
MAX_RANGE_M = 50_000.0
MAX_TURN_RADIUS_M = 50_000.0
MAX_NZ_G = 20.0


def _set_3d_spatial_aspect(ax):
    """Keep x and y on a common scale; let z auto-scale so altitude detail is visible."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Equalize x/y only (typical flight trajectory is mostly horizontal).
    xy_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2.0
    x_center = (xlim[0] + xlim[1]) / 2.0
    y_center = (ylim[0] + ylim[1]) / 2.0
    ax.set_xlim3d([x_center - xy_range, x_center + xy_range])
    ax.set_ylim3d([y_center - xy_range, y_center + xy_range])

    # Keep z on its own scale with modest padding so vertical structure is visible.
    z_range = zlim[1] - zlim[0]
    if z_range <= 0:
        z_range = 1.0
    z_center = (zlim[0] + zlim[1]) / 2.0
    pad = 0.05 * z_range
    ax.set_zlim3d([z_center - z_range / 2.0 - pad, z_center + z_range / 2.0 + pad])


def _find_representative_episode(run_dir: Path, task: str, controller: str, seed: int = 0, episode: int = 0) -> Optional[Path]:
    path = run_dir / "raw" / task / controller / f"seed_{seed:02d}" / f"episode_{episode:03d}.json"
    return path if path.exists() else None


def _load_episode(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_range(values, cap_m: float = 10_000.0):
    """Replace non-finite or physically implausible range values with NaN for plotting."""
    out = []
    for v in values:
        if v is None or not np.isfinite(v) or abs(v) > cap_m:
            out.append(np.nan)
        else:
            out.append(float(v))
    return out


def _get_controller_episodes(run_dir: Path, task: str, controllers: list) -> Dict[str, Path]:
    episodes = {}
    for controller in controllers:
        for seed in range(20):
            path = _find_representative_episode(run_dir, task, controller, seed=seed, episode=0)
            if path:
                episodes[controller] = path
                break
    return episodes


def _plot_fig6(run_dir: Path, controllers: list, out_png: Path) -> None:
    episodes = _get_controller_episodes(run_dir, "multi_waypoint", controllers)
    if not episodes:
        print("No multi-waypoint episodes found; skipping Figure 6")
        return

    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    first_waypoints = None

    for controller, json_path in episodes.items():
        obj = _load_episode(json_path)
        traj = obj.get("trajectory", [])
        if not traj:
            continue
        color = CONTROLLER_COLORS.get(controller, None)
        label = CONTROLLER_LABELS.get(controller, controller)

        xs = [r["own_pos_m"][0] for r in traj if r.get("own_pos_m")]
        ys = [r["own_pos_m"][1] for r in traj if r.get("own_pos_m")]
        zs = [r["own_pos_m"][2] if len(r["own_pos_m"]) > 2 else 0.0 for r in traj if r.get("own_pos_m")]
        ts = [r["time_s"] for r in traj]
        rng = _sanitize_range([r["range_m"] for r in traj])

        ax1.plot(xs, ys, zs, label=label, color=color, linewidth=1.5)
        ax2.plot(ts, rng, label=label, color=color, linewidth=1.5)

        # Segment-normalized error curves
        seg_map = {}
        for r in traj:
            seg_map.setdefault(r.get("active_waypoint_index", 0), []).append(r)
        for seg_idx, rows in seg_map.items():
            if not rows:
                continue
            t0 = rows[0]["time_s"]
            ax3.plot(
                [rr["time_s"] - t0 for rr in rows],
                _sanitize_range([rr["range_m"] for rr in rows]),
                alpha=0.35,
                color=color,
                label=label if seg_idx == 0 else None,
            )

        # Switch events for this controller (draw on Figure 6B).
        for r in traj:
            if r.get("switch_event"):
                ax2.axvline(r["time_s"], linestyle="--", alpha=0.25, color="gray")
        # Fallback to top-level switch_events if trajectory points lack the flag.
        for sw in obj.get("switch_events", []):
            ax2.axvline(sw["time_s"], linestyle="--", alpha=0.25, color="gray")

        if first_waypoints is None:
            first_waypoints = obj.get("waypoints", [])

    # Plot waypoint markers from the first available episode
    if first_waypoints:
        for i, wp in enumerate(first_waypoints):
            pos = wp.get("pos", [0.0, 0.0, 0.0])
            ax1.scatter([pos[0]], [pos[1]], [pos[2] if len(pos) > 2 else 0.0], marker="x", s=80, color="red", zorder=5)
            ax1.text(pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0, f" W{i}", fontsize=9, color="red")

    ax1.set_title("Figure 6A: 3D Spatial Trajectory")
    ax1.set_xlabel("x / m")
    ax1.set_ylabel("y / m")
    ax1.set_zlabel("z / m")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    _set_3d_spatial_aspect(ax1)

    ax2.set_title("Figure 6B: Active Target Range vs Time")
    ax2.set_xlabel("time / s")
    ax2.set_ylabel("range / m")
    ax2.set_ylim(bottom=0.0)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Figure 6C: Segment-normalized Error Curves")
    ax3.set_xlabel("segment time / s")
    ax3.set_ylabel("range / m")
    ax3.set_ylim(bottom=0.0)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Creator": "uav-vpp-guidance", "Title": "Figure 6 Multi-Waypoint Tracking"},
    )
    plt.close(fig)
    print(f"Saved Figure 6 to {out_png}")


def _plot_fig7(run_dir: Path, controllers: list, out_png: Path) -> None:
    episodes = _get_controller_episodes(run_dir, "sustained_turn", controllers)
    if not episodes:
        print("No sustained-turn episodes found; skipping Figure 7")
        return

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Try to read target position from the first trajectory point or task config
    target_pos = [2000.0, 0.0, 0.0]
    first_json_path = next(iter(episodes.values()), None)
    if first_json_path is not None:
        first_obj = _load_episode(first_json_path)
        first_traj = first_obj.get("trajectory", [])
        if first_traj and first_traj[0].get("target_pos_m"):
            target_pos = first_traj[0]["target_pos_m"]
        elif first_obj.get("waypoints"):
            # Multi-waypoint fallback: use first waypoint as target
            wp_pos = first_obj["waypoints"][0].get("pos", [0.0, 0.0, 0.0])
            target_pos = wp_pos if len(wp_pos) >= 3 else [wp_pos[0], wp_pos[1], 0.0]
    ppo_pid_plotted = False

    for controller, json_path in episodes.items():
        obj = _load_episode(json_path)
        traj = obj.get("trajectory", [])
        if not traj:
            continue
        color = CONTROLLER_COLORS.get(controller, None)
        label = CONTROLLER_LABELS.get(controller, controller)

        xs = [r["own_pos_m"][0] for r in traj if r.get("own_pos_m")]
        ys = [r["own_pos_m"][1] for r in traj if r.get("own_pos_m")]
        zs = [r["own_pos_m"][2] if len(r["own_pos_m"]) > 2 else 0.0 for r in traj if r.get("own_pos_m")]
        ts = [r["time_s"] for r in traj]
        nz = _sanitize_range([r.get("nz_g", np.nan) for r in traj], cap_m=MAX_NZ_G)
        radius = _sanitize_range([r.get("turn_radius_m", np.nan) for r in traj], cap_m=MAX_TURN_RADIUS_M)

        ax1.plot(xs, ys, zs, label=label, color=color, linewidth=1.5)
        ax2.plot(ts, nz, label=label, color=color, linewidth=1.5)
        ax3.plot(ts, radius, label=label, color=color, linewidth=1.5)

        if controller == "ppo_pid":
            agg = [r.get("aggressiveness") for r in traj]
            agg = [a if a is not None else np.nan for a in agg]
            ax4.plot(ts, agg, label="aggressiveness", linewidth=1.5, color="C0")
            gsc = [r.get("gain_scale") for r in traj]
            gsc = [g if g is not None else np.nan for g in gsc]
            ax4_twin = ax4.twinx()
            ax4_twin.plot(ts, gsc, label="gain_scale", linewidth=1.0, linestyle="--", color="C1", alpha=0.7)
            ax4_twin.set_ylabel("gain_scale", color="C1")
            ax4_twin.tick_params(axis="y", labelcolor="C1")
            ax4_twin.set_ylim(0.4, 1.6)
            ppo_pid_plotted = True

        # Try to read target position from the first trajectory point
        if traj and traj[0].get("target_pos_m"):
            target_pos = traj[0]["target_pos_m"]

    ax1.scatter([target_pos[0]], [target_pos[1]], [target_pos[2] if len(target_pos) > 2 else 0.0], marker="*", s=150, color="red", zorder=5)
    ax1.set_title("Figure 7A: Sustained Turn 3D Spatial Trajectory")
    ax1.set_xlabel("x / m")
    ax1.set_ylabel("y / m")
    ax1.set_zlabel("z / m")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    _set_3d_spatial_aspect(ax1)

    ax2.set_title("Figure 7B: NZ Overload vs Time")
    ax2.set_xlabel("time / s")
    ax2.set_ylabel("NZ / g")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Figure 7C: Turn Radius vs Time")
    ax3.set_xlabel("time / s")
    ax3.set_ylabel("radius / m")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    ax4.set_title("Figure 7D: Aggressiveness (PPO+PID only)")
    ax4.set_xlabel("time / s")
    ax4.set_ylabel("aggressiveness", color="C0")
    ax4.tick_params(axis="y", labelcolor="C0")
    ax4.set_ylim(-1.2, 1.2)
    ax4.grid(True, alpha=0.3)
    if ppo_pid_plotted:
        ax4.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Creator": "uav-vpp-guidance", "Title": "Figure 7 Sustained Turn"},
    )
    plt.close(fig)
    print(f"Saved Figure 7 to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot flight-control comparison figures")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run root directory")
    parser.add_argument(
        "--controllers",
        type=str,
        nargs="+",
        default=["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"],
        help="Controllers to plot",
    )
    args = parser.parse_args()

    figures_dir = args.run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_fig6(args.run_dir, args.controllers, figures_dir / "fig6_multi_waypoint.png")
    _plot_fig7(args.run_dir, args.controllers, figures_dir / "fig7_sustained_turn.png")


if __name__ == "__main__":
    main()
