#!/usr/bin/env python3
"""
Generate a synthetic flight-control comparison dataset for debugging Table 3
and the plotting pipeline.

The data is intentionally contrived so that:
  - PPO+PID outperforms PPO on both tasks,
  - Enhanced PID outperforms Baseline PID on both tasks,
  - there is enough seed/episode variation to exercise the statistical tests.

Usage:
    python scripts/generate_synthetic_flight_control_data.py \
        --run-dir outputs/flight_control_compare/_synthetic_demo
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from uav_vpp_guidance.evaluation.recorders import RunRecorder


CONTROLLERS = ["ppo_pid", "ppo", "enhanced_pid", "baseline_pid"]
SEEDS = list(range(5))
EPISODES = list(range(3))

# Controller biases (relative performance).
# Higher is better for the primary score: completed_waypoints / completed_orbits.
CONTROLLER_SCORE_BIAS = {
    "ppo_pid": 1.25,
    "ppo": 1.00,
    "enhanced_pid": 1.15,
    "baseline_pid": 1.00,
}

CONTROLLER_LABELS = {
    "ppo_pid": "PPO+PID",
    "ppo": "PPO",
    "enhanced_pid": "Enhanced PID",
    "baseline_pid": "Baseline PID",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _noisy(rng: random.Random, base: float, sigma: float) -> float:
    return max(0.0, rng.gauss(base, sigma))


def _make_waypoints(rng: random.Random) -> List[Dict[str, Any]]:
    """A simple 5-waypoint route in the xy-plane."""
    return [
        {"index": i, "pos": [float(x), float(y), 5000.0]}
        for i, (x, y) in enumerate(
            [
                (1000.0, 0.0),
                (2500.0, 800.0),
                (4000.0, -400.0),
                (5500.0, 600.0),
                (7000.0, 0.0),
            ]
        )
    ]


def _sample_completed_waypoints(rng: random.Random, controller: str) -> int:
    """Sample a realistic number of completed waypoints for this controller."""
    score_bias = CONTROLLER_SCORE_BIAS[controller]
    # Probability of completing each subsequent waypoint after the first.
    p_complete = 0.35 * score_bias
    # Always attempt the first waypoint; sample how many of the remaining 4 are reached.
    additional = sum(1 for _ in range(4) if rng.random() < p_complete)
    completed = 1 + additional
    # Best controller occasionally gets a free pass to full completion.
    if controller == "ppo_pid" and rng.random() < 0.60:
        completed = 5
    return max(0, min(5, completed))


def _multi_waypoint_trajectory(
    rng: random.Random,
    controller: str,
    waypoints: List[Dict[str, Any]],
    steps: int = 240,
    dt: float = 0.2,
) -> List[Dict[str, Any]]:
    """Generate a plausible multi-waypoint trajectory."""
    trajectory: List[Dict[str, Any]] = []
    score_bias = CONTROLLER_SCORE_BIAS[controller]
    # Per-episode speed variation.
    speed_base = 240.0 * score_bias * rng.uniform(0.92, 1.08)
    noise_base = 120.0 / score_bias

    target_completed = _sample_completed_waypoints(rng, controller)
    # Cap flight time roughly according to how far the controller gets.
    max_steps = int(steps * (target_completed / 5.0)) + 40
    max_steps = max(60, min(steps, max_steps))

    # Random initial offset.
    x0 = rng.uniform(-200.0, 200.0)
    y0 = rng.uniform(-200.0, 200.0)
    x, y = x0, y0
    t = 0.0
    active = 0

    for _ in range(max_steps):
        if active < len(waypoints) - 1:
            target = waypoints[active]["pos"]
            dx = target[0] - x
            dy = target[1] - y
            dist = math.hypot(dx, dy) + 1e-6
            if dist < 150.0:
                active += 1
                target = waypoints[active]["pos"]
                dx = target[0] - x
                dy = target[1] - y
                dist = math.hypot(dx, dy) + 1e-6

        # Once the sampled completion target is reached, drift slowly.
        if active >= target_completed and target_completed > 0:
            dx = rng.gauss(0.0, 50.0)
            dy = rng.gauss(0.0, 50.0)
            dist = math.hypot(dx, dy) + 1e-6

        vx = (dx / dist) * speed_base + rng.gauss(0.0, 15.0)
        vy = (dy / dist) * speed_base + rng.gauss(0.0, 15.0)
        x += vx * dt
        y += vy * dt
        # Mild altitude variation to make 3D plots interesting.
        z = 5000.0 + 80.0 * math.sin(0.05 * t) + rng.gauss(0.0, 5.0)
        t += dt

        speed = math.hypot(vx, vy)
        range_m = dist + rng.gauss(0.0, 30.0)
        nz = max(0.5, rng.gauss(2.5, 0.8))
        trajectory.append(
            {
                "time_s": round(t, 3),
                "own_pos_m": [round(x, 2), round(y, 2), round(z, 2)],
                "target_pos_m": [target[0], target[1], 5000.0],
                "range_m": max(0.0, range_m),
                "speed_mps": round(speed, 2),
                "nz_g": round(nz, 3),
                "active_waypoint_index": active,
                "saturation_flag": rng.random() < 0.05,
                "aggressiveness": (
                    round(rng.gauss(0.2, 0.3), 3) if controller == "ppo_pid" else None
                ),
                "gain_scale": (
                    round(rng.gauss(1.1, 0.1), 3) if controller == "ppo_pid" else None
                ),
            }
        )
    return trajectory


def _sustained_turn_trajectory(
    rng: random.Random,
    controller: str,
    steps: int = 450,
    dt: float = 0.2,
) -> List[Dict[str, Any]]:
    """Generate a circular sustained-turn trajectory with per-episode variation."""
    trajectory: List[Dict[str, Any]] = []
    score_bias = CONTROLLER_SCORE_BIAS[controller]
    # Per-episode variation in speed and radius to avoid zero variance.
    radius_base = (700.0 / score_bias) * rng.uniform(0.95, 1.05)
    speed_base = (220.0 * score_bias) * rng.uniform(0.95, 1.05)

    target_pos = [2000.0, 0.0, 5000.0]
    # Random phase start.
    theta = rng.uniform(0.0, 2.0 * math.pi)
    t = 0.0
    heading = math.degrees(theta) + 90.0

    for _ in range(steps):
        # Small independent perturbations so completed orbits vary across episodes.
        speed = speed_base + rng.gauss(0.0, 5.0)
        radius = radius_base + rng.gauss(0.0, 10.0)
        radius = max(100.0, radius)
        omega = speed / radius
        theta += omega * dt
        x = target_pos[0] + radius * math.cos(theta)
        y = target_pos[1] + radius * math.sin(theta)
        # Helical altitude variation for a nicer 3D trajectory.
        z = 5000.0 + 120.0 * math.sin(2.0 * theta) + rng.gauss(0.0, 5.0)
        t += dt
        heading = (math.degrees(theta) + 90.0) % 360.0

        nz = max(1.0, rng.gauss(2.8, 0.5))
        completed = max(0.0, (theta // (2.0 * math.pi)))

        trajectory.append(
            {
                "time_s": round(t, 3),
                "own_pos_m": [round(x, 2), round(y, 2), round(z, 2)],
                "target_pos_m": target_pos,
                "range_m": abs(radius),
                "speed_mps": round(speed, 2),
                "nz_g": round(nz, 3),
                "turn_radius_m": radius,
                "heading_deg": round(heading, 2),
                "completed_orbits": int(completed),
                "saturation_flag": rng.random() < 0.05,
                "aggressiveness": (
                    round(rng.gauss(0.15, 0.25), 3) if controller == "ppo_pid" else None
                ),
                "gain_scale": (
                    round(rng.gauss(1.05, 0.08), 3) if controller == "ppo_pid" else None
                ),
            }
        )
    return trajectory


def _multi_waypoint_stats(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranges = [r["range_m"] for r in trajectory]
    speeds = [r["speed_mps"] for r in trajectory]
    nz = [r["nz_g"] for r in trajectory]
    agg = [r["aggressiveness"] for r in trajectory if r["aggressiveness"] is not None]
    gsc = [r["gain_scale"] for r in trajectory if r["gain_scale"] is not None]
    sat = [r["saturation_flag"] for r in trajectory]
    return {
        "mean_track_error_m": float(np.mean(ranges)),
        "std_track_error_m": float(np.std(ranges, ddof=1)),
        "median_track_error_m": float(np.median(ranges)),
        "max_track_error_m": float(np.max(ranges)),
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "mean_nz_g": float(np.mean(nz)),
        "max_nz_g": float(np.max(nz)),
        "saturation_ratio": float(np.mean(sat)),
        "mean_aggressiveness": float(np.mean(agg)) if agg else None,
        "mean_gain_scale": float(np.mean(gsc)) if gsc else None,
    }


def _sustained_turn_stats(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    radii = [r["turn_radius_m"] for r in trajectory]
    speeds = [r["speed_mps"] for r in trajectory]
    nz = [r["nz_g"] for r in trajectory]
    agg = [r["aggressiveness"] for r in trajectory if r["aggressiveness"] is not None]
    gsc = [r["gain_scale"] for r in trajectory if r["gain_scale"] is not None]
    sat = [r["saturation_flag"] for r in trajectory]
    headings = [r["heading_deg"] for r in trajectory]
    times = [r["time_s"] for r in trajectory]

    # turn rate
    turn_rates = []
    for i in range(1, len(headings)):
        dh = headings[i] - headings[i - 1]
        dh = (dh + 180.0) % 360.0 - 180.0
        dt = times[i] - times[i - 1]
        if dt > 0:
            turn_rates.append(abs(dh) / dt)

    # Energy loss rate: slope of speed vs time (loss = -b).
    s = np.asarray(speeds, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(s) & np.isfinite(t)
    s, t = s[valid], t[valid]
    if len(s) >= 2:
        t_mean = float(np.mean(t))
        var_t = float(np.mean((t - t_mean) ** 2))
        if var_t > 1e-12:
            cov = float(np.mean((t - t_mean) * (s - np.mean(s))))
            energy_loss = max(0.0, float(-(cov / var_t)))
        else:
            energy_loss = 0.0
    else:
        energy_loss = 0.0

    return {
        "completed_orbits": int(trajectory[-1]["completed_orbits"]) if trajectory else 0,
        "avg_turn_rate_deg_s": float(np.mean(turn_rates)) if turn_rates else 0.0,
        "std_turn_rate_deg_s": float(np.std(turn_rates, ddof=1)) if len(turn_rates) > 1 else 0.0,
        "avg_turn_radius_m": float(np.mean(radii)),
        "std_turn_radius_m": float(np.std(radii, ddof=1)),
        "radius_std_m": float(np.std(radii, ddof=1)),
        "avg_speed_mps": float(np.mean(speeds)),
        "std_speed_mps": float(np.std(speeds, ddof=1)),
        "mean_nz_g": float(np.mean(nz)),
        "max_nz_g": float(np.max(nz)),
        "energy_loss_rate_mps2": energy_loss,
        "saturation_ratio": float(np.mean(sat)),
        "mean_aggressiveness": float(np.mean(agg)) if agg else None,
        "mean_gain_scale": float(np.mean(gsc)) if gsc else None,
    }


def _build_record(
    run_id: str,
    task: str,
    controller: str,
    seed: int,
    episode: int,
    trajectory: List[Dict[str, Any]],
    waypoints: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if task == "multi_waypoint":
        stats = _multi_waypoint_stats(trajectory)
        completed = sum(
            1
            for i, r in enumerate(trajectory)
            if i > 0 and r["active_waypoint_index"] != trajectory[i - 1]["active_waypoint_index"]
        ) + (1 if trajectory else 0)
        completed = min(5, max(0, completed))
    else:
        stats = _sustained_turn_stats(trajectory)
        completed = stats["completed_orbits"]

    total_time = trajectory[-1]["time_s"] if trajectory else 0.0
    return {
        "run_id": run_id,
        "task": task,
        "controller": controller,
        "seed": seed,
        "episode": episode,
        "backend": "simple",
        "strict_backend": False,
        "config_sha256": "synthetic",
        "git_commit": "synthetic",
        "success": completed > 0,
        "termination_reason": "success" if completed > 0 else "timeout",
        "steps": len(trajectory),
        "total_time_s": total_time,
        "total_reward": float(completed) * 100.0,
        "trajectory": trajectory,
        "statistics": stats,
        "completed_waypoints": int(completed) if task == "multi_waypoint" else None,
        "completed_orbits": int(completed) if task == "sustained_turn" else None,
        "switch_events": [],
        "waypoints": waypoints or [],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic flight-control comparison data")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/flight_control_compare/_synthetic_demo"),
        help="Output run directory",
    )
    parser.add_argument("--run-id", type=str, default="synthetic_demo", help="Run identifier")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    recorder = RunRecorder(run_dir)

    waypoints = _make_waypoints(random.Random(42))

    for task in ("multi_waypoint", "sustained_turn"):
        for controller in CONTROLLERS:
            for seed in SEEDS:
                rng = _rng(seed * 1000 + 7)
                for episode in EPISODES:
                    if task == "multi_waypoint":
                        traj = _multi_waypoint_trajectory(
                            _rng(seed * 10000 + episode), controller, waypoints=waypoints
                        )
                        rec = _build_record(
                            args.run_id, task, controller, seed, episode, traj, waypoints
                        )
                    else:
                        traj = _sustained_turn_trajectory(
                            _rng(seed * 10000 + episode), controller
                        )
                        rec = _build_record(
                            args.run_id, task, controller, seed, episode, traj
                        )
                    recorder.add(rec)

    recorder.write()

    # Write a minimal manifest so downstream tools know the backend.
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "backend": "simple",
                "controllers": CONTROLLERS,
                "n_episodes": len(recorder.records),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(recorder.records)} synthetic episodes in {run_dir}")


if __name__ == "__main__":
    main()
