"""Unit tests for aggregation and Table 3 export helpers."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uav_vpp_guidance.evaluation.flight_control_metrics import (
    compute_multi_waypoint_metrics,
    compute_sustained_turn_metrics,
)


def _make_multi_waypoint_trajectory():
    rng = np.random.default_rng(0)
    return [
        {
            "range_m": float(rng.uniform(400, 800)),
            "speed_mps": float(rng.uniform(200, 260)),
            "nz_g": float(rng.uniform(1.0, 5.0)),
            "aggressiveness": 0.1,
            "gain_scale": 1.05,
            "saturation_flag": False,
        }
        for _ in range(50)
    ]


def _make_sustained_turn_trajectory():
    rng = np.random.default_rng(1)
    return [
        {
            "time_s": float(i * 0.2),
            "turn_radius_m": float(rng.normal(1200, 100)),
            "speed_mps": float(rng.uniform(220, 250)),
            "nz_g": float(rng.uniform(3.0, 6.0)),
            "heading_deg": float(i * 3.0),
            "aggressiveness": -0.2,
            "gain_scale": 0.9,
            "saturation_flag": False,
        }
        for i in range(100)
    ]


def test_multi_waypoint_metrics():
    traj = _make_multi_waypoint_trajectory()
    stats = compute_multi_waypoint_metrics(traj)
    assert "mean_track_error_m" in stats
    assert "max_track_error_m" in stats
    assert "max_nz_g" in stats
    assert "mean_aggressiveness" in stats
    assert "mean_gain_scale" in stats
    assert stats["mean_aggressiveness"] == pytest.approx(0.1, abs=1e-6)


def test_sustained_turn_metrics():
    traj = _make_sustained_turn_trajectory()
    stats = compute_sustained_turn_metrics(traj)
    assert "avg_turn_rate_deg_s" in stats
    assert "avg_turn_radius_m" in stats
    assert "radius_std_m" in stats
    assert "avg_speed_mps" in stats
    assert "energy_loss_rate_mps2" in stats
    assert "mean_aggressiveness" in stats


def test_aggregate_script_runs(tmp_path: Path, monkeypatch):
    """End-to-end smoke test of the aggregation script on synthetic data."""
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw" / "multi_waypoint" / "ppo_pid" / "seed_00"
    raw_dir.mkdir(parents=True)

    traj = _make_multi_waypoint_trajectory()
    ep = {
        "run_id": "test",
        "task": "multi_waypoint",
        "controller": "ppo_pid",
        "seed": 0,
        "episode": 0,
        "backend": "jsbsim",
        "strict_backend": True,
        "config_sha256": "abc",
        "git_commit": "abc",
        "success": True,
        "termination_reason": "all_waypoints_completed",
        "steps": 50,
        "total_time_s": 10.0,
        "total_reward": 100.0,
        "trajectory": traj,
        "statistics": compute_multi_waypoint_metrics(traj),
        "completed_waypoints": 5,
    }
    (raw_dir / "episode_000.json").write_text(json.dumps(ep), encoding="utf-8")

    # Import the script under test and run it via subprocess to avoid path issues.
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_flight_control_results.py", f"--run-dir={run_dir}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "aggregate" / "summary.csv").exists()
    summary = pd.read_csv(run_dir / "aggregate" / "summary.csv")
    assert not summary.empty
    assert "completed_waypoints_mean" in summary.columns


def test_export_table3_script_runs(tmp_path: Path):
    """Smoke test of the Table 3 export script using synthetic summary data."""
    run_dir = tmp_path / "run"
    aggregate_dir = run_dir / "aggregate"
    tables_dir = run_dir / "tables"
    aggregate_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)

    summary_rows = [
        {
            "task": "multi_waypoint",
            "controller": "ppo_pid",
            "n": 20,
            "completed_waypoints_mean": 4.5,
            "completed_waypoints_std": 0.5,
            "completed_waypoints_ci95_low": 4.2,
            "completed_waypoints_ci95_high": 4.8,
            "mean_track_error_m_mean": 300.0,
            "mean_track_error_m_std": 50.0,
            "mean_speed_mps_mean": 250.0,
            "mean_nz_g_mean": 3.0,
            "mean_aggressiveness_mean": 0.1,
        },
        {
            "task": "multi_waypoint",
            "controller": "ppo",
            "n": 20,
            "completed_waypoints_mean": 3.0,
            "completed_waypoints_std": 0.8,
            "completed_waypoints_ci95_low": 2.6,
            "completed_waypoints_ci95_high": 3.4,
            "mean_track_error_m_mean": 500.0,
            "mean_track_error_m_std": 80.0,
            "mean_speed_mps_mean": 240.0,
            "mean_nz_g_mean": 2.8,
            "mean_aggressiveness_mean": np.nan,
        },
        {
            "task": "sustained_turn",
            "controller": "ppo_pid",
            "n": 20,
            "completed_orbits_mean": 2.5,
            "completed_orbits_std": 0.2,
            "avg_turn_rate_deg_s_mean": 18.0,
            "avg_turn_radius_m_mean": 1200.0,
            "radius_std_m_mean": 50.0,
            "avg_speed_mps_mean": 230.0,
            "mean_nz_g_mean": 4.0,
            "energy_loss_rate_mps2_mean": 0.5,
            "mean_aggressiveness_mean": 0.1,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(aggregate_dir / "summary.csv", index=False)

    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        [sys.executable, "scripts/export_table3.py", f"--run-dir={run_dir}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tables_dir / "table3_summary.md").exists()
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pass
    else:
        assert (tables_dir / "table3_summary.xlsx").exists()
    md = (tables_dir / "table3_summary.md").read_text(encoding="utf-8")
    assert "PPO+PID" in md
    assert "Multi-Waypoint" in md
