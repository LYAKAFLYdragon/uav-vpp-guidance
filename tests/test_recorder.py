"""Tests for the flight-control comparison recorder layer."""

import tempfile
from pathlib import Path

import numpy as np

from uav_vpp_guidance.evaluation.recorders import EpisodeRecorder, RunRecorder


def _make_info(own_pos, target_pos, completed_waypoints=0):
    return {
        "own_state": {
            "position_m": np.asarray(own_pos),
            "altitude_m": float(np.asarray(own_pos)[2]),
            "speed_mps": 250.0,
            "nz_g": 1.0,
            "roll_rad": 0.35,
        },
        "target_state": {"position_m": np.asarray(target_pos)},
        "range_m": float(np.linalg.norm(np.asarray(own_pos) - np.asarray(target_pos))),
        "nz_cmd": 1.0,
        "roll_rate_cmd": -0.4,
        "throttle_cmd": 0.95,
        "task_supervisor_state": "recovery",
        "task_supervisor_pause_orbit_tracking": True,
        "task_supervisor_recovery_active": True,
        "aggressiveness": 0.0,
        "gain_scale": 1.0,
        "saturation_flag": False,
        "active_waypoint_index": 0,
        "switch_events": [],
        "waypoints": [],
        "completed_waypoints": completed_waypoints,
        "completed_orbits": 0.0,
        "turn_radius_m": 1000.0,
        "orbit_direction": 1.0,
        "virtual_point": {"position_neu": np.asarray(target_pos)},
    }


def test_episode_recorder_multi_waypoint():
    recorder = EpisodeRecorder(
        run_id="r1",
        task="multi_waypoint",
        controller="ppo_pid",
        seed=0,
        episode=0,
        config={},
        config_sha256="abc",
        git_commit="def",
        backend="simple",
        strict_backend=False,
    )
    own = np.array([0.0, 0.0, 5000.0])
    tgt = np.array([1000.0, 0.0, 5000.0])
    info = _make_info(own, tgt)
    recorder.record_step(1, 0.2, info["own_state"], info["target_state"], info, 0.0)
    ep = recorder.finalize(
        steps=1,
        total_time_s=0.2,
        total_reward=0.0,
        termination_reason="timeout",
        success=False,
        final_position_m=own.tolist(),
        final_speed_mps=250.0,
        final_altitude_m=5000.0,
    )
    assert ep["run_id"] == "r1"
    assert ep["backend"] == "simple"
    assert ep["task"] == "multi_waypoint"
    assert "statistics" in ep
    assert "trajectory" in ep
    assert ep["completed_waypoints"] == 0
    point = ep["trajectory"][0]
    assert point["altitude_m"] == 5000.0
    assert point["roll_rad"] == 0.35
    assert point["throttle_cmd"] == 0.95
    assert point["task_supervisor_state"] == "recovery"
    assert point["task_supervisor_pause_orbit_tracking"] is True


def test_run_recorder_writes_files():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        recorder = RunRecorder(run_dir)
        recorder.add({"task": "multi_waypoint", "controller": "ppo_pid", "seed": 0, "episode": 0})
        recorder.write()
        assert (run_dir / "aggregate" / "episode_records.json").exists()
        assert (run_dir / "raw" / "multi_waypoint" / "ppo_pid" / "seed_00" / "episode_000.json").exists()
