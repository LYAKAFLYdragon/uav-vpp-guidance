"""Tests for the simplified missile engagement model."""
import numpy as np
import pytest

from uav_vpp_guidance.envs.missile_model import EngagementTracker, MissileModel


def test_missile_hits_static_target():
    config = {
        "speed_mps": 300.0,
        "max_g": 10.0,
        "kill_radius_m": 20.0,
        "max_range_m": 5000.0,
        "max_flight_time_s": 20.0,
        "navigation_constant": 3.0,
        "fov_deg": 60.0,
    }
    missile = MissileModel(
        launch_position_m=np.zeros(3),
        launch_velocity_mps=np.array([300.0, 0.0, 0.0]),
        target_uid="target",
        side="bandit",
        config=config,
    )
    target_pos = np.array([500.0, 0.0, 0.0])
    target_vel = np.zeros(3)
    for _ in range(200):
        missile.step(0.02, target_pos, target_vel)
        if missile.hit:
            break
    assert missile.hit
    assert not missile.missed


def test_missile_expires_when_target_too_far():
    config = {
        "speed_mps": 300.0,
        "max_g": 10.0,
        "kill_radius_m": 5.0,
        "max_range_m": 100.0,
        "max_flight_time_s": 1.0,
        "navigation_constant": 3.0,
        "fov_deg": 60.0,
    }
    missile = MissileModel(
        launch_position_m=np.zeros(3),
        launch_velocity_mps=np.array([300.0, 0.0, 0.0]),
        target_uid="target",
        side="bandit",
        config=config,
    )
    target_pos = np.array([0.0, 200.0, 0.0])
    target_vel = np.zeros(3)
    for _ in range(200):
        missile.step(0.02, target_pos, target_vel)
        if not missile.alive:
            break
    assert missile.missed


def test_engagement_tracker_launch_and_step():
    config = {
        "enabled": True,
        "max_range_m": 2000.0,
        "min_range_m": 100.0,
        "launch_cooldown_s": 0.0,
        "launch_probability": 1.0,
        "speed_mps": 600.0,
        "kill_radius_m": 15.0,
    }
    tracker = EngagementTracker(config)
    shooter = {
        "position_m": np.zeros(3),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    target = {
        "position_m": np.array([1000.0, 0.0, 0.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    missile = tracker.check_launch(shooter, target, side="bandit", sim_time=0.0)
    assert missile is not None
    for _ in range(300):
        tracker.step(0.02, {"own": target})
        hit = tracker.check_hits()
        if hit["own_hit"]:
            break
    assert tracker.check_hits()["own_hit"]
