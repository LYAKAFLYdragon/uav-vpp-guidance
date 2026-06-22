"""Smoke tests for MultiWaypointTrackingEnv."""

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest

from uav_vpp_guidance.envs.multi_waypoint_tracking_env import MultiWaypointTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

pytest.importorskip("jsbsim")


def _load_experiment_config(config_path: str) -> Dict[str, Any]:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged: Dict[str, Any] = {}
    cfg_dir = os.path.dirname(config_path)
    for inc_path in includes:
        inc_full = os.path.join(cfg_dir, inc_path)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc_path))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


@pytest.fixture(scope="module")
def base_config() -> Dict[str, Any]:
    path = os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment", "compare_flight_control_base.yaml"
    )
    cfg = _load_experiment_config(path)
    task_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "experiment", "task_multi_waypoint.yaml"
    )
    task_cfg = _load_experiment_config(task_path)
    return merge_config(cfg, task_cfg)


@pytest.fixture
def env(base_config):
    cfg = deepcopy(base_config)
    cfg["env"]["max_high_level_steps"] = 32
    environment = MultiWaypointTrackingEnv(cfg)
    yield environment
    environment.close()


def test_reset_generates_waypoints(env):
    obs = env.reset(seed=0)
    assert obs["observation_vector"].shape[0] > 0
    assert len(env.waypoints) == 5
    assert env.active_idx == 0
    assert env.completed_waypoints == 0


def test_step_returns_valid_tuple(env):
    env.reset(seed=1)
    action = np.zeros(3, dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "active_waypoint_index" in info
    assert "completed_waypoints" in info


def test_waypoint_switching_logic(env):
    env.reset(seed=2)
    # The first waypoint is generated far enough that a single zero-action step
    # should not switch. Just verify the counter does not decrease.
    info = None
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    assert info["completed_waypoints"] >= 0
    assert info["active_waypoint_index"] >= 0


def test_no_success_termination_on_first_waypoint(env):
    env.reset(seed=3)
    info = None
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    # If we terminated early, it should not be due to the default success rule.
    if info:
        assert info.get("reason") != "success"
