"""Smoke tests for BreakTurnEnv."""

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest

from uav_vpp_guidance.envs.break_turn_env import BreakTurnEnv
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
        os.path.dirname(__file__), "..", "config", "experiment", "task_break_turn.yaml"
    )
    task_cfg = _load_experiment_config(task_path)
    return merge_config(cfg, task_cfg)


@pytest.fixture
def env(base_config):
    cfg = deepcopy(base_config)
    cfg["backend"] = "simple"
    cfg["env"]["use_jsbsim"] = False
    cfg["env"]["strict_backend"] = False
    cfg["env"]["max_high_level_steps"] = 32
    environment = BreakTurnEnv(cfg)
    yield environment
    environment.close()


def test_reset_generates_trajectory(env):
    obs = env.reset(seed=0)
    assert obs["observation_vector"].shape[0] > 0
    assert len(env.trajectory) > 0
    assert env.trajectory[0]["speed_mps"] > 0


def test_trajectory_has_required_keys(env):
    env.reset(seed=1)
    point = env.trajectory[0]
    for key in ("time_s", "position_m", "velocity_vector_mps", "heading_deg", "speed_mps"):
        assert key in point


def test_step_returns_valid_tuple(env):
    env.reset(seed=2)
    action = np.zeros(3, dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "maneuver_type" in info
    assert "range_m" in info


def test_yo_yo_maneuver_type(base_config):
    cfg = deepcopy(base_config)
    cfg["backend"] = "simple"
    cfg["env"]["use_jsbsim"] = False
    cfg["env"]["strict_backend"] = False
    cfg["env"]["max_high_level_steps"] = 32
    cfg["task"]["break_turn"]["maneuver_type"] = "yo_yo"
    environment = BreakTurnEnv(cfg)
    environment.reset(seed=3)
    assert environment.maneuver_type == "yo_yo"
    assert len(environment.trajectory) > 0
    # Altitude should vary across the yo-yo trajectory.
    alts = [p["position_m"][2] for p in environment.trajectory]
    assert max(alts) > min(alts) + 1.0
    environment.close()


def test_no_success_termination(env):
    env.reset(seed=4)
    info = None
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    if info:
        assert info.get("reason") != "success"
