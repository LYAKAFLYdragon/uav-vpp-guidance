"""Smoke tests for SustainedTurnEnv."""

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest

from uav_vpp_guidance.envs.sustained_turn_env import SustainedTurnEnv
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
        os.path.dirname(__file__), "..", "config", "experiment", "task_sustained_turn.yaml"
    )
    task_cfg = _load_experiment_config(task_path)
    return merge_config(cfg, task_cfg)


@pytest.fixture
def env(base_config):
    cfg = deepcopy(base_config)
    cfg["env"]["max_high_level_steps"] = 32
    environment = SustainedTurnEnv(cfg)
    yield environment
    environment.close()


def test_reset_initializes_orbit(env):
    obs = env.reset(seed=0)
    assert obs["observation_vector"].shape[0] > 0
    assert env.orbit_direction in (-1.0, 1.0)
    assert len(env.radius_hist) == 0


def test_step_tracks_orbit_metrics(env):
    env.reset(seed=1)
    info = None
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    assert info is not None
    assert "completed_orbits" in info
    assert "turn_radius_m" in info


def test_no_success_termination(env):
    env.reset(seed=2)
    info = None
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        if terminated or truncated:
            break
    if info:
        assert info.get("reason") != "success"
