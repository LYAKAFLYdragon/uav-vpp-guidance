"""Smoke tests for the adversarial two-aircraft JSBSim environment."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest

from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

# Skip the whole module if JSBSim is unavailable.
pytest.importorskip("jsbsim")


def _load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load a config the same way the training scripts do."""
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
        os.path.dirname(__file__), "..", "config", "adversarial", "train_target.yaml"
    )
    return _load_experiment_config(path)


@pytest.fixture
def env(base_config):
    """Adversarial environment with default RL controllers."""
    cfg = deepcopy(base_config)
    # Keep episodes short for smoke tests.
    cfg["env"]["max_high_level_steps"] = 32
    environment = AdversarialJSBSimEnv(cfg)
    yield environment
    environment.close()


def test_default_reset_is_not_overlapping(env):
    """Without an explicit scenario the two aircraft must not start on top of each other."""
    p_obs, t_obs = env.reset(seed=0)

    p_pos = p_obs["own_state"]["position_neu"]
    t_pos = p_obs["target_state"]["position_neu"]
    distance = float(np.linalg.norm(np.asarray(p_pos) - np.asarray(t_pos)))

    assert distance > 1000.0, f"Aircraft start too close: {distance:.1f} m"
    # Tail-chase default should put the target ahead of the pursuer.
    assert p_obs["relative_state"]["range_m"] > 1000.0


def test_reset_and_step_return_shapes(env):
    """reset/step must return the documented 7-tuple."""
    p_obs, t_obs = env.reset(seed=1)
    assert p_obs["observation_vector"].shape == (16,)
    assert t_obs["observation_vector"].shape == (16,)

    p_action = np.zeros(3, dtype=np.float64)
    t_action = np.zeros(3, dtype=np.float64)
    result = env.step(p_action, t_action)
    assert len(result) == 7
    p_obs2, t_obs2, p_rew, t_rew, terminated, truncated, info = result
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(p_rew, float)
    assert isinstance(t_rew, float)
    assert "step" in info


def test_seed_reproducibility(base_config):
    """Same seed must yield the same initial observation."""
    cfg = deepcopy(base_config)
    cfg["env"]["max_high_level_steps"] = 4

    env_a = AdversarialJSBSimEnv(cfg)
    env_b = AdversarialJSBSimEnv(cfg)
    try:
        p_a, t_a = env_a.reset(seed=123)
        p_b, t_b = env_b.reset(seed=123)
        np.testing.assert_allclose(
            p_a["observation_vector"], p_b["observation_vector"], rtol=1e-6
        )
        np.testing.assert_allclose(
            t_a["observation_vector"], t_b["observation_vector"], rtol=1e-6
        )
    finally:
        env_a.close()
        env_b.close()


def test_truncated_on_timeout(base_config):
    """Episode must report truncated (not terminated) when max_steps is reached."""
    cfg = deepcopy(base_config)
    cfg["env"]["max_high_level_steps"] = 2

    env = AdversarialJSBSimEnv(cfg)
    try:
        p_obs, t_obs = env.reset(seed=7)
        terminated = truncated = False
        for _ in range(3):
            p_obs, t_obs, _, _, terminated, truncated, _ = env.step(
                np.zeros(3), np.zeros(3)
            )
            if terminated or truncated:
                break
        assert truncated is True, "Expected truncation at max_steps"
        assert terminated is False, "Timeout should be truncated, not terminated"
    finally:
        env.close()


def test_maneuver_library_target(base_config):
    """Target side can be driven by the maneuver library."""
    cfg = deepcopy(base_config)
    cfg["env"]["target_controller_type"] = "maneuver_library"
    cfg["env"]["bandit"] = {"fallback_maneuver": "straight_level"}
    cfg["env"]["max_high_level_steps"] = 4

    env = AdversarialJSBSimEnv(cfg)
    try:
        p_obs, t_obs = env.reset(seed=5)
        # Maneuver-library controller ignores the passed target_action.
        result = env.step(np.zeros(3), np.zeros(3))
        assert len(result) == 7
        info = result[-1]
        assert info["target_controller_type"] == "maneuver_library"
        assert "target_maneuver" in info
        assert info["target_maneuver"] is not None
    finally:
        env.close()


def test_expert_vpp_pursuer(base_config):
    """Pursuer side can be driven by ExpertVPPPolicy."""
    cfg = deepcopy(base_config)
    cfg["env"]["pursuer_controller_type"] = "expert_vpp"
    cfg["expert_vpp"] = {}
    cfg["env"]["max_high_level_steps"] = 4

    env = AdversarialJSBSimEnv(cfg)
    try:
        p_obs, t_obs = env.reset(seed=6)
        result = env.step(np.zeros(3), np.zeros(3))
        assert len(result) == 7
        info = result[-1]
        assert info["pursuer_controller_type"] == "expert_vpp"
    finally:
        env.close()
