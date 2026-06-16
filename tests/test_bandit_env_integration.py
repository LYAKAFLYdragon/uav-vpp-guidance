"""Integration tests for the JSBSim bandit maneuver baseline in CloseRangeTrackingEnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def _resolve_bandit_config(overrides: dict | None = None) -> dict:
    """Load config/env_bandit.yaml and apply optional overrides."""
    config_path = Path(__file__).parent.parent / "config" / "env_bandit.yaml"
    root_dir = config_path.parent
    base = load_yaml_config(str(config_path))
    includes = base.pop("includes", []) or []
    merged = {}
    for inc in includes:
        inc_path = root_dir / inc
        if inc_path.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_path)))
    config = merge_config(merged, base)
    if overrides:
        config = merge_config(config, overrides)
    return config


@pytest.fixture
def jsbsim_available():
    """Skip tests if JSBSim Python bindings or data are unavailable."""
    jsbsim = pytest.importorskip("jsbsim")
    data_dir = Path(__file__).parent.parent / "data" / "jsbsim"
    if not data_dir.is_dir() or not (data_dir / "aircraft").is_dir():
        pytest.skip("JSBSim data directory not found")
    return jsbsim


class TestBanditEnvIntegration:
    def test_jsbsim_bandit_reset_and_step(self, jsbsim_available):
        """Bandit-enabled JSBSim env can reset and step, and exposes bandit info."""
        config = _resolve_bandit_config({"env": {"max_high_level_steps": 10}})
        env = CloseRangeTrackingEnv(config)
        assert env._backend == "jsbsim"
        assert env._bandit_enabled is True
        assert env._bandit_controller is not None

        obs = env.reset()
        assert isinstance(obs, dict)
        assert "relative_state" in obs

        action = np.zeros(3, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(info, dict)
        assert "bandit_maneuver" in info
        assert info.get("backend") == "jsbsim"
        assert np.isfinite(reward)
        env.close()

    def test_simple_backend_disables_bandit(self):
        """Bandit is disabled when using the simple backend."""
        config = _resolve_bandit_config({"backend": "simple"})
        env = CloseRangeTrackingEnv(config)
        assert env._backend == "simple"
        assert env._bandit_enabled is False
        assert env._bandit_controller is None

        obs = env.reset()
        action = np.zeros(3, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info.get("bandit_maneuver") is None
        env.close()
