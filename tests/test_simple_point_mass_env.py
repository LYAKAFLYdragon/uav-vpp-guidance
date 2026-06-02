"""
Tests for SimplePointMassEnv.
"""

import pytest
import numpy as np
from uav_vpp_guidance.envs.simple_point_mass_env import SimplePointMassEnv


@pytest.fixture
def env():
    config = {"decision_freq": 5, "target_mode": "constant_velocity"}
    return SimplePointMassEnv(config)


class TestSimplePointMassEnv:
    def test_reset_returns_states(self, env):
        own, target = env.reset()
        assert isinstance(own, dict)
        assert isinstance(target, dict)

    def test_reset_with_seed(self, env):
        own, target = env.reset(seed=42)
        assert "position_m" in own
        assert "velocity_vector_mps" in own

    def test_step(self, env):
        env.reset()
        command = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
        own, target = env.step(command)
        assert "position_m" in own
        assert "velocity_vector_mps" in own
        assert "altitude_m" in own

    def test_get_state(self, env):
        env.reset()
        own, target = env.get_state()
        assert isinstance(own, dict)
        assert isinstance(target, dict)

    def test_own_state_fields(self, env):
        env.reset()
        own, _ = env.get_state()
        required = ["position_m", "velocity_vector_mps", "speed_mps",
                    "heading_rad", "pitch_rad", "roll_rad", "altitude_m", "nz"]
        for field in required:
            assert field in own, f"Missing field: {field}"

    def test_target_state_fields(self, env):
        env.reset()
        _, target = env.get_state()
        required = ["position_m", "velocity_vector_mps", "speed_mps",
                    "heading_rad", "altitude_m"]
        for field in required:
            assert field in target, f"Missing field: {field}"

    def test_target_constant_velocity(self, env):
        env.reset()
        _, target0 = env.get_state()
        pos0 = target0["position_m"].copy()
        env.step({"nz_cmd": 1.0})
        _, target1 = env.get_state()
        pos1 = target1["position_m"]
        assert not np.allclose(pos0, pos1)

    def test_target_sinusoidal_mode(self):
        env = SimplePointMassEnv({"decision_freq": 5, "target_mode": "sinusoidal"})
        env.reset()
        _, target0 = env.get_state()
        pos0 = target0["position_m"].copy()
        env.step({"nz_cmd": 1.0})
        _, target1 = env.get_state()
        pos1 = target1["position_m"]
        assert not np.allclose(pos0, pos1)
