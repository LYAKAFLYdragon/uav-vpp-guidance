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


class TestManeuverTargetModes:
    """Tests for new maneuver target dynamics (sinusoidal_weaving, bang_bang, barrel_roll)."""

    def test_sinusoidal_weaving(self):
        env = SimplePointMassEnv({
            "decision_freq": 5,
            "target_mode": "sinusoidal_weaving",
            "weaving_amplitude_g": 3.0,
            "weaving_frequency_rad_s": 1.0,
        })
        env.reset()
        _, target0 = env.get_state()
        pos0 = target0["position_m"].copy()
        heading0 = target0["heading_rad"]
        speed0 = target0["speed_mps"]
        for _ in range(20):
            env.step({"nz_cmd": 1.0})
        _, target1 = env.get_state()
        pos1 = target1["position_m"]
        assert not np.allclose(pos0, pos1)
        # Speed should remain approximately constant (pure lateral acceleration)
        assert abs(target1["speed_mps"] - speed0) < 5.0
        # Heading should have changed
        assert abs(target1["heading_rad"] - heading0) > 0.01

    def test_bang_bang(self):
        env = SimplePointMassEnv({
            "decision_freq": 5,
            "target_mode": "bang_bang",
            "bang_bang_max_g": 3.0,
            "bang_bang_switch_interval_s": 2.0,
        })
        env.reset()
        headings = []
        for _ in range(30):
            env.step({"nz_cmd": 1.0})
            _, target = env.get_state()
            headings.append(target["heading_rad"])
        # Bang-bang should produce sawtooth-like heading changes
        heading_range = max(headings) - min(headings)
        assert heading_range > 0.1

    def test_barrel_roll(self):
        env = SimplePointMassEnv({
            "decision_freq": 5,
            "target_mode": "barrel_roll",
            "barrel_roll_rate_rad_s": 0.5,
        })
        env.reset()
        _, target0 = env.get_state()
        alt0 = target0["altitude_m"]
        headings = []
        altitudes = []
        for _ in range(50):
            env.step({"nz_cmd": 1.0})
            _, target = env.get_state()
            headings.append(target["heading_rad"])
            altitudes.append(target["altitude_m"])
        # Should show significant heading change (circular motion)
        heading_range = max(headings) - min(headings)
        assert heading_range > 1.0
        # Should show altitude variation
        alt_range = max(altitudes) - min(altitudes)
        assert alt_range > 5.0

    def test_unknown_target_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown target_mode"):
            SimplePointMassEnv({"decision_freq": 5, "target_mode": "hyperspace_jump"})

    def test_backward_compatible_sinusoidal(self):
        """Legacy 'sinusoidal' mode should still work via target_dynamics."""
        env = SimplePointMassEnv({"decision_freq": 5, "target_mode": "sinusoidal"})
        env.reset()
        _, target0 = env.get_state()
        pos0 = target0["position_m"].copy()
        for _ in range(10):
            env.step({"nz_cmd": 1.0})
        _, target1 = env.get_state()
        pos1 = target1["position_m"]
        assert not np.allclose(pos0, pos1)
