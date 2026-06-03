"""
Unit tests for LOS-rate guidance.
"""

import math

import numpy as np
import pytest

from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance, _normalize_angle
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


class TestLOSRateGuidance:
    def test_init(self):
        guidance = LOSRateGuidance(config={})
        assert guidance.prev_command is None

    def test_init_with_config_params(self):
        config = {
            "gains": {"k_los": 2.0, "k_pos": 0.3},
            "params": {
                "distance_scale_m": 1000.0,
                "target_speed_mps": 200.0,
                "speed_error_scale_mps": 50.0,
                "base_throttle": 0.6,
                "epsilon": 1.0e-4,
                "base_nz": 1.0,
            },
        }
        guidance = LOSRateGuidance(config=config)
        assert guidance.k_los == 2.0
        assert guidance.k_pos == 0.3
        assert guidance.distance_scale_m == 1000.0
        assert guidance.target_speed_mps == 200.0
        assert guidance.epsilon == 1.0e-4

    def test_init_invalid_params(self):
        with pytest.raises(ValueError, match="distance_scale_m must be positive"):
            LOSRateGuidance(config={"params": {"distance_scale_m": 0.0}})
        with pytest.raises(ValueError, match="speed_error_scale_mps must be positive"):
            LOSRateGuidance(config={"params": {"speed_error_scale_mps": -1.0}})
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LOSRateGuidance(config={"params": {"epsilon": 0.0}})

    def test_reset(self):
        guidance = LOSRateGuidance(config={})
        guidance.prev_command = {"nz_cmd": 1.0}
        guidance.reset()
        assert guidance.prev_command is None

    def test_compute_command_basic(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains()
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, target_state, virtual_point, gains)
        assert isinstance(cmd, dict)
        assert "nz_cmd" in cmd
        assert "roll_rate_cmd" in cmd
        assert "throttle_cmd" in cmd
        assert -2.0 <= cmd["nz_cmd"] <= 7.0
        assert -1.5 <= cmd["roll_rate_cmd"] <= 1.5
        assert 0.0 <= cmd["throttle_cmd"] <= 1.0

    def test_compute_command_heading_error(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains(k_roll=2.0)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([0.0, 1000.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point, gains)
        # VP is to the right (east), so positive heading error -> positive roll_rate
        assert cmd["roll_rate_cmd"] > 0.1

    def test_compute_command_elevation_error(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains(k_los=2.0)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 6000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point, gains)
        # VP is above, so positive elevation -> nz_cmd > 1.0
        assert cmd["nz_cmd"] > 1.0

    def test_compute_command_zero_distance(self):
        """When own_state == virtual_point, should not crash."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([0.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    def test_compute_command_zero_speed(self):
        """When own_speed is zero, heading defaults to 0.0 and should not crash."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    def test_compute_command_missing_position_raises(self):
        guidance = LOSRateGuidance(config={})
        own_state = {"velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        with pytest.raises(ValueError, match="missing position field"):
            guidance.compute_command(own_state, None, virtual_point)

    def test_compute_command_missing_velocity_raises(self):
        guidance = LOSRateGuidance(config={})
        own_state = {"position_m": np.array([0.0, 0.0, 5000.0])}
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        with pytest.raises(ValueError, match="missing velocity field"):
            guidance.compute_command(own_state, None, virtual_point)

    def test_compute_command_invalid_state_type_raises(self):
        guidance = LOSRateGuidance(config={})
        with pytest.raises(TypeError, match="must be a dict"):
            guidance.compute_command("not_a_dict", None, {"position_m": np.zeros(3)})

    def test_compute_command_with_nan_velocity(self):
        """NaN in velocity should propagate to finite commands if speed is below epsilon."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([np.nan, np.nan, np.nan]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        # Speed = NaN, so heading defaults to 0.0; commands should still be finite
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    def test_normalize_angle_standard(self):
        assert _normalize_angle(0.0) == pytest.approx(0.0, abs=1e-9)
        assert _normalize_angle(math.pi) == pytest.approx(math.pi, abs=1e-9)
        # Python modulo returns non-negative, so -pi maps to pi (same direction)
        assert _normalize_angle(-math.pi) == pytest.approx(math.pi, abs=1e-9)
        assert _normalize_angle(3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9)
        assert _normalize_angle(-3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9)

    def test_normalize_angle_nan(self):
        """NaN should pass through without crashing."""
        result = _normalize_angle(np.nan)
        assert np.isnan(result)

    def test_config_from_guidance_yaml(self):
        """Ensure guidance.yaml config loads correctly with new params."""
        from uav_vpp_guidance.utils.config import load_yaml_config
        config = load_yaml_config("config/guidance.yaml")
        assert "guidance" in config
        assert "params" in config["guidance"]
        params = config["guidance"]["params"]
        assert "distance_scale_m" in params
        assert "target_speed_mps" in params
        assert "epsilon" in params
