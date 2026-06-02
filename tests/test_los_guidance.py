"""
Unit tests for LOS-rate guidance.
"""

import pytest
import numpy as np
from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


class TestLOSRateGuidance:
    def test_init(self):
        guidance = LOSRateGuidance(config={})
        assert guidance.prev_command is None

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
        cmd = guidance.compute_command(own_state, {}, virtual_point, gains)
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
        cmd = guidance.compute_command(own_state, {}, virtual_point, gains)
        # VP is above, so positive elevation -> nz_cmd > 1.0
        assert cmd["nz_cmd"] > 1.0
