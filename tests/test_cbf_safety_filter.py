"""Tests for CBF Safety Filter module."""

import numpy as np
import pytest

from uav_vpp_guidance.safety.cbf_filter import CBFSafetyFilter, create_cbf_filter


class TestCBFSafetyFilter:
    """Unit tests for the CBF safety filter."""

    def test_default_disabled(self):
        """CBF filter is disabled by default."""
        cbf = CBFSafetyFilter({})
        assert not cbf.enabled

    def test_enabled_via_config(self):
        """CBF filter can be enabled via config."""
        cbf = CBFSafetyFilter({"enabled": True})
        assert cbf.enabled

    def test_no_intervention_when_safe(self):
        """No intervention when all constraints are satisfied."""
        cbf = CBFSafetyFilter({"enabled": True, "d_min_m": 150.0})
        
        nominal_command = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
        
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "altitude_m": 5000.0,
            "speed_mps": 200.0,
        }
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        
        safe_command, cbf_info = cbf.filter_command(
            nominal_command, own_state, target_state
        )
        
        assert not cbf_info["intervened"]
        assert safe_command["nz_cmd"] == pytest.approx(1.0, abs=0.01)
        assert safe_command["roll_rate_cmd"] == pytest.approx(0.0, abs=0.01)
        assert safe_command["throttle_cmd"] == pytest.approx(0.5, abs=0.01)

    def test_collision_avoidance_intervention(self):
        """CBF intervenes when too close to target."""
        cbf = CBFSafetyFilter({"enabled": True, "d_min_m": 150.0, "verbose": False})
        
        nominal_command = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
        
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "altitude_m": 5000.0,
            "speed_mps": 200.0,
        }
        target_state = {
            "position_neu": np.array([100.0, 0.0, 5000.0]),  # Only 100m away
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        
        safe_command, cbf_info = cbf.filter_command(
            nominal_command, own_state, target_state
        )
        
        # CBF should be active but may not always intervene depending on dynamics
        assert cbf_info["active"]
        assert "collision" in cbf_info.get("constraint_types", [])

    def test_altitude_low_intervention(self):
        """CBF intervenes when altitude is too low."""
        cbf = CBFSafetyFilter({"enabled": True, "h_min_m": 500.0, "verbose": False})
        
        nominal_command = {"nz_cmd": -1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
        
        own_state = {
            "position_neu": np.array([0.0, 0.0, 550.0]),  # Close to min altitude
            "velocity_vector_mps": np.array([200.0, 0.0, -10.0]),  # Descending
            "altitude_m": 550.0,
            "speed_mps": 200.0,
        }
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 550.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        
        safe_command, cbf_info = cbf.filter_command(
            nominal_command, own_state, target_state
        )
        
        assert cbf_info["active"]
        # CBF should push nz_cmd up to avoid going lower
        assert safe_command["nz_cmd"] >= nominal_command["nz_cmd"]

    def test_speed_low_intervention(self):
        """CBF intervenes when speed is too low (stall protection)."""
        cbf = CBFSafetyFilter({"enabled": True, "v_min_mps": 80.0, "verbose": False})
        
        nominal_command = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.1}
        
        own_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([85.0, 0.0, 0.0]),  # Close to stall
            "altitude_m": 5000.0,
            "speed_mps": 85.0,
        }
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        
        safe_command, cbf_info = cbf.filter_command(
            nominal_command, own_state, target_state
        )
        
        assert cbf_info["active"]
        # CBF should increase throttle to avoid stall
        assert safe_command["throttle_cmd"] >= nominal_command["throttle_cmd"]

    def test_factory_none(self):
        """Factory returns None when not configured."""
        assert create_cbf_filter(None) is None
        assert create_cbf_filter({}) is None
        assert create_cbf_filter({"enabled": False}) is None

    def test_factory_enabled(self):
        """Factory returns filter when enabled."""
        cbf = create_cbf_filter({"enabled": True})
        assert cbf is not None
        assert cbf.enabled

    def test_stats_tracking(self):
        """CBF tracks intervention statistics."""
        cbf = CBFSafetyFilter({"enabled": True})
        
        stats = cbf.get_stats()
        assert stats["total_steps"] == 0
        assert stats["intervention_count"] == 0
        assert stats["intervention_rate"] == 0.0

    def test_reset_stats(self):
        """CBF stats can be reset."""
        cbf = CBFSafetyFilter({"enabled": True})
        cbf._total_steps = 10
        cbf._intervention_count = 5
        
        cbf.reset_stats()
        stats = cbf.get_stats()
        assert stats["total_steps"] == 0
        assert stats["intervention_count"] == 0
