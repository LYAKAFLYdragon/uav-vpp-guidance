"""
P1 migration tests for JSBSimEnv and CloseRangeTrackingEnv.

These tests verify that the migrated minimal closed-loop structures
initialize correctly and expose the expected interfaces.
"""

import pytest
import numpy as np
from uav_vpp_guidance.envs.jsbsim_env import lla2neu, neu2lla, JSBSimEnv, _JSBSimAircraft


class TestCoordinateConversion:
    """Tests for migrated LLA<->NEU conversion utilities."""

    def test_functions_exist(self):
        assert callable(lla2neu)
        assert callable(neu2lla)

    def test_roundtrip_with_pymap3d(self):
        """If pymap3d is available, verify round-trip conversion."""
        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        lon0, lat0, alt0 = 120.0, 60.0, 0.0
        # Small offset from origin
        n, e, u = 1000.0, 500.0, 200.0
        lon, lat, alt = neu2lla(n, e, u, lon0, lat0, alt0)
        n2, e2, u2 = lla2neu(lon, lat, alt, lon0, lat0, alt0)

        assert n2 == pytest.approx(n, abs=1e-6)
        assert e2 == pytest.approx(e, abs=1e-6)
        assert u2 == pytest.approx(u, abs=1e-6)


class TestJSBSimEnv:
    """Tests for migrated JSBSimEnv."""

    def test_init_reads_config(self):
        config = {
            "sim_freq": 60,
            "legacy_project_root": "E:/CloseAirCombat_control",
            "origin": (120.0, 60.0, 0.0),
        }
        env = JSBSimEnv(config)
        assert env.sim_freq == 60
        assert env.dt == pytest.approx(1 / 60, abs=1e-9)

    def test_add_and_reset_aircraft(self):
        """Verify that aircraft can be added and reset without crashing.

        This test requires a valid JSBSim data directory at the legacy root.
        If JSBSim fails to load, the test is skipped.
        """
        config = {
            "sim_freq": 60,
            "legacy_project_root": "E:/CloseAirCombat_control",
        }
        env = JSBSimEnv(config)
        env.add_aircraft("own", {"model": "f16"})

        try:
            states = env.reset({"own": {}})
        except RuntimeError as exc:
            if "JSBSim data directory not found" in str(exc):
                pytest.skip("Legacy JSBSim data directory not available")
            raise

        assert "own" in states
        assert "position_neu" in states["own"]
        assert "attitude_rpy" in states["own"]
        env.close()

    def test_step_runs_simulation(self):
        """Verify that step() advances simulation time."""
        config = {
            "sim_freq": 60,
            "legacy_project_root": "E:/CloseAirCombat_control",
        }
        env = JSBSimEnv(config)
        env.add_aircraft("own", {"model": "f16"})

        try:
            env.reset({"own": {}})
            t0 = env.get_state()["own"]["sim_time"]
            env.step()
            t1 = env.get_state()["own"]["sim_time"]
        except RuntimeError as exc:
            if "JSBSim data directory not found" in str(exc):
                pytest.skip("Legacy JSBSim data directory not available")
            raise

        assert t1 > t0
        env.close()


class TestCloseRangeTrackingEnv:
    """Tests for migrated CloseRangeTrackingEnv."""

    def test_init(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config = {
            "env": {
                "sim_freq": 60,
                "decision_freq": 5,
                "max_high_level_steps": 512,
                "aircraft_model": "f16",
                "legacy_project_root": "E:/CloseAirCombat_control",
            }
        }
        env = CloseRangeTrackingEnv(config)
        assert env.own_uid == "own"
        assert env.target_uid == "target"
        assert env._sim_steps_per_decision == 12
        env.close()

    def test_reset_and_step_minimal(self):
        """Verify minimal reset/step cycle.

        Skips if legacy JSBSim data is unavailable or pymap3d is missing.
        """
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        try:
            import pymap3d  # noqa: F401
        except ImportError:
            pytest.skip("pymap3d not installed")

        config = {
            "env": {
                "sim_freq": 60,
                "decision_freq": 5,
                "max_high_level_steps": 512,
                "aircraft_model": "f16",
                "legacy_project_root": "E:/CloseAirCombat_control",
            }
        }
        env = CloseRangeTrackingEnv(config)

        try:
            obs = env.reset()
            obs2, reward, terminated, truncated, info = env.step()
        except RuntimeError as exc:
            if "JSBSim data directory not found" in str(exc):
                pytest.skip("Legacy JSBSim data directory not available")
            raise

        assert "own_state" in obs2
        assert "target_state" in obs2
        assert "relative_state" in obs2
        assert "observation_vector" in obs2
        assert isinstance(reward, float)
        assert terminated is False
        assert isinstance(truncated, bool)
        env.close()
