"""
P1 migration tests for JSBSimEnv and CloseRangeTrackingEnv.

These tests verify that the migrated minimal closed-loop structures
initialize correctly and expose the expected interfaces.
"""

import os

import numpy as np
import pytest
from uav_vpp_guidance.envs.jsbsim_env import (
    _JSBSimAircraft,
    lla2neu,
    neu2lla,
    JSBSimEnv,
)


class _DummyJSBSimExec:
    def get_sim_time(self):
        return float("nan")


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

    def test_update_state_sanitizes_non_finite_values(self, caplog):
        aircraft = object.__new__(_JSBSimAircraft)
        aircraft.uid = "own"
        aircraft.lon0 = 120.0
        aircraft.lat0 = 60.0
        aircraft.alt0 = 0.0
        aircraft.origin = (aircraft.lon0, aircraft.lat0, aircraft.alt0)
        aircraft.jsbsim_exec = _DummyJSBSimExec()
        aircraft._state = {
            "position_neu": np.array([10.0, 20.0, 3000.0], dtype=np.float64),
            "position_m": np.array([10.0, 20.0, 3000.0], dtype=np.float64),
            "position_lla": np.array([120.1, 60.1, 3000.0], dtype=np.float64),
            "altitude_m": 3000.0,
            "attitude_rpy": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "velocity_ned": np.array([200.0, 0.0, 0.0], dtype=np.float64),
            "velocity_vector_mps": np.array([200.0, 0.0, -0.0], dtype=np.float64),
            "velocity_body": np.array([200.0, 0.0, 0.0], dtype=np.float64),
            "body_rates_rps": np.zeros(3, dtype=np.float64),
            "roll_rad": 0.1,
            "pitch_rad": 0.2,
            "yaw_rad": 0.3,
            "p_rps": 0.0,
            "q_rps": 0.0,
            "r_rps": 0.0,
            "nz_g": 0.9,
            "beta_rad": 0.0,
            "sideslip_rad": 0.0,
            "speed_mps": 200.0,
            "vt_mps": 200.0,
            "sim_time": 12.0,
        }
        values = {
            "position/long-gc-deg": float("nan"),
            "position/lat-geod-deg": 60.0,
            "position/h-sl-ft": float("inf"),
            "attitude/roll-rad": 0.0,
            "attitude/pitch-rad": 0.0,
            "attitude/heading-true-rad": 0.0,
            "velocities/v-north-fps": 700.0,
            "velocities/v-east-fps": 0.0,
            "velocities/v-down-fps": 0.0,
            "velocities/vt-fps": float("nan"),
            "velocities/u-fps": float("nan"),
            "velocities/v-fps": 0.0,
            "velocities/w-fps": 0.0,
            "velocities/p-rad_sec": 0.0,
            "velocities/q-rad_sec": 0.0,
            "velocities/r-rad_sec": 0.0,
            "accelerations/Nz": float("inf"),
        }
        aircraft.get_property_value = values.__getitem__

        with caplog.at_level("WARNING"):
            aircraft._update_state()

        state = aircraft.get_state()
        assert np.isfinite(state["position_neu"]).all()
        assert np.isfinite(state["attitude_rpy"]).all()
        np.testing.assert_allclose(state["position_neu"], [10.0, 20.0, 3000.0])
        assert state["altitude_m"] == pytest.approx(3000.0)
        assert state["speed_mps"] == pytest.approx(200.0)
        assert state["nz_g"] == pytest.approx(0.9)
        assert any("Non-finite JSBSim state" in rec.message for rec in caplog.records)

    def test_init_reads_config(self):
        root = os.environ.get("JSBSIM_ROOT", "")
        if not root:
            pytest.skip("JSBSIM_ROOT not set")
        config = {
            "sim_freq": 60,
            "legacy_project_root": root,
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
        root = os.environ.get("JSBSIM_ROOT", "")
        if not root:
            pytest.skip("JSBSIM_ROOT not set")
        config = {
            "sim_freq": 60,
            "legacy_project_root": root,
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
        root = os.environ.get("JSBSIM_ROOT", "")
        if not root:
            pytest.skip("JSBSIM_ROOT not set")
        config = {
            "sim_freq": 60,
            "legacy_project_root": root,
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

        root = os.environ.get("JSBSIM_ROOT", "")
        if not root:
            pytest.skip("JSBSIM_ROOT not set")
        config = {
            "env": {
                "sim_freq": 60,
                "decision_freq": 5,
                "max_high_level_steps": 512,
                "aircraft_model": "f16",
                "legacy_project_root": root,
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
                "legacy_project_root": os.environ.get("JSBSIM_ROOT", ""),
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
