"""
Tests for ActuatorDynamics command pipeline layer.
"""

import numpy as np
import pytest

from uav_vpp_guidance.flight_control.actuator_dynamics import ActuatorDynamics


class TestActuatorDynamics:
    def test_disabled_passes_command_through(self):
        dyn = ActuatorDynamics({"enabled": False}, dt=0.2)
        cmd = {"nz_cmd": 3.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.7}
        out = dyn.step(cmd)
        assert out == pytest.approx(cmd)

    def test_saturation_when_enabled(self):
        limits = {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        }
        dyn = ActuatorDynamics({"enabled": True, "limits": limits}, dt=0.2)
        out = dyn.step({"nz_cmd": 10.0, "roll_rate_cmd": -2.0, "throttle_cmd": 1.5})
        assert out["nz_cmd"] == pytest.approx(7.0)
        assert out["roll_rate_cmd"] == pytest.approx(-1.5)
        assert out["throttle_cmd"] == pytest.approx(1.0)

    def test_first_order_lag_step_response(self):
        # tau=0.2, dt=0.2 -> alpha=0.5. First step initializes state to input;
        # subsequent steps converge toward the input.
        dyn = ActuatorDynamics(
            {"enabled": True, "tau_s": {"nz_cmd": 0.2}}, dt=0.2
        )
        out = dyn.step({"nz_cmd": 2.0})
        assert out["nz_cmd"] == pytest.approx(2.0, abs=1e-6)
        out2 = dyn.step({"nz_cmd": 4.0})
        # y = 2 + 0.5 * (4 - 2) = 3.0
        assert out2["nz_cmd"] == pytest.approx(3.0, abs=1e-6)
        out3 = dyn.step({"nz_cmd": 4.0})
        # y = 3 + 0.5 * (4 - 3) = 3.5
        assert out3["nz_cmd"] == pytest.approx(3.5, abs=1e-6)

    def test_delay_steps(self):
        dyn = ActuatorDynamics(
            {"enabled": True, "delay_steps": 2}, dt=0.2
        )
        # Buffer not full yet -> pass through
        assert dyn.step({"nz_cmd": 1.0})["nz_cmd"] == pytest.approx(1.0)
        assert dyn.step({"nz_cmd": 2.0})["nz_cmd"] == pytest.approx(1.0)
        # Buffer full -> output is value from 2 steps ago
        assert dyn.step({"nz_cmd": 3.0})["nz_cmd"] == pytest.approx(1.0)
        assert dyn.step({"nz_cmd": 4.0})["nz_cmd"] == pytest.approx(2.0)

    def test_rate_limit(self):
        dyn = ActuatorDynamics(
            {"enabled": True, "rate_limit_per_s": {"nz_cmd": 10.0}}, dt=0.2
        )
        out = dyn.step({"nz_cmd": 0.0})
        assert out["nz_cmd"] == pytest.approx(0.0)
        # Max delta per step = 10 * 0.2 = 2.0
        out = dyn.step({"nz_cmd": 5.0})
        assert out["nz_cmd"] == pytest.approx(2.0)
        out = dyn.step({"nz_cmd": 5.0})
        assert out["nz_cmd"] == pytest.approx(4.0)
        out = dyn.step({"nz_cmd": 5.0})
        assert out["nz_cmd"] == pytest.approx(5.0)

    def test_reset_clears_state(self):
        dyn = ActuatorDynamics(
            {"enabled": True, "tau_s": {"nz_cmd": 0.2}}, dt=0.2
        )
        dyn.step({"nz_cmd": 2.0})
        dyn.step({"nz_cmd": 4.0})
        dyn.reset()
        out = dyn.step({"nz_cmd": 2.0})
        # After reset the lag state is gone, so output equals input again
        assert out["nz_cmd"] == pytest.approx(2.0, abs=1e-6)

    def test_multi_channel_independence(self):
        dyn = ActuatorDynamics(
            {"enabled": True, "tau_s": {"nz_cmd": 0.2}}, dt=0.2
        )
        out = dyn.step({"nz_cmd": 2.0, "roll_rate_cmd": 1.0})
        # First step initializes nz lag state to input
        assert out["nz_cmd"] == pytest.approx(2.0)
        # roll_rate_cmd has no tau -> passes through
        assert out["roll_rate_cmd"] == pytest.approx(1.0)
