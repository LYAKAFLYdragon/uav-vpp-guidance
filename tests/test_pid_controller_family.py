"""Tests for the named PID controller family."""

import numpy as np

from uav_vpp_guidance.flight_control import (
    BaselinePIDController,
    EnhancedPIDController,
    GainScheduledPIDController,
    HybridPPOPIDAdapter,
)


def _dummy_guidance_command():
    return {"nz_cmd": 2.0, "roll_rate_cmd": 0.1, "throttle_cmd": 0.7}


def _dummy_aircraft_state():
    return {
        "nz_g": 1.0,
        "p_rps": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "alpha_rad": 0.05,
        "beta_rad": 0.0,
        "speed_mps": 250.0,
        "altitude_m": 5000.0,
    }


def _assert_jsbsim_props(output):
    for key in ["fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"]:
        assert key in output
        assert np.isfinite(output[key])


def test_baseline_pid_controller():
    ctrl = BaselinePIDController({})
    out = ctrl.compute_actuator(_dummy_guidance_command(), _dummy_aircraft_state())
    _assert_jsbsim_props(out)


def test_enhanced_pid_controller():
    ctrl = EnhancedPIDController({})
    out = ctrl.compute_actuator(_dummy_guidance_command(), _dummy_aircraft_state())
    _assert_jsbsim_props(out)


def test_gain_scheduled_pid_controller():
    ctrl = GainScheduledPIDController({})
    out = ctrl.compute_actuator(_dummy_guidance_command(), _dummy_aircraft_state())
    _assert_jsbsim_props(out)


def test_hybrid_ppo_pid_adapter():
    ctrl = HybridPPOPIDAdapter({})
    out = ctrl.compute_actuator(
        _dummy_guidance_command(), _dummy_aircraft_state(), aggressiveness=0.5
    )
    _assert_jsbsim_props(out)
    assert ctrl.config["gain_schedule"]["source"] == "aggressiveness"


def test_baseline_disables_advanced_features():
    ctrl = BaselinePIDController({})
    assert ctrl.use_pid is True
    assert ctrl.use_rudder is False
    assert ctrl.use_aoa_protection is False
