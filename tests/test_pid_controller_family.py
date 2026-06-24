"""Tests for the named PID controller family."""

import math
import numpy as np
import pytest

from uav_vpp_guidance.flight_control import (
    BaselinePIDController,
    EnhancedPIDController,
    GainScheduledPIDController,
    HybridPPOPIDAdapter,
    RobustPIDController,
)
from uav_vpp_guidance.flight_control.actuator_interface import JSBSimActuatorInterface
from uav_vpp_guidance.flight_control.command_limiter import (
    DEFAULT_THROTTLE_MAX,
    DEFAULT_THROTTLE_MIN,
    clip_command,
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


def test_robust_pid_controller():
    ctrl = RobustPIDController({})
    out = ctrl.compute_actuator(_dummy_guidance_command(), _dummy_aircraft_state())
    _assert_jsbsim_props(out)


def test_robust_pid_enables_protections():
    ctrl = RobustPIDController({})
    assert ctrl.use_rudder is True
    assert ctrl.use_beta_suppression is True
    assert ctrl.use_aoa_protection is True
    assert ctrl.enable_altitude_hold is True
    # Bank-angle protection is intentionally disabled so lead-turn authority
    # is not restricted during crossing-angle intercepts.
    assert ctrl.enable_bank_angle_protection is False
    assert ctrl.max_bank_rad > math.radians(70.0)


def test_enhanced_pid_nz_anti_windup_holds_integrator_when_saturated():
    ctrl = EnhancedPIDController(
        {
            "alpha_filter": 1.0,
            "Kp_nz": 0.0,
            "Ki_nz": 0.3,
            "Kd_nz": 0.0,
            "nz_to_elevator_gain": 0.0,
            "elevator_max": 0.25,
            "integral_windup_limit": 10.0,
            "enable_dynamic_nz_limit": False,
            "enable_stall_protection": False,
            "enable_altitude_hold": False,
            "enable_speed_hold": False,
            "enable_energy_boost": False,
            "use_aoa_protection": False,
            "use_rudder": False,
        }
    )
    command = {"nz_cmd": 7.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.7}
    state = _dummy_aircraft_state()

    for _ in range(30):
        out = ctrl.compute_actuator(command, state)
        assert out["fcs/elevator-cmd-norm"] == pytest.approx(-0.25)

    assert ctrl._nz_integral == pytest.approx(ctrl.elevator_max / ctrl.Ki_nz)
    assert ctrl._nz_integral < ctrl.integral_windup_limit


def test_protection_arbitration_keeps_stall_limit_above_altitude_and_bank():
    ctrl = EnhancedPIDController(
        {
            "alpha_filter": 1.0,
            "Kp_nz": 0.0,
            "Ki_nz": 0.0,
            "Kd_nz": 0.0,
            "enable_dynamic_nz_limit": True,
            "enable_stall_protection": True,
            "enable_altitude_hold": True,
            "altitude_reference_m": 5000.0,
            "altitude_hold_gain": 0.01,
            "enable_bank_angle_protection": True,
            "max_bank_rad": math.radians(70.0),
            "bank_violation_threshold": 0,
            "bank_protection_nz_increment": 2.0,
            "bank_protection_nz_increment_max": 1.0,
            "enable_speed_hold": False,
            "enable_energy_boost": False,
            "use_aoa_protection": False,
        }
    )
    state = {
        **_dummy_aircraft_state(),
        "roll_rad": math.radians(85.0),
        "speed_mps": 140.0,
        "altitude_m": 800.0,
        "nz_g": 1.0,
    }
    out = ctrl.compute_actuator(
        {"nz_cmd": 7.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.7},
        state,
    )

    record = ctrl.command_history[-1]
    assert record["pid_errors"]["nz_error"] == pytest.approx(0.0)
    assert out["protection_flags"]["stall_limited"] is True
    assert out["protection_flags"]["bank_recovery"] is True
    assert out["fcs/aileron-cmd-norm"] < 0.0


def test_throttle_defaults_are_aligned_to_f16_effective_range():
    high = clip_command(
        {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 1.0},
        {"throttle_min": 0.0, "throttle_max": 1.0},
    )
    low = clip_command(
        {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.0},
        {"throttle_min": 0.0, "throttle_max": 1.0},
    )
    assert high["throttle_cmd"] == pytest.approx(DEFAULT_THROTTLE_MAX)
    assert low["throttle_cmd"] == pytest.approx(DEFAULT_THROTTLE_MIN)

    actuator = JSBSimActuatorInterface({"throttle_min": 0.0, "throttle_max": 1.0})
    actuator_out = actuator.command_to_jsbsim_properties(
        {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 1.0}
    )
    assert actuator_out["fcs/throttle-cmd-norm"] == pytest.approx(DEFAULT_THROTTLE_MAX)
    assert actuator_out["saturation_flag"] is True

    ctrl = EnhancedPIDController(
        {
            "alpha_filter": 1.0,
            "enable_speed_hold": False,
            "enable_energy_boost": False,
            "enable_altitude_hold": False,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        }
    )
    ctrl_out = ctrl.compute_actuator(
        {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 1.0},
        _dummy_aircraft_state(),
    )
    assert ctrl_out["fcs/throttle-cmd-norm"] == pytest.approx(DEFAULT_THROTTLE_MAX)
