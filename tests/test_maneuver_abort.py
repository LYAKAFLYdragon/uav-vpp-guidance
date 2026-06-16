"""Tests for maneuver envelope/abort guards and executor integration."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_vpp_guidance.maneuver_library import ManeuverLibrary
from uav_vpp_guidance.maneuver_library.base import FlightState, Maneuver, ManeuverSetpoint
from uav_vpp_guidance.maneuver_library.executor import ManeuverExecutor


@pytest.fixture
def healthy_state():
    return FlightState(
        t=0.0,
        position_m=np.array([0.0, 0.0, 5000.0]),
        velocity_mps=300.0,
        altitude_m=5000.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        p_rps=0.0,
        q_rps=0.0,
        r_rps=0.0,
        nz=1.0,
        alpha_rad=0.05,
        beta_rad=0.0,
        mach=0.85,
    )


def test_default_should_abort_respects_altitude_floor(healthy_state):
    maneuver = ManeuverLibrary.create("break_turn")
    assert not maneuver.should_abort(healthy_state)

    low_state = FlightState(**{**healthy_state.__dict__, "altitude_m": 800.0})
    assert maneuver.should_abort(low_state)


def test_default_should_abort_respects_speed_floor(healthy_state):
    maneuver = ManeuverLibrary.create("break_turn")
    slow_state = FlightState(**{**healthy_state.__dict__, "velocity_mps": 80.0})
    assert maneuver.should_abort(slow_state)


def test_abort_guard_hook_is_checked():
    class AbortAlways(Maneuver):
        name = "abort_always"

        def abort_guard(self, state: FlightState) -> bool:
            return True

    class AbortNever(Maneuver):
        name = "abort_never"

        def abort_guard(self, state: FlightState) -> bool:
            return False

    state = FlightState(velocity_mps=300.0, altitude_m=5000.0)
    assert AbortAlways({}).should_abort(state)
    assert not AbortNever({}).should_abort(state)


def test_executor_auto_aborts_on_envelope_violation(healthy_state):
    """The executor transitions to fallback when a maneuver should abort."""
    maneuver = ManeuverLibrary.create("break_turn")
    fallback = ManeuverLibrary.create("straight_level")

    executor = ManeuverExecutor()
    executor.set_fallback(fallback)
    assert executor.select(maneuver, healthy_state)

    # Drop below the maneuver's hard altitude floor.
    low_state = FlightState(**{**healthy_state.__dict__, "altitude_m": 800.0})
    cmd = executor.update(low_state, dt=0.05)
    active = executor.active_maneuver()
    assert active is not None
    assert active.name == fallback.name
    assert active.state.name == "EXECUTE"


def test_telemetry_records_abort_reason(healthy_state):
    from uav_vpp_guidance.maneuver_library.telemetry import ManeuverTelemetry

    telemetry = ManeuverTelemetry()
    maneuver = ManeuverLibrary.create("break_turn")
    fallback = ManeuverLibrary.create("straight_level")

    executor = ManeuverExecutor(telemetry=telemetry)
    executor.set_fallback(fallback)
    executor.select(maneuver, healthy_state)

    low_state = FlightState(**{**healthy_state.__dict__, "altitude_m": 800.0})
    executor.update(low_state, dt=0.05)

    assert telemetry.abort_count == 1
    record = telemetry.records[0]
    assert record.reason == "abort"
    assert record.abort_reason == "envelope"
