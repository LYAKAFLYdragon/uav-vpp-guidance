"""Unit tests for the maneuver library primitives."""
from __future__ import annotations

import math

import numpy as np
import pytest

from uav_vpp_guidance.maneuver_library import InnerLoopController, ManeuverLibrary
from uav_vpp_guidance.maneuver_library.base import FlightState


@pytest.fixture
def healthy_state():
    """A flight state that satisfies the entry conditions of most maneuvers."""
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


@pytest.fixture
def controller():
    return InnerLoopController()


@pytest.mark.parametrize("name", ManeuverLibrary.list_maneuvers())
def test_maneuver_can_be_created_and_updated(name, healthy_state, controller):
    """Every registered maneuver can be instantiated and produces bounded commands."""
    maneuver = ManeuverLibrary.create(name)
    assert maneuver.name == name

    # Entry condition should be a boolean.
    can_enter = maneuver.can_enter(healthy_state)
    assert isinstance(can_enter, bool)

    # If it can enter, running one update should produce a setpoint that the
    # inner-loop controller can turn into bounded commands.
    if can_enter:
        maneuver.enter(healthy_state)
        setpoint = maneuver.update(healthy_state, dt=0.05)
        cmd = controller.update(healthy_state, setpoint, dt=0.05)
        assert -1.0 <= cmd.elevator <= 1.0
        assert -1.0 <= cmd.aileron <= 1.0
        assert -1.0 <= cmd.rudder <= 1.0
        assert 0.0 <= cmd.throttle <= 1.0


def test_new_dogfight_maneuvers_are_registered():
    """The new dogfight-specific maneuvers are present in the library."""
    names = ManeuverLibrary.list_maneuvers()
    for name in (
        "break_turn",
        "displacement_roll",
        "vertical_scissors",
        "defensive_spiral",
        "extension",
        "jink",
    ):
        assert name in names


def test_break_turn_phases_and_completion():
    """break_turn progresses through entry/execute/exit and completes after rollout."""
    maneuver = ManeuverLibrary.create("break_turn", {"heading_change_deg": 90.0})
    state = FlightState(
        velocity_mps=300.0,
        altitude_m=5000.0,
        psi_rad=0.0,
        phi_rad=0.0,
        theta_rad=0.0,
        nz=1.0,
    )
    assert maneuver.can_enter(state)
    maneuver.enter(state)
    assert maneuver.phase() == "entry"
    assert not maneuver.is_complete(state)

    # Roll in and reach the target heading (within execute phase).
    state.psi_rad = math.radians(95.0)
    maneuver.update(state, dt=2.0)
    assert maneuver.phase() == "execute"
    assert not maneuver.is_complete(state)

    # Let the exit/roll-out phase finish.
    maneuver.update(state, dt=2.0)
    assert maneuver.phase() == "exit"
    assert maneuver.is_complete(state)


def test_maneuver_phase_exposes_internal_state():
    """Maneuvers with internal phase strings expose them through ``phase()``."""
    maneuver = ManeuverLibrary.create("high_yoyo")
    assert maneuver.phase() == "climb"


def test_displacement_roll_has_five_phases():
    """displacement_roll exposes entry -> roll_in -> hold -> roll_out -> exit."""
    maneuver = ManeuverLibrary.create(
        "displacement_roll",
        {"roll_target_deg": 90.0, "roll_rate_dps": 90.0, "hold_time_s": 1.0},
    )
    state = FlightState(velocity_mps=300.0, altitude_m=5000.0)
    assert maneuver.can_enter(state)
    maneuver.enter(state)
    assert maneuver.phase() == "entry"

    # Roll in phase at t=0.5s should be well into the roll.
    sp = maneuver.update(state, dt=0.5)
    assert maneuver.phase() == "roll_in"
    assert sp.phi_ref is not None
    assert abs(sp.phi_ref) > math.radians(20.0)

    # Hold phase at t=1.25s should be at the target bank.
    sp = maneuver.update(state, dt=0.75)
    assert maneuver.phase() == "hold"
    assert abs(sp.phi_ref) == pytest.approx(math.radians(90.0), abs=0.01)

    # Roll out phase at t=2.5s should be decreasing.
    sp = maneuver.update(state, dt=1.25)
    assert maneuver.phase() == "roll_out"
    assert abs(sp.phi_ref) < math.radians(90.0)
