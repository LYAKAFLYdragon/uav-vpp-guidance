"""Tests for the tactical maneuver library."""
import math

import numpy as np
import pytest

from uav_vpp_guidance.maneuver_library import (
    FlightState,
    InnerLoopController,
    ManeuverExecutor,
    ManeuverLibrary,
)


@pytest.fixture
def cruise_state():
    return FlightState(
        t=0.0,
        position_m=np.zeros(3),
        velocity_mps=250.0,
        altitude_m=5000.0,
        phi_rad=0.0,
        theta_rad=0.0,
        psi_rad=0.0,
        p_rps=0.0,
        q_rps=0.0,
        r_rps=0.0,
        nz=1.0,
    )


def test_library_registry():
    names = ManeuverLibrary.list_maneuvers()
    expected = [
        "barrel_roll",
        "coordinated_turn",
        "dive",
        "high_yoyo",
        "loop",
        "low_yoyo",
        "straight_level",
    ]
    assert set(expected).issubset(set(names))


def test_straight_level_setpoint(cruise_state):
    m = ManeuverLibrary.create("straight_level", {"duration_s": 2.0})
    assert m.can_enter(cruise_state)
    m.enter(cruise_state)
    sp = m.update(cruise_state, 0.05)
    assert sp.phi_ref == pytest.approx(0.0)
    assert sp.nz_ref == pytest.approx(1.0)
    assert sp.velocity_ref == pytest.approx(250.0)


def test_coordinated_turn_setpoint(cruise_state):
    m = ManeuverLibrary.create(
        "coordinated_turn",
        {"heading_change_deg": 90.0, "bank_angle_deg": 30.0},
    )
    assert m.can_enter(cruise_state)
    m.enter(cruise_state)
    sp = m.update(cruise_state, 0.05)
    assert sp.phi_ref == pytest.approx(math.radians(30.0))
    assert sp.nz_ref == pytest.approx(1.0 / math.cos(math.radians(30.0)))


def test_dive_completion(cruise_state):
    m = ManeuverLibrary.create("dive", {"target_altitude_m": 3000.0})
    m.enter(cruise_state)
    assert not m.is_complete(cruise_state)
    end_state = FlightState(
        t=10.0,
        altitude_m=2900.0,
        velocity_mps=300.0,
    )
    assert m.is_complete(end_state)


def test_controller_outputs_bounded(cruise_state):
    ctrl = InnerLoopController()
    from uav_vpp_guidance.maneuver_library.maneuvers.coordinated_turn import CoordinatedTurn

    m = CoordinatedTurn({"heading_change_deg": 90.0, "bank_angle_deg": 30.0})
    m.enter(cruise_state)
    sp = m.update(cruise_state, 0.05)
    cmd = ctrl.update(cruise_state, sp, 0.05)
    assert -1.0 <= cmd.elevator <= 1.0
    assert -1.0 <= cmd.aileron <= 1.0
    assert -1.0 <= cmd.rudder <= 1.0
    assert 0.0 <= cmd.throttle <= 1.0


def test_executor_runs_maneuver(cruise_state):
    from uav_vpp_guidance.maneuver_library.maneuvers.straight_level import StraightLevel

    executor = ManeuverExecutor()
    executor.select(StraightLevel({"duration_s": 0.2}), cruise_state)
    for i in range(10):
        state = FlightState(
            t=i * 0.05,
            velocity_mps=250.0,
            altitude_m=5000.0,
        )
        cmd = executor.update(state, 0.05)
        assert -1.0 <= cmd.aileron <= 1.0


def test_new_maneuvers_register():
    names = ManeuverLibrary.list_maneuvers()
    for name in ["scissors", "split_s", "immelmann"]:
        assert name in names
        m = ManeuverLibrary.create(name)
        assert m.name == name


def test_split_s_phases():
    from uav_vpp_guidance.maneuver_library.maneuvers.split_s import SplitS

    state = FlightState(velocity_mps=350.0, altitude_m=8000.0)
    m = SplitS({"entry_speed_mps": 350.0, "nz_pull": 5.0})
    assert m.can_enter(state)
    m.enter(state)
    assert m._phase == "roll"
    # Roll phase commands inverted wings.
    sp = m.update(state, 0.05)
    assert sp.phi_ref == pytest.approx(math.pi)


def test_immelmann_phases():
    from uav_vpp_guidance.maneuver_library.maneuvers.immelmann import Immelmann

    state = FlightState(velocity_mps=350.0, altitude_m=5000.0)
    m = Immelmann({"entry_speed_mps": 350.0, "nz_pull": 5.0})
    assert m.can_enter(state)
    m.enter(state)
    assert m._phase == "pull"
    sp = m.update(state, 0.05)
    assert sp.q_ref is not None


def test_scissors_reversal(cruise_state):
    from uav_vpp_guidance.maneuver_library.maneuvers.scissors import Scissors

    m = Scissors({"cycles": 1, "turn_angle_deg": 90.0})
    m.enter(cruise_state)
    sp = m.update(cruise_state, 0.05)
    assert abs(sp.phi_ref) == pytest.approx(math.radians(60.0))
    assert sp.throttle_ref == pytest.approx(0.6)
