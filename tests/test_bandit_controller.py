"""Tests for the bandit maneuver controller and selector (no JSBSim required)."""
import math

import numpy as np
import pytest

from uav_vpp_guidance.envs.bandit_controller import (
    BanditManeuverController,
    build_bandit_flight_state,
)
from uav_vpp_guidance.envs.bandit_selector import BanditManeuverSelector
from uav_vpp_guidance.envs.situation_evaluator import SituationEvaluator


@pytest.fixture
def own_state():
    return {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "altitude_m": 5000.0,
        "speed_mps": 200.0,
    }


@pytest.fixture
def bandit_state():
    return {
        "position_m": np.array([2000.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([-200.0, 0.0, 0.0]),
        "altitude_m": 5000.0,
        "speed_mps": 200.0,
        "sim_time": 0.0,
        "attitude_rpy": np.array([0.0, 0.0, math.radians(180.0)]),
        "body_rates_rps": np.zeros(3),
        "nz_g": 1.0,
    }


@pytest.fixture
def bandit_flight_state(bandit_state):
    return build_bandit_flight_state(bandit_state, 0.0, 0.0, 0.8)


def test_situation_evaluator_computes_geometry(own_state, bandit_state):
    eval_ = SituationEvaluator()
    sit = eval_.evaluate(own_state, bandit_state)
    assert "range_m" in sit
    assert "bandit_on_own_tail" in sit
    assert "weapon_zone" in sit


def test_selector_returns_allowed_maneuver(own_state, bandit_state):
    selector = BanditManeuverSelector({"difficulty": "easy"})
    maneuver = selector.select_initial(own_state, bandit_state, sim_time=0.0)
    assert maneuver is not None
    assert maneuver.name in selector.allowed_maneuvers


def test_controller_outputs_bounded_commands(own_state, bandit_state, bandit_flight_state):
    ctrl = BanditManeuverController({"difficulty": "easy"})
    ctrl.reset(own_state, bandit_state, bandit_flight_state, sim_time=0.0)
    cmd, info = ctrl.update(own_state, bandit_state, bandit_flight_state, sim_time=0.05, dt=0.05)
    assert -1.0 <= cmd.elevator <= 1.0
    assert -1.0 <= cmd.aileron <= 1.0
    assert -1.0 <= cmd.rudder <= 1.0
    assert 0.0 <= cmd.throttle <= 1.0
    assert "bandit_maneuver" in info


def test_build_flight_state_converts_fields(bandit_state):
    fs = build_bandit_flight_state(bandit_state, 0.1, -0.05, 0.8)
    assert fs.altitude_m == pytest.approx(5000.0)
    assert fs.velocity_mps == pytest.approx(200.0)
    assert fs.alpha_rad == pytest.approx(0.1)
    assert fs.beta_rad == pytest.approx(-0.05)
    assert fs.mach == pytest.approx(0.8)
