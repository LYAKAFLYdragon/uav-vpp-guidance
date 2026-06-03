"""Tests for ProportionalNavigationGuidance."""

import math

import numpy as np
import pytest

from uav_vpp_guidance.guidance.proportional_navigation import (
    ProportionalNavigationGuidance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_pn() -> ProportionalNavigationGuidance:
    return ProportionalNavigationGuidance()


@pytest.fixture
def own_state() -> dict:
    return {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }


@pytest.fixture
def virtual_point() -> dict:
    return {
        "position_m": np.array([1000.0, 0.0, 5000.0]),
    }


# ---------------------------------------------------------------------------
# Constructor / config
# ---------------------------------------------------------------------------


def test_default_config(default_pn):
    assert default_pn.navigation_constant == pytest.approx(3.0)
    assert default_pn.los_rate_filter_alpha == pytest.approx(0.3)
    assert default_pn.max_accel_mps2 == pytest.approx(100.0)
    assert default_pn.epsilon == pytest.approx(1.0e-6)


def test_custom_config():
    pn = ProportionalNavigationGuidance(
        {
            "gains": {"k_roll": 2.0, "k_speed": 0.5},
            "params": {
                "navigation_constant": 4.0,
                "los_rate_filter_alpha": 0.5,
                "max_accel_mps2": 150.0,
                "epsilon": 1.0e-4,
            },
        }
    )
    assert pn.navigation_constant == pytest.approx(4.0)
    assert pn.los_rate_filter_alpha == pytest.approx(0.5)
    assert pn.k_roll == pytest.approx(2.0)
    assert pn.epsilon == pytest.approx(1.0e-4)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_state(default_pn, own_state, virtual_point):
    default_pn.compute_command(own_state, None, virtual_point)
    default_pn.reset()
    assert default_pn._prev_los_vec is None
    assert default_pn._filtered_los_rate is None


# ---------------------------------------------------------------------------
# Basic command computation
# ---------------------------------------------------------------------------


def test_compute_command_structure(default_pn, own_state, virtual_point):
    cmd = default_pn.compute_command(own_state, None, virtual_point)
    assert set(cmd.keys()) == {"nz_cmd", "roll_rate_cmd", "throttle_cmd"}
    for v in cmd.values():
        assert isinstance(v, float)
        assert math.isfinite(v)


def test_head_on_zero_error(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    cmd = default_pn.compute_command(own, None, vp)
    # Head-on, no lateral error -> small roll_rate_cmd
    assert abs(cmd["roll_rate_cmd"]) < 0.5
    # Throttle should be reasonable
    assert 0.0 <= cmd["throttle_cmd"] <= 1.0


def test_lateral_offset_generates_roll(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 500.0, 5000.0])}
    cmd = default_pn.compute_command(own, None, vp)
    # Positive lateral offset should produce positive roll rate
    assert cmd["roll_rate_cmd"] > 0.0


def test_vertical_offset_generates_nz(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 0.0, 5500.0])}
    # Warm up LOS rate filter with a slightly different VP first
    default_pn.compute_command(
        own, None, {"position_m": np.array([900.0, 0.0, 5400.0])}
    )
    cmd = default_pn.compute_command(own, None, vp)
    # Higher VP should increase nz
    assert cmd["nz_cmd"] > 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_distance(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([0.0, 0.0, 5000.0])}
    cmd = default_pn.compute_command(own, None, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


def test_zero_velocity(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    cmd = default_pn.compute_command(own, None, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


def test_backward_velocity(default_pn):
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([-200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    cmd = default_pn.compute_command(own, None, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


def test_multiple_calls_filter_state(default_pn, own_state, virtual_point):
    # First call initializes filter
    cmd1 = default_pn.compute_command(own_state, None, virtual_point)
    # Second call with same state should produce similar but not identical cmd
    cmd2 = default_pn.compute_command(own_state, None, virtual_point)
    assert cmd1.keys() == cmd2.keys()
    for k in cmd1:
        assert math.isfinite(cmd2[k])


# ---------------------------------------------------------------------------
# Gain override
# ---------------------------------------------------------------------------


def test_gain_override():
    pn = ProportionalNavigationGuidance(
        {
            "gains": {"k_roll": 1.0, "k_speed": 0.2},
        }
    )

    class FakeGains:
        k_roll = 3.0
        k_speed = 0.8

    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 500.0, 5000.0])}
    cmd_default = pn.compute_command(own, None, vp)
    cmd_override = pn.compute_command(own, None, vp, gains=FakeGains())
    # Higher k_roll should produce larger roll_rate_cmd magnitude
    assert abs(cmd_override["roll_rate_cmd"]) > abs(cmd_default["roll_rate_cmd"])
