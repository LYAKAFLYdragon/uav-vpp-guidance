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
    assert default_pn.dt == pytest.approx(0.2)


def test_custom_config():
    pn = ProportionalNavigationGuidance(
        {
            "gains": {"k_roll": 2.0, "k_speed": 0.5},
            "params": {
                "navigation_constant": 4.0,
                "los_rate_filter_alpha": 0.5,
                "max_accel_mps2": 150.0,
                "epsilon": 1.0e-4,
                "dt": 0.1,
            },
        }
    )
    assert pn.navigation_constant == pytest.approx(4.0)
    assert pn.los_rate_filter_alpha == pytest.approx(0.5)
    assert pn.k_roll == pytest.approx(2.0)
    assert pn.epsilon == pytest.approx(1.0e-4)
    assert pn.dt == pytest.approx(0.1)


def test_invalid_dt_raises():
    with pytest.raises(ValueError, match="dt must be positive"):
        ProportionalNavigationGuidance({"params": {"dt": 0.0}})


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
# LOS rate dt scaling
# ---------------------------------------------------------------------------


def test_los_rate_scales_with_dt():
    """Smaller dt should produce larger LOS rate and more aggressive commands."""
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp1 = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    vp2 = {"position_m": np.array([1100.0, 50.0, 5050.0])}

    pn_fast = ProportionalNavigationGuidance({"params": {"dt": 0.05}})
    pn_slow = ProportionalNavigationGuidance({"params": {"dt": 0.4}})

    # Warm up both filters
    pn_fast.compute_command(own, None, vp1)
    pn_slow.compute_command(own, None, vp1)

    cmd_fast = pn_fast.compute_command(own, None, vp2)
    cmd_slow = pn_slow.compute_command(own, None, vp2)

    # Faster dt -> larger LOS rate -> larger acceleration -> higher nz
    assert cmd_fast["nz_cmd"] > cmd_slow["nz_cmd"]


def test_dt_epsilon_protection():
    """Very small dt should not explode due to epsilon protection."""
    pn = ProportionalNavigationGuidance({"params": {"dt": 1e-12, "epsilon": 1e-9}})
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    pn.compute_command(own, None, vp)
    cmd = pn.compute_command(own, None, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


def test_relative_velocity_uses_target_when_available():
    """PN should use target velocity for closing velocity when provided."""
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    target = {
        "position_m": np.array([2000.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([100.0, 0.0, 0.0]),
    }
    vp = {"position_m": np.array([2000.0, 0.0, 5000.0])}
    pn = ProportionalNavigationGuidance()
    # Warm filter
    pn.compute_command(own, target, vp)
    cmd = pn.compute_command(own, target, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


def test_relative_velocity_fallback_when_target_has_no_velocity():
    """PN should fallback to -own_vel when target lacks velocity field."""
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    target = {"position_m": np.array([2000.0, 0.0, 5000.0])}
    vp = {"position_m": np.array([2000.0, 0.0, 5000.0])}
    pn = ProportionalNavigationGuidance()
    pn.compute_command(own, target, vp)
    cmd = pn.compute_command(own, target, vp)
    assert math.isfinite(cmd["nz_cmd"])
    assert math.isfinite(cmd["roll_rate_cmd"])


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


# ---------------------------------------------------------------------------
# LOS-rate boundary jump bug (Stage 6H.3-B)
# ---------------------------------------------------------------------------


class TestLOSEstimateBoundary:
    def test_los_rate_crossing_pi_boundary(self):
        """LOS azimuth crossing from 359° to 1° should not produce huge rate."""
        pn = ProportionalNavigationGuidance({"params": {"dt": 0.2}})
        pn._prev_los_vec = np.array([
            np.cos(np.deg2rad(359)),
            np.sin(np.deg2rad(359)),
            0.0,
        ])
        pn._filtered_los_rate = np.zeros(3)
        los_unit = np.array([
            np.cos(np.deg2rad(1)),
            np.sin(np.deg2rad(1)),
            0.0,
        ])
        rate = pn._estimate_los_rate(los_unit)
        # 2° change over 0.2s => ~0.175 rad/s, should be small
        assert np.linalg.norm(rate[:2]) < 1.0, f"Rate too large: {rate}"

    def test_los_rate_crossing_minus_pi_boundary(self):
        """LOS azimuth crossing from -179° to +179° should not produce huge rate."""
        pn = ProportionalNavigationGuidance({"params": {"dt": 0.2}})
        pn._prev_los_vec = np.array([
            np.cos(np.deg2rad(-179)),
            np.sin(np.deg2rad(-179)),
            0.0,
        ])
        pn._filtered_los_rate = np.zeros(3)
        los_unit = np.array([
            np.cos(np.deg2rad(179)),
            np.sin(np.deg2rad(179)),
            0.0,
        ])
        rate = pn._estimate_los_rate(los_unit)
        assert np.linalg.norm(rate[:2]) < 1.0, f"Rate too large: {rate}"

    def test_los_rate_normal_small_change(self):
        """A normal 5° change should produce a reasonable rate."""
        pn = ProportionalNavigationGuidance({"params": {"dt": 0.2, "los_rate_filter_alpha": 1.0}})
        pn._prev_los_vec = np.array([
            np.cos(np.deg2rad(10)),
            np.sin(np.deg2rad(10)),
            0.0,
        ])
        pn._filtered_los_rate = np.zeros(3)
        los_unit = np.array([
            np.cos(np.deg2rad(15)),
            np.sin(np.deg2rad(15)),
            0.0,
        ])
        rate = pn._estimate_los_rate(los_unit)
        # 5° over 0.2s => ~0.436 rad/s
        assert 0.3 < np.linalg.norm(rate[:2]) < 0.6, f"Rate unexpected: {rate}"

    def test_los_rate_elevation_boundary(self):
        """Elevation near +/- 90° should not clip violently."""
        pn = ProportionalNavigationGuidance({"params": {"dt": 0.2}})
        pn._prev_los_vec = np.array([0.0, 0.999, 0.044])  # el ~ 2.5°
        pn._prev_los_vec /= np.linalg.norm(pn._prev_los_vec)
        pn._filtered_los_rate = np.zeros(3)
        los_unit = np.array([0.0, 0.999, 0.087])  # el ~ 5.0°
        los_unit /= np.linalg.norm(los_unit)
        rate = pn._estimate_los_rate(los_unit)
        # Small elevation change => small rate
        assert abs(rate[1]) < 0.3, f"Elevation rate too large: {rate[1]}"
