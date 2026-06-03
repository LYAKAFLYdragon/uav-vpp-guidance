"""Tests for HybridGuidance."""

import math

import numpy as np
import pytest

from uav_vpp_guidance.guidance.hybrid_guidance import HybridGuidance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_hybrid() -> HybridGuidance:
    return HybridGuidance()


@pytest.fixture
def own_state() -> dict:
    return {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }


@pytest.fixture
def virtual_point_far() -> dict:
    return {"position_m": np.array([10000.0, 0.0, 5000.0])}


@pytest.fixture
def virtual_point_near() -> dict:
    return {"position_m": np.array([500.0, 0.0, 5000.0])}


# ---------------------------------------------------------------------------
# Constructor / config
# ---------------------------------------------------------------------------


def test_default_config(default_hybrid):
    assert default_hybrid.mode == "range"
    assert default_hybrid.range_threshold_m == pytest.approx(3000.0)
    assert default_hybrid.blend_transition_m == pytest.approx(1000.0)
    assert default_hybrid.energy_speed_threshold_mps == pytest.approx(220.0)


def test_custom_config():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "blended",
                "range_threshold_m": 5000.0,
                "blend_transition_m": 2000.0,
                "energy_speed_threshold_mps": 180.0,
            },
        }
    )
    assert hyb.mode == "blended"
    assert hyb.range_threshold_m == pytest.approx(5000.0)


def test_unknown_mode_fallback(own_state, virtual_point_far):
    hyb = HybridGuidance({"params": {"hybrid_mode": "unknown"}})
    # Should not raise; falls back to range mode
    cmd = hyb.compute_command(own_state, None, virtual_point_far)
    assert set(cmd.keys()) == {"nz_cmd", "roll_rate_cmd", "throttle_cmd"}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset(default_hybrid):
    default_hybrid.reset()
    assert default_hybrid._active_law == "pn"


# ---------------------------------------------------------------------------
# Range mode
# ---------------------------------------------------------------------------


def test_range_mode_far_uses_pn(default_hybrid, own_state, virtual_point_far):
    cmd = default_hybrid.compute_command(own_state, None, virtual_point_far)
    assert default_hybrid._active_law == "pn"
    assert set(cmd.keys()) == {"nz_cmd", "roll_rate_cmd", "throttle_cmd"}
    for v in cmd.values():
        assert math.isfinite(v)


def test_range_mode_near_uses_los(default_hybrid, own_state, virtual_point_near):
    cmd = default_hybrid.compute_command(own_state, None, virtual_point_near)
    assert default_hybrid._active_law == "los"
    for v in cmd.values():
        assert math.isfinite(v)


def test_range_mode_at_threshold(default_hybrid, own_state):
    vp = {"position_m": np.array([3000.0, 0.0, 5000.0])}
    default_hybrid.compute_command(own_state, None, vp)
    # At exactly threshold (range == threshold), range < threshold is False -> PN
    assert default_hybrid._active_law == "pn"


# ---------------------------------------------------------------------------
# Energy mode
# ---------------------------------------------------------------------------


def test_energy_mode_low_speed_uses_los():
    hyb = HybridGuidance({"params": {"hybrid_mode": "energy"}})
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([150.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([10000.0, 0.0, 5000.0])}
    hyb.compute_command(own, None, vp)
    assert hyb._active_law == "los"


def test_energy_mode_high_speed_far_range_uses_pn():
    hyb = HybridGuidance({"params": {"hybrid_mode": "energy"}})
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([300.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([10000.0, 0.0, 5000.0])}
    hyb.compute_command(own, None, vp)
    assert hyb._active_law == "pn"


# ---------------------------------------------------------------------------
# Blended mode
# ---------------------------------------------------------------------------


def test_blended_mode_far_is_pure_pn():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "blended",
                "range_threshold_m": 3000.0,
                "blend_transition_m": 1000.0,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([10000.0, 0.0, 5000.0])}
    hyb.compute_command(own, None, vp)
    assert hyb._active_law == "pn"


def test_blended_mode_near_is_pure_los():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "blended",
                "range_threshold_m": 3000.0,
                "blend_transition_m": 1000.0,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp = {"position_m": np.array([500.0, 0.0, 5000.0])}
    hyb.compute_command(own, None, vp)
    assert hyb._active_law == "los"


def test_blended_mode_transition_zone():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "blended",
                "range_threshold_m": 3000.0,
                "blend_transition_m": 1000.0,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    # Range = 3000 -> exactly at threshold, should be in blend zone
    vp = {"position_m": np.array([3000.0, 0.0, 5000.0])}
    cmd = hyb.compute_command(own, None, vp)
    assert hyb._active_law == "blend"
    for v in cmd.values():
        assert math.isfinite(v)


def test_blended_output_between_pure_commands():
    from uav_vpp_guidance.guidance.los_rate_guidance import LOSRateGuidance
    from uav_vpp_guidance.guidance.proportional_navigation import (
        ProportionalNavigationGuidance,
    )

    config = {
        "params": {
            "hybrid_mode": "blended",
            "range_threshold_m": 3000.0,
            "blend_transition_m": 1000.0,
        }
    }
    hyb = HybridGuidance(config)
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    vp_mid = {"position_m": np.array([3000.0, 500.0, 5000.0])}

    # Compute pure LOS and pure PN for the SAME mid point using fresh instances
    los = LOSRateGuidance(config)
    pn = ProportionalNavigationGuidance(config)
    cmd_los = los.compute_command(own, None, vp_mid)
    # Warm up PN filter first
    pn.compute_command(own, None, {"position_m": np.array([2900.0, 450.0, 5000.0])})
    cmd_pn = pn.compute_command(own, None, vp_mid)
    cmd_mid = hyb.compute_command(own, None, vp_mid)

    # Blended command should lie between pure PN and pure LOS
    for key in ("nz_cmd", "roll_rate_cmd", "throttle_cmd"):
        lo = min(cmd_pn[key], cmd_los[key])
        hi = max(cmd_pn[key], cmd_los[key])
        assert (
            lo <= cmd_mid[key] <= hi
            or math.isclose(cmd_mid[key], lo, abs_tol=1e-6)
            or math.isclose(cmd_mid[key], hi, abs_tol=1e-6)
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_distance(default_hybrid, own_state):
    vp = {"position_m": np.array([0.0, 0.0, 5000.0])}
    cmd = default_hybrid.compute_command(own_state, None, vp)
    for v in cmd.values():
        assert math.isfinite(v)


def test_gain_override(default_hybrid, own_state):
    class FakeGains:
        k_roll = 3.0
        k_speed = 0.8

    # Use a VP with lateral offset so heading error is non-zero
    vp = {"position_m": np.array([500.0, 300.0, 5000.0])}
    cmd_default = default_hybrid.compute_command(own_state, None, vp)
    cmd_override = default_hybrid.compute_command(
        own_state, None, vp, gains=FakeGains()
    )
    # Near range uses LOS, which respects k_roll
    assert abs(cmd_override["roll_rate_cmd"]) > abs(cmd_default["roll_rate_cmd"])
