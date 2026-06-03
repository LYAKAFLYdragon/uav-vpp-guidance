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
    # Default min_dwell_steps=3, need multiple calls to switch
    for _ in range(3):
        cmd = default_hybrid.compute_command(own_state, None, virtual_point_near)
    assert default_hybrid._active_law == "los"
    for v in cmd.values():
        assert math.isfinite(v)


def test_range_mode_at_threshold(default_hybrid, own_state):
    vp = {"position_m": np.array([3000.0, 0.0, 5000.0])}
    default_hybrid.compute_command(own_state, None, vp)
    # At exactly threshold (range == threshold), range < threshold is False -> PN
    assert default_hybrid._active_law == "pn"


def test_range_mode_hysteresis_prevents_chatter():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "range",
                "range_threshold_m": 3000.0,
                "hysteresis_m": 500.0,
                "min_dwell_steps": 3,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    # Start far away -> PN
    hyb.compute_command(own, None, {"position_m": np.array([10000.0, 0.0, 5000.0])})
    assert hyb._active_law == "pn"

    # Move to just inside threshold (2750m) but above lower bound (2750 > 2500)
    # With hysteresis, PN should stay active until range < 2500
    hyb.compute_command(own, None, {"position_m": np.array([2750.0, 0.0, 5000.0])})
    assert hyb._active_law == "pn"

    # Move below lower bound -> should switch to LOS after dwell steps
    for _ in range(3):
        hyb.compute_command(own, None, {"position_m": np.array([2400.0, 0.0, 5000.0])})
    assert hyb._active_law == "los"


def test_range_mode_dwell_time_enforced():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "range",
                "range_threshold_m": 3000.0,
                "hysteresis_m": 500.0,
                "min_dwell_steps": 5,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    # Start in LOS (short range)
    for _ in range(5):
        hyb.compute_command(own, None, {"position_m": np.array([2000.0, 0.0, 5000.0])})
    assert hyb._active_law == "los"

    # Move to far range but only 2 steps -> should NOT switch yet
    for _ in range(2):
        hyb.compute_command(own, None, {"position_m": np.array([10000.0, 0.0, 5000.0])})
    assert hyb._active_law == "los"

    # After 5 total steps in pending -> switch to PN
    for _ in range(3):
        hyb.compute_command(own, None, {"position_m": np.array([10000.0, 0.0, 5000.0])})
    assert hyb._active_law == "pn"


def test_reset_clears_dwell_and_pending():
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "range",
                "range_threshold_m": 3000.0,
                "min_dwell_steps": 3,
            },
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "roll_rad": 0.0,
    }
    hyb.compute_command(own, None, {"position_m": np.array([2000.0, 0.0, 5000.0])})
    hyb.reset()
    assert hyb._active_law == "pn"
    assert hyb._steps_in_law == 0
    assert hyb._pending_law is None


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
    for _ in range(3):
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
    """Blended mode output should interpolate between PN and LOS."""
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

    # Warm up hybrid's internal PN filter (first call initializes filter)
    _ = hyb.compute_command(own, None, vp_mid)
    # Second call gets actual blended output with warmed-up filter
    cmd_mid = hyb.compute_command(own, None, vp_mid)

    # Compute pure LOS and pure PN with identically warmed-up state
    los = LOSRateGuidance(config)
    pn = ProportionalNavigationGuidance(config)
    cmd_los = los.compute_command(own, None, vp_mid)
    # Warm up PN filter with same VP
    pn.compute_command(own, None, vp_mid)
    cmd_pn = pn.compute_command(own, None, vp_mid)

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
# Position field parsing (defensive hardening)
# ---------------------------------------------------------------------------


def test_virtual_point_position_key_computes_range():
    """VPP outputs 'position'; HybridGuidance must read it correctly."""
    hyb = HybridGuidance({"params": {"hybrid_mode": "range", "min_dwell_steps": 1}})
    own = {
        "position": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    vp_far = {"position": np.array([10000.0, 0.0, 5000.0])}
    cmd = hyb.compute_command(own, None, vp_far)
    assert hyb._active_law == "pn"
    for v in cmd.values():
        assert math.isfinite(v)


def test_own_state_position_neu_no_zero_fallback():
    """own_state with 'position_neu' must not silently fallback to [0,0,0]."""
    hyb = HybridGuidance({"params": {"hybrid_mode": "range", "min_dwell_steps": 1}})
    own = {
        "position_neu": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    # VP at [3000,0,5000] -> range=3000, at threshold (PN side)
    vp = {"position_m": np.array([3000.0, 0.0, 5000.0])}
    hyb.compute_command(own, None, vp)
    # If fallback to zero had happened, range would be 3000 anyway,
    # but let's verify with a case where it matters.
    own_moved = {
        "position_neu": np.array([1000.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    vp_same = {"position_m": np.array([3000.0, 0.0, 5000.0])}
    cmd2 = hyb.compute_command(own_moved, None, vp_same)
    # Range should be 2000, so after dwell it switches to LOS
    assert hyb._active_law == "los"
    for v in cmd2.values():
        assert math.isfinite(v)


def test_blended_weight_changes_with_real_range():
    """Blended mode w should vary when range changes."""
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "blended",
                "range_threshold_m": 3000.0,
                "blend_transition_m": 1000.0,
            }
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    vp_far = {"position_m": np.array([10000.0, 0.0, 5000.0])}
    vp_near = {"position_m": np.array([500.0, 0.0, 5000.0])}

    cmd_far = hyb.compute_command(own, None, vp_far)
    cmd_near = hyb.compute_command(own, None, vp_near)

    # Far should be more PN-like (lower nz, smaller roll than near)
    # Near should be more LOS-like
    # We just assert they differ, proving range affects blending
    assert cmd_far["nz_cmd"] != pytest.approx(cmd_near["nz_cmd"], abs=1e-3)


def test_range_switch_at_threshold_correct():
    """Range mode must switch correctly near threshold with dwell."""
    hyb = HybridGuidance(
        {
            "params": {
                "hybrid_mode": "range",
                "range_threshold_m": 3000.0,
                "min_dwell_steps": 1,
            }
        }
    )
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
    }
    vp_far = {"position_m": np.array([4000.0, 0.0, 5000.0])}
    vp_near = {"position_m": np.array([2000.0, 0.0, 5000.0])}

    hyb.compute_command(own, None, vp_far)
    assert hyb._active_law == "pn"
    hyb.compute_command(own, None, vp_near)
    assert hyb._active_law == "los"


def test_missing_position_raises():
    """Missing position field must raise, not silently fallback."""
    hyb = HybridGuidance()
    own = {"velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
    vp = {"position_m": np.array([1000.0, 0.0, 5000.0])}
    with pytest.raises(ValueError, match="position"):
        hyb.compute_command(own, None, vp)


def test_position_ned_supported():
    """position_ned key must be supported."""
    hyb = HybridGuidance({"params": {"hybrid_mode": "range", "min_dwell_steps": 1}})
    own = {
        "position_ned": np.array([0.0, 0.0, -5000.0]),
        "velocity_ned": np.array([200.0, 0.0, 0.0]),
    }
    vp = {"position_ned": np.array([3000.0, 0.0, -5000.0])}
    cmd = hyb.compute_command(own, None, vp)
    assert hyb._active_law == "pn"
    for v in cmd.values():
        assert math.isfinite(v)


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
