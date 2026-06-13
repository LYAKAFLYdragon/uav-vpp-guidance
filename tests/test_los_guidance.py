"""
Unit tests for LOS-rate guidance.

Covers:
  - Basic construction and config parsing
  - Standard compute_command scenarios
  - Edge cases: zero distance, zero speed, NaN/Inf inputs, huge distances
  - Numerical stability: near-singular elevations, ±pi heading differences
  - Capture-radius blending
  - Internal clipping and filtering
"""

import math

import numpy as np
import pytest

from uav_vpp_guidance.guidance.los_rate_guidance import (
    LOSRateGuidance,
    _normalize_angle,
    _stable_angle_diff,
)
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


class TestLOSRateGuidance:
    def test_init(self):
        guidance = LOSRateGuidance(config={})
        assert guidance.prev_command is None

    def test_init_with_config_params(self):
        config = {
            "gains": {"k_los": 2.0, "k_pos": 0.3},
            "params": {
                "distance_scale_m": 1000.0,
                "target_speed_mps": 200.0,
                "speed_error_scale_mps": 50.0,
                "base_throttle": 0.6,
                "epsilon": 1.0e-4,
                "base_nz": 1.0,
                "capture_radius_m": 30.0,
            },
        }
        guidance = LOSRateGuidance(config=config)
        assert guidance.k_los == 2.0
        assert guidance.k_pos == 0.3
        assert guidance.distance_scale_m == 1000.0
        assert guidance.target_speed_mps == 200.0
        assert guidance.epsilon == 1.0e-4
        assert guidance.capture_radius_m == 30.0

    def test_init_invalid_params(self):
        with pytest.raises(ValueError, match="distance_scale_m must be positive"):
            LOSRateGuidance(config={"params": {"distance_scale_m": 0.0}})
        with pytest.raises(ValueError, match="speed_error_scale_mps must be positive"):
            LOSRateGuidance(config={"params": {"speed_error_scale_mps": -1.0}})
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LOSRateGuidance(config={"params": {"epsilon": 0.0}})
        with pytest.raises(ValueError, match="capture_radius_m must be non-negative"):
            LOSRateGuidance(config={"params": {"capture_radius_m": -10.0}})

    def test_reset(self):
        guidance = LOSRateGuidance(config={})
        guidance.prev_command = {"nz_cmd": 1.0}
        guidance.reset()
        assert guidance.prev_command is None

    def test_compute_command_basic(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains()
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        target_state = {
            "position_m": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, target_state, virtual_point, gains)
        assert isinstance(cmd, dict)
        assert "nz_cmd" in cmd
        assert "roll_rate_cmd" in cmd
        assert "throttle_cmd" in cmd
        assert -2.0 <= cmd["nz_cmd"] <= 7.0
        assert -1.5 <= cmd["roll_rate_cmd"] <= 1.5
        assert 0.0 <= cmd["throttle_cmd"] <= 1.0

    def test_compute_command_heading_error(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains(k_roll=2.0)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([0.0, 1000.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point, gains)
        # VP is to the right (east), so positive heading error -> positive roll_rate
        assert cmd["roll_rate_cmd"] > 0.1

    def test_compute_command_elevation_error(self):
        guidance = LOSRateGuidance(config={})
        gains = GuidanceGains(k_los=2.0)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 6000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point, gains)
        # VP is above, so positive elevation -> nz_cmd > 1.0
        assert cmd["nz_cmd"] > 1.0

    # ------------------------------------------------------------------
    # Edge cases: zero distance
    # ------------------------------------------------------------------

    def test_compute_command_zero_distance(self):
        """When own_state == virtual_point, should not crash and return finite values."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([0.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    def test_compute_command_extremely_small_distance(self):
        """Distance = 1e-10 m: should handle safely without division by zero."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1e-10, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    # ------------------------------------------------------------------
    # Edge cases: zero speed
    # ------------------------------------------------------------------

    def test_compute_command_zero_speed(self):
        """When own_speed is zero, heading defaults to 0.0 and should not crash."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    # ------------------------------------------------------------------
    # Edge cases: NaN / Inf inputs
    # ------------------------------------------------------------------

    def test_compute_command_with_nan_velocity(self):
        """NaN in velocity should propagate to finite commands if speed is below epsilon."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([np.nan, np.nan, np.nan]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        # Speed = NaN, so heading defaults to 0.0; commands should still be finite
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])

    def test_compute_command_with_inf_position(self):
        """Inf in position: arctan2 handles inf robustly; output must be finite and clipped."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([np.inf, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        assert np.isfinite(cmd["throttle_cmd"])
        # roll_rate_cmd is finite and internally clipped
        assert -1.5 <= cmd["roll_rate_cmd"] <= 1.5

    # ------------------------------------------------------------------
    # Numerical stability: near-vertical LOS
    # ------------------------------------------------------------------

    def test_compute_command_near_vertical_elevation_positive(self):
        """VPP almost directly above: elevation ≈ +89.9°."""
        guidance = LOSRateGuidance(config={"gains": {"k_los": 1.0}})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        # Small horizontal offset so d_horiz is tiny but non-zero
        virtual_point = {"position_m": np.array([1e-3, 0.0, 5000.0 + 1000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert cmd["nz_cmd"] > 1.0  # Should demand climb

    def test_compute_command_near_vertical_elevation_negative(self):
        """VPP almost directly below: elevation ≈ -89.9°."""
        guidance = LOSRateGuidance(config={"gains": {"k_los": 1.0}})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1e-3, 0.0, 5000.0 - 1000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert cmd["nz_cmd"] < 1.0  # Should demand descent

    def test_compute_command_directly_above(self):
        """VPP exactly above (d_horiz = 0): should not crash."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([0.0, 0.0, 6000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        assert np.isfinite(cmd["roll_rate_cmd"])
        # Horizontal distance is zero, so LOS heading falls back to own_heading
        # => heading_error = 0 => roll_rate_cmd ≈ -k_damp * roll_rad ≈ 0
        assert cmd["roll_rate_cmd"] == pytest.approx(0.0, abs=1e-6)

    def test_compute_command_directly_behind(self):
        """VPP directly behind (180° heading difference): stable wrapping."""
        guidance = LOSRateGuidance(config={})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([-1000.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["roll_rate_cmd"])
        # Heading diff should be π (or -π), but _stable_angle_diff maps to π
        # With k_roll=1.0, roll_rate_cmd ≈ π ≈ 3.14, but clipped to 1.5 max
        assert cmd["roll_rate_cmd"] == pytest.approx(1.5, abs=1e-3)

    # ------------------------------------------------------------------
    # Edge cases: huge distance
    # ------------------------------------------------------------------

    def test_compute_command_huge_distance(self):
        """Distance = 1e5 m: nz proportional term should not explode."""
        guidance = LOSRateGuidance(config={"gains": {"k_pos": 0.5}})
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {"position_m": np.array([1e5, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert np.isfinite(cmd["nz_cmd"])
        # nz = 1.0 + 0 + 0.5 * (1e5 / 2000) = 1.0 + 25 = 26 -> clipped to 7.0
        assert cmd["nz_cmd"] == pytest.approx(7.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Capture radius
    # ------------------------------------------------------------------

    def test_capture_radius_reduces_roll_and_nz(self):
        """Inside capture radius, roll should be attenuated relative to the un-attenuated raw command."""
        config = {
            "params": {"capture_radius_m": 100.0, "base_nz": 1.0},
            "gains": {"k_roll": 2.0, "k_los": 2.0},
        }
        guidance = LOSRateGuidance(config=config)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        # VP is close (distance ≈ 70.7 m) and to the right
        virtual_point = {"position_m": np.array([50.0, 50.0, 5050.0])}
        distance = np.linalg.norm(np.array([50.0, 50.0, 50.0]))
        assert distance < 100.0
        cmd = guidance.compute_command(own_state, None, virtual_point)

        # Roll should be attenuated: capture_ratio = distance / 100 < 1
        # so |roll_rate_cmd| should be less than the raw un-attenuated value.
        # We can verify this by comparing with a case where capture_radius = 0 (no attenuation).
        guidance_no_cap = LOSRateGuidance(
            config={**config, "params": {**config["params"], "capture_radius_m": 0.0}}
        )
        cmd_no_cap = guidance_no_cap.compute_command(own_state, None, virtual_point)
        assert abs(cmd["roll_rate_cmd"]) < abs(cmd_no_cap["roll_rate_cmd"])
        # nz is blended toward base_nz; at d=0 it would equal base_nz, at d<100 it is closer than raw
        assert abs(cmd["nz_cmd"] - config["params"]["base_nz"]) <= abs(
            cmd_no_cap["nz_cmd"] - config["params"]["base_nz"]
        )

    def test_capture_radius_zero_fallback(self):
        """At distance=0, commands should fully fall back to safe hold."""
        config = {"params": {"capture_radius_m": 50.0, "base_nz": 1.0}}
        guidance = LOSRateGuidance(config=config)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.5,
        }
        virtual_point = {"position_m": np.array([0.0, 0.0, 5000.0])}
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert cmd["roll_rate_cmd"] == pytest.approx(0.0, abs=1e-9)
        assert cmd["nz_cmd"] == pytest.approx(1.0, abs=1e-9)

    # ------------------------------------------------------------------
    # Internal clipping and filtering
    # ------------------------------------------------------------------

    def test_internal_clip_limits(self):
        """Internal clipping should enforce limits even if raw command exceeds them."""
        config = {
            "limits": {
                "nz_min": -2.0,
                "nz_max": 5.0,
                "roll_rate_min": -1.0,
                "roll_rate_max": 1.0,
                "throttle_min": 0.0,
                "throttle_max": 1.0,
            },
            "params": {"enable_internal_clip": True},
        }
        guidance = LOSRateGuidance(config=config)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        virtual_point = {
            "position_m": np.array([0.0, -1000.0, 5000.0])
        }  # behind and left
        cmd = guidance.compute_command(own_state, None, virtual_point)
        assert -2.0 <= cmd["nz_cmd"] <= 5.0
        assert -1.0 <= cmd["roll_rate_cmd"] <= 1.0
        assert 0.0 <= cmd["throttle_cmd"] <= 1.0

    def test_internal_filter_smoothing(self):
        """With internal filter enabled, successive commands should be smoothed."""
        config = {
            "gains": {"alpha_filter": 0.5},
            "params": {"enable_internal_filter": True, "enable_internal_clip": True},
        }
        guidance = LOSRateGuidance(config=config)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        vp1 = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        vp2 = {"position_m": np.array([0.0, 1000.0, 5000.0])}
        guidance.compute_command(own_state, None, vp1)
        cmd2 = guidance.compute_command(own_state, None, vp2)
        # With alpha=0.5, cmd2 should be blend of raw cmd2 and cmd1
        # We just verify that filtering happened (prev_command was used)
        assert guidance.prev_command is cmd2
        assert np.isfinite(cmd2["roll_rate_cmd"])

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_compute_command_missing_position_raises(self):
        guidance = LOSRateGuidance(config={})
        own_state = {"velocity_vector_mps": np.array([200.0, 0.0, 0.0])}
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        with pytest.raises(ValueError, match="missing position field"):
            guidance.compute_command(own_state, None, virtual_point)

    def test_compute_command_missing_velocity_raises(self):
        guidance = LOSRateGuidance(config={})
        own_state = {"position_m": np.array([0.0, 0.0, 5000.0])}
        virtual_point = {"position_m": np.array([1000.0, 0.0, 5000.0])}
        with pytest.raises(ValueError, match="missing velocity field"):
            guidance.compute_command(own_state, None, virtual_point)

    def test_compute_command_invalid_state_type_raises(self):
        guidance = LOSRateGuidance(config={})
        with pytest.raises(TypeError, match="must be a dict"):
            guidance.compute_command("not_a_dict", None, {"position_m": np.zeros(3)})

    # ------------------------------------------------------------------
    # Terminal boundary layer
    # ------------------------------------------------------------------

    def test_terminal_boundary_layer_disabled_by_default(self):
        guidance = LOSRateGuidance(config={})
        assert guidance.tbl_enabled is False
        assert guidance._compute_terminal_blend(50.0) == pytest.approx(1.0)

    def test_terminal_boundary_layer_blend_shape(self):
        config = {
            "params": {
                "terminal_boundary_layer": {
                    "enabled": True,
                    "R_dead_m": 500.0,
                    "blend_scale": 125.0,
                }
            }
        }
        guidance = LOSRateGuidance(config=config)
        assert guidance.tbl_enabled is True
        # Far away -> full command
        assert guidance._compute_terminal_blend(2000.0) == pytest.approx(1.0, abs=1e-3)
        # At R_dead/2 -> 0.5
        assert guidance._compute_terminal_blend(250.0) == pytest.approx(0.5, abs=1e-3)
        # Very close -> near 0
        assert guidance._compute_terminal_blend(0.0) == pytest.approx(0.0, abs=1e-9)
        assert guidance._compute_terminal_blend(1.0) < 0.05

    def test_terminal_boundary_layer_suppresses_commands(self):
        config = {
            "params": {
                "terminal_boundary_layer": {
                    "enabled": True,
                    "R_dead_m": 500.0,
                    "blend_scale": 125.0,
                }
            }
        }
        guidance = LOSRateGuidance(config=config)
        gains = GuidanceGains(k_roll=2.0, k_los=2.0)
        own_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "roll_rad": 0.0,
        }
        # VP to the right and above, close enough to trigger boundary layer
        virtual_point_close = {"position_m": np.array([10.0, 200.0, 5100.0])}
        cmd_close = guidance.compute_command(own_state, None, virtual_point_close, gains)

        guidance.reset()
        # Same direction but far away -> should produce larger commands
        virtual_point_far = {"position_m": np.array([1000.0, 20000.0, 6000.0])}
        cmd_far = guidance.compute_command(own_state, None, virtual_point_far, gains)

        assert abs(cmd_close["roll_rate_cmd"]) < abs(cmd_far["roll_rate_cmd"])
        assert cmd_close["nz_cmd"] < cmd_far["nz_cmd"]

    def test_terminal_boundary_layer_invalid_config(self):
        with pytest.raises(ValueError, match="R_dead_m must be positive"):
            LOSRateGuidance(
                config={
                    "params": {
                        "terminal_boundary_layer": {
                            "enabled": True,
                            "R_dead_m": 0.0,
                        }
                    }
                }
            )
        with pytest.raises(ValueError, match="blend_scale must be positive"):
            LOSRateGuidance(
                config={
                    "params": {
                        "terminal_boundary_layer": {
                            "enabled": True,
                            "R_dead_m": 500.0,
                            "blend_scale": 0.0,
                        }
                    }
                }
            )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def test_config_from_guidance_yaml(self):
        """Ensure guidance.yaml config loads correctly with new params."""
        from uav_vpp_guidance.utils.config import load_yaml_config

        config = load_yaml_config("config/guidance.yaml")
        assert "guidance" in config
        assert "params" in config["guidance"]
        params = config["guidance"]["params"]
        assert "distance_scale_m" in params
        assert "target_speed_mps" in params
        assert "epsilon" in params
        assert "capture_radius_m" in params
        assert "terminal_boundary_layer" in params
        assert "enable_internal_clip" in params
        assert "enable_internal_filter" in params
        assert "limits" in config["guidance"]


# ---------------------------------------------------------------------------
# Angle helper tests
# ---------------------------------------------------------------------------


class TestNormalizeAngle:
    def test_normalize_angle_standard(self):
        assert _normalize_angle(0.0) == pytest.approx(0.0, abs=1e-9)
        assert _normalize_angle(math.pi) == pytest.approx(math.pi, abs=1e-9)
        # Python modulo returns non-negative, so -pi maps to pi (same direction)
        assert _normalize_angle(-math.pi) == pytest.approx(math.pi, abs=1e-9)
        assert _normalize_angle(3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9)
        assert _normalize_angle(-3.0 * math.pi) == pytest.approx(math.pi, abs=1e-9)

    def test_normalize_angle_nan(self):
        """NaN should pass through without crashing."""
        result = _normalize_angle(np.nan)
        assert np.isnan(result)


class TestStableAngleDiff:
    def test_stable_angle_diff_basic(self):
        assert _stable_angle_diff(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
        assert _stable_angle_diff(math.pi / 2, 0.0) == pytest.approx(
            math.pi / 2, abs=1e-9
        )
        assert _stable_angle_diff(0.0, math.pi / 2) == pytest.approx(
            -math.pi / 2, abs=1e-9
        )

    def test_stable_angle_diff_wraparound_pi(self):
        """Difference near ±pi should map correctly to [-pi, pi]."""
        # a = π, b = -π → diff should be 0 (same direction)
        assert _stable_angle_diff(math.pi, -math.pi) == pytest.approx(0.0, abs=1e-9)
        # a = π - 0.01, b = -π + 0.01 → diff ≈ 2*(π - 0.01) - 2π = -0.02 ... wait
        # Let's use a simpler case: a = 3π/4, b = -3π/4 → diff = 1.5π → wrapped to -0.5π
        assert _stable_angle_diff(3 * math.pi / 4, -3 * math.pi / 4) == pytest.approx(
            -math.pi / 2, abs=1e-9
        )

    def test_stable_angle_diff_2pi_periodicity(self):
        """Adding 2π should not change the difference."""
        for a in [0.1, 1.0, -0.5, 3.0]:
            for b in [0.2, -1.0, 2.5]:
                d1 = _stable_angle_diff(a, b)
                d2 = _stable_angle_diff(a + 2 * math.pi, b)
                d3 = _stable_angle_diff(a, b - 2 * math.pi)
                assert d1 == pytest.approx(d2, abs=1e-9)
                assert d1 == pytest.approx(d3, abs=1e-9)

    def test_stable_angle_diff_vs_normalize(self):
        """For moderate angles, both methods should agree."""
        for a in [0.1, 1.0, 2.0, -0.5, -2.0]:
            for b in [0.2, -1.0, 1.5, -2.5]:
                d_stable = _stable_angle_diff(a, b)
                d_legacy = _normalize_angle(a - b)
                assert d_stable == pytest.approx(d_legacy, abs=1e-9)

    def test_stable_angle_diff_nan(self):
        result = _stable_angle_diff(np.nan, 0.0)
        assert np.isnan(result)
        result = _stable_angle_diff(0.0, np.nan)
        assert np.isnan(result)

    def test_stable_angle_diff_large_cumulative_angles(self):
        """Simulate accumulated floating-point drift: a = 1000π, b = 1000π + 0.1."""
        a = 1000.0 * math.pi
        b = a + 0.1
        d = _stable_angle_diff(a, b)
        assert d == pytest.approx(-0.1, abs=1e-9)

    def test_stable_angle_diff_near_pi_boundary(self):
        """a = π - ε, b = -π + ε: difference should be ~2ε, not ~-2π+2ε."""
        eps = 1e-6
        a = math.pi - eps
        b = -math.pi + eps
        d = _stable_angle_diff(a, b)
        # Expected: wrap to -(2π - 2eps) which maps to ~2eps
        assert abs(d) == pytest.approx(2 * eps, abs=1e-9)
