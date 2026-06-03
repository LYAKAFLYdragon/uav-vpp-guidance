"""Tests for CommandPostProcessor (overload_rollrate module)."""

import math

import numpy as np
import pytest

from uav_vpp_guidance.guidance.overload_rollrate import CommandPostProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_post() -> CommandPostProcessor:
    return CommandPostProcessor()


@pytest.fixture
def raw_command() -> dict:
    return {"nz_cmd": 3.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.6}


# ---------------------------------------------------------------------------
# Constructor / config loading
# ---------------------------------------------------------------------------


def test_default_config(default_post):
    assert default_post.nz_min == -2.0
    assert default_post.nz_max == 7.0
    assert default_post.enable_energy_comp is False
    assert default_post.enable_terminal_protection is True
    assert default_post.terminal_range_m == 500.0
    assert default_post.base_nz == 1.0


def test_custom_config():
    post = CommandPostProcessor(
        {
            "limits": {"nz_min": -1.0, "nz_max": 5.0, "roll_rate_max": 1.0},
            "post_process": {
                "enable_energy_compensation": True,
                "energy_k_nz": 0.1,
                "terminal_range_m": 200.0,
            },
        }
    )
    assert post.nz_max == 5.0
    assert post.enable_energy_comp is True
    assert post.energy_k_nz == pytest.approx(0.1)
    assert post.terminal_range_m == 200.0


def test_base_nz_from_guidance_params():
    post = CommandPostProcessor(
        {
            "params": {"base_nz": 1.5},
            "post_process": {},
        }
    )
    assert post.base_nz == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Global limits merging
# ---------------------------------------------------------------------------


def test_global_limits_read_from_top_level_config():
    """CommandPostProcessor must read limits from merged top-level config."""
    post = CommandPostProcessor(
        {
            "limits": {"nz_max": 4.0, "roll_rate_max": 0.8},
            "post_process": {},
        }
    )
    assert post.nz_max == pytest.approx(4.0)
    assert post.roll_rate_max == pytest.approx(0.8)
    cmd = {"nz_cmd": 10.0, "roll_rate_cmd": 1.5, "throttle_cmd": 0.5}
    result = post.process(cmd)
    assert result["nz_cmd"] == pytest.approx(4.0)
    assert result["roll_rate_cmd"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_process_invalid_type(default_post):
    with pytest.raises(TypeError):
        default_post.process("not a dict")


def test_process_missing_key(default_post):
    with pytest.raises(ValueError, match="nz_cmd"):
        default_post.process({"roll_rate_cmd": 0.0, "throttle_cmd": 0.0})


def test_process_non_finite_command(default_post, raw_command):
    raw_command["nz_cmd"] = float("nan")
    # Should NOT raise; logs warning and clamps later
    result = default_post.process(raw_command)
    assert math.isfinite(result["nz_cmd"])


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------


def test_saturation_nz_max(default_post):
    cmd = {"nz_cmd": 10.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    result = default_post.process(cmd)
    assert result["nz_cmd"] == pytest.approx(7.0)


def test_saturation_nz_min(default_post):
    cmd = {"nz_cmd": -5.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    result = default_post.process(cmd)
    assert result["nz_cmd"] == pytest.approx(-2.0)


def test_saturation_roll_rate(default_post):
    cmd = {"nz_cmd": 1.0, "roll_rate_cmd": 2.0, "throttle_cmd": 0.5}
    result = default_post.process(cmd)
    assert result["roll_rate_cmd"] == pytest.approx(1.5)


def test_saturation_throttle(default_post):
    cmd = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 1.5}
    result = default_post.process(cmd)
    assert result["throttle_cmd"] == pytest.approx(1.0)

    cmd["throttle_cmd"] = -0.2
    result = default_post.process(cmd)
    assert result["throttle_cmd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Terminal-phase protection
# ---------------------------------------------------------------------------


def test_terminal_protection_active(default_post, raw_command):
    rel = {"range_m": 250.0}
    result = default_post.process(raw_command, relative_state=rel)
    # nz and roll should be scaled down
    assert result["nz_cmd"] < raw_command["nz_cmd"]
    assert abs(result["roll_rate_cmd"]) < abs(raw_command["roll_rate_cmd"])


def test_terminal_protection_inactive(default_post, raw_command):
    rel = {"range_m": 600.0}
    result = default_post.process(raw_command, relative_state=rel)
    # Should be unchanged (before saturation)
    assert result["nz_cmd"] == pytest.approx(raw_command["nz_cmd"])
    assert result["roll_rate_cmd"] == pytest.approx(raw_command["roll_rate_cmd"])


def test_terminal_protection_at_zero_range(default_post, raw_command):
    rel = {"range_m": 0.0}
    result = default_post.process(raw_command, relative_state=rel)
    # nz should converge to base_nz (1.0), not be scaled below it
    assert result["nz_cmd"] == pytest.approx(default_post.base_nz, abs=1e-3)
    assert math.isfinite(result["nz_cmd"])


def test_terminal_protection_preserves_base_nz():
    """nz_cmd=1.0 at terminal range must not be scaled below 1g."""
    post = CommandPostProcessor(
        {
            "params": {"base_nz": 1.0},
            "post_process": {
                "enable_terminal_protection": True,
                "terminal_range_m": 500.0,
                "terminal_nz_scale": 0.5,
            },
        }
    )
    cmd = {"nz_cmd": 1.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.5}
    rel = {"range_m": 0.0}
    result = post.process(cmd, relative_state=rel)
    # With deviation scaling, nz=base_nz + epsilon*(nz-base_nz) ≈ base_nz
    assert result["nz_cmd"] == pytest.approx(post.base_nz, abs=1e-3)


def test_terminal_protection_scales_deviation():
    """Higher nz_cmd should be pulled toward base_nz at close range."""
    post = CommandPostProcessor(
        {
            "params": {"base_nz": 1.0},
            "post_process": {
                "enable_terminal_protection": True,
                "terminal_range_m": 500.0,
            },
        }
    )
    cmd_high = {"nz_cmd": 5.0, "roll_rate_cmd": 0.5, "throttle_cmd": 0.5}
    rel = {"range_m": 0.0}
    result = post.process(cmd_high, relative_state=rel)
    # Should be pulled toward base_nz
    assert result["nz_cmd"] < cmd_high["nz_cmd"]
    assert result["nz_cmd"] >= post.base_nz


# ---------------------------------------------------------------------------
# Load-roll coordination
# ---------------------------------------------------------------------------


def test_load_roll_coordination_active():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {
                "enable_load_roll_coordination": True,
                "coord_nz_threshold": 0.5,
                "coord_roll_scale": 0.5,
            },
        }
    )
    cmd = {"nz_cmd": 6.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.5}
    result = post.process(cmd)
    # High nz should reduce roll rate
    assert abs(result["roll_rate_cmd"]) < abs(cmd["roll_rate_cmd"])


def test_load_roll_coordination_inactive():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {
                "enable_load_roll_coordination": True,
                "coord_nz_threshold": 0.9,
                "coord_roll_scale": 0.5,
            },
        }
    )
    cmd = {"nz_cmd": 1.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.5}
    result = post.process(cmd)
    # Low nz -> no reduction
    assert result["roll_rate_cmd"] == pytest.approx(cmd["roll_rate_cmd"])


# ---------------------------------------------------------------------------
# Energy compensation
# ---------------------------------------------------------------------------


def test_energy_compensation_high_nz():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {
                "enable_energy_compensation": True,
                "energy_k_nz": 0.1,
                "energy_k_speed": 0.0,
            },
        }
    )
    cmd = {"nz_cmd": 5.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    own = {"velocity_ned": np.array([250.0, 0.0, 0.0])}
    result = post.process(cmd, own_state=own)
    # High nz should boost throttle
    assert result["throttle_cmd"] > cmd["throttle_cmd"]


def test_energy_compensation_low_speed():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {
                "enable_energy_compensation": True,
                "energy_k_nz": 0.0,
                "energy_k_speed": 0.01,
            },
        }
    )
    cmd = {"nz_cmd": 1.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    own = {"velocity_ned": np.array([100.0, 0.0, 0.0])}
    result = post.process(cmd, own_state=own)
    # Low speed should boost throttle
    assert result["throttle_cmd"] > cmd["throttle_cmd"]


def test_energy_compensation_no_own_state():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {"enable_energy_compensation": True},
        }
    )
    cmd = {"nz_cmd": 5.0, "roll_rate_cmd": 0.0, "throttle_cmd": 0.5}
    result = post.process(cmd)
    assert result["throttle_cmd"] == pytest.approx(cmd["throttle_cmd"])


# ---------------------------------------------------------------------------
# Integration: combined effects
# ---------------------------------------------------------------------------


def test_combined_terminal_and_coordination():
    post = CommandPostProcessor(
        {
            "limits": {},
            "post_process": {
                "enable_terminal_protection": True,
                "terminal_range_m": 500.0,
                "enable_load_roll_coordination": True,
                "coord_nz_threshold": 0.5,
                "coord_roll_scale": 0.5,
            },
        }
    )
    cmd = {"nz_cmd": 6.0, "roll_rate_cmd": 1.0, "throttle_cmd": 0.5}
    rel = {"range_m": 200.0}
    result = post.process(cmd, relative_state=rel)
    assert abs(result["nz_cmd"]) <= post.nz_max
    assert abs(result["roll_rate_cmd"]) <= post.roll_rate_max
    # Both terminal protection and load-roll coord should reduce roll
    assert abs(result["roll_rate_cmd"]) < abs(cmd["roll_rate_cmd"])
