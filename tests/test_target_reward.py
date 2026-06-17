"""Unit tests for the adversarial target reward calculator."""
from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.envs.adversarial_jsbsim_env import TargetRewardCalculator


def _make_states(range_m: float, range_rate: float, own_speed: float = 250.0):
    """Build minimal own/pursuer state dicts and a relative-state dict."""
    own_state = {
        "position_neu": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([own_speed, 0.0, 0.0]),
    }
    # pursuer_state content does not affect the tested paths because rel_state is explicit.
    pursuer_state = {
        "position_neu": np.array([range_m, 0.0, 5000.0]),
        "velocity_vector_mps": np.array([0.0, 0.0, 0.0]),
    }
    rel_state = {
        "range_m": range_m,
        "range_rate_mps": range_rate,
    }
    return own_state, pursuer_state, rel_state


def test_survival_bonus_is_positive():
    calc = TargetRewardCalculator()
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=0.0)
    reward, terms = calc.compute(own, pursuer, rel)
    assert terms["survival"] > 0.0
    assert reward > 0.0


def test_escaping_bonus_positive():
    """Positive range_rate from the target's perspective = escaping = positive bonus."""
    calc = TargetRewardCalculator()
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=100.0)
    reward, terms = calc.compute(own, pursuer, rel)
    assert terms["escaping"] > 0.0


def test_being_closed_on_has_no_escaping_bonus():
    """Negative range_rate from the target's perspective = pursuer closing."""
    calc = TargetRewardCalculator()
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=-100.0)
    reward, terms = calc.compute(own, pursuer, rel)
    assert terms["escaping"] == pytest.approx(0.0)


def test_captured_terminal_penalty():
    calc = TargetRewardCalculator()
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=0.0)
    term_info = {"is_success": True}
    reward, terms = calc.compute(own, pursuer, rel, term_info)
    assert terms["terminal"] == pytest.approx(-200.0)
    assert reward < -100.0


def test_crash_terminal_penalty():
    calc = TargetRewardCalculator()
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=0.0)
    term_info = {"is_crash": True}
    reward, terms = calc.compute(own, pursuer, rel, term_info)
    assert terms["terminal"] == pytest.approx(-300.0)


def test_timeout_terminal_bonus():
    calc = TargetRewardCalculator(
        {"target_reward": {"terminal_timeout": 50.0}}
    )
    own, pursuer, rel = _make_states(range_m=2000.0, range_rate=0.0)
    term_info = {"is_timeout": True}
    reward, terms = calc.compute(own, pursuer, rel, term_info)
    assert terms["terminal"] == pytest.approx(50.0)
