"""Tests for the bandit maneuver selector temporal stability constraints."""
from __future__ import annotations

import numpy as np
import pytest

from uav_vpp_guidance.envs.bandit_selector import BanditManeuverSelector


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
        "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        "altitude_m": 5000.0,
        "speed_mps": 200.0,
        "sim_time": 0.0,
        "attitude_rpy": np.zeros(3),
        "body_rates_rps": np.zeros(3),
        "nz_g": 1.0,
    }


def _seeded_selector(seed: int, **kwargs):
    config = {"seed": seed, "selector_noise": 0.0, "reaction_time_s": 0.0, **kwargs}
    return BanditManeuverSelector(config)


def test_dwell_time_prevents_early_switch(own_state, bandit_state):
    """The selector must hold a maneuver for at least ``min_dwell_time_s``."""
    selector = _seeded_selector(
        42, allowed_maneuvers=["coordinated_turn", "straight_level"]
    )
    selector._rule_select = lambda sit, bs: "coordinated_turn"
    initial = selector.select_initial(own_state, bandit_state, sim_time=0.0)
    assert initial is not None

    # Force the rule to desire a different maneuver, but elapsed time is short.
    selector._rule_select = lambda sit, bs: "straight_level"

    next_maneuver = selector.select(
        own_state,
        bandit_state,
        current_maneuver_name=initial.name,
        elapsed_in_maneuver=0.5,
        sim_time=0.5,
    )
    assert next_maneuver is None

    # After the dwell period has elapsed, a switch is allowed.
    next_maneuver = selector.select(
        own_state,
        bandit_state,
        current_maneuver_name=initial.name,
        elapsed_in_maneuver=3.0,
        sim_time=3.0,
    )
    assert next_maneuver is not None
    assert next_maneuver.name != initial.name


def test_cooldown_blocks_recently_used_maneuver():
    """A maneuver on cooldown is replaced by an allowed alternative."""
    selector = _seeded_selector(
        42,
        allowed_maneuvers=["coordinated_turn", "straight_level", "barrel_roll"],
        maneuver_cooldown_s=5.0,
    )
    # Put coordinated_turn on cooldown.
    selector._mark_cooldown("coordinated_turn", sim_time=0.0)
    assert selector._is_in_cooldown("coordinated_turn", sim_time=2.0)

    # When the rule asks for the cooled-down maneuver, it should be redirected.
    redirected = selector._apply_temporal_constraints(
        "coordinated_turn", sim_time=2.0, force=False
    )
    assert redirected != "coordinated_turn"
    assert redirected in selector.allowed_maneuvers

    # ``force=True`` bypasses the cooldown (used by emergency selection).
    forced = selector._apply_temporal_constraints(
        "coordinated_turn", sim_time=2.0, force=True
    )
    assert forced == "coordinated_turn"

    # After the cooldown expires the maneuver is selectable again.
    assert not selector._is_in_cooldown("coordinated_turn", sim_time=6.0)
    recovered = selector._apply_temporal_constraints(
        "coordinated_turn", sim_time=6.0, force=False
    )
    assert recovered == "coordinated_turn"


def test_emergency_bypasses_dwell_and_reaction_time(own_state, bandit_state):
    """Emergency selection must be allowed before dwell/reaction windows expire."""
    selector = _seeded_selector(
        42,
        allowed_maneuvers=["break_turn", "barrel_roll"],
        reaction_time_s=10.0,
        min_dwell_time_s=10.0,
    )
    selector._rule_select = lambda sit, bs: "barrel_roll"
    initial = selector.select_initial(own_state, bandit_state, sim_time=0.0)
    assert initial is not None

    # Force the emergency detector to fire regardless of geometry.
    selector.is_emergency = lambda situation: True
    emergency = selector.select_emergency(own_state, bandit_state, sim_time=0.1)
    assert emergency is not None
    assert emergency.name in selector.emergency_maneuvers


def test_history_anti_repetition_avoids_exact_repeat(own_state, bandit_state):
    """A maneuver in the recent history is replaced by a same-class alternative."""
    selector = _seeded_selector(
        0,
        allowed_maneuvers=["coordinated_turn", "straight_level"],
        history_window=3,
        history_switch_prob=1.0,
        min_dwell_time_s=0.5,
    )
    selector._rule_select = lambda sit, bs: "coordinated_turn"
    first = selector.select_initial(own_state, bandit_state, sim_time=0.0)
    assert first.name == "coordinated_turn"

    # With probability 1.0 the selector must avoid the exact repeat.
    second = selector.select(
        own_state,
        bandit_state,
        current_maneuver_name=first.name,
        elapsed_in_maneuver=3.0,
        sim_time=3.0,
    )
    assert second is not None
    assert second.name != "coordinated_turn"
    assert second.name in selector.allowed_maneuvers


def test_selector_noise_is_seeded(own_state, bandit_state):
    """With identical seeds the stochastic noise path is reproducible."""
    config = {
        "seed": 123,
        "selector_noise": 1.0,
        "allowed_maneuvers": ["coordinated_turn", "straight_level"],
        "min_dwell_time_s": 0.0,
        "reaction_time_s": 0.0,
    }
    s1 = BanditManeuverSelector(config)
    s2 = BanditManeuverSelector(config)

    s1._rule_select = lambda sit, bs: "coordinated_turn"
    s2._rule_select = lambda sit, bs: "coordinated_turn"

    m1 = s1.select(
        own_state,
        bandit_state,
        current_maneuver_name=None,
        elapsed_in_maneuver=0.0,
        sim_time=0.0,
    )
    m2 = s2.select(
        own_state,
        bandit_state,
        current_maneuver_name=None,
        elapsed_in_maneuver=0.0,
        sim_time=0.0,
    )
    assert m1 is not None
    assert m2 is not None
    assert m1.name == m2.name
