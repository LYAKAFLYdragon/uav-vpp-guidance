from __future__ import annotations

import pytest

from uav_vpp_guidance.evaluation.terminal_outcome_reward import (
    TerminalOutcomeRewardError,
    terminal_outcome_reward,
)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("target_killed", 1.0),
        ("target_crash_or_out_of_bounds", 1.0),
        ("timeout_hp_advantage", 1.0),
        ("timeout_hp_disadvantage", -1.0),
        ("timeout_draw", -0.25),
        ("ego_crash_or_out_of_bounds", -2.0),
        ("mutual_aircraft_failure", -1.25),
    ],
)
def test_terminal_reason_mapping_is_exact(reason: str, expected: float):
    assert terminal_outcome_reward(reason, terminal=True).reward == expected


def test_target_crash_is_a_win_and_nonterminal_steps_receive_no_terminal_reward():
    assert terminal_outcome_reward("target_crash_or_out_of_bounds", terminal=True).outcome == "win"
    assert terminal_outcome_reward("ego_crash_or_out_of_bounds", terminal=False).reward == 0.0


def test_unknown_terminal_reason_is_rejected():
    with pytest.raises(TerminalOutcomeRewardError, match="unmapped"):
        terminal_outcome_reward("opaque_new_reason", terminal=True)
