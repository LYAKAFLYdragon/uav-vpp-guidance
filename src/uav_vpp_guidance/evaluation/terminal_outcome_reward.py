"""Frozen terminal-outcome reward semantics for the V1 re-entry pilot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TerminalOutcomeRewardError(ValueError):
    """Raised when a terminal outcome cannot be mapped unambiguously."""


@dataclass(frozen=True)
class TerminalOutcomeRewardConfig:
    """Small, final-step-only reward terms declared before pilot execution."""

    win_bonus: float = 1.0
    loss_penalty: float = 1.0
    draw_penalty: float = 0.25
    ego_crash_oob_supplemental_penalty: float = 1.0


@dataclass(frozen=True)
class TerminalOutcomeReceipt:
    reason: str
    outcome: str
    ego_crash_or_oob: bool
    reward: float


_OUTCOMES = {
    "target_killed": ("win", False),
    "target_crash_or_out_of_bounds": ("win", False),
    "timeout_hp_advantage": ("win", False),
    "ego_crash_or_out_of_bounds": ("loss", True),
    "timeout_hp_disadvantage": ("loss", False),
    "timeout_draw": ("draw", False),
    "mutual_aircraft_failure": ("draw", True),
}


def terminal_outcome_reward(
    reason: Any,
    *,
    terminal: bool,
    config: TerminalOutcomeRewardConfig | None = None,
) -> TerminalOutcomeReceipt:
    """Return the preregistered reward receipt; nonterminal steps always get zero."""

    if not terminal:
        return TerminalOutcomeReceipt(
            reason="nonterminal",
            outcome="nonterminal",
            ego_crash_or_oob=False,
            reward=0.0,
        )
    normalized = str(reason or "")
    if normalized not in _OUTCOMES:
        raise TerminalOutcomeRewardError(f"unmapped terminal reason: {normalized!r}")
    outcome, ego_crash = _OUTCOMES[normalized]
    settings = config or TerminalOutcomeRewardConfig()
    if outcome == "win":
        reward = float(settings.win_bonus)
    elif outcome == "loss":
        reward = -float(settings.loss_penalty)
    else:
        reward = -float(settings.draw_penalty)
    if ego_crash:
        reward -= float(settings.ego_crash_oob_supplemental_penalty)
    return TerminalOutcomeReceipt(
        reason=normalized,
        outcome=outcome,
        ego_crash_or_oob=ego_crash,
        reward=reward,
    )
