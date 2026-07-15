"""Read-only V8-phase and terminal-reward annotation for a future pilot."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig
from uav_vpp_guidance.evaluation.raw_si_reentry_phase import (
    RawSIReentryPhaseConfig,
    RawSIReentryPhaseTracker,
)
from uav_vpp_guidance.evaluation.terminal_outcome_reward import (
    TerminalOutcomeRewardConfig,
    terminal_outcome_reward,
)


RANGE_REFERENCE_M = 5000.0
RANGE_RATE_REFERENCE_MPS = 200.0


class ReentryRewardPreflightError(ValueError):
    """Raised when a continuous ledger violates the new pilot contract."""


def annotate_ledger(
    ledger: dict[str, Any],
    *,
    terminal_reason: str,
    first_pass_event: Mapping[str, Any],
    reward_config: TerminalOutcomeRewardConfig | None = None,
) -> dict[str, int | float]:
    """Append V8 phase and terminal reward fields without changing actions."""

    steps = ledger.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ReentryRewardPreflightError("ledger must contain at least one step")
    tracker = RawSIReentryPhaseTracker(
        RawSIReentryPhaseConfig(first_pass=RawSIFirstPassConfig(**dict(first_pass_event)))
    )
    reentry_66d_steps = 0
    terminal_reward_total = 0.0
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ReentryRewardPreflightError("ledger step must be mutable mapping")
        base = np.asarray(step.get("base_observation_16d"), dtype=np.float64)
        if base.shape != (16,) or not np.isfinite(base).all():
            raise ReentryRewardPreflightError("invalid 16-D base observation")
        raw_range_m = float(base[0] * RANGE_REFERENCE_M)
        raw_rate_mps = float(base[1] * RANGE_RATE_REFERENCE_MPS)
        phase = tracker.update(raw_range_m, raw_rate_mps, step_index=index)
        recorded_first_pass = bool(step.get("raw_si_first_pass_latched"))
        if recorded_first_pass != tracker.first_pass_complete:
            raise ReentryRewardPreflightError("raw first-pass replay mismatch")
        receipt = terminal_outcome_reward(
            terminal_reason,
            terminal=index == len(steps) - 1,
            config=reward_config,
        )
        step["pilot_v1_raw_range_m"] = raw_range_m
        step["pilot_v1_raw_range_rate_mps"] = raw_rate_mps
        step["pilot_v1_phase"] = phase
        step["pilot_v1_first_pass_latched"] = tracker.first_pass_complete
        step["pilot_v1_terminal_reward"] = receipt.reward
        step["pilot_v1_terminal_outcome"] = receipt.outcome
        terminal_reward_total += receipt.reward
        if phase == "re_entry" and step.get("observation_66d") is not None:
            reentry_66d_steps += 1
    ledger["pilot_v1_reward_contract"] = {
        "phase_contract": "raw_physical_first_pass_aware_v1",
        "range_reference_m": RANGE_REFERENCE_M,
        "range_rate_reference_mps": RANGE_RATE_REFERENCE_MPS,
        "terminal_reason": terminal_reason,
        "terminal_reward_final_step_only": True,
    }
    return {
        "reentry_66d_steps": reentry_66d_steps,
        "terminal_reward_total": terminal_reward_total,
    }
