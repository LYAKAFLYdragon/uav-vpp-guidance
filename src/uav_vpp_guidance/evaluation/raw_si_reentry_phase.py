"""Physical-unit, first-pass-aware phase labels for new re-entry collectors.

This contract intentionally does not modify the historical ``PhaseTracker``.
That tracker is part of the preserved P4 evidence and was fed normalized
observations in V5--V7.  New collectors must pass raw metres and metres per
second to this module and record the resulting labels separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import (
    RawSIFirstPassConfig,
    RawSIFirstPassEvent,
    RawSIFirstPassTracker,
)


class RawSIReentryPhaseError(ValueError):
    """Raised when a physical raw-SI phase input is invalid."""


@dataclass(frozen=True)
class RawSIReentryPhaseConfig:
    """Frozen physical thresholds for a future non-learning phase probe."""

    merge_range_m: float = 1000.0
    reentry_rate_mps: float = -25.0
    first_pass: RawSIFirstPassConfig = RawSIFirstPassConfig()

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.merge_range_m)) or float(self.merge_range_m) <= 0.0:
            raise RawSIReentryPhaseError("merge_range_m must be a finite positive physical distance")
        if not math.isfinite(float(self.reentry_rate_mps)) or float(self.reentry_rate_mps) >= 0.0:
            raise RawSIReentryPhaseError("reentry_rate_mps must be a finite negative physical rate")


class RawSIReentryPhaseTracker:
    """Classify pre-merge, post-merge, and re-entry from raw physical units.

    ``re_entry`` is impossible before the independent first-pass tracker has
    latched.  That prevents the original inbound approach after merge from
    being mistaken for a second approach.
    """

    def __init__(self, config: RawSIReentryPhaseConfig | None = None) -> None:
        self.config = config or RawSIReentryPhaseConfig()
        self._seen_merge = False
        self._first_pass_tracker = RawSIFirstPassTracker(self.config.first_pass)

    @property
    def first_pass_event(self) -> RawSIFirstPassEvent | None:
        return self._first_pass_tracker.event

    @property
    def first_pass_complete(self) -> bool:
        return self._first_pass_tracker.first_pass_complete

    def update(self, range_m: Any, range_rate_mps: Any, *, step_index: int) -> str:
        """Return one phase label from raw range [m] and range-rate [m/s]."""

        try:
            physical_range_m = float(range_m)
            physical_rate_mps = float(range_rate_mps)
        except (TypeError, ValueError) as error:
            raise RawSIReentryPhaseError("range_m and range_rate_mps must be numeric") from error
        if not math.isfinite(physical_range_m) or physical_range_m < 0.0:
            raise RawSIReentryPhaseError("range_m must be finite and non-negative")
        if not math.isfinite(physical_rate_mps):
            raise RawSIReentryPhaseError("range_rate_mps must be finite")

        if physical_range_m <= self.config.merge_range_m:
            self._seen_merge = True
        self._first_pass_tracker.update(physical_rate_mps, step_index=step_index)
        if not self._seen_merge:
            return "pre_merge"
        if not self._first_pass_tracker.first_pass_complete:
            return "post_merge"
        if physical_rate_mps <= self.config.reentry_rate_mps:
            return "re_entry"
        return "post_merge"
