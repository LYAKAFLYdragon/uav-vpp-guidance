"""Deterministic raw-SI first-pass event contract for future collectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


class RawSIFirstPassEventError(ValueError):
    """Raised when a raw-SI first-pass event contract is invalid."""


@dataclass(frozen=True)
class RawSIFirstPassConfig:
    """A stable close-then-open event, independent of env-specific flags."""

    closing_rate_threshold_mps: float = 0.0
    opening_rate_threshold_mps: float = 0.0
    minimum_closing_steps: int = 3
    minimum_opening_steps: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            ("closing_rate_threshold_mps", self.closing_rate_threshold_mps),
            ("opening_rate_threshold_mps", self.opening_rate_threshold_mps),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise RawSIFirstPassEventError(f"{name} must be a finite non-negative number")
        for name, value in (
            ("minimum_closing_steps", self.minimum_closing_steps),
            ("minimum_opening_steps", self.minimum_opening_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RawSIFirstPassEventError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RawSIFirstPassEvent:
    """One latched physical pass event derived only from range-rate history."""

    event_step_index: int
    closing_steps_before_event: int
    opening_steps_at_event: int
    range_rate_mps: float


class RawSIFirstPassTracker:
    """Latch once after a confirmed approach is followed by stable separation."""

    def __init__(self, config: RawSIFirstPassConfig | None = None) -> None:
        self.config = config or RawSIFirstPassConfig()
        self._closing_steps = 0
        self._opening_steps = 0
        self._event: RawSIFirstPassEvent | None = None

    @property
    def event(self) -> RawSIFirstPassEvent | None:
        return self._event

    @property
    def first_pass_complete(self) -> bool:
        return self._event is not None

    def update(self, range_rate_mps: Any, *, step_index: int) -> RawSIFirstPassEvent | None:
        if self._event is not None:
            return self._event
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise RawSIFirstPassEventError("step_index must be a non-negative integer")
        try:
            range_rate = float(range_rate_mps)
        except (TypeError, ValueError) as error:
            raise RawSIFirstPassEventError("range_rate_mps must be numeric") from error
        if not math.isfinite(range_rate):
            raise RawSIFirstPassEventError("range_rate_mps must be finite")
        if range_rate < -self.config.closing_rate_threshold_mps:
            self._closing_steps += 1
        if self._closing_steps >= self.config.minimum_closing_steps and range_rate >= self.config.opening_rate_threshold_mps:
            self._opening_steps += 1
            if self._opening_steps >= self.config.minimum_opening_steps:
                self._event = RawSIFirstPassEvent(
                    event_step_index=step_index,
                    closing_steps_before_event=self._closing_steps,
                    opening_steps_at_event=self._opening_steps,
                    range_rate_mps=range_rate,
                )
        else:
            self._opening_steps = 0
        return self._event


def replay_raw_si_first_pass(
    steps: Sequence[Mapping[str, Any]], config: RawSIFirstPassConfig | None = None
) -> RawSIFirstPassEvent | None:
    """Replay one ledger's raw-SI range rates without reading env flags."""

    tracker = RawSIFirstPassTracker(config)
    for index, step in enumerate(steps):
        raw = step.get("raw_si_phase_input") if isinstance(step, Mapping) else None
        if not isinstance(raw, Mapping) or "range_rate_mps" not in raw:
            raise RawSIFirstPassEventError("ledger step lacks raw_si_phase_input.range_rate_mps")
        event = tracker.update(raw["range_rate_mps"], step_index=index)
        if event is not None:
            return event
    return None
