"""Non-learning recovery tactic for the local empirical capability envelope.

This module only decides a temporary specialist mode and a geometry target. It
does not generate VPP commands and does not alter the existing H0 controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional


@dataclass(frozen=True)
class EnvelopePoint:
    target_x_m: float
    target_y_abs_m: float


@dataclass(frozen=True)
class RecoveryDecision:
    phase: str
    mode: str
    target: Optional[EnvelopePoint]
    elapsed_steps: int
    dwell_steps: int
    reason: str


class CapabilityEnvelopeRecoveryPolicy:
    """State machine for a bounded, non-learning envelope-recovery tactic."""

    def __init__(
        self,
        stable_points: Iterable[EnvelopePoint],
        *,
        nominal_mode: str = "h0_current_geometry",
        recovery_mode: str = "crossing",
        x_tolerance_m: float = 25.0,
        y_tolerance_m: float = 25.0,
        max_recovery_steps: int = 48,
        required_dwell_steps: int = 10,
        max_clipping_rate: float = 0.30,
        min_altitude_m: float = 500.0,
        max_range_m: float = 12000.0,
    ) -> None:
        points = tuple(stable_points)
        if not points:
            raise ValueError("stable_points must not be empty")
        if max_recovery_steps < 1:
            raise ValueError("max_recovery_steps must be positive")
        if required_dwell_steps < 1:
            raise ValueError("required_dwell_steps must be positive")
        self.stable_points = points
        self.nominal_mode = str(nominal_mode)
        self.recovery_mode = str(recovery_mode)
        self.x_tolerance_m = float(x_tolerance_m)
        self.y_tolerance_m = float(y_tolerance_m)
        self.max_recovery_steps = int(max_recovery_steps)
        self.required_dwell_steps = int(required_dwell_steps)
        self.max_clipping_rate = float(max_clipping_rate)
        self.min_altitude_m = float(min_altitude_m)
        self.max_range_m = float(max_range_m)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._elapsed_steps = 0
        self._dwell_steps = 0
        self._target: Optional[EnvelopePoint] = None

    @property
    def target(self) -> Optional[EnvelopePoint]:
        return self._target

    def _validate(self, state: dict) -> Optional[str]:
        required = ("target_x_m", "target_y_m", "altitude_m", "range_m", "clipping_rate")
        for key in required:
            value = state.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return f"nonfinite_{key}"
        if bool(state.get("backend_fallback_occurred", False)):
            return "backend_fallback"
        if bool(state.get("terminal", False)):
            return "terminal"
        if not bool(state.get("envelope_condition_valid", False)):
            return "envelope_condition_unverified"
        if float(state["altitude_m"]) < self.min_altitude_m:
            return "altitude_below_minimum"
        if float(state["range_m"]) > self.max_range_m:
            return "range_above_maximum"
        if float(state["clipping_rate"]) > self.max_clipping_rate:
            return "clipping_above_limit"
        return None

    def _inside(self, target_x_m: float, target_y_m: float) -> bool:
        y_abs = abs(float(target_y_m))
        return any(
            abs(float(point.target_x_m) - float(target_x_m)) <= self.x_tolerance_m
            and abs(float(point.target_y_abs_m) - y_abs) <= self.y_tolerance_m
            for point in self.stable_points
        )

    def _nearest_target(self, target_x_m: float, target_y_m: float) -> EnvelopePoint:
        y_abs = abs(float(target_y_m))
        return min(
            self.stable_points,
            key=lambda point: (
                ((float(point.target_x_m) - float(target_x_m)) / max(self.x_tolerance_m, 1.0)) ** 2
                + ((float(point.target_y_abs_m) - y_abs) / max(self.y_tolerance_m, 1.0)) ** 2,
                float(point.target_x_m),
                float(point.target_y_abs_m),
            ),
        )

    def decide(self, state: dict) -> RecoveryDecision:
        """Return the next tactic decision from current relative geometry."""
        safety_reason = self._validate(state)
        if safety_reason is not None:
            return RecoveryDecision(
                phase="abort",
                mode=self.nominal_mode,
                target=self._target,
                elapsed_steps=self._elapsed_steps,
                dwell_steps=self._dwell_steps,
                reason=safety_reason,
            )

        target_x_m = float(state["target_x_m"])
        target_y_m = float(state["target_y_m"])
        inside = self._inside(target_x_m, target_y_m)

        if not self._active and inside:
            return RecoveryDecision(
                phase="nominal",
                mode=self.nominal_mode,
                target=None,
                elapsed_steps=0,
                dwell_steps=0,
                reason="already_inside_stable_envelope",
            )

        if not self._active:
            self._active = True
            self._elapsed_steps = 0
            self._dwell_steps = 0
            self._target = self._nearest_target(target_x_m, target_y_m)

        if not inside:
            if self._elapsed_steps >= self.max_recovery_steps:
                return RecoveryDecision(
                    phase="abort",
                    mode=self.nominal_mode,
                    target=self._target,
                    elapsed_steps=self._elapsed_steps,
                    dwell_steps=self._dwell_steps,
                    reason="recovery_step_budget_exhausted",
                )
            self._elapsed_steps += 1
            self._dwell_steps = 0
            return RecoveryDecision(
                phase="recover",
                mode=self.recovery_mode,
                target=self._target,
                elapsed_steps=self._elapsed_steps,
                dwell_steps=0,
                reason="outside_stable_envelope",
            )

        self._dwell_steps += 1
        if self._dwell_steps >= self.required_dwell_steps:
            decision = RecoveryDecision(
                phase="complete",
                mode=self.nominal_mode,
                target=self._target,
                elapsed_steps=self._elapsed_steps,
                dwell_steps=self._dwell_steps,
                reason="stable_envelope_dwell_complete",
            )
            self._active = False
            self._target = None
            return decision
        return RecoveryDecision(
            phase="hold",
            mode=self.recovery_mode,
            target=self._target,
            elapsed_steps=self._elapsed_steps,
            dwell_steps=self._dwell_steps,
            reason="inside_stable_envelope_dwell",
        )
