"""Pure offline F06 multi-rate scheduling and delay evidence; not runtime control."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isclose, log, pi, sin
from typing import Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np

Resampling = Literal["zero_order_hold", "linear"]


@dataclass(frozen=True)
class MultiRateSchedule:
    """Offline schedule contract measured in seconds.

    All stages start at t=0 and use the low-level period as the integer clock
    quantum. At coincident ticks the order is high-level, guidance/filter,
    then low-level application.
    """

    high_level_period_s: float = 0.2
    guidance_period_s: float = 0.2
    low_level_period_s: float = 0.2
    resampling: Resampling = "zero_order_hold"
    filter_alpha: float = 0.3

    def __post_init__(self) -> None:
        values = (
            self.high_level_period_s,
            self.guidance_period_s,
            self.low_level_period_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all schedule periods must be finite and positive")
        if self.resampling not in {"zero_order_hold", "linear"}:
            raise ValueError("resampling must be 'zero_order_hold' or 'linear'")
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        for name, period in (
            ("high_level", self.high_level_period_s),
            ("guidance", self.guidance_period_s),
        ):
            ratio = period / self.low_level_period_s
            if not isclose(ratio, round(ratio), abs_tol=1.0e-9):
                raise ValueError(
                    f"{name}_period_s / low_level_period_s must be an integer "
                    "unless an explicit asynchronous resampler is approved"
                )
        ratio = self.high_level_period_s / self.guidance_period_s
        inverse_ratio = self.guidance_period_s / self.high_level_period_s
        if not (
            isclose(ratio, round(ratio), abs_tol=1.0e-9)
            or isclose(inverse_ratio, round(inverse_ratio), abs_tol=1.0e-9)
        ):
            raise ValueError(
                "high-level and guidance periods must have an integer ratio; "
                "otherwise an explicit asynchronous resampler is required"
            )

    @property
    def high_level_rate_hz(self) -> float:
        return 1.0 / self.high_level_period_s

    @property
    def guidance_rate_hz(self) -> float:
        return 1.0 / self.guidance_period_s

    @property
    def low_level_rate_hz(self) -> float:
        return 1.0 / self.low_level_period_s

    @property
    def filter_time_constant_s(self) -> float:
        """Discrete-equivalent time constant for y=a*x+(1-a)*y_prev."""
        return -self.guidance_period_s / log(1.0 - self.filter_alpha)


LEGACY_SINGLE_RATE = MultiRateSchedule()
CANDIDATE_MULTI_RATE = MultiRateSchedule(
    high_level_period_s=0.1,
    guidance_period_s=0.05,
    low_level_period_s=0.01,
    filter_alpha=1.0 - exp(-0.05 / (-0.2 / log(0.7))),
)


@dataclass
class ScheduleTrace:
    high_level_timestamps_s: List[float] = field(default_factory=list)
    guidance_timestamps_s: List[float] = field(default_factory=list)
    filter_timestamps_s: List[float] = field(default_factory=list)
    low_level_timestamps_s: List[float] = field(default_factory=list)
    applied_commands: List[Tuple[float, float]] = field(default_factory=list)
    telemetry: List[Dict[str, float]] = field(default_factory=list)


class DeterministicMultiRateScheduler:
    """Pure scheduler model with no imports from protected runtime modules."""

    def __init__(self, schedule: MultiRateSchedule):
        self.schedule = schedule
        self.reset()

    def reset(self) -> None:
        self._action = 0.0
        self._filtered_command = 0.0
        self._filter_initialized = False
        self._action_timestamp_s = 0.0
        self._guidance_timestamp_s = 0.0
        self._tick = 0

    @staticmethod
    def _is_tick(time_s: float, period_s: float) -> bool:
        return isclose(time_s / period_s, round(time_s / period_s), abs_tol=1.0e-9)

    def run(
        self,
        duration_s: float,
        high_level: Callable[[float], float],
        guidance: Callable[[float, float], float] = lambda action, _time: action,
    ) -> ScheduleTrace:
        """Run [0, duration_s) and return exact invocation timestamps.

        The filter advances only when guidance advances. Low-level ticks apply
        the latest filtered command by zero-order hold. Telemetry is timestamped
        at the application tick and carries the command source ages.
        """
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        self.reset()
        trace = ScheduleTrace()
        ticks = int(round(duration_s / self.schedule.low_level_period_s))
        if not isclose(ticks * self.schedule.low_level_period_s, duration_s, abs_tol=1.0e-9):
            raise ValueError("duration_s must be an integer number of low-level ticks")
        for tick in range(ticks):
            time_s = tick * self.schedule.low_level_period_s
            high_due = self._is_tick(time_s, self.schedule.high_level_period_s)
            guidance_due = self._is_tick(time_s, self.schedule.guidance_period_s)
            if high_due:
                self._action = float(high_level(time_s))
                self._action_timestamp_s = time_s
                trace.high_level_timestamps_s.append(time_s)
            if guidance_due:
                raw_command = float(guidance(self._action, time_s))
                if self._filter_initialized:
                    self._filtered_command = (
                        self.schedule.filter_alpha * raw_command
                        + (1.0 - self.schedule.filter_alpha) * self._filtered_command
                    )
                else:
                    self._filtered_command = raw_command
                    self._filter_initialized = True
                self._guidance_timestamp_s = time_s
                trace.guidance_timestamps_s.append(time_s)
                trace.filter_timestamps_s.append(time_s)
            trace.low_level_timestamps_s.append(time_s)
            trace.applied_commands.append((time_s, self._filtered_command))
            trace.telemetry.append(
                {
                    "timestamp_s": time_s,
                    "action_timestamp_s": self._action_timestamp_s,
                    "guidance_timestamp_s": self._guidance_timestamp_s,
                    "action_age_s": time_s - self._action_timestamp_s,
                    "guidance_age_s": time_s - self._guidance_timestamp_s,
                    "filtered_command": self._filtered_command,
                }
            )
        return trace


def resample_command(
    previous_value: float,
    next_value: float | None,
    fraction: float,
    mode: Resampling,
) -> float:
    """Define hold/interpolation independently of scheduler callbacks.

    Linear interpolation is valid only when an explicit next sample exists;
    realtime policy actions therefore use zero-order hold unless a separately
    approved predictive source supplies the next action.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if mode == "zero_order_hold":
        return float(previous_value)
    if mode == "linear":
        if next_value is None:
            raise ValueError("linear interpolation requires an explicit next sample")
        return float((1.0 - fraction) * previous_value + fraction * next_value)
    raise ValueError(f"unsupported resampling mode: {mode}")


def sinusoid_response(
    schedule: MultiRateSchedule,
    frequency_hz: float,
    *,
    duration_s: float = 20.0,
) -> Dict[str, float]:
    """Measure discrete command-to-low-level delay/phase for an offline trace.

    It includes scheduler hold plus the documented command filter only. It
    deliberately excludes unmeasured simulator/actuator/sensor effects, which
    are reserved for task 6's F03/F11 evidence budget.
    """
    if frequency_hz <= 0.0 or frequency_hz >= 0.45 * schedule.guidance_rate_hz:
        raise ValueError("frequency_hz must be positive and below the guidance Nyquist guard")
    trace = DeterministicMultiRateScheduler(schedule).run(
        duration_s,
        high_level=lambda time_s: sin(2.0 * pi * frequency_hz * time_s),
    )
    time_s = np.asarray([sample[0] for sample in trace.applied_commands])
    applied = np.asarray([sample[1] for sample in trace.applied_commands])
    warmup = time_s >= duration_s / 2.0
    omega = 2.0 * pi * frequency_hz
    basis = np.column_stack((np.sin(omega * time_s[warmup]), np.cos(omega * time_s[warmup])))
    sin_coef, cos_coef = np.linalg.lstsq(basis, applied[warmup], rcond=None)[0]
    amplitude = float(np.hypot(sin_coef, cos_coef))
    phase_rad = float(np.arctan2(cos_coef, sin_coef))
    delay_s = float(-phase_rad / omega)
    while delay_s < 0.0:
        delay_s += 1.0 / frequency_hz
    return {
        "frequency_hz": float(frequency_hz),
        "amplitude_ratio": amplitude,
        "phase_lag_rad": float(-phase_rad),
        "apparent_delay_s": delay_s,
        "filter_time_constant_s": schedule.filter_time_constant_s,
        "scope": "offline scheduler hold plus command filter; excludes actuator, backend, sensor, and communication-delay effects",
    }


def compare_delay_spectra(
    baseline: MultiRateSchedule = LEGACY_SINGLE_RATE,
    candidate: MultiRateSchedule = CANDIDATE_MULTI_RATE,
    frequencies_hz: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
) -> Dict[str, object]:
    """Produce deterministic baseline/candidate sweep evidence."""
    supported = tuple(
        frequency for frequency in frequencies_hz
        if frequency < 0.45 * min(baseline.guidance_rate_hz, candidate.guidance_rate_hz)
    )
    return {
        "baseline": [sinusoid_response(baseline, frequency) for frequency in supported],
        "candidate": [sinusoid_response(candidate, frequency) for frequency in supported],
        "excluded_frequencies_hz": [float(f) for f in frequencies_hz if f not in supported],
    }


def trace_to_dict(trace: ScheduleTrace) -> Dict[str, object]:
    """Return a JSON-safe exact scheduler trace."""
    return {
        "high_level_timestamps_s": trace.high_level_timestamps_s,
        "guidance_timestamps_s": trace.guidance_timestamps_s,
        "filter_timestamps_s": trace.filter_timestamps_s,
        "low_level_timestamps_s": trace.low_level_timestamps_s,
        "applied_commands": [
            {"timestamp_s": time_s, "command": command}
            for time_s, command in trace.applied_commands
        ],
        "telemetry": trace.telemetry,
    }
