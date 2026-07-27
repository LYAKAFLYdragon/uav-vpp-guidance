"""Pure offline F16 mode-switch state-machine evidence; no runtime dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


CANONICAL_ASPECT_THRESHOLD_DEG = 25.0
CANONICAL_RANGE_THRESHOLD_M = 3000.0
CANONICAL_CLOSING_SPEED_THRESHOLD_MPS = 50.0


class ModeState(str, Enum):
    """Offline candidate states, deliberately separate from the environment latch."""

    VPP = "vpp"
    DIRECT_TRACK_PN = "direct_track_pn"
    DISABLED = "disabled"
    FAILSAFE = "failsafe"


@dataclass(frozen=True)
class ModeSwitchParameters:
    """Canonical parameter ownership for offline F16 candidate evaluation."""

    enabled: bool = False
    aspect_threshold_deg: float = CANONICAL_ASPECT_THRESHOLD_DEG
    range_threshold_m: float = CANONICAL_RANGE_THRESHOLD_M
    closing_speed_threshold_mps: float = CANONICAL_CLOSING_SPEED_THRESHOLD_MPS
    crossing_aspect_threshold_deg: float | None = None
    release_aspect_threshold_deg: float = 30.0
    release_range_threshold_m: float = 3300.0
    release_closing_speed_threshold_mps: float = 40.0
    min_active_dwell_steps: int = 2
    max_active_steps: int | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ModeSwitchParameters":
        """Resolve explicit values first; omitted threshold has one canonical 25° owner."""
        values = dict(config or {})
        params = cls(
            enabled=bool(values.get("enabled", False)),
            aspect_threshold_deg=float(values.get("aspect_threshold_deg", CANONICAL_ASPECT_THRESHOLD_DEG)),
            range_threshold_m=float(values.get("range_threshold_m", CANONICAL_RANGE_THRESHOLD_M)),
            closing_speed_threshold_mps=float(values.get("closing_speed_threshold_mps", CANONICAL_CLOSING_SPEED_THRESHOLD_MPS)),
            crossing_aspect_threshold_deg=(None if values.get("crossing_aspect_threshold_deg") is None else float(values["crossing_aspect_threshold_deg"])),
            release_aspect_threshold_deg=float(values.get("release_aspect_threshold_deg", values.get("aspect_threshold_deg", CANONICAL_ASPECT_THRESHOLD_DEG) + 5.0)),
            release_range_threshold_m=float(values.get("release_range_threshold_m", values.get("range_threshold_m", CANONICAL_RANGE_THRESHOLD_M) * 1.1)),
            release_closing_speed_threshold_mps=float(values.get("release_closing_speed_threshold_mps", max(0.0, values.get("closing_speed_threshold_mps", CANONICAL_CLOSING_SPEED_THRESHOLD_MPS) - 10.0))),
            min_active_dwell_steps=int(values.get("min_active_dwell_steps", 2)),
            max_active_steps=(None if values.get("max_active_steps") is None else int(values["max_active_steps"])),
        )
        if params.min_active_dwell_steps < 0:
            raise ValueError("min_active_dwell_steps must be non-negative")
        if params.max_active_steps is not None and params.max_active_steps < 1:
            raise ValueError("max_active_steps must be at least one when configured")
        return params


@dataclass(frozen=True)
class ModeEvent:
    """Geometry sample consumed by the pure candidate models."""

    range_m: float
    range_rate_mps: float
    aspect_abs_deg: float

    @property
    def closing_speed_mps(self) -> float:
        return abs(self.range_rate_mps)


@dataclass(frozen=True)
class Transition:
    previous_state: str
    state: str
    transition: str
    reason: str
    mode_switch_effective: bool
    virtual_point_source: str
    effective_guidance_mode: str
    active_steps: int


def _finite_event(event: ModeEvent) -> bool:
    return all(isfinite(value) for value in (event.range_m, event.range_rate_mps, event.aspect_abs_deg))


def entry_reason(event: ModeEvent, params: ModeSwitchParameters) -> str | None:
    """Return an entry reason only when the frozen gate conditions hold."""
    if not _finite_event(event):
        return "invalid_geometry"
    if event.range_m > params.range_threshold_m:
        return "range_above_entry_threshold"
    if event.range_rate_mps > 0.0:
        return "opening_range_rate"
    if event.closing_speed_mps < params.closing_speed_threshold_mps:
        return "closing_speed_below_entry_threshold"
    if event.aspect_abs_deg <= params.aspect_threshold_deg:
        return "entry_low_aspect"
    if params.crossing_aspect_threshold_deg is not None and abs(event.aspect_abs_deg - 90.0) <= params.crossing_aspect_threshold_deg:
        return "entry_crossing"
    return None


def hold_reason(event: ModeEvent, params: ModeSwitchParameters) -> str | None:
    """Use wider release thresholds to make hysteresis explicit and deterministic."""
    if not _finite_event(event):
        return None
    if event.range_m > params.release_range_threshold_m:
        return None
    if event.range_rate_mps > 0.0:
        return None
    if event.closing_speed_mps < params.release_closing_speed_threshold_mps:
        return None
    if event.aspect_abs_deg <= params.release_aspect_threshold_deg:
        return "hold_low_aspect_hysteresis"
    if params.crossing_aspect_threshold_deg is not None and abs(event.aspect_abs_deg - 90.0) <= params.crossing_aspect_threshold_deg + 5.0:
        return "hold_crossing_hysteresis"
    return None


class LegacyPermanentLatch:
    """Faithful offline model of the current episode-only latch behavior."""

    name = "legacy_permanent_latch"

    def __init__(self, params: ModeSwitchParameters):
        self.params = params
        self.reset()

    def reset(self) -> Transition:
        previous = getattr(self, "state", ModeState.VPP)
        self.state, self.active_steps = ModeState.VPP, 0
        return self._transition(previous, "reset", "episode_reset")

    def step(self, event: ModeEvent) -> Transition:
        previous = self.state
        if not self.params.enabled:
            self.state = ModeState.DISABLED
            return self._transition(previous, "disabled", "mode_switch_disabled")
        if not _finite_event(event):
            self.state = ModeState.FAILSAFE
            return self._transition(previous, "failsafe", "invalid_geometry")
        reason = entry_reason(event, self.params)
        if reason is not None:
            self.state = ModeState.DIRECT_TRACK_PN
        if self.state is ModeState.DIRECT_TRACK_PN:
            self.active_steps += 1
            return self._transition(previous, "enter" if reason else "hold", reason or "latched_until_reset")
        return self._transition(previous, "remain_vpp", "entry_guard_not_met")

    def _transition(self, previous: ModeState, transition: str, reason: str) -> Transition:
        effective = self.state is ModeState.DIRECT_TRACK_PN
        return Transition(previous.value, self.state.value, transition, reason, effective, "direct_track" if effective else "vpp_policy", "proportional_navigation" if effective else "los_rate", self.active_steps)


class ReleaseableLatch(LegacyPermanentLatch):
    """Comparison candidate: releases immediately when the hold guard clears."""

    name = "releaseable_latch"

    def step(self, event: ModeEvent) -> Transition:
        previous = self.state
        if not self.params.enabled:
            self.state = ModeState.DISABLED
            return self._transition(previous, "disabled", "mode_switch_disabled")
        if not _finite_event(event):
            self.state = ModeState.FAILSAFE
            return self._transition(previous, "failsafe", "invalid_geometry")
        reason = entry_reason(event, self.params)
        if self.state is ModeState.VPP and reason is not None:
            self.state, self.active_steps = ModeState.DIRECT_TRACK_PN, 1
            return self._transition(previous, "enter", reason)
        if self.state is ModeState.DIRECT_TRACK_PN:
            hold = hold_reason(event, self.params)
            if hold is None:
                self.state, self.active_steps = ModeState.VPP, 0
                return self._transition(previous, "release", "release_guard_met")
            self.active_steps += 1
            return self._transition(previous, "hold", hold)
        return self._transition(previous, "remain_vpp", "entry_guard_not_met")


class ExplicitModeStateMachine(ReleaseableLatch):
    """Candidate with dwell, release, re-entry, reset, and timeout/failsafe telemetry."""

    name = "explicit_state_machine"

    def step(self, event: ModeEvent) -> Transition:
        previous = self.state
        if not self.params.enabled:
            self.state = ModeState.DISABLED
            return self._transition(previous, "disabled", "mode_switch_disabled")
        if not _finite_event(event):
            self.state = ModeState.FAILSAFE
            return self._transition(previous, "failsafe", "invalid_geometry")
        if self.state is ModeState.FAILSAFE:
            return self._transition(previous, "hold_failsafe", "reset_required_after_invalid_geometry")
        reason = entry_reason(event, self.params)
        if self.state is ModeState.VPP and reason is not None:
            self.state, self.active_steps = ModeState.DIRECT_TRACK_PN, 1
            return self._transition(previous, "enter", reason)
        if self.state is ModeState.DIRECT_TRACK_PN:
            self.active_steps += 1
            if self.params.max_active_steps is not None and self.active_steps > self.params.max_active_steps:
                self.state = ModeState.FAILSAFE
                return self._transition(previous, "timeout_failsafe", "max_active_steps_exceeded")
            hold = hold_reason(event, self.params)
            if self.active_steps <= self.params.min_active_dwell_steps:
                return self._transition(previous, "hold_dwell", "minimum_active_dwell")
            if hold is None:
                self.state, self.active_steps = ModeState.VPP, 0
                return self._transition(previous, "release", "release_guard_met")
            return self._transition(previous, "hold", hold)
        return self._transition(previous, "remain_vpp", "entry_guard_not_met")


def _trace(model: LegacyPermanentLatch, events: list[ModeEvent]) -> list[dict[str, object]]:
    return [asdict(model.step(event)) for event in events]


def compare_candidates(config: Mapping[str, Any] | None = None) -> dict[str, object]:
    """Compare permanent latch, immediate release, and explicit state-machine candidates."""
    params = ModeSwitchParameters.from_config({"enabled": True, **dict(config or {})})
    traces = {
        "non_trigger": [ModeEvent(2000.0, -120.0, 45.0)],
        "entry_hold_release_reentry": [
            ModeEvent(2000.0, -120.0, 10.0),
            ModeEvent(2100.0, -100.0, 28.0),
            ModeEvent(3600.0, -100.0, 28.0),
            ModeEvent(2000.0, -120.0, 10.0),
        ],
        "crossing": [ModeEvent(2000.0, -120.0, 90.0)],
        "invalid": [ModeEvent(float("nan"), -120.0, 10.0)],
    }
    models = (LegacyPermanentLatch, ReleaseableLatch, ExplicitModeStateMachine)
    return {
        "canonical_defaults": asdict(ModeSwitchParameters.from_config({"enabled": True})),
        "resolved_parameters": asdict(params),
        "candidate_traces": {
            model.name: {name: _trace(model(params), events) for name, events in traces.items()}
            for model in models
        },
        "comparison": {
            "legacy_release_worthy_state": "remains direct_track_pn until reset",
            "releaseable_release_worthy_state": "returns to vpp immediately after hold guard clears",
            "state_machine_release_worthy_state": "holds for configured dwell then returns to vpp when release guard clears",
            "runtime_selection": "none; offline evidence only",
        },
        "scope": "pure deterministic transition model; does not import or modify tracking_env, runtime defaults, or configuration",
    }
