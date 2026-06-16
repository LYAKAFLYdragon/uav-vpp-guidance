"""Maneuver executor: select, run, and transition between maneuvers."""
from __future__ import annotations

from typing import Optional

from .base import ControlCommand, FlightState, Maneuver, ManeuverSetpoint, ManeuverState
from .controller import InnerLoopController
from .telemetry import ManeuverTelemetry


class ManeuverExecutor:
    """Runs the active maneuver and converts its setpoints to actuator commands."""

    def __init__(
        self,
        controller: Optional[InnerLoopController] = None,
        telemetry: Optional[ManeuverTelemetry] = None,
    ):
        self.controller = controller or InnerLoopController()
        self.telemetry = telemetry
        self._current: Optional[Maneuver] = None
        self._fallback: Optional[Maneuver] = None
        self._last_command = ControlCommand()

    def set_fallback(self, maneuver: Maneuver):
        """Set a safe fallback maneuver used after completion or abort."""
        self._fallback = maneuver

    def select(self, maneuver: Maneuver, state: FlightState, reason: str = "switch") -> bool:
        """Select a new maneuver if entry conditions are satisfied."""
        if not maneuver.can_enter(state):
            return False
        self._current = maneuver
        self.controller.reset()
        maneuver.enter(state)
        if self.telemetry is not None:
            self.telemetry.start(maneuver.name, state.t, reason=reason)
        return True

    def update(self, state: FlightState, dt: float) -> ControlCommand:
        """Step the active maneuver and return actuator commands."""
        if self._current is None:
            if self._fallback is not None:
                self.select(self._fallback, state, reason="fallback")
            else:
                return ControlCommand()

        maneuver = self._current
        if maneuver.is_complete(state):
            maneuver.state = ManeuverState.COMPLETE
            if self.telemetry is not None:
                self.telemetry.end(state.t, "complete")
            self._transition_to_fallback(state)
            return self.update(state, dt)

        # Automatic envelope/abort-guard check.  If the maneuver reports it
        # cannot continue safely, transition to the fallback maneuver.
        if maneuver.should_abort(state):
            self.abort(state, reason="envelope")
            return self.update(state, dt)

        setpoint = maneuver.update(state, dt)
        if self.telemetry is not None:
            self.telemetry.set_phase(maneuver.phase())
        self._last_command = self.controller.update(state, setpoint, dt)
        return self._last_command

    def abort(self, state: FlightState, reason: str = "envelope"):
        """Abort the active maneuver and transition to fallback."""
        if self._current is not None:
            self._current.abort()
            if self.telemetry is not None:
                self.telemetry.end(state.t, "abort", abort_reason=reason)
        self._transition_to_fallback(state)

    def active_maneuver(self) -> Optional[Maneuver]:
        return self._current

    def is_idle(self) -> bool:
        return self._current is None or self._current.state == ManeuverState.COMPLETE

    def _transition_to_fallback(self, state: FlightState):
        if self._fallback is not None and self._fallback is not self._current:
            self.select(self._fallback, state, reason="fallback")
        else:
            self._current = None
