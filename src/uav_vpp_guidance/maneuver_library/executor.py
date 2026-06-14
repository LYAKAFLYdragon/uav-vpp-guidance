"""Maneuver executor: select, run, and transition between maneuvers."""
from __future__ import annotations

from typing import Optional

from .base import ControlCommand, FlightState, Maneuver, ManeuverSetpoint, ManeuverState
from .controller import InnerLoopController


class ManeuverExecutor:
    """Runs the active maneuver and converts its setpoints to actuator commands."""

    def __init__(self, controller: Optional[InnerLoopController] = None):
        self.controller = controller or InnerLoopController()
        self._current: Optional[Maneuver] = None
        self._fallback: Optional[Maneuver] = None
        self._last_command = ControlCommand()

    def set_fallback(self, maneuver: Maneuver):
        """Set a safe fallback maneuver used after completion or abort."""
        self._fallback = maneuver

    def select(self, maneuver: Maneuver, state: FlightState) -> bool:
        """Select a new maneuver if entry conditions are satisfied."""
        if not maneuver.can_enter(state):
            return False
        self._current = maneuver
        self.controller.reset()
        maneuver.enter(state)
        return True

    def update(self, state: FlightState, dt: float) -> ControlCommand:
        """Step the active maneuver and return actuator commands."""
        if self._current is None:
            if self._fallback is not None:
                self.select(self._fallback, state)
            else:
                return ControlCommand()

        maneuver = self._current
        if maneuver.is_complete(state):
            maneuver.state = ManeuverState.COMPLETE
            self._transition_to_fallback(state)
            return self.update(state, dt)

        setpoint = maneuver.update(state, dt)
        self._last_command = self.controller.update(state, setpoint, dt)
        return self._last_command

    def active_maneuver(self) -> Optional[Maneuver]:
        return self._current

    def is_idle(self) -> bool:
        return self._current is None or self._current.state == ManeuverState.COMPLETE

    def _transition_to_fallback(self, state: FlightState):
        if self._fallback is not None and self._fallback is not self._current:
            self.select(self._fallback, state)
        else:
            self._current = None
