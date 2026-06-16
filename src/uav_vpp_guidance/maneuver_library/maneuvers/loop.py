"""Vertical loop maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Loop(Maneuver):
    """Fly a vertical circle (loop) in the pitch plane.

    Parameters
    ----------
    entry_speed_mps : float
        True airspeed at loop entry (m/s).  Default 350 m/s.
    nz_target : float
        Target load factor during the pull (g).  Default 5.0 g.
    throttle : float
        Throttle setting during the maneuver.  Default 1.0.
    min_speed_mps : float
        Speed floor; commands are softened if speed drops below this (m/s).
        Default 150 m/s.
    max_alpha_deg : float
        Angle-of-attack ceiling (deg).  Default 25 deg.
    """

    name = "loop"

    GRAVITY = 9.80665

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.entry_speed = self.params.get("entry_speed_mps", 350.0)
        self.nz_target = self.params.get("nz_target", 4.0)
        self.throttle = self.params.get("throttle", 1.0)
        self.min_speed = self.params.get("min_speed_mps", 150.0)
        self.max_alpha = math.radians(self.params.get("max_alpha_deg", 25.0))
        self._start_altitude: float | None = None
        self._min_elapsed: float = 0.0

    def can_enter(self, state: FlightState) -> bool:
        return (
            state.velocity_mps >= self.entry_speed * 0.9
            and state.altitude_m >= 2500.0
            and self.nz_target <= 7.0
        )

    def enter(self, state: FlightState):
        super().enter(state)
        self._start_altitude = state.altitude_m
        # One full loop at the nominal radius takes roughly this long.
        r = self.entry_speed**2 / ((self.nz_target - 1.0) * self.GRAVITY)
        self._min_elapsed = 2.0 * math.pi * r / max(self.entry_speed, 50.0) * 0.5

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        # Use load-factor command only; the controller will modulate elevator
        # as speed changes, which is more energy-conservative than a fixed
        # pitch-rate command at low speed.
        return ManeuverSetpoint(
            phi_ref=0.0,
            nz_ref=self.nz_target,
            throttle_ref=self.throttle,
            min_speed_mps=self.min_speed,
            max_alpha_rad=self.max_alpha,
            min_altitude_m=1500.0,
        )

    def is_complete(self, state: FlightState) -> bool:
        if self._start_altitude is None:
            return False
        # Complete when back near level at or above the entry altitude.
        near_level = abs(state.theta_rad) <= math.radians(10.0)
        altitude_regained = state.altitude_m >= self._start_altitude - 100.0
        return self._elapsed >= self._min_elapsed and near_level and altitude_regained
