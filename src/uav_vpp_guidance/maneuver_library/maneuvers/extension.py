"""Energy extension / separation maneuver."""
from __future__ import annotations

import math

from ..base import FlightState, Maneuver, ManeuverSetpoint


class Extension(Maneuver):
    """Execute an extension: level or slightly descending flight at full throttle
    to increase separation from an opponent.

    Unlike ``dive`` this maneuver does not recover at a target altitude; it
    continues until the configured duration elapses or until a hard envelope
    floor is approached.  It is useful for disengaging from a close-range
    merge or for running down an opponent from behind.

    Parameters
    ----------
    dive_angle_deg : float
        Slight descent angle (deg).  Default 5 deg; set to 0 for level extension.
    throttle : float
        Throttle setting.  Default 1.0.
    velocity_ref_mps : float
        Target airspeed (m/s).  Default 350 m/s.
    max_duration_s : float
        Maximum maneuver duration (s).  Default 6.0 s.
    min_altitude_m : float
        Hard altitude floor (m).  Default 1500 m.
    min_speed_mps : float
        Hard speed floor (m/s).  Default 160 m/s.
    """

    name = "extension"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.dive_angle = math.radians(self.params.get("dive_angle_deg", 5.0))
        self.throttle = self.params.get("throttle", 1.0)
        self.velocity_ref = self.params.get("velocity_ref_mps", 350.0)
        self.max_duration = self.params.get("max_duration_s", 6.0)
        self.min_altitude = self.params.get("min_altitude_m", 1500.0)
        self.min_speed = self.params.get("min_speed_mps", 160.0)

    def can_enter(self, state: FlightState) -> bool:
        return (
            state.velocity_mps >= 150.0
            and state.altitude_m >= self.min_altitude + 500.0
        )

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        super().update(state, dt)
        return ManeuverSetpoint(
            phi_ref=0.0,
            theta_ref=-self.dive_angle,
            velocity_ref=self.velocity_ref,
            throttle_ref=self.throttle,
            min_altitude_m=self.min_altitude,
            min_speed_mps=self.min_speed,
        )

    def is_complete(self, state: FlightState) -> bool:
        return self._elapsed >= self.max_duration
