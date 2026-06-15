"""Base classes for the maneuver library."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ManeuverState(Enum):
    """Finite states of a maneuver execution."""

    IDLE = auto()
    ENTRY = auto()
    EXECUTE = auto()
    EXIT = auto()
    COMPLETE = auto()
    ABORT = auto()


@dataclass
class FlightState:
    """Normalized aircraft state used by the maneuver library.

    Units are SI unless otherwise noted.  The library is backend-agnostic;
    JSBSim-specific properties must be converted before constructing this
    object.
    """

    t: float = 0.0
    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_mps: float = 0.0
    altitude_m: float = 0.0
    phi_rad: float = 0.0
    theta_rad: float = 0.0
    psi_rad: float = 0.0
    p_rps: float = 0.0
    q_rps: float = 0.0
    r_rps: float = 0.0
    nz: float = 1.0
    alpha_rad: float = 0.0
    beta_rad: float = 0.0
    mach: float = 0.0

    def speed_mps(self) -> float:
        return self.velocity_mps

    def heading_rad(self) -> float:
        return self.psi_rad


@dataclass
class ControlCommand:
    """Normalized actuator commands sent to JSBSim."""

    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def clamp(self) -> "ControlCommand":
        return ControlCommand(
            elevator=float(np.clip(self.elevator, -1.0, 1.0)),
            aileron=float(np.clip(self.aileron, -1.0, 1.0)),
            rudder=float(np.clip(self.rudder, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
        )


@dataclass
class ManeuverSetpoint:
    """Setpoint produced by a maneuver primitive.

    The inner-loop controller is responsible for tracking these setpoints and
    producing normalized surface commands.
    """

    phi_ref: Optional[float] = None
    theta_ref: Optional[float] = None
    psi_rate_ref: Optional[float] = None
    q_ref: Optional[float] = None
    p_ref: Optional[float] = None
    r_ref: Optional[float] = None
    nz_ref: Optional[float] = None
    throttle_ref: Optional[float] = None
    velocity_ref: Optional[float] = None

    # Optional flight-envelope limits for the controller to enforce.
    min_altitude_m: Optional[float] = None
    min_speed_mps: Optional[float] = None
    max_alpha_rad: Optional[float] = None


class Maneuver:
    """Abstract base class for a maneuver primitive."""

    name: str = "maneuver"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self.state = ManeuverState.IDLE
        self._start_t: Optional[float] = None
        self._elapsed = 0.0

    def can_enter(self, state: FlightState) -> bool:
        """Return True if the maneuver can safely start from ``state``."""
        return True

    def enter(self, state: FlightState):
        """Initialize maneuver state when selected."""
        self.state = ManeuverState.EXECUTE
        self._start_t = state.t
        self._elapsed = 0.0

    def update(self, state: FlightState, dt: float) -> ManeuverSetpoint:
        """Compute the next setpoint for the maneuver.

        Subclasses implement the core geometry/energy logic here.
        """
        self._elapsed += dt
        return ManeuverSetpoint()

    def is_complete(self, state: FlightState) -> bool:
        """Return True when the maneuver has achieved its objective."""
        return False

    def abort(self):
        """Trigger an abort; the executor should return to a safe fallback."""
        self.state = ManeuverState.ABORT

    def _heading_error(self, target: float, current: float) -> float:
        """Shortest signed heading error in [-pi, pi]."""
        err = (target - current + math.pi) % (2.0 * math.pi) - math.pi
        return err
