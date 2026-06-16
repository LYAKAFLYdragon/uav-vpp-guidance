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

    def _envelope_limit(
        self,
        param_keys: tuple[str, ...],
        attr_aliases: tuple[str, ...] = (),
    ) -> Optional[float]:
        """Resolve a numeric envelope limit from params or subclass attributes.

        Searches ``self.params`` first, then instance attributes (including
        common aliases such as ``min_altitude`` for ``min_altitude_m``).  This
        lets existing maneuvers that store limits as ``self.min_altitude``
        participate in the automatic abort guard without extra boilerplate.
        """
        for key in param_keys:
            value = self.params.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        for attr in attr_aliases:
            value = getattr(self, attr, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def _refresh_envelope_limits(self):
        """Re-read envelope limits so subclass attributes are visible."""
        self._min_altitude_m = self._envelope_limit(
            ("min_altitude_m",), ("min_altitude",)
        )
        self._min_speed_mps = self._envelope_limit(
            ("min_speed_mps",), ("min_speed",)
        )
        self._max_alpha_rad = self._envelope_limit(
            ("max_alpha_rad", "max_alpha_deg"), ("max_alpha",)
        )
        if "max_alpha_deg" in self.params and self._max_alpha_rad is not None:
            # If the param was given in degrees, convert to radians.
            self._max_alpha_rad = math.radians(self._max_alpha_rad)

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

    def abort_guard(self, state: FlightState) -> bool:
        """Subclass hook for maneuver-specific abort conditions.

        Return True to force an abort (e.g., target overshoot, unsafe attitude).
        """
        return False

    def should_abort(self, state: FlightState) -> bool:
        """Return True if the active maneuver must abort for safety reasons.

        Checks, in order:
        1. Maneuver-specific abort guard (``abort_guard``).
        2. Hard altitude floor (``min_altitude_m``).
        3. Hard speed floor (``min_speed_mps``).
        4. Angle-of-attack ceiling (``max_alpha_rad``).

        Already-terminal states (COMPLETE, ABORT) never abort again.
        """
        if self.state in (ManeuverState.COMPLETE, ManeuverState.ABORT):
            return False
        # Subclasses set limits as attributes after ``super().__init__``; refresh
        # at decision time so those values are visible.
        self._refresh_envelope_limits()
        if self.abort_guard(state):
            return True
        if self._min_altitude_m is not None and state.altitude_m < self._min_altitude_m:
            return True
        if self._min_speed_mps is not None and state.velocity_mps < self._min_speed_mps:
            return True
        if self._max_alpha_rad is not None and state.alpha_rad > self._max_alpha_rad:
            return True
        return False

    def phase(self) -> str:
        """Return the current maneuver phase as a human-readable string."""
        internal = getattr(self, "_phase", None)
        if internal is not None:
            return str(internal)
        return self.state.name.lower()

    def _heading_error(self, target: float, current: float) -> float:
        """Shortest signed heading error in [-pi, pi]."""
        err = (target - current + math.pi) % (2.0 * math.pi) - math.pi
        return err
