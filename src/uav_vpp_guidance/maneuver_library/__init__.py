"""Tactical maneuver library for JSBSim F-16.

This package provides parameterized, finite-state maneuver primitives
(straight flight, coordinated turn, dive, loop, barrel roll, high/low yo-yo)
that can be executed by an inner-loop attitude controller in JSBSim.
"""

from .base import FlightState, ControlCommand, ManeuverSetpoint, Maneuver, ManeuverState
from .controller import InnerLoopController
from .executor import ManeuverExecutor
from .library import ManeuverLibrary
from .telemetry import ManeuverTelemetry

__all__ = [
    "FlightState",
    "ControlCommand",
    "ManeuverSetpoint",
    "Maneuver",
    "ManeuverState",
    "InnerLoopController",
    "ManeuverExecutor",
    "ManeuverLibrary",
    "ManeuverTelemetry",
]
