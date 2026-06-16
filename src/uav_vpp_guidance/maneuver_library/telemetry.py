"""Telemetry/audit logging for maneuver-library execution.

This module provides a lightweight recorder for phase sequences, switch
reasons, abort reasons, and flight-envelope events. It is intentionally
separate from the control logic so it can be enabled/disabled without
affecting behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ManeuverRecord:
    """One entry in the maneuver execution log."""

    name: str
    start_t: float
    end_t: Optional[float] = None
    reason: str = "active"  # active | complete | abort | switch | emergency
    phase: str = "entry"
    abort_reason: Optional[str] = None


class ManeuverTelemetry:
    """Record maneuver history for regression tests and post-hoc analysis."""

    def __init__(self):
        self.records: List[ManeuverRecord] = []
        self._active: Optional[ManeuverRecord] = None
        self.switch_count = 0
        self.emergency_count = 0
        self.abort_count = 0

    def start(self, name: str, sim_time: float, reason: str = "switch"):
        """Record the start of a new maneuver."""
        if self._active is not None:
            self.end(sim_time, reason)
        self._active = ManeuverRecord(
            name=name,
            start_t=sim_time,
            reason="active",
            phase="entry",
        )
        self.records.append(self._active)
        if reason == "emergency":
            self.emergency_count += 1
        elif reason == "switch":
            self.switch_count += 1

    def end(self, sim_time: float, reason: str, abort_reason: Optional[str] = None):
        """Close the active maneuver record."""
        if self._active is None:
            return
        self._active.end_t = sim_time
        self._active.reason = reason
        self._active.abort_reason = abort_reason
        if reason == "abort":
            self.abort_count += 1
        self._active = None

    def set_phase(self, phase: str):
        """Update the phase of the active maneuver."""
        if self._active is not None:
            self._active.phase = phase

    def active_name(self) -> Optional[str]:
        """Return the name of the currently active maneuver, if any."""
        return self._active.name if self._active is not None else None

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "switch_count": self.switch_count,
            "emergency_count": self.emergency_count,
            "abort_count": self.abort_count,
            "maneuver_history": [
                {
                    "name": r.name,
                    "start_t": r.start_t,
                    "end_t": r.end_t,
                    "reason": r.reason,
                    "phase": r.phase,
                    "abort_reason": r.abort_reason,
                }
                for r in self.records
            ],
        }

    def reset(self):
        """Clear all records."""
        self.records.clear()
        self._active = None
        self.switch_count = 0
        self.emergency_count = 0
        self.abort_count = 0
