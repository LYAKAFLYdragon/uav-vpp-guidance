"""Pure offline F02 two-loop roll-control evidence; not runtime control code."""
from dataclasses import dataclass, replace
from math import atan, isfinite
from typing import Dict, List

import numpy as np

F04_ROLL_RATE_LIMITS_RADPS = (-1.5, 1.5)


@dataclass(frozen=True)
class F02CandidateParameters:
    """Proposed offline-only two-loop controller parameters in SI units."""

    gravity_mps2: float = 9.80665
    heading_gain_per_s: float = 1.0
    max_turn_rate_radps: float = 0.35
    max_bank_rad: float = np.deg2rad(45.0)
    bank_gain_per_s: float = 2.0
    roll_rate_damping_gain: float = 0.8
    min_airspeed_mps: float = 80.0
    max_airspeed_mps: float = 350.0


class OfflineF02TwoLoopController:
    """Stateless candidate: heading demand -> bank -> damped roll-rate command."""

    def __init__(self, parameters: F02CandidateParameters = F02CandidateParameters()):
        self.parameters = parameters
        self.last_command = None

    def reset(self) -> None:
        """Clear diagnostic state; the candidate has no integrator to wind up."""
        self.last_command = None

    def command(
        self, heading_error_rad: float, roll_rad: float, roll_rate_radps: float, airspeed_mps: float
    ) -> Dict[str, float]:
        values = (heading_error_rad, roll_rad, roll_rate_radps, airspeed_mps)
        if not all(isfinite(float(value)) for value in values):
            result = self._safe_hold()
            self.last_command = result
            return result

        p = self.parameters
        speed = float(np.clip(airspeed_mps, p.min_airspeed_mps, p.max_airspeed_mps))
        heading_rate_raw = p.heading_gain_per_s * float(heading_error_rad)
        heading_rate_cmd = float(np.clip(heading_rate_raw, -p.max_turn_rate_radps, p.max_turn_rate_radps))
        bank_raw = atan(speed * heading_rate_cmd / p.gravity_mps2)
        bank_cmd = float(np.clip(bank_raw, -p.max_bank_rad, p.max_bank_rad))
        bank_error = bank_cmd - float(roll_rad)
        attitude_contribution = p.bank_gain_per_s * bank_error
        damping_contribution = -p.roll_rate_damping_gain * float(roll_rate_radps)
        roll_rate_raw = attitude_contribution + damping_contribution
        roll_rate_cmd = float(np.clip(roll_rate_raw, *F04_ROLL_RATE_LIMITS_RADPS))
        result = {
            "airspeed_used_mps": speed,
            "heading_rate_raw_radps": heading_rate_raw,
            "heading_rate_cmd_radps": heading_rate_cmd,
            "bank_raw_rad": bank_raw,
            "bank_cmd_rad": bank_cmd,
            "bank_error_rad": bank_error,
            "attitude_contribution_radps": attitude_contribution,
            "damping_contribution_radps": damping_contribution,
            "roll_rate_raw_radps": roll_rate_raw,
            "roll_rate_cmd_radps": roll_rate_cmd,
            "saturated": abs(roll_rate_raw - roll_rate_cmd) > 1.0e-12,
            "safe_hold": False,
        }
        self.last_command = result
        return result

    def _safe_hold(self) -> Dict[str, float]:
        return {
            "airspeed_used_mps": 0.0, "heading_rate_raw_radps": 0.0, "heading_rate_cmd_radps": 0.0,
            "bank_raw_rad": 0.0, "bank_cmd_rad": 0.0, "bank_error_rad": 0.0,
            "attitude_contribution_radps": 0.0, "damping_contribution_radps": 0.0,
            "roll_rate_raw_radps": 0.0, "roll_rate_cmd_radps": 0.0,
            "saturated": False, "safe_hold": True,
        }
