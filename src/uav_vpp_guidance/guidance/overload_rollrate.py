"""
Command post-processor: saturation, energy compensation, and load-roll coordination.

This module receives raw guidance commands (e.g. from LOSRateGuidance or PN),
applies final saturation, optional energy compensation, and overload-roll
coordination, then returns flight-ready commands.

All magic numbers are config-driven. Input validation and NaN protection
mirror the hardened style of los_rate_guidance.py.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CommandPostProcessor:
    """
    Post-process raw guidance commands before sending to the actuator backend.

    Responsibilities:
    1. Validate incoming command dict.
    2. Apply configurable saturation limits.
    3. Optional energy compensation (throttle boost when nz is high).
    4. Optional load-roll coordination (reduce roll rate when nz nears limit).
    5. Terminal-phase protection (limit aggressive commands at close range).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        limits = config.get("limits", {})
        params = config.get("post_process", {})
        guidance_params = config.get("params", {})

        # Saturation limits
        self.nz_min = float(limits.get("nz_min", -2.0))
        self.nz_max = float(limits.get("nz_max", 7.0))
        self.roll_rate_min = float(limits.get("roll_rate_min", -1.5))
        self.roll_rate_max = float(limits.get("roll_rate_max", 1.5))
        self.throttle_min = float(limits.get("throttle_min", 0.0))
        self.throttle_max = float(limits.get("throttle_max", 1.0))

        # Energy compensation
        self.enable_energy_comp = bool(params.get("enable_energy_compensation", False))
        self.energy_k_nz = float(params.get("energy_k_nz", 0.05))
        self.energy_k_speed = float(params.get("energy_k_speed", 0.01))

        # Load-roll coordination
        self.enable_load_roll_coord = bool(
            params.get("enable_load_roll_coordination", False)
        )
        self.coord_nz_threshold = float(params.get("coord_nz_threshold", 0.8))
        self.coord_roll_scale = float(params.get("coord_roll_scale", 0.5))

        # Terminal-phase protection
        self.enable_terminal_protection = bool(
            params.get("enable_terminal_protection", True)
        )
        self.terminal_range_m = float(params.get("terminal_range_m", 500.0))
        self.terminal_nz_scale = float(params.get("terminal_nz_scale", 0.7))
        self.terminal_roll_scale = float(params.get("terminal_roll_scale", 0.8))
        self.base_nz = float(guidance_params.get("base_nz", params.get("base_nz", 1.0)))

        self.epsilon = float(params.get("epsilon", 1.0e-6))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        raw_command: Dict[str, float],
        own_state: Optional[Dict[str, Any]] = None,
        target_state: Optional[Dict[str, Any]] = None,
        relative_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Post-process a raw guidance command.

        Args:
            raw_command (dict): Must contain keys 'nz_cmd', 'roll_rate_cmd',
                'throttle_cmd'.
            own_state (dict, optional): Own aircraft state.
            target_state (dict, optional): Target aircraft state (unused but
                reserved for future extensions).
            relative_state (dict, optional): Relative geometry dict, expected
                to contain 'range_m' if terminal-phase protection is enabled.

        Returns:
            dict: Finalized command dictionary.
        """
        self._validate_command(raw_command)

        nz = float(raw_command["nz_cmd"])
        roll = float(raw_command["roll_rate_cmd"])
        throttle = float(raw_command["throttle_cmd"])

        # Terminal-phase protection
        if self.enable_terminal_protection and relative_state is not None:
            range_m = relative_state.get("range_m")
            if range_m is not None and np.isfinite(range_m):
                nz, roll = self._apply_terminal_protection(nz, roll, float(range_m))

        # Load-roll coordination
        if self.enable_load_roll_coord:
            roll = self._apply_load_roll_coordination(nz, roll)

        # Saturation
        nz = float(np.clip(nz, self.nz_min, self.nz_max))
        roll = float(np.clip(roll, self.roll_rate_min, self.roll_rate_max))
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))

        # Energy compensation
        if self.enable_energy_comp and own_state is not None:
            throttle = self._apply_energy_compensation(nz, throttle, own_state)

        return {
            "nz_cmd": float(nz),
            "roll_rate_cmd": float(roll),
            "throttle_cmd": float(throttle),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_command(self, command: Dict[str, float]) -> None:
        if not isinstance(command, dict):
            raise TypeError(f"command must be dict, got {type(command).__name__}")
        for key in ("nz_cmd", "roll_rate_cmd", "throttle_cmd"):
            if key not in command:
                raise ValueError(f"command missing required key: {key}")
            val = command[key]
            if val is not None and not np.isfinite(val):
                logger.warning(
                    f"command['{key}'] is non-finite ({val}), will be clamped"
                )
                command[key] = 0.0

    def _apply_terminal_protection(
        self, nz: float, roll: float, range_m: float
    ) -> tuple:
        """Scale down aggressive commands when inside terminal range.

        Instead of scaling the absolute nz_cmd (which would distort 1g level
        flight), we scale the deviation from base_nz so that the command
        converges to base_nz as range -> 0.
        """
        if range_m > self.terminal_range_m:
            return nz, roll
        # Linear scale from epsilon at range=0 to 1.0 at boundary
        scale = max(
            self.epsilon,
            min(1.0, range_m / self.terminal_range_m),
        )
        # Scale deviation from base_nz: nz = base_nz + scale * (nz - base_nz)
        nz = self.base_nz + scale * (nz - self.base_nz)
        # Roll still scales toward zero
        roll_factor = scale + (1.0 - scale) * self.terminal_roll_scale
        return nz, roll * roll_factor

    def _apply_load_roll_coordination(self, nz: float, roll: float) -> float:
        """Reduce roll rate when nz is near its limits (structural/energy protection)."""
        # Normalize nz to [0,1] relative to positive limit (most common case)
        nz_norm = abs(nz) / max(abs(self.nz_max), self.epsilon)
        if nz_norm > self.coord_nz_threshold:
            excess = (nz_norm - self.coord_nz_threshold) / (
                1.0 - self.coord_nz_threshold
            )
            scale = 1.0 - self.coord_roll_scale * excess
            scale = max(0.0, scale)
            return roll * scale
        return roll

    def _apply_energy_compensation(
        self, nz: float, throttle: float, own_state: Dict[str, Any]
    ) -> float:
        """Boost throttle when high g-load is demanded to prevent energy bleed."""
        vel = own_state.get("velocity_ned")
        if vel is None:
            vel = own_state.get("velocity_vector_mps")
        if vel is None:
            return throttle
        speed = float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))
        if not np.isfinite(speed):
            return throttle

        # High nz -> more thrust needed; low speed -> more thrust needed
        nz_demand = max(0.0, abs(nz) - 1.0)  # excess above 1g
        speed_deficit = max(0.0, 250.0 - speed)
        delta = self.energy_k_nz * nz_demand + self.energy_k_speed * speed_deficit
        return float(np.clip(throttle + delta, self.throttle_min, self.throttle_max))
