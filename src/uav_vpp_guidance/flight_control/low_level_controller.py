"""
Low-level controller interface.

Translates high-level guidance commands (nz, roll rate, throttle)
into actuator inputs for JSBSim via the actuator interface.

Features:
- Command filtering
- Saturation handling
- NaN / inf protection
- Control history recording
"""

import numpy as np
from .actuator_interface import JSBSimActuatorInterface


class LowLevelController:
    """
    Low-level controller translating guidance commands to JSBSim inputs.

    First-order filter on each command channel for smoothness,
    then maps to JSBSim actuator properties via ActuatorInterface.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Flight control configuration.
                Keys: alpha_filter (float), actuator config dict.
        """
        self.config = config or {}
        self.alpha = self.config.get("alpha_filter", 0.3)
        self.actuator = JSBSimActuatorInterface(self.config.get("actuator", {}))

        # Filter state
        self._prev_nz = 1.0
        self._prev_roll_rate = 0.0
        self._prev_throttle = 0.7

        # History for diagnostics
        self.command_history = []

    def reset(self):
        """Reset controller internal state."""
        self._prev_nz = 1.0
        self._prev_roll_rate = 0.0
        self._prev_throttle = 0.7
        self.command_history.clear()

    def compute_actuator(self, guidance_command: dict, aircraft_state: dict = None) -> dict:
        """
        Compute actuator commands from guidance commands.

        Args:
            guidance_command (dict): Keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
            aircraft_state (dict, optional): Current aircraft state.

        Returns:
            dict: Actuator command dictionary for JSBSim, including:
                - fcs/elevator-cmd-norm
                - fcs/aileron-cmd-norm
                - fcs/rudder-cmd-norm
                - fcs/throttle-cmd-norm
                - saturation_flag
                - filtered_command (dict of raw filtered values)
        """
        # Sanitize inputs
        nz = self._safe_float(guidance_command.get("nz_cmd", 1.0))
        roll_rate = self._safe_float(guidance_command.get("roll_rate_cmd", 0.0))
        throttle = self._safe_float(guidance_command.get("throttle_cmd", 0.7))

        # First-order filter
        nz_filt = self.alpha * nz + (1.0 - self.alpha) * self._prev_nz
        roll_rate_filt = self.alpha * roll_rate + (1.0 - self.alpha) * self._prev_roll_rate
        throttle_filt = self.alpha * throttle + (1.0 - self.alpha) * self._prev_throttle

        self._prev_nz = nz_filt
        self._prev_roll_rate = roll_rate_filt
        self._prev_throttle = throttle_filt

        filtered = {
            "nz_cmd": float(nz_filt),
            "roll_rate_cmd": float(roll_rate_filt),
            "throttle_cmd": float(throttle_filt),
        }

        # Map to JSBSim properties
        jsbsim_props = self.actuator.command_to_jsbsim_properties(filtered, aircraft_state)

        # Record history
        record = {
            "guidance": guidance_command,
            "filtered": filtered,
            "jsbsim": {k: v for k, v in jsbsim_props.items() if k.startswith("fcs/")},
            "saturation_flag": jsbsim_props.get("saturation_flag", False),
        }
        self.command_history.append(record)

        # Return merged dict
        result = dict(jsbsim_props)
        result["filtered_command"] = filtered
        return result

    @staticmethod
    def _safe_float(value):
        """Protect against NaN and inf."""
        if value is None:
            return 0.0
        v = float(value)
        if not np.isfinite(v):
            return 0.0
        return v
