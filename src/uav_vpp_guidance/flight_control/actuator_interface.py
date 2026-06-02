"""
Actuator interface for JSBSim.

Maps normalized or physical guidance commands to JSBSim property names.

Provides:
- Command normalization and scaling
- Saturation detection
- NaN / inf protection
"""

import numpy as np


class JSBSimActuatorInterface:
    """
    Maps guidance commands (nz_cmd, roll_rate_cmd, throttle_cmd)
    to JSBSim actuator properties.

    For the F-16 model:
    - elevator: [-1, 1] normalized, maps roughly to ±7g
    - aileron:  [-1, 1] normalized, maps roughly to ±1.5 rad/s roll rate
    - rudder:   [-1, 1] normalized (not used in current guidance)
    - throttle: [0, 1] normalized
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): Configuration with gain mappings.
        """
        self.config = config or {}
        # Gains: how to map physical command to normalized JSBSim input
        self.nz_gain = self.config.get("nz_to_elevator_gain", 1.0 / 7.0)
        self.roll_rate_gain = self.config.get("roll_rate_to_aileron_gain", 1.0 / 1.5)
        self.rudder_from_roll = self.config.get("rudder_from_roll_gain", 0.0)

        # Limits
        self.elevator_min = self.config.get("elevator_min", -1.0)
        self.elevator_max = self.config.get("elevator_max", 1.0)
        self.aileron_min = self.config.get("aileron_min", -1.0)
        self.aileron_max = self.config.get("aileron_max", 1.0)
        self.rudder_min = self.config.get("rudder_min", -1.0)
        self.rudder_max = self.config.get("rudder_max", 1.0)
        self.throttle_min = self.config.get("throttle_min", 0.0)
        self.throttle_max = self.config.get("throttle_max", 1.0)

    def command_to_jsbsim_properties(self, command: dict, aircraft_state: dict = None) -> dict:
        """
        Convert guidance command to JSBSim property dictionary.

        Args:
            command (dict): Keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
            aircraft_state (dict, optional): Current aircraft state.

        Returns:
            dict: JSBSim property-value mapping with keys:
                - fcs/elevator-cmd-norm
                - fcs/aileron-cmd-norm
                - fcs/rudder-cmd-norm
                - fcs/throttle-cmd-norm
                - saturation_flag (bool)
        """
        # Extract and sanitize inputs
        nz_cmd = self._safe_float(command.get("nz_cmd", 1.0))
        roll_rate_cmd = self._safe_float(command.get("roll_rate_cmd", 0.0))
        throttle_cmd = self._safe_float(command.get("throttle_cmd", 0.7))

        # Map to normalized JSBSim commands
        elevator = -nz_cmd * self.nz_gain
        aileron = roll_rate_cmd * self.roll_rate_gain
        rudder = 0.0
        if aircraft_state is not None and self.rudder_from_roll != 0.0:
            roll = aircraft_state.get("roll_rad", 0.0)
            rudder = -self.rudder_from_roll * roll

        # Clip to limits
        elevator_clipped = np.clip(elevator, self.elevator_min, self.elevator_max)
        aileron_clipped = np.clip(aileron, self.aileron_min, self.aileron_max)
        rudder_clipped = np.clip(rudder, self.rudder_min, self.rudder_max)
        throttle_clipped = np.clip(throttle_cmd, self.throttle_min, self.throttle_max)

        # Detect saturation
        saturation_flag = (
            abs(elevator - elevator_clipped) > 1e-6 or
            abs(aileron - aileron_clipped) > 1e-6 or
            abs(rudder - rudder_clipped) > 1e-6 or
            abs(throttle_cmd - throttle_clipped) > 1e-6
        )

        return {
            "fcs/elevator-cmd-norm": float(elevator_clipped),
            "fcs/aileron-cmd-norm": float(aileron_clipped),
            "fcs/rudder-cmd-norm": float(rudder_clipped),
            "fcs/throttle-cmd-norm": float(throttle_clipped),
            "saturation_flag": bool(saturation_flag),
            "elevator_raw": float(elevator),
            "aileron_raw": float(aileron),
            "rudder_raw": float(rudder),
        }

    @staticmethod
    def _safe_float(value):
        """Protect against NaN and inf."""
        if value is None:
            return 0.0
        v = float(value)
        if not np.isfinite(v):
            return 0.0
        return v
