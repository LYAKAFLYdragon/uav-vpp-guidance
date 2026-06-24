"""
Command limiting utilities.

Enforces physical and safety limits on guidance commands.
"""

import numpy as np


DEFAULT_THROTTLE_MIN = 0.4
DEFAULT_THROTTLE_MAX = 0.9


def effective_throttle_limits(limits):
    """Return throttle limits constrained to the F-16 effective command envelope."""
    limits = limits or {}
    throttle_min = max(
        float(limits.get("throttle_min", DEFAULT_THROTTLE_MIN)),
        DEFAULT_THROTTLE_MIN,
    )
    throttle_max = min(
        float(limits.get("throttle_max", DEFAULT_THROTTLE_MAX)),
        DEFAULT_THROTTLE_MAX,
    )
    if throttle_min > throttle_max:
        throttle_min, throttle_max = DEFAULT_THROTTLE_MIN, DEFAULT_THROTTLE_MAX
    return throttle_min, throttle_max


def clip_command(command, limits):
    """
    Clip nz_cmd, roll_rate_cmd, throttle_cmd according to configured limits.

    Args:
        command (dict): Command dictionary with keys 'nz_cmd', 'roll_rate_cmd', 'throttle_cmd'.
        limits (dict): Limit dictionary with keys 'nz_min', 'nz_max',
                       'roll_rate_min', 'roll_rate_max',
                       'throttle_min', 'throttle_max'.

    Returns:
        dict: Clipped command dictionary.
    """
    throttle_min, throttle_max = effective_throttle_limits(limits)
    clipped = {
        "nz_cmd": np.clip(
            command.get("nz_cmd", 1.0),
            limits.get("nz_min", -2.0),
            limits.get("nz_max", 7.0),
        ),
        "roll_rate_cmd": np.clip(
            command.get("roll_rate_cmd", 0.0),
            limits.get("roll_rate_min", -1.5),
            limits.get("roll_rate_max", 1.5),
        ),
        "throttle_cmd": np.clip(
            command.get("throttle_cmd", 0.5),
            throttle_min,
            throttle_max,
        ),
    }
    return clipped
