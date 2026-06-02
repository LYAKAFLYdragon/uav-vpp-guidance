"""
Command limiting utilities.

Enforces physical and safety limits on guidance commands.
"""

import numpy as np


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
            limits.get("throttle_min", 0.0),
            limits.get("throttle_max", 1.0),
        ),
    }
    return clipped
