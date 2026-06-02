"""
Overload and roll-rate command generation utilities.

TODO: Migrate from legacy NDI controller or guidance modules.
"""

import numpy as np


def compute_normal_overload(los_rate, closing_velocity, gains):
    """
    Compute normal overload command from LOS rate.

    Args:
        los_rate (float): Line-of-sight rate in rad/s.
        closing_velocity (float): Closing velocity in m/s.
        gains (GuidanceGains): Guidance gains.

    Returns:
        float: Normal overload command (nz_cmd, in g).
    """
    # TODO: Implement overload command generation.
    raise NotImplementedError


def compute_roll_rate_command(heading_error, roll_angle, gains):
    """
    Compute roll-rate command from heading error.

    Args:
        heading_error (float): Heading error in radians.
        roll_angle (float): Current roll angle in radians.
        gains (GuidanceGains): Guidance gains.

    Returns:
        float: Roll-rate command in rad/s.
    """
    # TODO: Implement roll-rate command generation.
    raise NotImplementedError
