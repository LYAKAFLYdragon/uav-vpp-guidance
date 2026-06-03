"""
Energy management / compensation for guidance commands.

TODO: Review if legacy project contains energy compensation logic.
  If not, implement as an optional module.
"""


def compute_energy_compensation(own_state, target_state, gains):
    """
    Compute throttle adjustment based on total energy error.

    Args:
        own_state (dict): Own aircraft state.
        target_state (dict): Target aircraft state.
        gains (GuidanceGains): Guidance gains.

    Returns:
        float: Throttle delta in [-1, 1].
    """
    # TODO: Implement energy-based throttle compensation if needed.
    raise NotImplementedError
