"""
Pursuit point priors (lag, lead, pure pursuit).

TODO: Extract pursuit geometry from legacy project or reference texts.
"""



def lag_pursuit_point(target_state, d_ref):
    """
    Generate virtual point behind target aircraft.

    Args:
        target_state (dict): Target aircraft state.
        d_ref (float): Longitudinal offset behind target (positive = behind).

    Returns:
        np.ndarray: Virtual point position [x, y, z] in meters.
    """
    # TODO: Implement lag pursuit geometry.
    raise NotImplementedError


def lead_pursuit_point(target_state, prediction_time, d_ref):
    """
    Generate virtual point near predicted future position of target aircraft.

    Args:
        target_state (dict): Target aircraft state.
        prediction_time (float): Prediction horizon in seconds.
        d_ref (float): Additional longitudinal offset.

    Returns:
        np.ndarray: Virtual point position [x, y, z] in meters.
    """
    # TODO: Implement lead pursuit geometry with simple prediction.
    raise NotImplementedError


def pure_pursuit_point(target_state):
    """
    Generate virtual point at current target position.

    Args:
        target_state (dict): Target aircraft state.

    Returns:
        np.ndarray: Virtual point position [x, y, z] in meters.
    """
    # TODO: Return current target position as virtual point.
    raise NotImplementedError
