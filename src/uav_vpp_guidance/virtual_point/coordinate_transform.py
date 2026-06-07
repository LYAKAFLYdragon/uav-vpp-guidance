"""
Coordinate transforms for virtual point generation.

Converts offsets in target body / heading frame to world coordinates.
"""



def heading_to_rotation_matrix(heading_deg, pitch_deg=0.0, roll_deg=0.0):
    """
    Construct a rotation matrix from heading-pitch-roll angles (Z-Y-X convention).

    Args:
        heading_deg (float): Heading angle in degrees.
        pitch_deg (float): Pitch angle in degrees.
        roll_deg (float): Roll angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # TODO: Implement rotation matrix or reuse from legacy project.
    raise NotImplementedError


def target_relative_to_world(offset, target_state):
    """
    Convert a target-relative offset to world coordinates.

    Args:
        offset (np.ndarray): [dx, dy, dz] relative to target heading frame.
        target_state (dict): Target aircraft state with position and attitude.

    Returns:
        np.ndarray: World coordinates [x, y, z].
    """
    # TODO: Implement coordinate transform.
    raise NotImplementedError
