"""
Virtual point smoothing utilities.

TODO: Add temporal smoothing to avoid jitter in virtual point position.
"""



class VirtualPointSmoother:
    """
    Exponential moving average smoother for virtual point trajectories.
    """

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha (float): Smoothing factor in [0, 1]; higher = more responsive.
        """
        self.alpha = alpha
        self._prev_point = None

    def reset(self):
        """Reset smoothing state."""
        self._prev_point = None

    def smooth(self, point):
        """
        Apply exponential smoothing.

        Args:
            point (np.ndarray): Current virtual point position.

        Returns:
            np.ndarray: Smoothed virtual point position.
        """
        if self._prev_point is None:
            self._prev_point = point.copy()
            return point
        smoothed = self.alpha * point + (1.0 - self.alpha) * self._prev_point
        self._prev_point = smoothed.copy()
        return smoothed
