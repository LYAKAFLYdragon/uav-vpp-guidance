"""
Command filtering to smooth noisy or discontinuous guidance commands.
"""

import numpy as np


class FirstOrderCommandFilter:
    """
    First-order low-pass filter for command signals.

    y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    """

    def __init__(self, alpha):
        """
        Args:
            alpha (float): Smoothing factor in (0, 1].
        """
        self.alpha = alpha
        self.prev = None

    def reset(self):
        """Reset filter state."""
        self.prev = None

    def filter(self, command):
        """
        Apply first-order filtering.

        Args:
            command (float or np.ndarray): Raw command.

        Returns:
            float or np.ndarray: Filtered command.
        """
        if self.prev is None:
            self.prev = command
            return command
        smoothed = self.alpha * command + (1.0 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed
