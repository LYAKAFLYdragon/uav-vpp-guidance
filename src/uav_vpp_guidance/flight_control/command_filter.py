"""
Command filtering to smooth noisy or discontinuous guidance commands.
"""



class FirstOrderCommandFilter:
    """
    First-order low-pass filter for a single command channel.

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


class MultiChannelCommandFilter:
    """
    Independent first-order filters for multiple command channels.

    Each channel maintains its own filter state, preventing cross-channel
    pollution (e.g. nz_cmd history affecting roll_rate_cmd smoothing).
    """

    def __init__(self, alpha, channels=("nz_cmd", "roll_rate_cmd", "throttle_cmd")):
        """
        Args:
            alpha (float): Smoothing factor in (0, 1].
            channels (tuple): Names of command channels to filter.
        """
        self.channels = channels
        self._filters = {ch: FirstOrderCommandFilter(alpha) for ch in channels}

    def reset(self):
        """Reset all channel filter states."""
        for f in self._filters.values():
            f.reset()

    def filter(self, command_dict):
        """
        Apply filtering to each channel independently.

        Args:
            command_dict (dict): Mapping channel_name -> raw command value.

        Returns:
            dict: Mapping channel_name -> filtered command value.
        """
        result = {}
        for ch in self.channels:
            raw = command_dict.get(ch, 0.0)
            result[ch] = float(self._filters[ch].filter(raw))
        return result
