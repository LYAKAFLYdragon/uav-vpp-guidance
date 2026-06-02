"""
Bounded guidance-gain search space.

TODO: Define exact bounds based on flight mechanics constraints.
"""

import numpy as np


class GainSpace:
    """
    Bounded guidance-gain search space.
    """

    def __init__(self, bounds):
        """
        Args:
            bounds (dict): Mapping from gain name to [min, max] list.
        """
        self.bounds = bounds
        self.names = list(bounds.keys())
        self.low = np.array([bounds[n][0] for n in self.names], dtype=np.float32)
        self.high = np.array([bounds[n][1] for n in self.names], dtype=np.float32)

    def sample(self, n, seed=None):
        """
        Sample n random gain vectors uniformly.

        Args:
            n (int): Number of samples.
            seed (int, optional): Random seed.

        Returns:
            np.ndarray: Array of shape (n, dim).
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self.low, self.high, size=(n, len(self.names)))

    def clip(self, gains):
        """
        Clip gain vector to bounds.

        Args:
            gains (np.ndarray): Gain vector or array.

        Returns:
            np.ndarray: Clipped gain vector.
        """
        return np.clip(gains, self.low, self.high)

    def vector_to_gains(self, vector):
        """
        Convert a flat vector to a gain dictionary.

        Args:
            vector (np.ndarray): Gain vector.

        Returns:
            dict: Named gain dictionary.
        """
        return {name: float(vector[i]) for i, name in enumerate(self.names)}

    def gains_to_vector(self, gains):
        """
        Convert a gain dictionary to a flat vector.

        Args:
            gains (dict): Named gain dictionary.

        Returns:
            np.ndarray: Gain vector.
        """
        return np.array([gains[n] for n in self.names], dtype=np.float32)
