"""
Guidance gain dataclass.

Defines the tunable guidance gain parameters.
"""

from dataclasses import dataclass


@dataclass
class GuidanceGains:
    """
    Container for guidance law gains.

    Attributes:
        k_los: LOS-rate gain.
        k_pos: Position error gain.
        k_damp: Damping gain.
        k_roll: Roll-rate gain.
        k_speed: Speed tracking gain.
        k_energy: Energy compensation gain.
        alpha_filter: Command filter smoothing factor.
    """
    k_los: float = 1.0
    k_pos: float = 0.5
    k_damp: float = 0.2
    k_roll: float = 1.0
    k_speed: float = 0.2
    k_energy: float = 0.1
    alpha_filter: float = 0.3

    def as_vector(self):
        """
        Return gains as a flat list.

        Returns:
            list: Ordered gain values.
        """
        return [
            self.k_los,
            self.k_pos,
            self.k_damp,
            self.k_roll,
            self.k_speed,
            self.k_energy,
            self.alpha_filter,
        ]
