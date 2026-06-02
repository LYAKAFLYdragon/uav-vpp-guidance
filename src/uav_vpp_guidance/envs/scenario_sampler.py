"""
Scenario sampling for initial conditions.

TODO: Migrate scenario initialization logic from legacy project:
  E:/CloseAirCombat_control/runner/jsbsim_runner.py or envs/JSBSim/__init__.py
"""

from dataclasses import dataclass


@dataclass
class AircraftInitState:
    """Initial state of an aircraft."""
    position_m: tuple
    velocity_mps: float
    heading_deg: float
    altitude_m: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


@dataclass
class Scenario:
    """A single scenario configuration."""
    name: str
    own_init: AircraftInitState
    target_init: AircraftInitState
    target_policy: str = "rule_based"


class ScenarioSampler:
    """
    Sample initial configurations for favorable, neutral, disadvantage, and challenging cases.

    TODO: Extract scenario definitions from legacy runner initialization
    or from legacy config.py.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Scenario configuration dictionary.
        """
        self.config = config

    def sample(self, scenario_type=None, seed=None):
        """
        Sample a scenario.

        Args:
            scenario_type (str, optional): One of 'favorable', 'neutral', 'disadvantage', 'challenging'.
            seed (int, optional): Random seed.

        Returns:
            Scenario: Sampled scenario object.
        """
        # TODO: Implement scenario sampling based on legacy initial states.
        raise NotImplementedError("Implement scenario sampling based on legacy initial states.")
