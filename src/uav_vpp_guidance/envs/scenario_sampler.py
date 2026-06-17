"""
Scenario sampling for initial conditions.

Self-contained JSBSim integration.
"""

from dataclasses import dataclass

import numpy as np

from .scenario_registry import ScenarioRegistry


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


def make_scenario_sampler(cfg: dict):
    """
    Build a lightweight scenario sampler.

    Supported cfg keys:
      enabled: bool (default True)
      source: "registry_set" | "explicit_list" (default "registry_set")
      set: str, used when source="registry_set" (default "smoke_test")
      names: list[str], used when source="explicit_list"
      seed: int (default None -> system random)

    Returns an object with a ``sample()`` method, or None if disabled.
    """
    if not cfg.get("enabled", True):
        return None

    source = cfg.get("source", "registry_set")
    if source == "registry_set":
        pool = list(ScenarioRegistry.get_set(cfg.get("set", "smoke_test")).values())
    elif source == "explicit_list":
        pool = []
        for name in cfg.get("names", []):
            scenario = ScenarioRegistry.get(name)
            if scenario is not None:
                pool.append(scenario)
    else:
        pool = []

    if not pool:
        return None

    rng = np.random.default_rng(cfg.get("seed"))

    class _Sampler:
        """Uniform random sampler over a frozen scenario pool."""

        def sample(self):
            idx = rng.integers(len(pool))
            return pool[idx]

    return _Sampler()
