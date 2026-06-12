"""
Curriculum learning utilities for UAV-VPP guidance training.

The scheduler progressively exposes the agent to harder scenarios
(static / favorable -> random maneuver -> scripted opponent -> crossing)
and only advances when a success-rate gate is satisfied.
"""

from typing import Dict, List, Optional


class CurriculumScheduler:
    """
    Gradually increase scenario difficulty during training.

    Configuration schema (YAML) example::

        curriculum:
          stages:
            - name: static_target
              scenario_names: [favorable, neutral]
              success_threshold: 0.80
              min_episodes: 100
              weights:
                favorable: 0.6
                neutral: 0.4
            - name: crossing_geometry
              scenario_names: [favorable, neutral, challenging]
              success_threshold: 0.30
              min_episodes: 500
          gate_mode: min   # "min" requires all allowed scenarios to pass threshold

    Args:
        config (dict): Curriculum configuration. May be the top-level config
            dict (looked up under ``curriculum``) or the ``curriculum`` subdict.
    """

    DEFAULT_STAGES = [
        {
            "name": "static_target",
            "scenario_names": ["favorable", "neutral"],
            "success_threshold": 0.80,
            "min_episodes": 100,
        },
        {
            "name": "random_maneuver",
            "scenario_names": ["favorable", "neutral", "disadvantage"],
            "success_threshold": 0.60,
            "min_episodes": 200,
        },
        {
            "name": "scripted_opponent",
            "scenario_names": ["favorable", "neutral", "disadvantage", "bank_to_turn"],
            "success_threshold": 0.50,
            "min_episodes": 300,
        },
        {
            "name": "crossing_geometry",
            "scenario_names": ["favorable", "neutral", "disadvantage", "challenging"],
            "success_threshold": 0.30,
            "min_episodes": 500,
        },
    ]

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        # Allow passing the whole config dict or the curriculum subdict
        self.config = config.get("curriculum", config)
        self.stages: List[dict] = self.config.get("stages", self.DEFAULT_STAGES)
        if not self.stages:
            raise ValueError("Curriculum stages list is empty")
        self.gate_mode = self.config.get("gate_mode", "min")
        self.current_level = 0
        self.episodes_in_stage = 0
        self.total_episodes = 0

    @property
    def current_stage(self) -> dict:
        """Return the current stage dictionary."""
        return self.stages[self.current_level]

    @property
    def current_stage_name(self) -> str:
        """Return the name of the current stage."""
        return str(self.current_stage.get("name", f"stage_{self.current_level}"))

    @property
    def allowed_scenario_names(self) -> List[str]:
        """Scenario names allowed in the current stage."""
        return list(self.current_stage.get("scenario_names", []))

    def get_current_scenario_weights(self, all_scenarios: Dict[str, dict]) -> Dict[str, float]:
        """
        Get normalized sampling weights for each scenario at the current level.

        Scenarios not listed in the current stage receive weight 0.0.
        Weights from the stage configuration are honored; otherwise a uniform
        distribution over allowed scenarios is returned.

        Args:
            all_scenarios (dict): Mapping from scenario name to scenario spec.

        Returns:
            dict: Scenario name -> sampling weight (sums to 1.0 over all keys).
        """
        allowed = self.allowed_scenario_names
        if not allowed:
            # Fallback: allow everything uniformly
            n = max(1, len(all_scenarios))
            return {name: 1.0 / n for name in all_scenarios}

        stage_weights = self.current_stage.get("weights", {}) or {}
        raw_weights = {}
        for name in all_scenarios:
            if name in allowed:
                raw_weights[name] = float(stage_weights.get(name, 1.0))
            else:
                raw_weights[name] = 0.0

        total = sum(raw_weights.values())
        if total <= 0:
            # Guard against misconfiguration
            n = max(1, len(allowed))
            return {name: (1.0 / n if name in allowed else 0.0) for name in all_scenarios}

        return {name: w / total for name, w in raw_weights.items()}

    def update(self, performance_metrics: dict) -> bool:
        """
        Update curriculum level based on recent performance.

        Args:
            performance_metrics (dict): Must contain:
              - ``n_episodes`` (int): episodes completed since last call.
              - ``per_scenario_success_rates`` (dict): scenario name -> SR.

        Returns:
            bool: True if the scheduler advanced to a harder stage.
        """
        n_episodes = int(performance_metrics.get("n_episodes", 0))
        self.episodes_in_stage += n_episodes
        self.total_episodes += n_episodes

        stage = self.current_stage
        min_episodes = int(stage.get("min_episodes", 0))
        threshold = float(stage.get("success_threshold", 1.0))

        if self.episodes_in_stage < min_episodes:
            return False

        per_sr = performance_metrics.get("per_scenario_success_rates", {})
        allowed = self.allowed_scenario_names
        sr_values = [float(per_sr.get(name, 0.0)) for name in allowed]

        if self.gate_mode == "min":
            passed = bool(sr_values and min(sr_values) >= threshold)
        elif self.gate_mode == "mean":
            passed = bool(sr_values and (sum(sr_values) / len(sr_values)) >= threshold)
        else:
            raise ValueError(f"Unknown curriculum gate_mode: {self.gate_mode}")

        if passed and self.current_level < len(self.stages) - 1:
            self.current_level += 1
            self.episodes_in_stage = 0
            return True
        return False

    def state_dict(self) -> dict:
        """Serialize scheduler state."""
        return {
            "current_level": self.current_level,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
        }

    def load_state_dict(self, state: dict):
        """Restore scheduler state."""
        self.current_level = int(state.get("current_level", 0))
        self.episodes_in_stage = int(state.get("episodes_in_stage", 0))
        self.total_episodes = int(state.get("total_episodes", 0))
