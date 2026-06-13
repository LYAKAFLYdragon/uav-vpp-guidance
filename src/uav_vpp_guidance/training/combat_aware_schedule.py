"""
Combat-aware intentional update schedule.

Maps air-combat geometry features (range, aspect angle, altitude/energy
difference, etc.) to per-update actor/critic budget scalers. The phase rules
are deliberately simple and interpretable, matching the discussion in
"IntentionalRL实验计划与论文撰写.md".

Phases:
  - search_approach    : far range, exploration is cheap
  - merge_maneuver     : medium range, rapid geometry change
  - advantage_position : own aircraft is in a favorable tail-position
  - disadvantage_defense: enemy is in a threatening position behind ownship
  - terminal           : very close to termination / success boundary
"""

import math
from typing import Dict, Optional

import numpy as np


class CombatAwareSchedule:
    """
    Rule-based schedule that adjusts intentional-update budgets by air-combat phase.

    Args:
        config (dict): Configuration dictionary. Expected keys:
          - ``eta_actor`` (float): base actor update budget.
          - ``eta_critic`` (float): base critic update budget.
          - ``range_thresholds_m`` (list): [medium, far] thresholds in meters.
          - ``terminal_range_m`` (float): range below which episode is near end.
          - ``aspect_threshold_deg`` (float): angular window for tail/head geometry.
          - ``phase_scales`` (dict): optional override of default phase scales.
    """

    DEFAULT_PHASE_SCALES = {
        "search_approach": {"actor": 1.5, "critic": 1.0},
        "merge_maneuver": {"actor": 1.0, "critic": 1.0},
        "advantage_position": {"actor": 0.5, "critic": 0.5},
        "disadvantage_defense": {"actor": 0.5, "critic": 1.5},
        "terminal": {"actor": 0.3, "critic": 0.5},
    }

    FEATURE_KEYS = [
        "range_m",
        "ata_rad",
        "aa_rad",
        "altitude_diff_m",
        "speed_diff_mps",
        "range_rate_mps",
        "missile_threat",
    ]

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        self.eta_actor_base = float(config.get("eta_actor", 0.01))
        self.eta_critic_base = float(config.get("eta_critic", 0.1))

        range_thresh = config.get("range_thresholds_m", [3000.0, 6000.0])
        self.range_medium_m = float(range_thresh[0])
        self.range_far_m = float(range_thresh[1])
        self.terminal_range_m = float(config.get("terminal_range_m", 1200.0))
        self.aspect_threshold_deg = float(config.get("aspect_threshold_deg", 30.0))

        self.phase_scales = dict(self.DEFAULT_PHASE_SCALES)
        if "phase_scales" in config:
            self.phase_scales.update(config["phase_scales"])

    def classify(self, features: dict) -> str:
        """
        Classify a single set of geometry features into a combat phase.

        Args:
            features (dict): Must contain at least ``range_m`` and ``aa_rad``.

        Returns:
            str: Phase name.
        """
        range_m = float(features.get("range_m", float("inf")))
        aa_rad = float(features.get("aa_rad", 0.0))
        aa_deg = abs(np.rad2deg(aa_rad))
        aspect_window = self.aspect_threshold_deg

        # Terminal dominates if we are very close to the success envelope
        if range_m <= self.terminal_range_m:
            return "terminal"

        if range_m > self.range_far_m:
            return "search_approach"

        if range_m > self.range_medium_m:
            return "merge_maneuver"

        # Close-range phases depend mainly on aspect angle
        if aa_deg <= aspect_window:
            # Own nose roughly points at target (tail-chase or head-on merge)
            return "advantage_position"
        if abs(aa_deg - 180.0) <= aspect_window:
            # Own tail points at target -> target is behind us
            return "disadvantage_defense"

        return "merge_maneuver"

    def get_eta_scales(self, features: dict) -> Dict[str, float]:
        """
        Return actor/critic budget scalers for a single state.

        Args:
            features (dict): Geometry features.

        Returns:
            dict: ``{"actor": scale, "critic": scale}``.
        """
        phase = self.classify(features)
        return self.phase_scales.get(phase, {"actor": 1.0, "critic": 1.0})

    def get_eta(self, features: dict) -> Dict[str, float]:
        """
        Return absolute actor/critic eta values for a single state.

        Args:
            features (dict): Geometry features.

        Returns:
            dict: ``{"actor": eta_actor, "critic": eta_critic}``.
        """
        scales = self.get_eta_scales(features)
        return {
            "actor": self.eta_actor_base * scales["actor"],
            "critic": self.eta_critic_base * scales["critic"],
        }

    def get_batch_eta(self, features_list) -> Dict[str, float]:
        """
        Return mean actor/critic eta values for a batch of states.

        Args:
            features_list: Iterable of feature dicts or a (N, F) array. Arrays
                are interpreted using :attr:`FEATURE_KEYS` in order.

        Returns:
            dict: ``{"actor": mean_eta_actor, "critic": mean_eta_critic}``.
        """
        if isinstance(features_list, np.ndarray):
            features_list = self._array_to_features(features_list)

        actor_etas = []
        critic_etas = []
        for feat in features_list:
            eta = self.get_eta(feat)
            actor_etas.append(eta["actor"])
            critic_etas.append(eta["critic"])

        if not actor_etas:
            return {"actor": self.eta_actor_base, "critic": self.eta_critic_base}

        return {
            "actor": float(np.mean(actor_etas)),
            "critic": float(np.mean(critic_etas)),
        }

    def get_batch_scales(self, features_list) -> Dict[str, float]:
        """
        Return mean actor/critic scale factors for a batch.

        Args:
            features_list: Iterable of feature dicts or a (N, F) array.

        Returns:
            dict: ``{"actor": mean_scale, "critic": mean_scale}``.
        """
        if isinstance(features_list, np.ndarray):
            features_list = self._array_to_features(features_list)

        actor_scales = []
        critic_scales = []
        for feat in features_list:
            scales = self.get_eta_scales(feat)
            actor_scales.append(scales["actor"])
            critic_scales.append(scales["critic"])

        if not actor_scales:
            return {"actor": 1.0, "critic": 1.0}

        return {
            "actor": float(np.mean(actor_scales)),
            "critic": float(np.mean(critic_scales)),
        }

    def _array_to_features(self, arr: np.ndarray) -> list:
        """Convert a (N, F) feature array into a list of feature dicts."""
        arr = np.atleast_2d(arr)
        features = []
        for row in arr:
            feat = {}
            for i, key in enumerate(self.FEATURE_KEYS):
                if i < len(row):
                    feat[key] = row[i]
            features.append(feat)
        return features
