"""
RuleEngine

Maps tactical state and geometry to a maneuver intent.

V0 intents:
  - PURE_PURSUIT
  - LEAD_PURSUIT
  - LAG_PURSUIT
  - HIGH_ANGLE_CUT
  - EXTEND
  - ALTITUDE_RECOVER
"""

import numpy as np


class ManeuverIntent:
    PURE_PURSUIT = "PURE_PURSUIT"
    LEAD_PURSUIT = "LEAD_PURSUIT"
    LAG_PURSUIT = "LAG_PURSUIT"
    HIGH_ANGLE_CUT = "HIGH_ANGLE_CUT"
    EXTEND = "EXTEND"
    ALTITUDE_RECOVER = "ALTITUDE_RECOVER"


class RuleEngine:
    """
    Priority-based rule engine for maneuver intent selection.
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.close_range_m = cfg.get("close_range_m", 700.0)
        self.high_closure_rate_mps = cfg.get("high_closure_rate_mps", 120.0)
        self.offensive_aspect_threshold_deg = cfg.get("offensive_aspect_threshold_deg", 35.0)
        self.lead_range_threshold_m = cfg.get("lead_range_threshold_m", 1500.0)

    def select_intent(self, situation_result, rel_state):
        """
        Select maneuver intent based on situation evaluation.

        Args:
            situation_result: output from SituationEvaluator.evaluate()
            rel_state: relative geometry dict

        Returns:
            dict with keys:
                intent: str
                rule_id: str
                fallback_reason: str or None
        """
        tactical_state = situation_result["tactical_state"]
        diagnostics = situation_result["diagnostics"]
        range_m = diagnostics["range_m"]
        closure_rate = diagnostics["closure_rate"]
        aspect_deg = diagnostics["aspect_deg"]
        altitude_safe = diagnostics["altitude_safe"]

        # Rule priority: UNSAFE > DEFENSIVE > STRONG_OFFENSIVE > OFFENSIVE > NEUTRAL
        if tactical_state == "UNSAFE":
            return {
                "intent": ManeuverIntent.ALTITUDE_RECOVER,
                "rule_id": "R1_unsafe",
                "fallback_reason": None,
            }

        if tactical_state == "DEFENSIVE":
            if altitude_safe:
                return {
                    "intent": ManeuverIntent.EXTEND,
                    "rule_id": "R2_defensive_extend",
                    "fallback_reason": None,
                }
            else:
                return {
                    "intent": ManeuverIntent.ALTITUDE_RECOVER,
                    "rule_id": "R3_defensive_altitude",
                    "fallback_reason": None,
                }

        if tactical_state == "STRONG_OFFENSIVE":
            if closure_rate > self.high_closure_rate_mps or range_m < self.close_range_m:
                return {
                    "intent": ManeuverIntent.LAG_PURSUIT,
                    "rule_id": "R4_strong_offensive_lag",
                    "fallback_reason": None,
                }
            else:
                return {
                    "intent": ManeuverIntent.PURE_PURSUIT,
                    "rule_id": "R5_strong_offensive_pure",
                    "fallback_reason": None,
                }

        if tactical_state == "OFFENSIVE":
            if aspect_deg > self.offensive_aspect_threshold_deg and range_m > self.lead_range_threshold_m:
                return {
                    "intent": ManeuverIntent.LEAD_PURSUIT,
                    "rule_id": "R6_offensive_lead",
                    "fallback_reason": None,
                }
            else:
                return {
                    "intent": ManeuverIntent.PURE_PURSUIT,
                    "rule_id": "R7_offensive_pure",
                    "fallback_reason": None,
                }

        # NEUTRAL or fallback
        return {
            "intent": ManeuverIntent.HIGH_ANGLE_CUT,
            "rule_id": "R8_neutral_high_angle_cut",
            "fallback_reason": None,
        }
