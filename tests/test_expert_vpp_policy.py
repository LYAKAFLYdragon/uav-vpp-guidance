"""
Tests for ExpertVPPPolicy.

Covers:
  - SituationEvaluator
  - RuleEngine
  - ExpertVPPPolicy action generation
  - Output bounds and shape
  - Diagnostic logging
"""

import pytest
import numpy as np

from uav_vpp_guidance.expert_system.situation_evaluator import SituationEvaluator, TacticalState
from uav_vpp_guidance.expert_system.rule_engine import RuleEngine, ManeuverIntent
from uav_vpp_guidance.expert_system.expert_vpp_policy import ExpertVPPPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_rel_state(range_m=2000.0, aa_rad=np.deg2rad(45.0), ata_rad=np.deg2rad(135.0),
                   range_rate_mps=-50.0, altitude_diff_m=0.0):
    """Build a relative geometry dict."""
    return {
        "range_m": range_m,
        "aa_rad": aa_rad,
        "ata_rad": ata_rad,
        "range_rate_mps": range_rate_mps,
        "altitude_diff_m": altitude_diff_m,
    }


def make_own_state(position_m=None, velocity_vector_mps=None, altitude_m=5000.0):
    if position_m is None:
        position_m = np.array([0.0, 0.0, altitude_m])
    if velocity_vector_mps is None:
        velocity_vector_mps = np.array([200.0, 0.0, 0.0])
    return {
        "position_m": np.asarray(position_m, dtype=np.float64),
        "velocity_vector_mps": np.asarray(velocity_vector_mps, dtype=np.float64),
        "altitude_m": altitude_m,
    }


def make_target_state(position_m=None, velocity_vector_mps=None, altitude_m=5000.0):
    if position_m is None:
        position_m = np.array([2000.0, 0.0, altitude_m])
    if velocity_vector_mps is None:
        velocity_vector_mps = np.array([200.0, 0.0, 0.0])
    return {
        "position_m": np.asarray(position_m, dtype=np.float64),
        "velocity_vector_mps": np.asarray(velocity_vector_mps, dtype=np.float64),
        "altitude_m": altitude_m,
    }


# ---------------------------------------------------------------------------
# SituationEvaluator tests
# ---------------------------------------------------------------------------

class TestSituationEvaluator:
    def test_favorable_returns_offensive(self):
        ev = SituationEvaluator()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state(range_m=800.0, aa_rad=np.deg2rad(15.0), range_rate_mps=-30.0)
        result = ev.evaluate(own, target, rel)
        assert result["tactical_state"] == TacticalState.STRONG_OFFENSIVE

    def test_low_altitude_returns_unsafe(self):
        ev = SituationEvaluator()
        own = make_own_state(altitude_m=200.0)
        target = make_target_state(altitude_m=5000.0)
        rel = make_rel_state()
        result = ev.evaluate(own, target, rel)
        assert result["tactical_state"] == TacticalState.UNSAFE

    def test_large_aa_returns_defensive(self):
        ev = SituationEvaluator()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state(aa_rad=np.deg2rad(150.0))
        result = ev.evaluate(own, target, rel)
        assert result["tactical_state"] == TacticalState.DEFENSIVE

    def test_neutral_for_moderate_angles(self):
        ev = SituationEvaluator()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state(range_m=2000.0, aa_rad=np.deg2rad(80.0))
        result = ev.evaluate(own, target, rel)
        assert result["tactical_state"] == TacticalState.NEUTRAL

    def test_scores_are_finite(self):
        ev = SituationEvaluator()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state()
        result = ev.evaluate(own, target, rel)
        scores = result["scores"]
        for k in ["angle", "range", "energy", "altitude", "closure"]:
            assert np.isfinite(scores[k])
        assert np.isfinite(result["situation_score"])

    def test_missing_fields_graceful(self):
        ev = SituationEvaluator()
        own = {"position_m": np.array([0.0, 0.0, 5000.0])}
        target = {"position_m": np.array([2000.0, 0.0, 5000.0])}
        rel = {"range_m": 2000.0}
        result = ev.evaluate(own, target, rel)
        assert result["tactical_state"] is not None
        assert np.isfinite(result["situation_score"])


# ---------------------------------------------------------------------------
# RuleEngine tests
# ---------------------------------------------------------------------------

class TestRuleEngine:
    def test_unsafe_priority(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.UNSAFE,
            "diagnostics": {"range_m": 500.0, "closure_rate": 50.0, "aspect_deg": 20.0, "altitude_safe": False},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.ALTITUDE_RECOVER

    def test_strong_offensive_close_range_lag(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.STRONG_OFFENSIVE,
            "diagnostics": {"range_m": 500.0, "closure_rate": 150.0, "aspect_deg": 15.0, "altitude_safe": True},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.LAG_PURSUIT

    def test_strong_offensive_far_pure(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.STRONG_OFFENSIVE,
            "diagnostics": {"range_m": 1000.0, "closure_rate": 50.0, "aspect_deg": 15.0, "altitude_safe": True},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.PURE_PURSUIT

    def test_offensive_high_aspect_lead(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.OFFENSIVE,
            "diagnostics": {"range_m": 2000.0, "closure_rate": 50.0, "aspect_deg": 45.0, "altitude_safe": True},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.LEAD_PURSUIT

    def test_defensive_extend(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.DEFENSIVE,
            "diagnostics": {"range_m": 800.0, "closure_rate": 50.0, "aspect_deg": 140.0, "altitude_safe": True},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.EXTEND

    def test_neutral_high_angle_cut(self):
        engine = RuleEngine()
        situation = {
            "tactical_state": TacticalState.NEUTRAL,
            "diagnostics": {"range_m": 2000.0, "closure_rate": 50.0, "aspect_deg": 90.0, "altitude_safe": True},
        }
        rel = make_rel_state()
        result = engine.select_intent(situation, rel)
        assert result["intent"] == ManeuverIntent.HIGH_ANGLE_CUT


# ---------------------------------------------------------------------------
# ExpertVPPPolicy tests
# ---------------------------------------------------------------------------

class TestExpertVPPPolicy:
    def test_output_shape_and_bounds(self):
        policy = ExpertVPPPolicy()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state()
        action = policy.get_action(own, target, rel)
        assert action.shape == (3,)
        assert action.dtype == np.float64
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_output_no_nan_inf(self):
        policy = ExpertVPPPolicy()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state()
        action = policy.get_action(own, target, rel)
        assert np.all(np.isfinite(action))

    def test_diagnostics_populated(self):
        policy = ExpertVPPPolicy()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state()
        action = policy.get_action(own, target, rel)
        diag = policy.get_last_diagnostics()
        assert diag["expert_enabled"] is True
        assert diag["expert_tactical_state"] in vars(TacticalState).values()
        assert diag["expert_maneuver_intent"] in vars(ManeuverIntent).values()
        assert "expert_rule_id" in diag
        assert "expert_action_x" in diag
        assert "expert_action_y" in diag
        assert "expert_action_z" in diag

    def test_unsafe_outputs_altitude_recover(self):
        policy = ExpertVPPPolicy()
        own = make_own_state(altitude_m=200.0)
        target = make_target_state(altitude_m=5000.0)
        rel = make_rel_state(altitude_diff_m=4800.0)
        action = policy.get_action(own, target, rel)
        diag = policy.get_last_diagnostics()
        assert diag["expert_tactical_state"] == TacticalState.UNSAFE
        assert diag["expert_maneuver_intent"] == ManeuverIntent.ALTITUDE_RECOVER
        assert action[2] > 0.0

    def test_strong_offensive_lag_when_close(self):
        policy = ExpertVPPPolicy()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state(range_m=500.0, range_rate_mps=-150.0, aa_rad=np.deg2rad(15.0))
        action = policy.get_action(own, target, rel)
        diag = policy.get_last_diagnostics()
        assert diag["expert_tactical_state"] == TacticalState.STRONG_OFFENSIVE
        assert diag["expert_maneuver_intent"] == ManeuverIntent.LAG_PURSUIT
        assert action[0] < 0.0

    def test_reset_history(self):
        policy = ExpertVPPPolicy()
        own = make_own_state()
        target = make_target_state()
        rel = make_rel_state()
        policy.get_action(own, target, rel)
        assert len(policy._history) == 1
        policy.reset_history()
        assert len(policy._history) == 0

    def test_compatible_with_virtual_point_generator(self):
        """Output format compatible with VirtualPointGenerator (3-D [-1,1])."""
        policy = ExpertVPPPolicy()
        for scenario in [
            {"range_m": 800.0, "aa_rad": np.deg2rad(10.0)},
            {"range_m": 3000.0, "aa_rad": np.deg2rad(90.0)},
            {"range_m": 500.0, "aa_rad": np.deg2rad(150.0)},
        ]:
            own = make_own_state()
            target = make_target_state()
            rel = make_rel_state(**scenario)
            action = policy.get_action(own, target, rel)
            assert len(action) == 3
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_target_velocity_frame_conversion(self):
        """When target moves along +y, lead/lag should rotate accordingly."""
        policy = ExpertVPPPolicy()
        own = make_own_state(position_m=np.array([0.0, 0.0, 5000.0]))
        target = make_target_state(
            position_m=np.array([0.0, 2000.0, 5000.0]),
            velocity_vector_mps=np.array([0.0, 200.0, 0.0]),
        )
        rel = make_rel_state(range_m=2000.0, aa_rad=np.deg2rad(45.0))
        action = policy.get_action(own, target, rel)
        # Target heading is +y, so lead (+x in TV frame) should map to +y in world
        diag = policy.get_last_diagnostics()
        if diag["expert_maneuver_intent"] == ManeuverIntent.LEAD_PURSUIT:
            # Allow tolerance for rotation
            assert action[1] > 0.1
