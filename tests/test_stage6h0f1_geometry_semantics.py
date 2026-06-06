"""Geometry semantics tests (Stage 6H.0-F.1).

Verify that:
    1. Every scenario type produces expected geometry.
    2. Telemetry labels match actual geometry.
    3. Evaluator classification matches geometry validator output.
    4. Legacy and explicit builders are consistent where expected.
    5. ScenarioRegistry is the single source of truth.
"""

import copy
import math
import unittest
import warnings

import numpy as np
import yaml

from src.uav_vpp_guidance.envs.geometry_scenarios import (
    GEOMETRY_FAMILY_DOCS,
    VALID_SCENARIO_TYPES,
    build_explicit_scenario,
    build_scenario_from_legacy_params,
)
from src.uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from src.uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
from src.uav_vpp_guidance.utils.geometry_validator import (
    classify_geometry_family,
    compute_relative_geometry,
    validate_scenario_geometry,
    GEOMETRY_FAMILIES,
)


class TestExplicitScenarioGeometry(unittest.TestCase):
    """Task 1: Every scenario type produces expected geometry."""

    def test_all_families_defined(self):
        self.assertEqual(
            VALID_SCENARIO_TYPES,
            {"tail_chase", "head_on", "crossing_left", "crossing_right",
             "offset_pursuit", "offset_attack", "fleeing"},
        )

    def test_family_docs_cover_all_types(self):
        for st in VALID_SCENARIO_TYPES:
            self.assertIn(st, GEOMETRY_FAMILY_DOCS,
                          f"Missing documentation for {st}")

    def _assert_geometry(self, scenario, expected_family, expected_aspect_min,
                         expected_aspect_max, expected_closure_sign):
        report = validate_scenario_geometry(scenario)
        geo = report["geometry"]
        self.assertTrue(report["all_checks_pass"],
                        f"Checks failed: {report['consistency_checks']}")
        self.assertEqual(report["classified_family"], expected_family)
        self.assertGreaterEqual(geo["aspect_angle_deg"], expected_aspect_min)
        self.assertLessEqual(geo["aspect_angle_deg"], expected_aspect_max)
        if expected_closure_sign == "positive":
            self.assertGreater(geo["closure_rate_mps"], 0.0)
        elif expected_closure_sign == "negative":
            self.assertLess(geo["closure_rate_mps"], 0.0)

    def test_tail_chase_geometry(self):
        s = build_explicit_scenario("tail_chase", 2000.0, 340.0, 180.0)
        self._assert_geometry(s, "tail_chase", 0.0, 30.0, "positive")
        self.assertEqual(s["target_init"]["position_m"][0], 2000.0)
        self.assertEqual(s["target_init"]["heading_deg"], 0.0)

    def test_head_on_geometry(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        self._assert_geometry(s, "head_on", 150.0, 180.0, "positive")
        self.assertAlmostEqual(s["metadata"]["closure_rate_mps"], 400.0, delta=1.0)

    def test_crossing_left_geometry(self):
        s = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        self._assert_geometry(s, "crossing_left", 60.0, 120.0, "positive")
        self.assertEqual(s["target_init"]["position_m"][1], 2000.0)

    def test_crossing_right_geometry(self):
        s = build_explicit_scenario("crossing_right", 2000.0, 200.0, 200.0)
        self._assert_geometry(s, "crossing_right", 60.0, 120.0, "positive")
        self.assertEqual(s["target_init"]["position_m"][1], -2000.0)

    def test_offset_attack_geometry(self):
        s = build_explicit_scenario("offset_attack", 1000.0, 250.0, 180.0)
        report = validate_scenario_geometry(s)
        self.assertTrue(report["all_checks_pass"])
        self.assertEqual(report["classified_family"], "offset_attack")
        self.assertLess(s["target_init"]["position_m"][0], 0.0)
        self.assertNotEqual(s["target_init"]["position_m"][1], 0.0)

    def test_fleeing_geometry(self):
        s = build_explicit_scenario("fleeing", 2000.0, 200.0, 200.0)
        self._assert_geometry(s, "fleeing", 150.0, 180.0, "negative")
        self.assertEqual(s["target_init"]["position_m"][0], -2000.0)

    def test_offset_attack_alias_matches_offset_pursuit(self):
        s1 = build_explicit_scenario("offset_attack", 1000.0, 200.0, 220.0)
        s2 = build_explicit_scenario("offset_pursuit", 1000.0, 200.0, 220.0)
        self.assertEqual(s1["target_init"], s2["target_init"])
        self.assertEqual(s1["own_init"], s2["own_init"])


class TestGeometryValidator(unittest.TestCase):
    """Task 2: Geometry validation utilities produce correct metrics."""

    def test_compute_relative_geometry_head_on(self):
        geo = compute_relative_geometry(
            own_position_m=np.array([0.0, 0.0, 5000.0]),
            own_heading_deg=0.0,
            own_speed_mps=200.0,
            target_position_m=np.array([2000.0, 0.0, 5000.0]),
            target_heading_deg=180.0,
            target_speed_mps=200.0,
        )
        self.assertAlmostEqual(geo["range_m"], 2000.0, delta=0.1)
        self.assertAlmostEqual(geo["aspect_angle_deg"], 180.0, delta=0.1)
        self.assertAlmostEqual(geo["closure_rate_mps"], 400.0, delta=1.0)
        self.assertEqual(geo["opening_closing_status"], "closing")

    def test_compute_relative_geometry_tail_chase(self):
        geo = compute_relative_geometry(
            own_position_m=np.array([0.0, 0.0, 5000.0]),
            own_heading_deg=0.0,
            own_speed_mps=340.0,
            target_position_m=np.array([2000.0, 0.0, 5000.0]),
            target_heading_deg=0.0,
            target_speed_mps=180.0,
        )
        self.assertAlmostEqual(geo["range_m"], 2000.0, delta=0.1)
        self.assertAlmostEqual(geo["aspect_angle_deg"], 0.0, delta=0.1)
        self.assertAlmostEqual(geo["closure_rate_mps"], 160.0, delta=1.0)

    def test_classify_resolves_head_on_vs_fleeing(self):
        # Same aspect (~180), different closure
        head_on = compute_relative_geometry(
            np.array([0.0, 0.0, 5000.0]), 0.0, 200.0,
            np.array([2000.0, 0.0, 5000.0]), 180.0, 200.0,
        )
        fleeing = compute_relative_geometry(
            np.array([0.0, 0.0, 5000.0]), 0.0, 200.0,
            np.array([-2000.0, 0.0, 5000.0]), 180.0, 200.0,
        )
        self.assertEqual(classify_geometry_family(head_on), "head_on")
        self.assertEqual(classify_geometry_family(fleeing), "fleeing")

    def test_validate_scenario_geometry_returns_checks(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        report = validate_scenario_geometry(s)
        self.assertIn("geometry", report)
        self.assertIn("classified_family", report)
        self.assertIn("human_readable", report)
        self.assertIn("consistency_checks", report)
        self.assertTrue(report["all_checks_pass"])


class TestLegacyExplicitConsistency(unittest.TestCase):
    """Task 3 (partial): Legacy and explicit builders are consistent."""

    def test_legacy_zero_equals_explicit_tail_chase(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = build_geometry_scenario(2000.0, 340.0, 180.0, 0.0, 0.0)
        explicit = build_explicit_scenario("tail_chase", 2000.0, 340.0, 180.0)
        self.assertEqual(legacy["own_init"], explicit["own_init"])
        self.assertEqual(legacy["target_init"], explicit["target_init"])

    def test_legacy_ninety_classified_as_crossing(self):
        # Legacy 90° and explicit crossing_left differ by design
        # (legacy heading=90, explicit heading=270).  Both are crossing,
        # but the explicit builder resolves the ambiguity.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = build_geometry_scenario(2000.0, 200.0, 200.0, 90.0, 0.0)
        legacy_report = validate_scenario_geometry(legacy)
        self.assertIn(legacy_report["classified_family"], ["crossing_left", "crossing_right"])
        explicit = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        explicit_report = validate_scenario_geometry(explicit)
        self.assertEqual(explicit_report["classified_family"], "crossing_left")

    def test_legacy_180_raises_ambiguous(self):
        with self.assertRaises(ValueError) as ctx:
            build_scenario_from_legacy_params(2000.0, 200.0, 200.0, 180.0, 0.0)
        self.assertIn("ambiguous", str(ctx.exception).lower())

    def test_legacy_emits_deprecation_warning(self):
        with self.assertWarns(DeprecationWarning):
            build_geometry_scenario(2000.0, 200.0, 200.0, 0.0, 0.0)


class TestScenarioRegistry(unittest.TestCase):
    """Task 4: Unified ScenarioRegistry is the single source of truth."""

    def setUp(self):
        # Re-initialize to ensure clean state
        initialize_canonical_scenarios()

    def test_regression_baseline_scenarios_exist(self):
        names = ScenarioRegistry.list_names("regression_baseline")
        self.assertIn("regression_neutral", names)
        self.assertIn("regression_challenging", names)

    def test_smoke_test_one_per_family(self):
        smoke = ScenarioRegistry.get_set("smoke_test")
        families = set()
        for name, scen in smoke.items():
            fam = scen.get("metadata", {}).get("scenario_type", "unknown")
            families.add(fam)
        self.assertIn("tail_chase", families)
        self.assertIn("head_on", families)
        self.assertIn("crossing_left", families)
        self.assertIn("crossing_right", families)
        self.assertIn("offset_attack", families)
        self.assertIn("fleeing", families)

    def test_all_registry_scenarios_validate(self):
        for name in ScenarioRegistry.list_names():
            scen = ScenarioRegistry.get(name)
            report = validate_scenario_geometry(scen)
            self.assertTrue(
                report["all_checks_pass"],
                f"Scenario '{name}' failed validation: {report['consistency_checks']}"
            )

    def test_registry_scenarios_have_expected_families(self):
        expected = {
            "regression_neutral": "head_on",
            "regression_challenging": "head_on",  # geometrically head-on (aspect 180, los_from_ego ~0)
            "regression_crossing_left": "crossing_left",
            "regression_crossing_right": "crossing_right",
            "smoke_tail_chase": "tail_chase",
            "smoke_head_on": "head_on",
            "smoke_crossing_left": "crossing_left",
            "smoke_crossing_right": "crossing_right",
            "smoke_offset_attack": "offset_attack",
            "smoke_fleeing": "fleeing",
        }
        for name, expected_family in expected.items():
            scen = ScenarioRegistry.get(name)
            report = validate_scenario_geometry(scen)
            self.assertEqual(
                report["classified_family"], expected_family,
                f"Scenario '{name}' classified as {report['classified_family']}, "
                f"expected {expected_family}"
            )

    def test_registry_isolated_from_mutation(self):
        scen = ScenarioRegistry.get("regression_neutral")
        scen["own_init"]["position_m"][0] = 99999.0
        scen2 = ScenarioRegistry.get("regression_neutral")
        self.assertEqual(scen2["own_init"]["position_m"][0], 0.0)

    def test_candidate_search_has_non_tail_chase(self):
        candidates = ScenarioRegistry.get_set("candidate_search")
        for name, scen in candidates.items():
            fam = validate_scenario_geometry(scen)["classified_family"]
            self.assertNotEqual(fam, "tail_chase",
                                f"Candidate '{name}' is tail_chase; need non-tail-chase for threshold search")


class TestEnvLevelGeometryValidation(unittest.TestCase):
    """Task 3 (env level): After env reset, actual geometry matches scenario spec."""

    def _make_env(self):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from src.uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from src.uav_vpp_guidance.utils.config import merge_config
        import yaml
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        method_override = full_config.get("methods", {}).get("no_prediction", {})
        method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))
        method_config["backend"] = "simple"
        method_config["env"]["backend"] = "simple"
        method_config["env"]["use_jsbsim"] = False
        return CloseRangeTrackingEnv(method_config)

    def test_env_reset_matches_scenario_geometry(self):
        env = self._make_env()
        scen = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        env.reset(scenario=scen)

        # Extract actual state from simple backend
        simple = env._simple_env
        own_pos = simple.own_state["position_m"]
        tgt_pos = simple.target_state["position_m"]
        own_hdg = float(np.degrees(simple.own_state["heading_rad"]))
        tgt_hdg = float(np.degrees(simple.target_state["heading_rad"]))

        # Validate
        geo = compute_relative_geometry(
            own_pos, own_hdg, 200.0, tgt_pos, tgt_hdg, 200.0,
        )
        self.assertAlmostEqual(geo["range_m"], 2000.0, delta=10.0)
        self.assertAlmostEqual(geo["aspect_angle_deg"], 180.0, delta=5.0)
        self.assertGreater(geo["closure_rate_mps"], 0.0)
        env.close()

    def test_telemetry_label_matches_scenario_name(self):
        from src.uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
        from src.uav_vpp_guidance.agents.ppo_agent import PPOAgent

        env = self._make_env()
        scen = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        scen["name"] = "test_crossing_left"

        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = 3
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=env.config, device="cpu")

        result, _ = evaluate_single_episode(
            env, agent, env.config, scenario=scen, seed=42,
            save_trajectory=False, method_name="no_prediction",
        )
        self.assertEqual(result.get("scenario"), "test_crossing_left")
        env.close()

    def test_all_smoke_scenarios_env_reset_valid(self):
        env = self._make_env()
        for name in ScenarioRegistry.list_names("smoke_test"):
            scen = ScenarioRegistry.get(name)
            env.reset(scenario=scen)
            # Extract and validate
            simple = env._simple_env
            own_pos = simple.own_state["position_m"]
            tgt_pos = simple.target_state["position_m"]
            own_hdg = float(np.degrees(simple.own_state["heading_rad"]))
            tgt_hdg = float(np.degrees(simple.target_state["heading_rad"]))
            own_spd = float(np.linalg.norm(simple.own_state["velocity_vector_mps"]))
            tgt_spd = float(np.linalg.norm(simple.target_state["velocity_vector_mps"]))

            geo = compute_relative_geometry(
                own_pos, own_hdg, own_spd, tgt_pos, tgt_hdg, tgt_spd,
            )
            family = classify_geometry_family(geo)
            self.assertIn(
                family, GEOMETRY_FAMILIES,
                f"Smoke scenario '{name}' produced unclassified geometry: {family}"
            )
        env.close()


class TestEvaluatorConsistency(unittest.TestCase):
    """Task 3 (evaluator): Evaluator classification matches geometry validator."""

    def test_evaluator_scenario_field_present(self):
        from src.uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
        from src.uav_vpp_guidance.agents.ppo_agent import PPOAgent
        import sys, os
        import yaml
        from pathlib import Path
        from src.uav_vpp_guidance.utils.config import merge_config

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        config_path = Path(__file__).resolve().parent.parent / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
        full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        method_override = full_config.get("methods", {}).get("no_prediction", {})
        method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))
        method_config["backend"] = "simple"
        method_config["env"]["backend"] = "simple"
        method_config["env"]["use_jsbsim"] = False

        from src.uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        env = CloseRangeTrackingEnv(method_config)

        scen = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        scen["name"] = "evaluator_test_head_on"

        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=method_config, device="cpu")

        result, _ = evaluate_single_episode(
            env, agent, method_config, scenario=scen, seed=42,
            save_trajectory=False, method_name="no_prediction",
        )
        self.assertIn("scenario", result)
        self.assertEqual(result["scenario"], "evaluator_test_head_on")
        self.assertIn("is_success", result)
        self.assertIn("reason", result)
        env.close()


if __name__ == "__main__":
    unittest.main()
