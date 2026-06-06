"""Tests for explicit geometry scenario builder (Stage 6H.0-F)."""

import math
import unittest

from src.uav_vpp_guidance.envs.geometry_scenarios import (
    VALID_SCENARIO_TYPES,
    build_explicit_scenario,
    build_scenario_from_legacy_params,
)


class TestTailChase(unittest.TestCase):
    def test_target_ahead_same_direction(self):
        s = build_explicit_scenario("tail_chase", 2000.0, 340.0, 180.0)
        own = s["own_init"]
        tgt = s["target_init"]
        self.assertEqual(own["position_m"], [0.0, 0.0, 5000.0])
        self.assertEqual(tgt["position_m"][0], 2000.0)
        self.assertEqual(tgt["heading_deg"], 0.0)

    def test_positive_closure_when_ego_faster(self):
        s = build_explicit_scenario("tail_chase", 2000.0, 340.0, 180.0)
        meta = s["metadata"]
        self.assertEqual(meta["scenario_type"], "tail_chase")
        self.assertGreater(meta["closure_rate_mps"], 0.0)
        self.assertTrue(meta["expected_feasible_flag"])

    def test_negative_closure_when_ego_slower(self):
        s = build_explicit_scenario("tail_chase", 2000.0, 150.0, 180.0)
        meta = s["metadata"]
        self.assertLess(meta["closure_rate_mps"], 0.0)


class TestHeadOn(unittest.TestCase):
    def test_target_ahead_opposite_direction(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        own = s["own_init"]
        tgt = s["target_init"]
        self.assertEqual(tgt["position_m"][0], 2000.0)
        self.assertEqual(tgt["heading_deg"], 180.0)

    def test_high_closing_speed(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        meta = s["metadata"]
        self.assertAlmostEqual(meta["closure_rate_mps"], 400.0, delta=1.0)
        self.assertEqual(meta["aspect_angle_deg"], 180.0)


class TestCrossing(unittest.TestCase):
    def test_crossing_left(self):
        s = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        tgt = s["target_init"]
        self.assertEqual(tgt["position_m"][1], 2000.0)
        self.assertEqual(tgt["heading_deg"], 270.0)
        meta = s["metadata"]
        self.assertEqual(meta["scenario_type"], "crossing_left")

    def test_crossing_right(self):
        s = build_explicit_scenario("crossing_right", 2000.0, 200.0, 200.0)
        tgt = s["target_init"]
        self.assertEqual(tgt["position_m"][1], -2000.0)
        self.assertEqual(tgt["heading_deg"], 90.0)
        meta = s["metadata"]
        self.assertEqual(meta["scenario_type"], "crossing_right")

    def test_perpendicular_relative_velocity(self):
        s = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        meta = s["metadata"]
        # Closure should be non-zero because target has x-component toward ego
        # when heading=270, velocity is [-0, -200], so x-component is 0
        # Actually target at [0, 2000] heading 270 -> velocity [0, -200]
        # ego at [0,0] heading 0 -> velocity [200, 0]
        # rel_vel = [0-200, -200-0] = [-200, -200]
        # los_unit = [0, 1]
        # range_rate = dot([-200, -200], [0, 1]) = -200
        # closure_rate = 200
        self.assertGreater(meta["closure_rate_mps"], 0.0)


class TestOffsetPursuit(unittest.TestCase):
    def test_target_behind_with_offset(self):
        s = build_explicit_scenario("offset_pursuit", 1000.0, 200.0, 220.0)
        tgt = s["target_init"]
        self.assertLess(tgt["position_m"][0], 0.0)  # behind ego
        self.assertNotEqual(tgt["position_m"][1], 0.0)  # lateral offset
        meta = s["metadata"]
        self.assertEqual(meta["scenario_type"], "offset_pursuit")


class TestFleeing(unittest.TestCase):
    def test_target_behind_moving_away(self):
        s = build_explicit_scenario("fleeing", 2000.0, 200.0, 200.0)
        own = s["own_init"]
        tgt = s["target_init"]
        self.assertEqual(tgt["position_m"][0], -2000.0)
        self.assertEqual(tgt["heading_deg"], 180.0)

    def test_negative_closure(self):
        s = build_explicit_scenario("fleeing", 2000.0, 200.0, 200.0)
        meta = s["metadata"]
        self.assertLess(meta["closure_rate_mps"], 0.0)
        self.assertEqual(meta["aspect_angle_deg"], 180.0)


class TestAspectConvention(unittest.TestCase):
    def test_tail_chase_aspect_zero(self):
        s = build_explicit_scenario("tail_chase", 2000.0, 250.0, 180.0)
        self.assertEqual(s["metadata"]["aspect_angle_deg"], 0.0)

    def test_head_on_aspect_180(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        self.assertEqual(s["metadata"]["aspect_angle_deg"], 180.0)

    def test_crossing_left_aspect_90(self):
        s = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        self.assertAlmostEqual(s["metadata"]["aspect_angle_deg"], 90.0, delta=0.1)

    def test_crossing_right_aspect_90(self):
        s = build_explicit_scenario("crossing_right", 2000.0, 200.0, 200.0)
        self.assertAlmostEqual(s["metadata"]["aspect_angle_deg"], 90.0, delta=0.1)


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_outputs(self):
        s1 = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0, altitude_diff_m=200.0)
        s2 = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0, altitude_diff_m=200.0)
        self.assertEqual(s1["own_init"], s2["own_init"])
        self.assertEqual(s1["target_init"], s2["target_init"])


class TestSerialization(unittest.TestCase):
    def test_no_ndarrays(self):
        s = build_explicit_scenario("head_on", 2000.0, 200.0, 200.0)
        for key in ["position_m", "velocity_mps", "heading_deg"]:
            val = s["own_init"][key]
            self.assertIsInstance(val, (list, float, int))
            if isinstance(val, list):
                for v in val:
                    self.assertIsInstance(v, (float, int))

    def test_yaml_safe(self):
        import yaml
        s = build_explicit_scenario("crossing_left", 2000.0, 200.0, 200.0)
        dumped = yaml.safe_dump(s)
        loaded = yaml.safe_load(dumped)
        self.assertEqual(loaded["name"], "crossing_left")
        self.assertEqual(loaded["own_init"]["position_m"], [0.0, 0.0, 5000.0])


class TestLegacyWrapper(unittest.TestCase):
    def test_legacy_zero_becomes_tail_chase(self):
        s = build_scenario_from_legacy_params(2000.0, 340.0, 180.0, 0.0, 0.0)
        self.assertEqual(s["name"], "tail_chase")

    def test_legacy_180_raises_ambiguous(self):
        with self.assertRaises(ValueError) as ctx:
            build_scenario_from_legacy_params(2000.0, 200.0, 200.0, 180.0, 0.0)
        self.assertIn("ambiguous", str(ctx.exception).lower())

    def test_legacy_intermediate_angle_fallback(self):
        s = build_scenario_from_legacy_params(2000.0, 200.0, 200.0, 45.0, 0.0)
        self.assertIn(s["name"], ["crossing_left", "crossing_right"])


class TestValidation(unittest.TestCase):
    def test_invalid_scenario_type(self):
        with self.assertRaises(ValueError):
            build_explicit_scenario("not_a_type", 2000.0, 200.0, 200.0)

    def test_all_valid_types_build(self):
        for st in VALID_SCENARIO_TYPES:
            s = build_explicit_scenario(st, 2000.0, 200.0, 200.0)
            self.assertIn("metadata", s)
            self.assertIn("aspect_angle_deg", s["metadata"])


if __name__ == "__main__":
    unittest.main()
