"""Stage 6G.5D-R: Latch robustness contract tests.

These tests verify that the episode-level mode-switch latch behaves
correctly across edge cases: persistence, reset, state isolation,
telemetry, and negative controls.
"""

import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestLatchPersistenceAndReset(unittest.TestCase):
    """Latch must persist through an episode and reset on env.reset()."""

    def _make_env(self, mode_switch_cfg=None, guidance_mode="proportional_navigation"):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        import yaml

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode"] = guidance_mode
        cfg["guidance"]["direct_track_mode"] = False
        if mode_switch_cfg is not None:
            cfg["guidance"]["mode_switch"] = mode_switch_cfg
        else:
            cfg["guidance"]["mode_switch"] = {
                "enabled": True,
                "aspect_threshold_deg": 15.0,
                "range_threshold_m": 3000.0,
                "closing_speed_threshold_mps": 100.0,
            }
        return CloseRangeTrackingEnv(cfg)

    def test_latch_persists_when_aspect_exceeds_threshold(self):
        """Gate activates on step 1; even if aspect later exceeds threshold,
        latch keeps mode-switch active for the rest of the episode."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env()
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            # Step 1: gate should be active
            _, _, _, _, info1 = env.step(np.zeros(3))
            self.assertTrue(info1["mode_switch_effective"], "Gate should activate on step 1")
            self.assertEqual(info1["effective_guidance_mode"], "proportional_navigation")
            self.assertEqual(info1["virtual_point_source"], "direct_track")

            # Step 2: latch should keep it active even if pre-step aspect > threshold
            _, _, _, _, info2 = env.step(np.zeros(3))
            self.assertTrue(info2["mode_switch_effective"], "Latch should persist on step 2")
            self.assertEqual(info2["virtual_point_source"], "direct_track")
            self.assertIn(info2["mode_switch_reason"], ("gate_active", "latched"))
        finally:
            env.close()

    def test_latch_resets_on_env_reset(self):
        """After env.reset(), latch must be cleared."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env()
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            env.step(np.zeros(3))
            self.assertTrue(env._mode_switch_latched, "Latch should be active after step 1")

            env.reset(scenario=scenario, seed=1)
            self.assertFalse(env._mode_switch_latched, "Latch must reset on env.reset()")
        finally:
            env.close()

    def test_pn_guidance_state_resets_on_env_reset(self):
        """_guidance_pn internal LOS filter state must be reset between episodes."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env()
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            env.step(np.zeros(3))
            # After step 1, _guidance_pn should have stored _prev_los_vec
            self.assertIsNotNone(env._guidance_pn._prev_los_vec)

            env.reset(scenario=scenario, seed=1)
            # After reset, _prev_los_vec should be cleared
            self.assertIsNone(env._guidance_pn._prev_los_vec)
        finally:
            env.close()


class TestLatchNegativeControls(unittest.TestCase):
    """Latch must NOT activate when gate conditions are not met."""

    def _make_env(self, mode_switch_cfg):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        import yaml

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode"] = "proportional_navigation"
        cfg["guidance"]["direct_track_mode"] = False
        cfg["guidance"]["mode_switch"] = mode_switch_cfg
        return CloseRangeTrackingEnv(cfg)

    def test_latch_does_not_activate_for_high_aspect(self):
        """High aspect angle should keep gate inactive."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env({
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        })
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=90, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            _, _, _, _, info = env.step(np.zeros(3))
            self.assertFalse(info["mode_switch_effective"], "Gate should NOT activate for 90° aspect")
            self.assertFalse(env._mode_switch_latched, "Latch must stay inactive")
            self.assertEqual(info["virtual_point_source"], "vpp_policy")
        finally:
            env.close()

    def test_latch_does_not_activate_for_low_closing_speed(self):
        """Low closing speed should keep gate inactive."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env({
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 200.0,  # very high threshold
        })
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=150, target_speed_mps=120,  # closing speed = 30
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            _, _, _, _, info = env.step(np.zeros(3))
            self.assertFalse(info["mode_switch_effective"], "Gate should NOT activate for low closing speed")
            self.assertFalse(env._mode_switch_latched, "Latch must stay inactive")
        finally:
            env.close()

    def test_latch_does_not_activate_for_long_range(self):
        """Range above threshold should keep gate inactive."""
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        env = self._make_env({
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 1000.0,  # very short threshold
            "closing_speed_threshold_mps": 100.0,
        })
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            _, _, _, _, info = env.step(np.zeros(3))
            self.assertFalse(info["mode_switch_effective"], "Gate should NOT activate when range > threshold")
            self.assertFalse(env._mode_switch_latched, "Latch must stay inactive")
        finally:
            env.close()


class TestLatchBehaviorWithVppElsewhere(unittest.TestCase):
    """mode_switch_vpp_elsewhere must use VPP before gate and PN after latch."""

    def test_vpp_before_gate_pn_after_latch(self):
        """Monkey-patch gate to delay activation by 2 steps;
        verify virtual_point_source toggles correctly."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
        import yaml

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode"] = "los_rate"
        cfg["guidance"]["direct_track_mode"] = False
        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)

        # Delay gate activation: first 2 steps inactive, then active
        call_count = [0]
        orig_gate = env._evaluate_mode_switch_gate

        def delayed_gate(rel_state):
            call_count[0] += 1
            if call_count[0] <= 2:
                return False, "delayed"
            return orig_gate(rel_state)

        env._evaluate_mode_switch_gate = delayed_gate
        try:
            # Step 1: gate inactive → VPP
            _, _, _, _, info1 = env.step(np.zeros(3))
            self.assertFalse(info1["mode_switch_effective"], "Gate should be inactive on step 1")
            self.assertEqual(info1["virtual_point_source"], "vpp_policy", "Should use VPP before gate")

            # Step 2: gate still inactive → VPP
            _, _, _, _, info2 = env.step(np.zeros(3))
            self.assertFalse(info2["mode_switch_effective"], "Gate should be inactive on step 2")
            self.assertEqual(info2["virtual_point_source"], "vpp_policy", "Should use VPP before gate")

            # Step 3: gate activates → latch → direct-track PN
            _, _, _, _, info3 = env.step(np.zeros(3))
            self.assertTrue(info3["mode_switch_effective"], "Gate should activate on step 3")
            self.assertEqual(info3["virtual_point_source"], "direct_track", "Should switch to direct-track PN")
            self.assertEqual(info3["effective_guidance_mode"], "proportional_navigation")

            # Step 4: latch persists
            _, _, _, _, info4 = env.step(np.zeros(3))
            self.assertTrue(info4["mode_switch_effective"], "Latch should persist on step 4")
            self.assertEqual(info4["virtual_point_source"], "direct_track")
        finally:
            env.close()


class TestLatchDefaultHoldPolicy(unittest.TestCase):
    """Default latch behavior is hold-for-episode; no implicit exit."""

    def test_default_policy_is_hold_for_episode(self):
        """Without explicit exit config, latch never deactivates mid-episode."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
        import yaml

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode"] = "proportional_navigation"
        cfg["guidance"]["direct_track_mode"] = False
        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
            # No exit_policy key → default is hold-for-episode
        }
        env = CloseRangeTrackingEnv(cfg)
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            # Activate gate
            env.step(np.zeros(3))
            self.assertTrue(env._mode_switch_latched)

            # Step multiple times; latch should never deactivate
            for _ in range(5):
                _, _, _, _, info = env.step(np.zeros(3))
                self.assertTrue(info["mode_switch_effective"], "Latch should not deactivate mid-episode")
                self.assertTrue(env._mode_switch_latched)
        finally:
            env.close()


class TestLatchTelemetryFields(unittest.TestCase):
    """Telemetry must contain all mode-switch and latch related fields."""

    def test_telemetry_contains_all_latch_fields(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
        import yaml

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode"] = "proportional_navigation"
        cfg["guidance"]["direct_track_mode"] = False
        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            _, _, _, _, info = env.step(np.zeros(3))
            required_keys = [
                "mode_switch_requested",
                "mode_switch_effective",
                "mode_switch_reason",
                "virtual_point_source",
                "effective_guidance_mode",
                "direct_track_mode_requested",
                "direct_track_mode_effective",
            ]
            for key in required_keys:
                self.assertIn(key, info, f"Telemetry missing required key: {key}")

            self.assertTrue(info["mode_switch_requested"])
            self.assertTrue(info["mode_switch_effective"])
            self.assertIn(info["mode_switch_reason"], ("gate_active", "latched"))
            self.assertEqual(info["virtual_point_source"], "direct_track")
            self.assertEqual(info["effective_guidance_mode"], "proportional_navigation")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
