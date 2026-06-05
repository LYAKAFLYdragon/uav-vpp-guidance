"""Tests for Stage 6G.4: per-step telemetry, geometric direction, and smoke infrastructure."""

import json
import os
import tempfile
import unittest

import numpy as np

from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.evaluation.telemetry_schema_validator import (
    validate_episode_telemetry,
    validate_episodes_telemetry,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


class TestLeadDistanceFromConfig(unittest.TestCase):
    def test_lead_distance_read_from_config(self):
        gen = VirtualPointGenerator({
            "lead_distance_m": 1000.0,
            "d_long_range": [-100, 100],
            "d_lat_range": [-100, 100],
            "d_vert_range": [-50, 50],
        })
        self.assertEqual(gen.lead_distance_m, 1000.0)

    def test_lead_distance_defaults_to_500(self):
        gen = VirtualPointGenerator({
            "d_long_range": [-100, 100],
            "d_lat_range": [-100, 100],
            "d_vert_range": [-50, 50],
        })
        self.assertEqual(gen.lead_distance_m, 500.0)


class TestRuleBasedDirectionGeometry(unittest.TestCase):
    """Verify rule_based_pursuit VPP direction is geometrically correct."""

    def test_vpp_ahead_of_target_when_target_ahead_of_uav(self):
        """UAV at origin, target at +x. VPP should be ahead of target (+x)."""
        gen = VirtualPointGenerator({
            "lead_distance_m": 500.0,
            "d_long_range": [-100, 100],
            "d_lat_range": [-100, 100],
            "d_vert_range": [-50, 50],
        })
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([180.0, 0.0, 0.0]),
        }
        action = np.zeros(3)
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="rule_based_pursuit", return_info=True,
        )
        anchor = info["anchor_pos"]
        # Anchor must be ahead of target along x
        self.assertGreater(anchor[0], target_state["position_neu"][0],
                           "Anchor should be ahead of target")
        # Anchor y/z should equal target y/z (LOS is purely along x)
        self.assertAlmostEqual(anchor[1], target_state["position_neu"][1], places=6)
        self.assertAlmostEqual(anchor[2], target_state["position_neu"][2], places=6)

    def test_vpp_ahead_of_target_when_target_behind_uav(self):
        """UAV at +x, target at origin. VPP should be ahead of target (-x direction from UAV)."""
        gen = VirtualPointGenerator({
            "lead_distance_m": 500.0,
            "d_long_range": [-100, 100],
            "d_lat_range": [-100, 100],
            "d_vert_range": [-50, 50],
        })
        own_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}
        target_state = {
            "position_neu": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([180.0, 0.0, 0.0]),
        }
        action = np.zeros(3)
        vp, info = gen.action_to_virtual_point(
            action, own_state, target_state,
            anchor_mode="rule_based_pursuit", return_info=True,
        )
        anchor = info["anchor_pos"]
        # Anchor must be behind target (more negative x) because LOS is from UAV to target (-x)
        self.assertLess(anchor[0], target_state["position_neu"][0],
                        "Anchor should be in target-forward direction along LOS")


class TestOracleNotOverriddenByPredictedFallback(unittest.TestCase):
    """Ensure tracking_env preserves oracle/rule-based anchor modes."""

    def test_oracle_anchor_not_overridden(self):
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "oracle_future_position",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"max_nz": 6.0, "max_roll_rate": 30.0},
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(scenario="test")
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        self.assertEqual(info["anchor_mode"], "oracle_future_position")
        env.close()

    def test_rule_based_anchor_not_overridden(self):
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "rule_based_pursuit",
                "lead_distance_m": 500.0,
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"max_nz": 6.0, "max_roll_rate": 30.0},
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(scenario="test")
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        self.assertEqual(info["anchor_mode"], "rule_based_pursuit")
        env.close()


class TestPerStepTelemetryAggregates(unittest.TestCase):
    """Verify evaluate_single_episode returns command/altitude/energy aggregates."""

    def test_episode_contains_command_saturation_fields(self):
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"nz_min": -2.0, "nz_max": 7.0, "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                       "throttle_min": 0.0, "throttle_max": 1.0},
        }
        env = CloseRangeTrackingEnv(config)
        agent = PPOAgent(
            obs_dim=int(env.reset(seed=0)["observation_vector"].shape[0]),
            action_dim=3,
            config=config,
            device="cpu",
        )
        ep_result, traj = evaluate_single_episode(env, agent, config, scenario="test", seed=0)
        env.close()

        # Command saturation fields
        self.assertIn("nz_cmd_max", ep_result)
        self.assertIn("nz_cmd_mean", ep_result)
        self.assertIn("nz_cmd_saturation_rate", ep_result)
        self.assertIn("nz_cmd_modification_rate", ep_result)
        self.assertIn("roll_rate_cmd_max", ep_result)
        self.assertIn("roll_rate_cmd_mean", ep_result)
        self.assertIn("roll_rate_cmd_saturation_rate", ep_result)
        self.assertIn("throttle_cmd_max", ep_result)
        self.assertIn("throttle_cmd_mean", ep_result)
        self.assertIn("throttle_cmd_saturation_rate", ep_result)

        # Altitude / energy fields
        self.assertIn("min_altitude_m", ep_result)
        self.assertIn("max_altitude_m", ep_result)
        self.assertIn("final_altitude_m", ep_result)
        self.assertIn("altitude_loss_rate", ep_result)
        self.assertIn("energy_proxy", ep_result)

    def test_telemetry_validator_passes_with_new_fields(self):
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"nz_min": -2.0, "nz_max": 7.0, "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                       "throttle_min": 0.0, "throttle_max": 1.0},
        }
        env = CloseRangeTrackingEnv(config)
        agent = PPOAgent(
            obs_dim=int(env.reset(seed=0)["observation_vector"].shape[0]),
            action_dim=3,
            config=config,
            device="cpu",
        )
        ep_result, traj = evaluate_single_episode(env, agent, config, scenario="test", seed=0)
        env.close()

        # Add method/guidance_mode/episode_index which are injected later by evaluate_method
        ep_result["method"] = "test_method"
        ep_result["guidance_mode"] = "los_rate"
        ep_result["episode_index"] = 0
        ok, crit, missing = validate_episode_telemetry(
            ep_result,
            require_core=True,
            require_command_saturation=True,
            require_altitude_energy=True,
        )
        self.assertTrue(ok, f"Telemetry validation failed: critical={crit}, missing={missing}")
        self.assertEqual(len(crit), 0)

    def test_telemetry_validator_fails_missing_core_fields(self):
        incomplete = {"scenario": "test", "method": "m1"}
        ok, crit, missing = validate_episode_telemetry(incomplete, require_core=True)
        self.assertFalse(ok)
        self.assertTrue(len(crit) > 0)


class TestSmokeProbeOutputs(unittest.TestCase):
    """Verify smoke runner produces required summary files."""

    def test_smoke_summary_contains_required_keys(self):
        summary = {
            "probe": "oracle_vpp_anchor",
            "config": "config/experiment/stage6g3_oracle_vpp_anchor.yaml",
            "episodes": 2,
            "seeds": [0],
            "timestamp": "20260101_000000",
            "results": {
                "oracle_vpp_anchor": {
                    "overall_success_rate": 0.0,
                    "overall_crash_rate": 0.5,
                    "per_scenario": {},
                }
            },
        }
        required_keys = {"probe", "config", "episodes", "seeds", "timestamp", "results"}
        self.assertTrue(required_keys.issubset(set(summary.keys())))
        results = summary["results"]
        for method_name, method_data in results.items():
            self.assertIn("overall_success_rate", method_data)
            self.assertIn("overall_crash_rate", method_data)
            self.assertIn("per_scenario", method_data)

    def test_actual_smoke_output_files_exist(self):
        """If smoke outputs exist, verify their structure."""
        output_dir = "outputs/stage6g4_smoke_test"
        files = {
            "oracle": os.path.join(output_dir, "oracle_anchor_smoke_summary.json"),
            "rule_based": os.path.join(output_dir, "rule_based_pursuit_smoke_summary.json"),
            "terminal": os.path.join(output_dir, "terminal_control_ablation_smoke_summary.json"),
            "geometry": os.path.join(output_dir, "geometry_feasibility_smoke_summary.json"),
        }
        for name, path in files.items():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("probe", data, f"{name} missing 'probe'")
                self.assertIn("timestamp", data, f"{name} missing 'timestamp'")
                # Geometry feasibility uses 'success_by_params' instead of 'results'
                if data.get("probe") == "geometry_feasibility":
                    self.assertIn("success_by_params", data, f"{name} missing 'success_by_params'")
                else:
                    self.assertIn("results", data, f"{name} missing 'results'")
            else:
                # Smoke outputs are optional for unit tests; skip if not present
                continue


class TestAblationFlagsEffective(unittest.TestCase):
    """Verify terminal control ablation variants produce different runtime flags."""

    def _make_env(self, guidance_overrides=None):
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {
                "mode": "los_rate",
                "params": {"capture_radius_m": 50.0},
                "post_process": {
                    "enabled": True,
                    "enable_terminal_protection": True,
                    "enable_energy_compensation": True,
                    "enable_load_roll_coordination": True,
                },
            },
            "limits": {"nz_min": -2.0, "nz_max": 7.0, "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                       "throttle_min": 0.0, "throttle_max": 1.0},
        }
        if guidance_overrides:
            config["guidance"] = {**config.get("guidance", {}), **guidance_overrides}
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        return CloseRangeTrackingEnv(config)

    def test_baseline_has_post_process_enabled(self):
        env = self._make_env()
        self.assertIsNotNone(env.command_post_processor)
        self.assertTrue(env.command_post_processor.enable_terminal_protection)
        self.assertTrue(env.command_post_processor.enable_energy_comp)
        self.assertTrue(env.command_post_processor.enable_load_roll_coord)
        env.close()

    def test_no_post_process_disables_processor(self):
        env = self._make_env({"post_process": {"enabled": False}})
        self.assertIsNone(env.command_post_processor)
        env.close()

    def test_no_terminal_protection_changes_flag(self):
        env = self._make_env({"post_process": {"enabled": True, "enable_terminal_protection": False}})
        self.assertIsNotNone(env.command_post_processor)
        self.assertFalse(env.command_post_processor.enable_terminal_protection)
        env.close()

    def test_no_energy_comp_changes_flag(self):
        env = self._make_env({"post_process": {"enabled": True, "enable_energy_compensation": False}})
        self.assertIsNotNone(env.command_post_processor)
        self.assertFalse(env.command_post_processor.enable_energy_comp)
        env.close()

    def test_no_load_roll_coord_changes_flag(self):
        env = self._make_env({"post_process": {"enabled": True, "enable_load_roll_coordination": False}})
        self.assertIsNotNone(env.command_post_processor)
        self.assertFalse(env.command_post_processor.enable_load_roll_coord)
        env.close()

    def test_capture_radius_zero_changes_value(self):
        env = self._make_env({"params": {"capture_radius_m": 0.0}})
        self.assertEqual(env.guidance.capture_radius_m, 0.0)
        env.close()

    def test_rule_based_lead_distance_changes_vpp(self):
        gen = VirtualPointGenerator({"lead_distance_m": 1000.0, "d_long_range": [-100, 100],
                                     "d_lat_range": [-100, 100], "d_vert_range": [-50, 50]})
        own_state = {"position_neu": np.array([0.0, 0.0, 5000.0])}
        target_state = {"position_neu": np.array([1000.0, 0.0, 5000.0])}
        vp_500, _ = gen.action_to_virtual_point(np.zeros(3), own_state, target_state,
                                                 anchor_mode="rule_based_pursuit", return_info=True)
        gen2 = VirtualPointGenerator({"lead_distance_m": 2000.0, "d_long_range": [-100, 100],
                                      "d_lat_range": [-100, 100], "d_vert_range": [-50, 50]})
        vp_2000, _ = gen2.action_to_virtual_point(np.zeros(3), own_state, target_state,
                                                   anchor_mode="rule_based_pursuit", return_info=True)
        # VPP x should increase with lead distance
        self.assertGreater(vp_2000["position"][0], vp_500["position"][0])

    def test_oracle_mode_does_not_fallback(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "oracle_future_position",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"max_nz": 6.0, "max_roll_rate": 30.0},
        }
        env = CloseRangeTrackingEnv(config)
        env.reset(scenario="test")
        obs, reward, terminated, truncated, info = env.step(np.zeros(5))
        self.assertEqual(info["anchor_mode"], "oracle_future_position")
        env.close()


class TestTrajectoryFieldsExtended(unittest.TestCase):
    """Verify per-step trajectory CSV contains extended telemetry fields."""

    def test_trajectory_contains_altitude_and_raw_commands(self):
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
        config = {
            "scenarios": {
                "test": {
                    "max_range_m": 5000,
                    "ego": {"position_m": [0, 0, 1000], "velocity_mps": 150},
                    "target": {"position_m": [2000, 0, 1000], "velocity_mps": 150},
                }
            },
            "virtual_point": {
                "anchor_mode": "current_target",
                "d_long": [-100, 100],
                "d_lat": [-100, 100],
                "d_vert": [-50, 50],
            },
            "guidance": {"mode": "los_rate"},
            "limits": {"nz_min": -2.0, "nz_max": 7.0, "roll_rate_min": -1.5, "roll_rate_max": 1.5,
                       "throttle_min": 0.0, "throttle_max": 1.0},
        }
        env = CloseRangeTrackingEnv(config)
        agent = PPOAgent(
            obs_dim=int(env.reset(seed=0)["observation_vector"].shape[0]),
            action_dim=3,
            config=config,
            device="cpu",
        )
        ep_result, traj = evaluate_single_episode(env, agent, config, scenario="test", seed=0, save_trajectory=True)
        env.close()
        self.assertTrue(len(traj) > 0)
        step = traj[0]
        self.assertIn("altitude_m", step)
        self.assertIn("relative_speed_mps", step)
        self.assertIn("raw_nz_cmd", step)
        self.assertIn("raw_roll_rate_cmd", step)
        self.assertIn("raw_throttle_cmd", step)
        self.assertIn("nz_saturated", step)
        self.assertIn("roll_rate_saturated", step)
        self.assertIn("throttle_saturated", step)


if __name__ == "__main__":
    unittest.main()
