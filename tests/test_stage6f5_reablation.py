"""
Tests for Stage 6F.5 Scenario Redesign & Maneuvering Target Re-Ablation.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from uav_vpp_guidance.utils.config import load_yaml_config


class TestScenarioFeasibilityChecker(unittest.TestCase):
    """Scenario feasibility checker must correctly identify infeasible geometries."""

    def test_feasible_geometry_favorable_is_feasible(self):
        from scripts.check_scenario_feasibility import compute_scenario_feasibility
        scenario = {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 250.0, "heading_deg": 0.0},
            "target_init": {"position_m": [800.0, 0.0, 5000.0], "velocity_mps": 180.0, "heading_deg": 0.0},
        }
        result = compute_scenario_feasibility("favorable", scenario, max_range_m=12000.0)
        self.assertTrue(result["feasible"])
        self.assertGreater(result["closure_rate_mps"], 0)

    def test_infeasible_negative_closure_rate(self):
        from scripts.check_scenario_feasibility import compute_scenario_feasibility
        # Ego heading AWAY from target, target behind and faster -> closure rate negative
        scenario = {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 180.0, "heading_deg": 0.0},
            "target_init": {"position_m": [-1000.0, 0.0, 5000.0], "velocity_mps": 220.0, "heading_deg": 180.0},
        }
        result = compute_scenario_feasibility("disadvantage", scenario, max_range_m=8000.0)
        self.assertFalse(result["feasible"])
        self.assertLessEqual(result["closure_rate_mps"], 0)
        self.assertIn("closure_rate <= 0", result["warnings"])

    def test_low_closure_rate_flagged(self):
        from scripts.check_scenario_feasibility import compute_scenario_feasibility
        # Ego chasing target but target is faster and heading away
        scenario = {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 180.0, "heading_deg": 0.0},
            "target_init": {"position_m": [-1000.0, 0.0, 5000.0], "velocity_mps": 220.0, "heading_deg": 180.0},
        }
        result = compute_scenario_feasibility("stern_chase", scenario, max_range_m=8000.0)
        self.assertFalse(result["feasible"])
        self.assertIn("closure_rate <= 0", result["warnings"])

    def test_large_turn_angle_flagged(self):
        from scripts.check_scenario_feasibility import compute_scenario_feasibility
        scenario = {
            "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
            "target_init": {"position_m": [-500.0, 500.0, 5000.0], "velocity_mps": 220.0, "heading_deg": 90.0},
        }
        result = compute_scenario_feasibility("stern_conversion", scenario, max_range_m=12000.0)
        self.assertFalse(result["feasible"])
        self.assertGreater(result["required_turn_angle_deg"], 90.0)


class TestStage6F5ConfigSchema(unittest.TestCase):
    """Stage 6F.5 config files must have correct schema."""

    def test_feasible_geometry_config_exists(self):
        self.assertTrue(os.path.exists("config/experiment/stage6f5_feasible_geometry.yaml"))

    def test_maneuvering_target_config_exists(self):
        self.assertTrue(os.path.exists("config/experiment/stage6f5_maneuvering_target.yaml"))

    def test_feasible_geometry_has_five_methods(self):
        config = load_yaml_config("config/experiment/stage6f5_feasible_geometry.yaml")
        methods = config.get("methods", {})
        self.assertEqual(len(methods), 5)
        expected = {"no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"}
        self.assertEqual(set(methods.keys()), expected)

    def test_maneuvering_target_has_sinusoidal_mode(self):
        config = load_yaml_config("config/experiment/stage6f5_maneuvering_target.yaml")
        target_mode = config.get("env", {}).get("target_mode")
        self.assertEqual(target_mode, "sinusoidal")

    def test_feasible_geometry_has_scenario_metadata(self):
        config = load_yaml_config("config/experiment/stage6f5_feasible_geometry.yaml")
        for name, sc in config.get("scenarios", {}).items():
            with self.subTest(scenario=name):
                meta = sc.get("metadata", {})
                self.assertIn("scenario_name", meta)
                self.assertIn("initial_range_m", meta)
                self.assertIn("closure_rate_mps", meta)
                self.assertIn("expected_feasible", meta)
                self.assertIn("rationale", meta)

    def test_maneuvering_target_has_target_acceleration_rms(self):
        config = load_yaml_config("config/experiment/stage6f5_maneuvering_target.yaml")
        for name, sc in config.get("scenarios", {}).items():
            with self.subTest(scenario=name):
                meta = sc.get("metadata", {})
                self.assertIn("target_acceleration_rms", meta)
                self.assertGreater(meta["target_acceleration_rms"], 0)

    def test_feasible_geometry_increased_max_range(self):
        config = load_yaml_config("config/experiment/stage6f5_feasible_geometry.yaml")
        max_range = config.get("env", {}).get("max_range_m")
        self.assertGreater(max_range, 8000.0)

    def test_maneuvering_target_increased_max_range(self):
        config = load_yaml_config("config/experiment/stage6f5_maneuvering_target.yaml")
        max_range = config.get("env", {}).get("max_range_m")
        self.assertGreater(max_range, 8000.0)


class TestManeuveringTargetNonzeroAcceleration(unittest.TestCase):
    """Sinusoidal target mode must produce non-zero true acceleration."""

    def test_sinusoidal_target_has_lateral_acceleration(self):
        from uav_vpp_guidance.envs.simple_point_mass_env import SimplePointMassEnv
        env = SimplePointMassEnv({"target_mode": "sinusoidal"})
        env.target_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "altitude_m": 5000.0,
        }
        env.time = 0.0
        # Step target multiple times and compute acceleration from velocity differences
        vels = []
        for _ in range(20):
            env._update_target(None)
            vels.append(env.target_state["velocity_vector_mps"].copy())
            env.time += env.dt
        vels = np.array(vels)
        accs = np.diff(vels, axis=0) / env.dt
        acc_norms = np.linalg.norm(accs, axis=1)
        self.assertGreater(np.max(acc_norms), 0.1,
                           "Sinusoidal target should produce non-zero acceleration")

    def test_constant_velocity_target_has_zero_acceleration(self):
        from uav_vpp_guidance.envs.simple_point_mass_env import SimplePointMassEnv
        env = SimplePointMassEnv({"target_mode": "constant_velocity"})
        env.target_state = {
            "position_m": np.array([0.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
            "altitude_m": 5000.0,
        }
        vels = []
        for _ in range(20):
            env._update_target(None)
            vels.append(env.target_state["velocity_vector_mps"].copy())
        vels = np.array(vels)
        accs = np.diff(vels, axis=0) / env.dt
        acc_norms = np.linalg.norm(accs, axis=1)
        self.assertLess(np.max(acc_norms), 1e-6,
                        "Constant-velocity target should have zero acceleration")


class TestCVCAExpectedDifferentUnderManeuvering(unittest.TestCase):
    """CA predictor should produce different predictions than CV under maneuvering target."""

    def test_ca_predicts_different_from_cv_under_sinusoidal(self):
        from uav_vpp_guidance.trajectory_prediction.constant_acceleration import ConstantAccelerationPredictor
        from uav_vpp_guidance.trajectory_prediction.constant_velocity import ConstantVelocityPredictor

        ca = ConstantAccelerationPredictor(lookahead_time_s=1.0)
        cv = ConstantVelocityPredictor(lookahead_time_s=1.0)

        # Simulate a target with sinusoidal lateral motion
        current_target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 10.0, 0.0]),
        }

        # Build history with non-zero acceleration
        vel_scale = 300.0
        history_seq = np.zeros((5, 16))
        # velocity features at indices 3:6
        history_seq[:, 3] = (200.0 + np.array([0, 2, 4, 6, 8])) / vel_scale
        history_seq[:, 4] = (10.0 + np.array([0, 1, 2, 3, 4])) / vel_scale
        history_seq[:, 5] = 0.0

        ca_pred, _, ca_info = ca.predict(history_seq=history_seq, current_target_state=current_target_state)
        cv_pred, _, cv_info = cv.predict(history_seq=None, current_target_state=current_target_state)

        self.assertIsNotNone(ca_pred)
        self.assertIsNotNone(cv_pred)
        diff = np.linalg.norm(ca_pred - cv_pred)
        self.assertGreater(diff, 1.0,
                           f"CA and CV should differ under maneuvering target (diff={diff})")

    def test_ca_falls_back_to_cv_with_no_history(self):
        from uav_vpp_guidance.trajectory_prediction.constant_acceleration import ConstantAccelerationPredictor
        from uav_vpp_guidance.trajectory_prediction.constant_velocity import ConstantVelocityPredictor

        ca = ConstantAccelerationPredictor(lookahead_time_s=1.0)
        cv = ConstantVelocityPredictor(lookahead_time_s=1.0)

        current_target_state = {
            "position_neu": np.array([1000.0, 0.0, 5000.0]),
            "velocity_vector_mps": np.array([200.0, 0.0, 0.0]),
        }

        ca_pred, _, ca_info = ca.predict(history_seq=None, current_target_state=current_target_state)
        cv_pred, _, cv_info = cv.predict(history_seq=None, current_target_state=current_target_state)

        self.assertTrue(ca_info.get("fallback"))
        np.testing.assert_allclose(ca_pred, cv_pred, atol=1e-9)


class TestStage6F5RunnerDryRun(unittest.TestCase):
    """Stage 6F.5 runner dry-run must produce commands for both suites."""

    def test_feasible_geometry_dry_run(self):
        import io
        from contextlib import redirect_stdout
        from scripts.run_stage6f5_reablation import main as runner_main

        f = io.StringIO()
        with redirect_stdout(f):
            old_argv = sys.argv
            sys.argv = [
                "run_stage6f5_reablation.py",
                "--suite", "feasible_geometry",
                "--dry-run",
                "--training-seeds", "0", "1",
                "--evaluation-seeds", "0",
                "--episodes-per-scenario", "25",
            ]
            try:
                runner_main()
            finally:
                sys.argv = old_argv

        output = f.getvalue()
        self.assertIn("Suite: feasible_geometry", output)
        self.assertIn("Scenarios: ['favorable', 'neutral', 'disadvantage', 'challenging']", output)
        for method in ("no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"):
            self.assertIn(method, output)
        self.assertNotIn("--allow-random-policy", output)

    def test_maneuvering_target_dry_run(self):
        import io
        from contextlib import redirect_stdout
        from scripts.run_stage6f5_reablation import main as runner_main

        f = io.StringIO()
        with redirect_stdout(f):
            old_argv = sys.argv
            sys.argv = [
                "run_stage6f5_reablation.py",
                "--suite", "maneuvering_target",
                "--dry-run",
                "--training-seeds", "0",
                "--evaluation-seeds", "0",
                "--episodes-per-scenario", "25",
            ]
            try:
                runner_main()
            finally:
                sys.argv = old_argv

        output = f.getvalue()
        self.assertIn("Suite: maneuvering_target", output)
        self.assertIn("Scenarios: ['weaving_pursuit', 'weaving_headon', 'weaving_offset', 'weaving_disadvantage']", output)
        for method in ("no_prediction", "cv_prediction", "ca_prediction", "lstm_frozen", "gru_frozen"):
            self.assertIn(method, output)

    def test_runner_rejects_invalid_suite(self):
        import io
        from contextlib import redirect_stdout
        from scripts.run_stage6f5_reablation import main as runner_main

        f = io.StringIO()
        with redirect_stdout(f):
            old_argv = sys.argv
            sys.argv = [
                "run_stage6f5_reablation.py",
                "--suite", "invalid_suite",
                "--dry-run",
            ]
            try:
                with self.assertRaises(SystemExit):
                    runner_main()
            finally:
                sys.argv = old_argv


class TestPaperTableUsesSampleStd(unittest.TestCase):
    """Paper-ready tables must use sample standard deviation (ddof=1)."""

    def test_aggregate_script_uses_sample_std(self):
        from scripts.aggregate_stage6f_results import _safe_std
        vals = [1.0, 2.0, 3.0]
        # sample std of [1,2,3] = sqrt(((0-1)^2 + (1-0)^2 + (2-0)^2)/2) = sqrt(2/2) = 1.0
        result = _safe_std(vals)
        expected = np.std(vals, ddof=1)
        self.assertAlmostEqual(result, expected, places=6)

    def test_deep_audit_stability_uses_sample_std(self):
        from scripts.analyze_stage6f_deep_audit import compute_stability_metrics
        import pandas as pd

        seed_df = pd.DataFrame([
            {"method": "test", "training_seed": 0, "success_rate": 0.5, "mean_return": 10.0},
            {"method": "test", "training_seed": 1, "success_rate": 0.25, "mean_return": 5.0},
            {"method": "test", "training_seed": 2, "success_rate": 0.25, "mean_return": 7.0},
        ])
        stability = compute_stability_metrics(seed_df)
        self.assertEqual(len(stability), 1)
        row = stability.iloc[0]
        expected_sr_std = np.std([0.5, 0.25, 0.25], ddof=1)
        self.assertAlmostEqual(row["success_rate_std"], expected_sr_std, places=6)


class TestStage6F5AnalysisScript(unittest.TestCase):
    """Analysis script must produce expected tables even with partial data."""

    def test_analysis_handles_empty_data(self):
        from scripts.analyze_stage6f5_results import build_overall_summary
        import pandas as pd

        empty_df = pd.DataFrame(columns=["method", "scenario", "is_success", "return", "mean_env_prediction_error_m"])
        result = build_overall_summary(empty_df)
        self.assertTrue(result.empty)

    def test_cv_ca_delta_computed_correctly(self):
        from scripts.analyze_stage6f5_results import build_cv_ca_delta
        import pandas as pd

        df = pd.DataFrame([
            {"method": "cv_prediction", "scenario": "s1", "is_success": True, "return": 100.0, "mean_env_prediction_error_m": 10.0},
            {"method": "cv_prediction", "scenario": "s1", "is_success": False, "return": -100.0, "mean_env_prediction_error_m": 15.0},
            {"method": "ca_prediction", "scenario": "s1", "is_success": True, "return": 120.0, "mean_env_prediction_error_m": 8.0},
            {"method": "ca_prediction", "scenario": "s1", "is_success": True, "return": 110.0, "mean_env_prediction_error_m": 9.0},
        ])
        delta = build_cv_ca_delta(df)
        self.assertEqual(len(delta), 1)
        self.assertAlmostEqual(delta.iloc[0]["cv_success_rate"], 0.5)
        self.assertAlmostEqual(delta.iloc[0]["ca_success_rate"], 1.0)
        self.assertAlmostEqual(delta.iloc[0]["delta_success_rate"], 0.5)

    def test_neural_vs_classical_computed_correctly(self):
        from scripts.analyze_stage6f5_results import build_neural_vs_classical
        import pandas as pd

        df = pd.DataFrame([
            {"method": "no_prediction", "scenario": "s1", "is_success": False, "return": -50.0},
            {"method": "cv_prediction", "scenario": "s1", "is_success": True, "return": 50.0},
            {"method": "lstm_frozen", "scenario": "s1", "is_success": True, "return": 100.0},
            {"method": "gru_frozen", "scenario": "s1", "is_success": True, "return": 90.0},
        ])
        result = build_neural_vs_classical(df)
        self.assertEqual(len(result), 1)
        # classical = no_pred + cv = 1/2 success; neural = lstm + gru = 2/2 success
        self.assertAlmostEqual(result.iloc[0]["classical_success_rate"], 0.5)
        self.assertAlmostEqual(result.iloc[0]["neural_success_rate"], 1.0)
        self.assertAlmostEqual(result.iloc[0]["delta_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
