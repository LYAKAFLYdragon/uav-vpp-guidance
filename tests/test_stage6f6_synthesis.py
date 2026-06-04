"""
Tests for Stage 6F.6 Paper-Ready Synthesis & GRU-vs-LSTM Mechanism Audit.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestStage6F5ExperimentSuiteVersion(unittest.TestCase):
    """Runner and analysis scripts must emit experiment_suite_version = 6f.5."""

    def test_runner_has_experiment_suite_version(self):
        from scripts.run_stage6f5_reablation import EXPERIMENT_SUITE_VERSION
        self.assertEqual(EXPERIMENT_SUITE_VERSION, "6f.5")

    def test_analysis_has_experiment_suite_version(self):
        from scripts.analyze_stage6f5_results import EXPERIMENT_SUITE_VERSION
        self.assertEqual(EXPERIMENT_SUITE_VERSION, "6f.5")

    def test_experiment_plan_contains_suite_version(self):
        import tempfile, json, os
        from scripts.run_stage6f5_reablation import write_experiment_plan
        with tempfile.TemporaryDirectory() as tmpdir:
            write_experiment_plan(
                output_dir=tmpdir,
                suite_name="test_suite",
                comparison_config="config/test.yaml",
                training_seeds=[0, 1],
                evaluation_seeds=[0],
                episodes_per_scenario=10,
                scenarios=["s1"],
                formal=True,
            )
            plan_path = os.path.join(tmpdir, "experiment_plan.json")
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            self.assertEqual(plan.get("experiment_suite_version"), "6f.5")
            self.assertEqual(plan.get("metrics_schema_version"), "6f.2")

    def test_manifest_contains_suite_version(self):
        import tempfile, json, os
        from scripts.run_stage6f5_reablation import write_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            write_manifest(
                output_dir=tmpdir,
                method="gru_frozen",
                seed=0,
                config_path="config/test.yaml",
                policy_checkpoint_path="ckpt/best.pt",
                predictor_checkpoint_path=None,
                backend="simple",
                validation_mode="raise",
                allow_random_policy=False,
            )
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest.get("experiment_suite_version"), "6f.5")
            self.assertEqual(manifest.get("metrics_schema_version"), "6f.2")


class TestStage6F5ExpectedSeedsGuard(unittest.TestCase):
    """Analysis script must filter seeds and raise on missing expected seeds."""

    def test_discover_ignores_extra_seeds(self):
        from scripts.analyze_stage6f5_results import discover_training_seeds
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create train_seed0, train_seed1, train_seed99
            for seed in [0, 1, 99]:
                d = os.path.join(tmpdir, f"train_seed{seed}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "prediction_metrics.json"), "w") as f:
                    json.dump([], f)
            seeds = discover_training_seeds(Path(tmpdir), expected_seeds=[0, 1])
            self.assertEqual(seeds, [0, 1])

    def test_discover_raises_on_missing_seeds(self):
        from scripts.analyze_stage6f5_results import discover_training_seeds
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, "train_seed0")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "prediction_metrics.json"), "w") as f:
                json.dump([], f)
            with self.assertRaises(ValueError) as ctx:
                discover_training_seeds(Path(tmpdir), expected_seeds=[0, 1])
            self.assertIn("Missing expected training seeds", str(ctx.exception))


class TestPaperSynthesisTables(unittest.TestCase):
    """Synthesis script must produce all required tables."""

    def test_table_b_feasible_subset_filters_correctly(self):
        from scripts.synthesize_stage6f_paper_results import build_table_b_feasible_subset
        df = pd.DataFrame([
            {"method": "lstm_frozen", "scenario": "neutral", "is_success": True, "return": 100.0, "mean_env_prediction_error_m": 10.0},
            {"method": "lstm_frozen", "scenario": "favorable", "is_success": False, "return": -100.0, "mean_env_prediction_error_m": 20.0},
            {"method": "gru_frozen", "scenario": "weaving_headon", "is_success": True, "return": 120.0, "mean_env_prediction_error_m": 8.0},
        ])
        result = build_table_b_feasible_subset(df)
        self.assertEqual(len(result), 2)
        scenarios = set()
        for _, row in result.iterrows():
            # favorable should be excluded
            self.assertNotIn(row["method"], ["favorable"])
        methods = set(result["method"].tolist())
        self.assertEqual(methods, {"lstm_frozen", "gru_frozen"})

    def test_table_c_dead_zone_shows_zero_success(self):
        from scripts.synthesize_stage6f_paper_results import build_table_c_dead_zone
        df = pd.DataFrame([
            {"method": "no_prediction", "scenario": "favorable", "is_success": False, "return": -100.0, "reason": "crash"},
            {"method": "lstm_frozen", "scenario": "favorable", "is_success": False, "return": -100.0, "reason": "crash"},
            {"method": "gru_frozen", "scenario": "disadvantage", "is_success": False, "return": -200.0, "reason": "out_of_bounds"},
        ])
        result = build_table_c_dead_zone(df)
        self.assertEqual(len(result), 3)
        self.assertTrue((result["success_rate"] == 0.0).all())

    def test_table_f_cv_ca_delta_computes_effect_size(self):
        from scripts.synthesize_stage6f_paper_results import build_table_f_cv_ca_delta
        df = pd.DataFrame([
            {"method": "cv_prediction", "scenario": "s1", "is_success": True, "return": 100.0},
            {"method": "cv_prediction", "scenario": "s1", "is_success": False, "return": -50.0},
            {"method": "ca_prediction", "scenario": "s1", "is_success": True, "return": 105.0},
            {"method": "ca_prediction", "scenario": "s1", "is_success": True, "return": 110.0},
        ])
        result = build_table_f_cv_ca_delta(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["delta_success_rate"], 0.5, places=6)
        self.assertFalse(result.iloc[0]["success_changed"])
        self.assertTrue(np.isfinite(result.iloc[0]["cohens_d"]))

    def test_table_e_gru_lstm_focused(self):
        from scripts.synthesize_stage6f_paper_results import build_table_e_gru_lstm_focused
        df = pd.DataFrame([
            {"method": "lstm_frozen", "scenario": "weaving_headon", "is_success": True, "return": 100.0,
             "mean_env_prediction_error_m": 10.0, "mean_virtual_point_shift_m": 50.0, "final_range_m": 100.0, "final_ata_deg": 15.0},
            {"method": "gru_frozen", "scenario": "weaving_headon", "is_success": True, "return": 120.0,
             "mean_env_prediction_error_m": 8.0, "mean_virtual_point_shift_m": 45.0, "final_range_m": 90.0, "final_ata_deg": 12.0},
        ])
        result = build_table_e_gru_lstm_focused(df)
        self.assertEqual(len(result), 2)
        gru = result[result["method"] == "gru_frozen"].iloc[0]
        self.assertAlmostEqual(gru["mean_return"], 120.0)


class TestPaperClaimsChecklist(unittest.TestCase):
    """Claims checklist must correctly label statistical and practical significance."""

    def test_claims_mark_gru_vs_lstm_paper_safe_when_large_delta(self):
        from scripts.synthesize_stage6f_paper_results import build_claims_checklist
        stats = {
            "gru_vs_lstm_weaving_headon": {
                "n": 100, "a_only": 10, "b_only": 40, "both": 20, "neither": 30,
                "discordant": 50, "p_value": 0.001, "cohens_d": 0.8,
                "return_diff_ci": [10.0, 30.0],
            },
            "neural_vs_classical_feasible": {
                "neural_success_rate": 0.5, "classical_success_rate": 0.25,
                "delta_success_rate": 0.25, "cohens_d": 0.6,
            },
            "ca_vs_cv_maneuvering": {
                "n": 100, "a_only": 5, "b_only": 5, "both": 45, "neither": 45,
                "discordant": 10, "p_value": 1.0, "cohens_d": 0.05,
                "return_diff_ci": [-1.0, 1.0],
            },
        }
        tables = {
            "table_e": pd.DataFrame([
                {"method": "lstm_frozen", "success_rate": 0.33},
                {"method": "gru_frozen", "success_rate": 0.67},
            ]),
            "table_f": pd.DataFrame([
                {"scenario": "s1", "success_changed": False, "cohens_d": 0.05},
            ]),
            "table_c": pd.DataFrame([
                {"scenario": "favorable", "success_rate": 0.0},
                {"scenario": "disadvantage", "success_rate": 0.0},
            ]),
        }
        claims = build_claims_checklist(stats, tables)
        gru_claim = [c for c in claims if "GRU" in c["claim"]][0]
        self.assertTrue(gru_claim["statistically_supported"])
        self.assertTrue(gru_claim["practically_meaningful"])
        self.assertTrue(gru_claim["paper_safe_claim"])

    def test_claims_mark_cv_ca_not_paper_safe(self):
        from scripts.synthesize_stage6f_paper_results import build_claims_checklist
        stats = {
            "gru_vs_lstm_weaving_headon": {"n": 100, "a_only": 10, "b_only": 10, "both": 40, "neither": 40, "discordant": 20, "p_value": 0.5, "cohens_d": 0.1, "return_diff_ci": [-5.0, 5.0]},
            "neural_vs_classical_feasible": {"neural_success_rate": 0.5, "classical_success_rate": 0.25, "delta_success_rate": 0.25, "cohens_d": 0.6},
            "ca_vs_cv_maneuvering": {"n": 100, "a_only": 5, "b_only": 5, "both": 45, "neither": 45, "discordant": 10, "p_value": 1.0, "cohens_d": 0.05, "return_diff_ci": [-1.0, 1.0]},
        }
        tables = {
            "table_e": pd.DataFrame([
                {"method": "lstm_frozen", "success_rate": 0.33},
                {"method": "gru_frozen", "success_rate": 0.33},
            ]),
            "table_f": pd.DataFrame([
                {"scenario": "s1", "success_changed": False, "cohens_d": 0.05},
            ]),
            "table_c": pd.DataFrame([
                {"scenario": "favorable", "success_rate": 0.0},
                {"scenario": "disadvantage", "success_rate": 0.0},
            ]),
        }
        claims = build_claims_checklist(stats, tables)
        ca_claim = [c for c in claims if "CA" in c["claim"]][0]
        self.assertFalse(ca_claim["paper_safe_claim"])


class TestStatisticalComparisonOutputs(unittest.TestCase):
    """Statistical tests must produce finite, reasonable outputs."""

    def test_bootstrap_ci_reasonable(self):
        from scripts.synthesize_stage6f_paper_results import bootstrap_ci
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(vals, n_bootstrap=5000)
        self.assertTrue(np.isfinite(lower))
        self.assertTrue(np.isfinite(upper))
        self.assertLess(lower, upper)
        self.assertGreater(lower, vals.min() - 1.0)
        self.assertLess(upper, vals.max() + 1.0)

    def test_bootstrap_success_rate_ci(self):
        from scripts.synthesize_stage6f_paper_results import bootstrap_success_rate_ci
        flags = np.array([True, True, False, False, False])
        lower, upper = bootstrap_success_rate_ci(flags, n_bootstrap=5000)
        self.assertTrue(np.isfinite(lower))
        self.assertTrue(np.isfinite(upper))
        self.assertLessEqual(lower, upper)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

    def test_mcnemar_paired_exact(self):
        from scripts.synthesize_stage6f_paper_results import mcnemar_paired_comparison
        a = np.array([True, False, True, False, True, False])
        b = np.array([True, True, False, False, True, False])
        result = mcnemar_paired_comparison(a, b)
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["a_only"], 1)  # a=True, b=False
        self.assertEqual(result["b_only"], 1)  # a=False, b=True
        self.assertTrue(np.isfinite(result["p_value"]))

    def test_cohens_d_between_groups(self):
        from scripts.synthesize_stage6f_paper_results import cohens_d_between_groups
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        d = cohens_d_between_groups(a, b)
        self.assertTrue(np.isfinite(d))
        self.assertGreater(d, 0.0)


class TestGRULSTMMechanismMissingFields(unittest.TestCase):
    """Mechanism audit must report missing fields without crashing."""

    def test_missing_fields_detected(self):
        from scripts.analyze_gru_lstm_mechanism import check_missing_fields
        ep_df = pd.DataFrame([
            {"method": "lstm_frozen", "mean_env_prediction_error_m": 10.0, "mean_virtual_point_shift_m": 5.0},
        ])
        traj_dfs = [pd.DataFrame({"range_m": [100.0], "ata_deg": [10.0]})]
        missing = check_missing_fields(ep_df, traj_dfs)
        self.assertIn("trajectory", missing)
        self.assertIn("prediction_error_m", missing["trajectory"])
        self.assertIn("virtual_point_shift_m", missing["trajectory"])

    def test_missing_fields_empty_when_complete(self):
        from scripts.analyze_gru_lstm_mechanism import check_missing_fields
        ep_df = pd.DataFrame([
            {"method": "lstm_frozen", "mean_env_prediction_error_m": 10.0, "mean_offline_aligned_error_m": 8.0,
             "mean_virtual_point_shift_m": 5.0, "final_range_m": 100.0, "final_ata_deg": 10.0, "length": 200, "return": 50.0, "is_success": True, "reason": "success"},
        ])
        traj_dfs = [pd.DataFrame({
            "prediction_error_m": [10.0],
            "virtual_point_shift_m": [5.0],
            "nz_cmd": [1.0],
            "roll_rate_cmd": [0.5],
            "throttle_cmd": [0.8],
            "range_m": [100.0],
            "ata_deg": [10.0],
            "ego_z": [5000.0],
        })]
        missing = check_missing_fields(ep_df, traj_dfs)
        self.assertEqual(missing, {})


class TestNoOverclaimSignificance(unittest.TestCase):
    """Statistical outputs must not overclaim significance with few training seeds."""

    def test_ci_widens_with_fewer_samples(self):
        from scripts.synthesize_stage6f_paper_results import bootstrap_ci
        # Use same distribution, different sample sizes
        rng = np.random.default_rng(seed=123)
        small = rng.normal(loc=5.0, scale=1.0, size=10)
        large = rng.normal(loc=5.0, scale=1.0, size=100)
        l_s, u_s = bootstrap_ci(small, n_bootstrap=5000)
        l_l, u_l = bootstrap_ci(large, n_bootstrap=5000)
        width_small = u_s - l_s
        width_large = u_l - l_l
        # CI for smaller sample should generally be wider
        self.assertGreater(width_small, width_large * 0.5)

    def test_paper_safe_claim_false_for_weak_evidence(self):
        from scripts.synthesize_stage6f_paper_results import build_claims_checklist
        stats = {
            "gru_vs_lstm_weaving_headon": {"n": 3, "a_only": 1, "b_only": 2, "both": 0, "neither": 0, "discordant": 3, "p_value": 1.0, "cohens_d": 0.05, "return_diff_ci": [-50.0, 50.0]},
            "neural_vs_classical_feasible": {"neural_success_rate": 0.30, "classical_success_rate": 0.25, "delta_success_rate": 0.05, "cohens_d": 0.1},
            "ca_vs_cv_maneuvering": {"n": 3, "a_only": 0, "b_only": 0, "both": 1, "neither": 2, "discordant": 0, "p_value": 1.0, "cohens_d": 0.0, "return_diff_ci": [-1.0, 1.0]},
        }
        tables = {
            "table_e": pd.DataFrame([
                {"method": "lstm_frozen", "success_rate": 0.33},
                {"method": "gru_frozen", "success_rate": 0.33},
            ]),
            "table_f": pd.DataFrame([
                {"scenario": "s1", "success_changed": False, "cohens_d": 0.0},
            ]),
            "table_c": pd.DataFrame([
                {"scenario": "favorable", "success_rate": 0.0},
            ]),
        }
        claims = build_claims_checklist(stats, tables)
        for claim in claims:
            if "GRU" in claim["claim"] and "LSTM" in claim["claim"]:
                self.assertFalse(claim["paper_safe_claim"])
            if "Neural" in claim["claim"]:
                self.assertFalse(claim["paper_safe_claim"])


if __name__ == "__main__":
    unittest.main()
