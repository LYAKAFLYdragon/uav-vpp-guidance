"""
Tests for Stage 6G Guidance-Law Limitation Probe and paper hardening.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestStage6F6RemoteArtifacts(unittest.TestCase):
    """Remote branch must contain Stage 6F.6 artifacts."""

    def test_synthesize_script_exists(self):
        self.assertTrue(os.path.exists("scripts/synthesize_stage6f_paper_results.py"))

    def test_mechanism_script_exists(self):
        self.assertTrue(os.path.exists("scripts/analyze_gru_lstm_mechanism.py"))

    def test_stage6f6_test_file_exists(self):
        self.assertTrue(os.path.exists("tests/test_stage6f6_synthesis.py"))

    def test_runner_has_experiment_suite_version(self):
        from scripts.run_stage6f5_reablation import EXPERIMENT_SUITE_VERSION
        self.assertEqual(EXPERIMENT_SUITE_VERSION, "6f.5")

    def test_analysis_has_expected_seeds_guard(self):
        from scripts.analyze_stage6f5_results import discover_training_seeds
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extra seed should be ignored
            for seed in [0, 1, 99]:
                d = os.path.join(tmpdir, f"train_seed{seed}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "prediction_metrics.json"), "w") as f:
                    json.dump([], f)
            seeds = discover_training_seeds(Path(tmpdir), expected_seeds=[0, 1])
            self.assertEqual(seeds, [0, 1])


class TestPaperSafeClaimsDoNotUsePValueOnly(unittest.TestCase):
    """Paper-safe claims must require cross-seed consistency, not just p-value."""

    def test_gru_lstm_not_safe_when_cross_seed_inconsistent(self):
        from scripts.synthesize_stage6f_paper_results import build_claims_checklist
        stats = {
            "gru_vs_lstm_weaving_headon": {
                "n": 450, "a_only": 10, "b_only": 100, "both": 50, "neither": 290,
                "discordant": 110, "p_value": 1e-20, "cohens_d": 0.8,
                "return_diff_ci": [50.0, 100.0],
                "gru_per_seed_sr": {0: 1.0, 1: 1.0, 2: 0.0},
                "lstm_per_seed_sr": {0: 1.0, 1: 0.0, 2: 0.0},
                "gru_gt_lstm_all_seeds": False,
                "gru_ge_lstm_all_seeds": True,
                "n_training_seeds": 3,
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
        gru_claim = [c for c in claims if "GRU is more robust" in c["claim"]][0]
        # Even with tiny p-value, cross-seed inconsistency should block paper_safe
        self.assertFalse(gru_claim["paper_safe_claim"])
        self.assertIn("cross-seed", gru_claim["evidence"].lower())

    def test_gru_lstm_safe_when_strictly_consistent(self):
        from scripts.synthesize_stage6f_paper_results import build_claims_checklist
        stats = {
            "gru_vs_lstm_weaving_headon": {
                "n": 450, "a_only": 10, "b_only": 100, "both": 50, "neither": 290,
                "discordant": 110, "p_value": 1e-20, "cohens_d": 0.8,
                "return_diff_ci": [50.0, 100.0],
                "gru_per_seed_sr": {0: 1.0, 1: 0.8, 2: 0.6},
                "lstm_per_seed_sr": {0: 0.5, 1: 0.4, 2: 0.3},
                "gru_gt_lstm_all_seeds": True,
                "gru_ge_lstm_all_seeds": True,
                "n_training_seeds": 3,
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
                {"method": "lstm_frozen", "success_rate": 0.40},
                {"method": "gru_frozen", "success_rate": 0.80},
            ]),
            "table_f": pd.DataFrame([
                {"scenario": "s1", "success_changed": False, "cohens_d": 0.05},
            ]),
            "table_c": pd.DataFrame([
                {"scenario": "favorable", "success_rate": 0.0},
            ]),
        }
        claims = build_claims_checklist(stats, tables)
        gru_claim = [c for c in claims if "GRU is more robust" in c["claim"]][0]
        self.assertTrue(gru_claim["paper_safe_claim"])


class TestPaperNarrativeContainsLimitations(unittest.TestCase):
    """Paper narrative and sections must explicitly discuss limitations."""

    def test_results_section_mentions_tail_chase_limitation(self):
        path = "outputs/paper/stage6_results_section.md"
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("guidance-law limitation", content.lower())
        self.assertIn("tail-chase", content.lower())

    def test_limitations_section_exists(self):
        path = "outputs/paper/stage6_limitations_section.md"
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("training seed", content.lower())
        self.assertIn("kinematic", content.lower())

    def test_discussion_section_mentions_cross_seed(self):
        path = "outputs/paper/stage6_discussion_section.md"
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("cross-seed", content.lower())
        self.assertIn("episode-level", content.lower())


class TestGuidanceProbeConfig(unittest.TestCase):
    """Probe script must generate valid temporary configs."""

    def test_build_probe_config_overrides_guidance_mode(self):
        from scripts.run_stage6g_guidance_limitation_probe import build_probe_config
        import tempfile, yaml, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            base = {
                "guidance": {"mode": "los_rate", "gains": {"k_los": 1.0}},
                "scenarios": {
                    "favorable": {"init": {}},
                    "disadvantage": {"init": {}},
                },
            }
            yaml.dump(base, f)
            temp_path = f.name
        try:
            probe = build_probe_config(temp_path, "proportional_navigation", "favorable")
            self.assertEqual(probe["guidance"]["mode"], "proportional_navigation")
            self.assertEqual(list(probe["scenarios"].keys()), ["favorable"])
        finally:
            os.unlink(temp_path)

    def test_probe_rejects_unknown_scenario(self):
        from scripts.run_stage6g_guidance_limitation_probe import build_probe_config
        import tempfile, yaml, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump({"scenarios": {"favorable": {}}}, f)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError):
                build_probe_config(temp_path, "los_rate", "nonexistent")
        finally:
            os.unlink(temp_path)


class TestGuidanceProbeDryRun(unittest.TestCase):
    """Probe dry-run must produce commands for all guidance x scenario combinations."""

    def test_dry_run_produces_all_combinations(self):
        import io
        from contextlib import redirect_stdout
        from scripts.run_stage6g_guidance_limitation_probe import main

        f = io.StringIO()
        with redirect_stdout(f):
            old_argv = sys.argv
            sys.argv = [
                "run_stage6g_guidance_limitation_probe.py",
                "--dry-run",
                "--episodes-per-scenario", "10",
                "--eval-seeds", "0", "1", "2",
            ]
            try:
                main()
            finally:
                sys.argv = old_argv

        output = f.getvalue()
        self.assertIn("los_rate", output)
        self.assertIn("proportional_navigation", output)
        self.assertIn("hybrid", output)
        self.assertIn("favorable", output)
        self.assertIn("disadvantage", output)
        self.assertIn("weaving_pursuit", output)
        self.assertIn("weaving_disadvantage", output)
        # 3 modes x 4 scenarios = 12 dry-run blocks
        self.assertGreaterEqual(output.count("[DRY-RUN]"), 12)


if __name__ == "__main__":
    unittest.main()
