"""Stage 6H.0-lite: Threshold optimization preflight contract tests."""

import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestThresholdGridSize(unittest.TestCase):
    """Full search space must contain 1152 configs."""

    def test_full_grid_size(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        space = mod.SEARCH_SPACE
        total = 1
        for values in space.values():
            total *= len(values)
        self.assertEqual(total, 1152, f"Expected 1152 configs, got {total}")


class TestThresholdSamplingDeterministic(unittest.TestCase):
    """Sampling with same seed must produce identical configs."""

    def test_latin_hypercube_deterministic(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        configs_a = mod._sample_configs(30, "latin_hypercube", seed=42)
        configs_b = mod._sample_configs(30, "latin_hypercube", seed=42)
        self.assertEqual(len(configs_a), 30)
        self.assertEqual(len(configs_b), 30)
        for a, b in zip(configs_a, configs_b):
            self.assertEqual(a, b)

    def test_random_deterministic(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        configs_a = mod._sample_configs(20, "random", seed=7)
        configs_b = mod._sample_configs(20, "random", seed=7)
        for a, b in zip(configs_a, configs_b):
            self.assertEqual(a, b)


class TestThresholdDryRunDoesNotNeedCheckpoints(unittest.TestCase):
    """Dry-run must not require checkpoint or regression baseline file."""

    def test_dry_run_produces_plan_and_configs(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_dryrun"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=10,
            sampling_method="random",
            seed=0,
            dry_run=True,
            allow_random_policy=False,
        )
        self.assertTrue((output_dir / "threshold_search_plan.json").exists())
        self.assertTrue((output_dir / "threshold_configs.csv").exists())
        self.assertTrue(result["dry_run"])


class TestThresholdAcceptanceCriteria(unittest.TestCase):
    """Acceptance criteria constants must be sensible."""

    def test_candidate_threshold_is_95_percent(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertEqual(mod.ACCEPTANCE["candidate_min_success_rate"], 0.95)
        self.assertEqual(mod.ACCEPTANCE["regression_max_degradation_pp"], 5.0)
        self.assertEqual(mod.ACCEPTANCE["negative_control_max_false_activation_rate"], 0.05)


class TestRegressionBaselineRequiredForSearch(unittest.TestCase):
    """Real run must reject if regression baseline file is missing."""

    def test_missing_baseline_warns_and_proceeds(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_missing_baseline"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        # After 6H.0-R: missing baseline is a warning, not an error
        result = mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=2,
            sampling_method="random",
            seed=0,
            dry_run=False,
            episodes_per_point=1,
            eval_seeds=[0],
        )
        self.assertTrue(output_dir.exists())
        self.assertIn("regression_baseline_missing", result)
        self.assertTrue(result["regression_baseline_missing"])


class TestPaperSafeThresholdClaimScoped(unittest.TestCase):
    """README must not contain universal threshold claims."""

    def test_no_universal_sufficient_language(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("sufficient for all", readme.lower())
        self.assertNotIn("universally sufficient", readme.lower())
        # Should contain scoped language
        self.assertIn("not robust across near-aspect", readme.lower())


class TestConftestXfailRegistryEmpty(unittest.TestCase):
    """conftest.py must have empty PREEXISTING_FAILURES."""

    def test_preexisting_failures_is_empty(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("conftest", str(PROJECT_ROOT / "tests" / "conftest.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.PREEXISTING_FAILURES, {}, "PREEXISTING_FAILURES must be empty after 6G.5D-R cleanup")


class TestReadmeStageStatusUpdated(unittest.TestCase):
    """README stage table must reflect 6G.5D-R complete and 6H.0-lite ready."""

    def test_stage_table_has_correct_status(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("6G.5D-R | ✅ Complete", readme)
        self.assertIn("6H.0-lite | 🧪 Ready / Preflight", readme)
        self.assertIn("6H (full) | ⏳ Gated", readme)


class TestThresholdRunnerHelpSmoke(unittest.TestCase):
    """Threshold search runner must expose --help."""

    def test_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"
        self.assertTrue(script_path.exists())
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"--help failed: {result.stderr}")
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("--sample-size", result.stdout)
        self.assertIn("--sampling-method", result.stdout)


class TestBaselineSearchHelpSmoke(unittest.TestCase):
    """Regression baseline search runner must expose --help."""

    def test_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "find_stage6h0_regression_baseline.py"
        self.assertTrue(script_path.exists())
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"--help failed: {result.stderr}")
        self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
