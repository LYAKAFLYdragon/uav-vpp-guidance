"""Stage 6H.0-lite: Threshold optimization preflight contract tests.

Stage 6H.0-F.3: Registry migration tests.
"""

import csv
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_THRESHOLD_CKPT = PROJECT_ROOT / "outputs" / "experiments" / "no_prediction_vpp_ppo_seed0" / "checkpoints" / "best.pt"


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
        self.assertEqual(total, 5760, f"Expected 5760 configs (6H.1 redesign with crossing_aspect_threshold_deg=0.0), got {total}")


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
        # Verify sets are from ScenarioRegistry
        self.assertIn("candidate_set", result)
        self.assertIn("negative_set", result)
        self.assertTrue(len(result["candidate_scenarios"]) > 0)
        self.assertTrue(len(result["negative_scenarios"]) > 0)


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


class TestRegistryOnlyScenarioSource(unittest.TestCase):
    """Threshold search must use ScenarioRegistry exclusively."""

    def test_no_legacy_builder_imported(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        source = Path(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py").read_text(encoding="utf-8")
        self.assertNotIn("build_geometry_scenario", source,
                         "Legacy build_geometry_scenario must not be used")
        self.assertIn("ScenarioRegistry", source,
                      "ScenarioRegistry must be used")
        self.assertIn("_scenario_from_registry", source,
                      "Registry helper must exist")

    def test_candidate_set_from_registry(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_registry_candidates"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        result = mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=2,
            sampling_method="random",
            seed=0,
            candidate_set="candidate_search",
            negative_set="negative_control",
            dry_run=True,
        )
        self.assertEqual(result["candidate_set"], "candidate_search")
        self.assertEqual(result["negative_set"], "negative_control")
        self.assertTrue(len(result["candidate_scenarios"]) > 0)


class TestRegressionBaselineRequiredForSearch(unittest.TestCase):
    """Real run must reject if regression baseline file is missing in formal mode."""

    def test_exploratory_mode_runs_without_baseline(self):
        if not _THRESHOLD_CKPT.exists():
            self.skipTest(f"Checkpoint not found: {_THRESHOLD_CKPT}")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_exploratory"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        result = mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=2,
            sampling_method="random",
            seed=0,
            dry_run=False,
            episodes_per_point=1,
            eval_seeds=[0],
            mode="exploratory",
        )
        self.assertTrue(output_dir.exists())
        self.assertEqual(result["mode"], "exploratory")
        self.assertTrue(result["regression_baseline_missing"])

    def test_formal_mode_requires_baseline(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_formal"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        with self.assertRaises(RuntimeError) as ctx:
            mod.run_threshold_search(
                output_dir=str(output_dir),
                sample_size=2,
                sampling_method="random",
                seed=0,
                dry_run=False,
                episodes_per_point=1,
                eval_seeds=[0],
                mode="formal",
            )
        self.assertIn("Regression baseline file is required", str(ctx.exception))

    def test_formal_mode_with_baseline_file_runs(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        baseline_file = PROJECT_ROOT / "outputs" / "stage6h0f2_formal_baseline" / "regression_baseline.csv"
        if not baseline_file.exists():
            self.skipTest("Baseline file not yet generated")

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_formal_with_baseline"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        result = mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=2,
            sampling_method="random",
            seed=0,
            dry_run=False,
            episodes_per_point=1,
            eval_seeds=[0],
            mode="formal",
            regression_baseline_file=str(baseline_file),
        )
        self.assertTrue(output_dir.exists())
        self.assertEqual(result["mode"], "formal")
        self.assertFalse(result["regression_baseline_missing"])


class TestRawEpisodesSchema(unittest.TestCase):
    """raw_episodes.csv must include required telemetry fields."""

    def test_csv_has_geometry_family_and_guidance_mode(self):
        if not _THRESHOLD_CKPT.exists():
            self.skipTest(f"Checkpoint not found: {_THRESHOLD_CKPT}")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_stage6h0_lite_threshold_search",
            str(PROJECT_ROOT / "scripts" / "run_stage6h0_lite_threshold_search.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        output_dir = PROJECT_ROOT / "outputs" / "test_stage6h0_schema"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        mod.run_threshold_search(
            output_dir=str(output_dir),
            sample_size=2,
            sampling_method="random",
            seed=0,
            dry_run=False,
            episodes_per_point=1,
            eval_seeds=[0],
            mode="exploratory",
        )

        csv_path = output_dir / "raw_episodes.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            self.assertIn("scenario_id", fields)
            self.assertIn("geometry_family", fields)
            self.assertIn("scenario_type", fields)
            self.assertIn("effective_guidance_mode", fields)
            self.assertIn("mode_switch_effective", fields)


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
    """README stage table must reflect current status."""

    def test_stage_table_has_correct_status(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("6G.5D-R | ✅ Complete", readme)
        self.assertIn("6H.0-lite | 🧪 Ready / Preflight", readme)
        self.assertIn("6H.1 | ✅ Complete", readme)
        self.assertIn("6I | ✅ Complete", readme)
        self.assertIn("8B | ✅ Complete", readme)
        self.assertIn("8C | ✅ Complete", readme)
        self.assertIn("9A | ✅ Complete", readme)


class TestThresholdRunnerHelpSmoke(unittest.TestCase):
    """Threshold search runner must expose --help with new CLI args."""

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
        self.assertIn("--candidate-set", result.stdout)
        self.assertIn("--negative-set", result.stdout)
        self.assertIn("--mode", result.stdout)


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
