"""Stage 6G.5A: Wide geometry smoke runner contract tests."""

import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure src on path for importing helpers
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from uav_vpp_guidance.utils.geometry_scenario import (
    build_geometry_scenario,
    compute_geometry_metadata,
    build_full_grid,
    sample_grid,
)

# scripts/ is not a package; load runner dynamically
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "run_stage6g5_geometry_smoke",
    str(PROJECT_ROOT / "scripts" / "run_stage6g5_geometry_smoke.py"),
)
_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner_mod)
run_geometry_smoke = _runner_mod.run_geometry_smoke


class TestGeometryGridSize(unittest.TestCase):
    """The full parameter grid must have 324 combinations."""

    def test_grid_size(self):
        grid = {
            "initial_range_m": [1200, 2000, 3200],
            "ego_speed_mps": [220, 280, 340],
            "target_speed_mps": [120, 160, 200],
            "aspect_angle_deg": [0, 30, 60, 90],
            "altitude_diff_m": [-500, 0, 500],
        }
        points = build_full_grid(grid)
        self.assertEqual(len(points), 324, "3×3×3×4×3 grid must yield 324 combos")


class TestGeometrySamplingDeterministic(unittest.TestCase):
    """Same seed must yield identical samples."""

    def test_random_sampling_deterministic(self):
        grid = {
            "initial_range_m": [1200, 2000, 3200],
            "ego_speed_mps": [220, 280, 340],
            "target_speed_mps": [120, 160, 200],
            "aspect_angle_deg": [0, 30, 60, 90],
            "altitude_diff_m": [-500, 0, 500],
        }
        s1 = sample_grid(grid, 40, "random", seed=42)
        s2 = sample_grid(grid, 40, "random", seed=42)
        self.assertEqual(s1, s2)

    def test_latin_hypercube_sampling_deterministic(self):
        grid = {
            "initial_range_m": [1200, 2000, 3200],
            "ego_speed_mps": [220, 280, 340],
            "target_speed_mps": [120, 160, 200],
            "aspect_angle_deg": [0, 30, 60, 90],
            "altitude_diff_m": [-500, 0, 500],
        }
        s1 = sample_grid(grid, 40, "latin_hypercube", seed=42)
        s2 = sample_grid(grid, 40, "latin_hypercube", seed=42)
        self.assertEqual(s1, s2)


class TestAspectAngleScenarioBuilder(unittest.TestCase):
    """Scenario builder must produce correct tail-chase, crossing, and altitude offsets."""

    def test_aspect_zero_is_tail_chase(self):
        scen = build_geometry_scenario(
            initial_range_m=2000,
            ego_speed_mps=280,
            target_speed_mps=160,
            aspect_angle_deg=0,
            altitude_diff_m=0,
        )
        own = scen["own_init"]
        tgt = scen["target_init"]
        self.assertEqual(own["heading_deg"], 0.0)
        self.assertEqual(tgt["heading_deg"], 0.0)
        self.assertAlmostEqual(tgt["position_m"][0], 2000.0, places=3)
        self.assertAlmostEqual(tgt["position_m"][1], 0.0, places=3)
        self.assertEqual(tgt["position_m"][2], 5000.0)

    def test_aspect_ninety_is_crossing(self):
        scen = build_geometry_scenario(
            initial_range_m=2000,
            ego_speed_mps=280,
            target_speed_mps=160,
            aspect_angle_deg=90,
            altitude_diff_m=0,
        )
        tgt = scen["target_init"]
        self.assertEqual(tgt["heading_deg"], 90.0)
        self.assertAlmostEqual(tgt["position_m"][0], 0.0, places=3)
        self.assertAlmostEqual(tgt["position_m"][1], 2000.0, places=3)

    def test_altitude_diff_applied(self):
        scen = build_geometry_scenario(
            initial_range_m=2000,
            ego_speed_mps=280,
            target_speed_mps=160,
            aspect_angle_deg=0,
            altitude_diff_m=-300,
        )
        own = scen["own_init"]
        tgt = scen["target_init"]
        self.assertEqual(own["position_m"][2], 5000.0)
        self.assertEqual(tgt["position_m"][2], 4700.0)

    def test_geometry_metadata_closure_rate(self):
        meta = compute_geometry_metadata({
            "initial_range_m": 2000,
            "ego_speed_mps": 280,
            "target_speed_mps": 160,
            "aspect_angle_deg": 0,
            "altitude_diff_m": 0,
        })
        self.assertAlmostEqual(meta["closure_rate_mps"], 120.0, places=2)
        self.assertTrue(meta["expected_feasible_flag"])


class TestDryRunDoesNotRequireCheckpoints(unittest.TestCase):
    """Dry-run must succeed without any checkpoint files."""

    def test_dry_run_produces_plan_and_points(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_dryrun"
        # Clean previous test artifacts
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

        plan = run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=40,
            sampling_method="random",
            seed=0,
            episodes_per_point=3,
            eval_seeds=[0],
            dry_run=True,
        )
        self.assertTrue((output_dir / "geometry_smoke_plan.json").exists())
        self.assertTrue((output_dir / "geometry_smoke_points.csv").exists())
        self.assertTrue((output_dir / "resolved_config.yaml").exists())
        self.assertTrue((output_dir / "geometry_smoke_summary.md").exists())
        self.assertEqual(plan["sampled_points_count"], 40)

        # Verify plan content
        with open(output_dir / "geometry_smoke_plan.json", "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        self.assertEqual(plan_data["dry_run"], True)
        self.assertEqual(plan_data["total_grid_size"], 324)

        # Verify points CSV has 40 rows
        with open(output_dir / "geometry_smoke_points.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 40)

        # Verify columns exist
        expected_cols = {
            "initial_range_m", "ego_speed_mps", "target_speed_mps",
            "aspect_angle_deg", "altitude_diff_m",
            "closure_rate_mps", "estimated_time_to_capture_s", "expected_feasible_flag",
        }
        self.assertTrue(expected_cols.issubset(set(rows[0].keys())))


class TestResolvedConfigSaved(unittest.TestCase):
    """Resolved config must record effective parameters."""

    def test_resolved_config_contains_smoke_params(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_resolved"
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

        run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=25,
            sampling_method="latin_hypercube",
            seed=7,
            episodes_per_point=2,
            eval_seeds=[0, 1],
            dry_run=True,
        )
        resolved_path = output_dir / "resolved_config.yaml"
        self.assertTrue(resolved_path.exists())
        with open(resolved_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["experiment"]["sample_size"], 25)
        self.assertEqual(cfg["experiment"]["sampling_method"], "latin_hypercube")
        self.assertEqual(cfg["experiment"]["seed"], 7)
        self.assertEqual(cfg["experiment"]["episodes_per_point"], 2)
        self.assertEqual(cfg["experiment"]["eval_seeds"], [0, 1])


class TestGeometryMetadataColumns(unittest.TestCase):
    """Points CSV must include derived metadata columns."""

    def test_columns_present(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_metadata"
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

        run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=10,
            dry_run=True,
        )
        with open(output_dir / "geometry_smoke_points.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        self.assertIn("closure_rate_mps", row)
        self.assertIn("estimated_time_to_capture_s", row)
        self.assertIn("expected_feasible_flag", row)


class TestBilevelStillBlockedUnlessSuccess(unittest.TestCase):
    """Markdown summary must declare bilevel blocked until feasible candidate found."""

    def test_summary_contains_bilevel_status(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_bilevel"
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

        run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=10,
            dry_run=True,
        )
        md_path = output_dir / "geometry_smoke_summary.md"
        self.assertTrue(md_path.exists())
        content = md_path.read_text(encoding="utf-8")
        self.assertIn("Bilevel unblocked candidate", content)
        # In dry-run with no real episodes, successes list is empty → should be false
        self.assertIn("false", content.lower())


class TestCIPullRequestIncludesFeatureBranch(unittest.TestCase):
    """CI must trigger on feature branch pull requests."""

    def test_pr_branches_include_feature(self):
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
        self.assertTrue(ci_path.exists())
        content = ci_path.read_text(encoding="utf-8")
        has_wildcard = '"feature/**"' in content
        has_exact = "feature/los-guidance-deep-hardening" in content
        self.assertTrue(
            has_wildcard or has_exact,
            "CI pull_request branches must include feature branches",
        )


class TestReadmeBilevelBlockedNotice(unittest.TestCase):
    """README must contain the bilevel gate notice."""

    def test_readme_mentions_bilevel_gated(self):
        readme_path = PROJECT_ROOT / "README.md"
        self.assertTrue(readme_path.exists(), "README.md missing")
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("bilevel", content.lower(), "README must mention bilevel")
        has_blocked = "blocked" in content.lower()
        has_gated = "gated" in content.lower()
        self.assertTrue(
            has_blocked or has_gated,
            "README must state that bilevel is blocked or gated",
        )


class TestScriptHelpSmoke(unittest.TestCase):
    """Stage 6G.5 script must expose --help."""

    def test_geometry_smoke_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "run_stage6g5_geometry_smoke.py"
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
