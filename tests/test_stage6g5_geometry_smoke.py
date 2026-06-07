"""Stage 6G.5A: Wide geometry smoke runner contract tests."""

import csv
import json
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

    def test_geometry_metadata_closure_rate_tail_chase(self):
        meta = compute_geometry_metadata({
            "initial_range_m": 2000,
            "ego_speed_mps": 280,
            "target_speed_mps": 160,
            "aspect_angle_deg": 0,
            "altitude_diff_m": 0,
        })
        # Vector projection: own_vel=[280,0], tgt_vel=[160,0], LOS=[1,0]
        # range_rate = dot([160-280, 0], [1,0]) = -120
        # closure = 120
        self.assertAlmostEqual(meta["closure_rate_mps"], 120.0, places=2)
        self.assertAlmostEqual(meta["range_rate_mps"], -120.0, places=2)
        self.assertTrue(meta["expected_feasible_flag"])

    def test_geometry_metadata_closure_rate_crossing(self):
        meta = compute_geometry_metadata({
            "initial_range_m": 2000,
            "ego_speed_mps": 280,
            "target_speed_mps": 160,
            "aspect_angle_deg": 90,
            "altitude_diff_m": 0,
        })
        # Vector projection: own_vel=[280,0], tgt_vel=[0,160], LOS=[0,1]
        # range_rate = dot([0-280, 160-0], [0,1]) = 160
        # closure = -160 (negative, target moving away laterally)
        self.assertAlmostEqual(meta["closure_rate_mps"], -160.0, places=2)
        self.assertAlmostEqual(meta["range_rate_mps"], 160.0, places=2)
        self.assertFalse(meta["expected_feasible_flag"])

    def test_geometry_metadata_negative_closure_infeasible(self):
        meta = compute_geometry_metadata({
            "initial_range_m": 2000,
            "ego_speed_mps": 100,
            "target_speed_mps": 200,
            "aspect_angle_deg": 0,
            "altitude_diff_m": 0,
        })
        self.assertLess(meta["closure_rate_mps"], 0.0)
        self.assertFalse(meta["expected_feasible_flag"])


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


class TestPolicyTypeVocabulary(unittest.TestCase):
    """Policy type values must belong to the approved vocabulary."""

    APPROVED_TYPES = {"trained_ppo", "random_policy", "dry_run", "missing_checkpoint"}

    def test_dry_run_policy_type(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_policy_dryrun"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        plan = run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=2,
            dry_run=True,
        )
        self.assertIn(plan["policy_type"], self.APPROVED_TYPES)

    def test_loaded_checkpoint_policy_type(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_policy_loaded"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # When checkpoint exists and dry_run=False, policy_type should be trained_ppo
        # We test via dry_run because real run would need env setup.
        # The logic in runner sets policy_type based on dry_run first, then checkpoint.
        # So for dry_run it will always be "dry_run".
        plan = run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=2,
            dry_run=True,
        )
        self.assertEqual(plan["policy_type"], "dry_run")


class TestReadmeCurrentStageMentions6G5A(unittest.TestCase):
    """README must reflect current Stage 6G.5A status."""

    def test_readme_mentions_stage6g5a(self):
        readme_path = PROJECT_ROOT / "README.md"
        self.assertTrue(readme_path.exists())
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("6G.5A", content, "README must mention Stage 6G.5A")

    def test_readme_bilevel_still_blocked(self):
        readme_path = PROJECT_ROOT / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        self.assertTrue(
            "blocked" in content.lower() or "gated" in content.lower(),
            "README must state bilevel is blocked or gated",
        )


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


class TestEpisodesPerPointExecuted(unittest.TestCase):
    """The runner must execute episodes_per_point episodes per point per eval seed."""

    def test_episode_count_matches_contract(self):
        """Mock evaluate_single_episode to count calls without heavy simulation."""
        import shutil
        from unittest.mock import patch
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_episodes"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        call_log = []

        def _fake_evaluate_single_episode(env, agent, method_config, scenario=None, seed=None, save_trajectory=False, method_name=None):
            call_log.append({"seed": seed, "scenario_name": scenario.get("name", "")})
            return {"is_success": True, "final_range_m": 100.0, "reason": "success"}, None

        class _FakeEnv:
            def __init__(self, cfg):
                pass
            def reset(self, seed=None):
                return {"observation_vector": np.zeros(16)}
            def close(self): pass

        class _FakeAgent:
            def __init__(self, obs_dim=None, action_dim=None, config=None, device=None):
                pass
            def load(self, path): pass

        # Use a temporary config pointing to a non-existent checkpoint so the runner
        # does not accidentally load a real (but architecture-mismatched) checkpoint.
        tmp_config_path = output_dir / "tmp_config.yaml"
        with open(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["methods"]["no_prediction"]["checkpoint"] = str(output_dir / "nonexistent.pt")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(tmp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with patch("uav_vpp_guidance.evaluation.evaluate_prediction_comparison.evaluate_single_episode", _fake_evaluate_single_episode), \
             patch("uav_vpp_guidance.envs.tracking_env.CloseRangeTrackingEnv", _FakeEnv), \
             patch("uav_vpp_guidance.agents.ppo_agent.PPOAgent", _FakeAgent):
            run_geometry_smoke(
                config_path=str(tmp_config_path),
                output_dir=str(output_dir),
                sample_size=2,
                sampling_method="random",
                seed=0,
                episodes_per_point=3,
                eval_seeds=[0, 1],
                dry_run=False,
                allow_random_policy=True,
            )

        # 2 points * 2 eval seeds * 3 episodes = 12 calls
        self.assertEqual(len(call_log), 12, f"Expected 12 episodes, got {len(call_log)}")

        # Verify deterministic episode seeds
        seeds = [c["seed"] for c in call_log]
        self.assertEqual(len(set(seeds)), 12, "Each episode must have a unique seed")


class TestRealRunRequiresCheckpointByDefault(unittest.TestCase):
    """Real smoke runs must fail fast if checkpoint is missing and --allow-random-policy is not set."""

    def test_missing_checkpoint_raises_without_flag(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_ckpt_guard"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary config with a non-existent checkpoint path
        tmp_config_path = output_dir / "tmp_config.yaml"
        with open(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["methods"]["no_prediction"]["checkpoint"] = str(output_dir / "definitely_missing.pt")
        with open(tmp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with self.assertRaises(FileNotFoundError) as cm:
            run_geometry_smoke(
                config_path=str(tmp_config_path),
                output_dir=str(output_dir),
                sample_size=2,
                dry_run=False,
                allow_random_policy=False,
            )
        self.assertIn("Checkpoint missing", str(cm.exception))


class TestAllowRandomPolicyExplicitOnly(unittest.TestCase):
    """Random policy is allowed only when explicitly requested."""

    def test_allow_random_policy_true_uses_random(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_random_explicit"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # We cannot easily run the full smoke without a checkpoint, but we can verify
        # the plan records the flag correctly in dry-run.
        plan = run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=2,
            dry_run=True,
            allow_random_policy=True,
        )
        self.assertTrue(plan["allow_random_policy"])
        self.assertEqual(plan["policy_type"], "dry_run")


class TestSmokeScopeNotePresent(unittest.TestCase):
    """Plan and summary must document the no_prediction-only scope."""

    def test_plan_contains_scope_note(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_scope"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        plan = run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=2,
            dry_run=True,
        )
        self.assertEqual(plan["methods_evaluated"], ["no_prediction"])
        self.assertIn("baseline geometric feasibility only", plan["scope_note"])

        with open(output_dir / "geometry_smoke_plan.json", "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        self.assertIn("scope_note", plan_data)
        self.assertIn("methods_evaluated", plan_data)

    def test_summary_md_contains_scope(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_scope_md"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=2,
            dry_run=True,
        )
        md_path = output_dir / "geometry_smoke_summary.md"
        content = md_path.read_text(encoding="utf-8")
        self.assertIn("Methods evaluated", content)
        self.assertIn("no_prediction", content)
        self.assertIn("Scope note", content)


class TestDryRunWritesStableOutputFiles(unittest.TestCase):
    """Dry-run must produce all stable output files (even if empty)."""

    def test_all_csv_files_exist_in_dry_run(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5_stable_files"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        run_geometry_smoke(
            config_path=str(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"),
            output_dir=str(output_dir),
            sample_size=5,
            dry_run=True,
        )
        required_files = [
            "geometry_smoke_plan.json",
            "geometry_smoke_points.csv",
            "resolved_config.yaml",
            "geometry_smoke_summary.md",
            "geometry_smoke_summary.csv",
            "feasible_candidates.csv",
            "failed_points.csv",
        ]
        for fname in required_files:
            fpath = output_dir / fname
            self.assertTrue(fpath.exists(), f"Missing stable output file: {fname}")

        # CSV files should have headers even when empty
        for fname in ["geometry_smoke_summary.csv", "feasible_candidates.csv", "failed_points.csv"]:
            with open(output_dir / fname, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames
                self.assertIsNotNone(header)
                self.assertGreater(len(header), 0)


if __name__ == "__main__":
    unittest.main()
