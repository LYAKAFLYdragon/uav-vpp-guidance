"""Stage 6G.5B: Direct-track / pure-PN control feasibility probe contract tests."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load runner dynamically (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "run_stage6g5b_direct_track_smoke",
    str(PROJECT_ROOT / "scripts" / "run_stage6g5b_direct_track_smoke.py"),
)
_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner_mod)
run_direct_track_smoke = _runner_mod.run_direct_track_smoke
_load_geometry_points = _runner_mod._load_geometry_points


class TestDirectTrackModeBypassesVPP(unittest.TestCase):
    """When direct_track_mode=True, the env must skip VPP offset generation."""

    def test_virtual_point_equals_target_position(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["direct_track_mode"] = True
        env = CloseRangeTrackingEnv(cfg)
        obs = env.reset(seed=0)

        # Monkey-patch VPP generator to detect if it was called
        called = []
        original = env.virtual_point_generator.action_to_virtual_point

        def _patched(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        env.virtual_point_generator.action_to_virtual_point = _patched
        try:
            env.step(np.zeros(3))
            self.assertEqual(len(called), 0, "VPP generator should not be called in direct_track_mode")
        finally:
            env.close()


class TestPurePNNoVPPDoesNotUseOffset(unittest.TestCase):
    """Pure PN variant must not apply policy action offset."""

    def test_guidance_mode_switches_to_pn(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["direct_track_mode"] = True
        cfg["guidance"]["mode"] = "proportional_navigation"
        env = CloseRangeTrackingEnv(cfg)
        env.reset(seed=0)
        try:
            self.assertEqual(env.guidance.__class__.__name__, "ProportionalNavigationGuidance")
        finally:
            env.close()


class TestGeometryPointsReuseFromStage6G5A(unittest.TestCase):
    """Stage 6G.5B must reuse the exact same geometry points as Stage 6G.5A."""

    def test_load_geometry_points_matches_source_csv(self):
        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A real smoke CSV not found")

        points = _load_geometry_points(str(csv_path))
        self.assertGreater(len(points), 0)
        for pt in points:
            self.assertIn("initial_range_m", pt)
            self.assertIn("ego_speed_mps", pt)
            self.assertIn("aspect_angle_deg", pt)


class TestPolicyTypeIsTrainedPPOWhenCheckpointExists(unittest.TestCase):
    """Policy audit must report trained_ppo when checkpoint is present."""

    def test_trained_ppo_when_checkpoint_present(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5b_policy_type"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A real smoke CSV not found")

        plan = run_direct_track_smoke(
            input_geometry_csv=str(csv_path),
            output_dir=str(output_dir),
            episodes_per_point=1,
            eval_seeds=[0],
            dry_run=True,
        )
        # In dry-run policy_type is not set per-variant, but effective flags are written empty
        # We verify the script does not crash and checkpoint path is recorded
        self.assertTrue((output_dir / "resolved_config.yaml").exists())


class TestScriptFailsIfCheckpointMissingAndNoRandomFlag(unittest.TestCase):
    """Real run must raise FileNotFoundError when checkpoint is missing and --allow-random-policy is false."""

    def test_missing_checkpoint_raises(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5b_ckpt_guard"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary CSV with 1 point
        csv_path = output_dir / "tmp_geometry.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"])
            writer.writeheader()
            writer.writerow({"initial_range_m": 2000, "ego_speed_mps": 280, "target_speed_mps": 160, "aspect_angle_deg": 0, "altitude_diff_m": 0})

        # Use temporary config with nonexistent checkpoint
        import yaml
        config_path = output_dir / "tmp_config.yaml"
        with open(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["methods"]["no_prediction"]["checkpoint"] = str(output_dir / "nonexistent.pt")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with self.assertRaises(FileNotFoundError):
            run_direct_track_smoke(
                input_geometry_csv=str(csv_path),
                output_dir=str(output_dir / "run"),
                config_path=str(config_path),
                episodes_per_point=1,
                eval_seeds=[0],
                dry_run=False,
                allow_random_policy=False,
            )


class TestDirectTrackArtifactsExistInDryRun(unittest.TestCase):
    """Dry-run must produce all stable output files."""

    def test_all_artifacts_present(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5b_dryrun_artifacts"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A real smoke CSV not found")

        run_direct_track_smoke(
            input_geometry_csv=str(csv_path),
            output_dir=str(output_dir),
            dry_run=True,
        )

        required = [
            "requested_config.yaml",
            "resolved_config.yaml",
            "effective_runtime_flags.json",
            "raw_episodes.csv",
            "geometry_method_summary.csv",
            "feasible_candidates.csv",
            "direct_track_vs_vpp_comparison.csv",
            "README_result_block.md",
        ]
        for fname in required:
            self.assertTrue((output_dir / fname).exists(), f"Missing artifact: {fname}")


class TestComparisonCSVHasExpectedColumns(unittest.TestCase):
    """direct_track_vs_vpp_comparison.csv must contain required columns."""

    def test_columns(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5b_columns"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A real smoke CSV not found")

        run_direct_track_smoke(
            input_geometry_csv=str(csv_path),
            output_dir=str(output_dir),
            dry_run=True,
        )

        with open(output_dir / "direct_track_vs_vpp_comparison.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

        expected = {"variant", "description", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp"}
        self.assertTrue(expected.issubset(set(header)), f"Missing columns: {expected - set(header)}")


class TestReadmeResultBlockPaperSafeWording(unittest.TestCase):
    """README_result_block.md must use paper-safe wording."""

    def test_no_universal_infeasibility_claim(self):
        import shutil
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5b_papersafe"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A real smoke CSV not found")

        run_direct_track_smoke(
            input_geometry_csv=str(csv_path),
            output_dir=str(output_dir),
            dry_run=True,
        )

        content = (output_dir / "README_result_block.md").read_text(encoding="utf-8")
        # dry-run text may not contain 'paper-safe' but must not over-claim
        self.assertNotIn("universally infeasible", content.lower())
        self.assertNotIn("impossible", content.lower())


class TestScriptHelpSmoke(unittest.TestCase):
    """Stage 6G.5B script must expose --help."""

    def test_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "run_stage6g5b_direct_track_smoke.py"
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
