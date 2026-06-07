"""Stage 6G.5C: Pure-PN candidate confirmation & VPP failure diagnosis contract tests."""

import csv
import importlib.util
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "run_stage6g5c_candidate_confirmation",
    str(PROJECT_ROOT / "scripts" / "run_stage6g5c_candidate_confirmation.py"),
)
_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner_mod)
run_candidate_confirmation = _runner_mod.run_candidate_confirmation
_load_geometry_points = _runner_mod._load_geometry_points
_select_candidate_points = _runner_mod._select_candidate_points


class TestCandidatePointsLoadedCorrectly(unittest.TestCase):
    """Candidate points pt20/pt29/pt38 must resolve to valid geometry rows."""

    def test_select_candidate_points_from_csv(self):
        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A geometry CSV not found")
        points = _load_geometry_points(str(csv_path))
        selected = _select_candidate_points(points, ["pt20", "pt29", "pt38"])
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0][0], "pt20")
        self.assertEqual(selected[1][0], "pt29")
        self.assertEqual(selected[2][0], "pt38")
        for _, pt in selected:
            self.assertIn("initial_range_m", pt)
            self.assertIn("ego_speed_mps", pt)

    def test_candidate_geometry_matches_summary(self):
        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A geometry CSV not found")
        points = _load_geometry_points(str(csv_path))
        selected = _select_candidate_points(points, ["pt20", "pt29", "pt38"])
        for cid, pt in selected:
            self.assertEqual(pt["initial_range_m"], 2000)
            self.assertEqual(pt["ego_speed_mps"], 340)
            self.assertEqual(pt["aspect_angle_deg"], 0)


class TestPurePnNoVppBypassesVPP(unittest.TestCase):
    """pure_pn_no_vpp must bypass VPP generator."""

    def test_vpp_generator_not_called(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["direct_track_mode"] = True
        cfg["guidance"]["mode"] = "proportional_navigation"
        env = CloseRangeTrackingEnv(cfg)
        env.reset(seed=0)
        called = []
        original = env.virtual_point_generator.action_to_virtual_point

        def _patched(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        env.virtual_point_generator.action_to_virtual_point = _patched
        try:
            env.step(np.zeros(3))
            self.assertEqual(len(called), 0, "VPP generator should not be called in pure_pn_no_vpp")
        finally:
            env.close()


class TestVppPolicyPnGuidanceUsesVPP(unittest.TestCase):
    """vpp_policy_pn_guidance must use VPP generator but with PN guidance."""

    def test_guidance_is_pn_and_vpp_called(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["direct_track_mode"] = False
        cfg["guidance"]["mode"] = "proportional_navigation"
        env = CloseRangeTrackingEnv(cfg)
        env.reset(seed=0)
        called = []
        original = env.virtual_point_generator.action_to_virtual_point

        def _patched(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        env.virtual_point_generator.action_to_virtual_point = _patched
        try:
            env.step(np.zeros(3))
            self.assertEqual(len(called), 1, "VPP generator should be called in vpp_policy_pn_guidance")
            self.assertEqual(env.guidance.__class__.__name__, "ProportionalNavigationGuidance")
        finally:
            env.close()


class TestHybridNoVppUsesHybridGuidance(unittest.TestCase):
    """hybrid_no_vpp must use direct_track_mode and hybrid guidance."""

    def test_hybrid_mode(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["direct_track_mode"] = True
        cfg["guidance"]["mode"] = "hybrid"
        env = CloseRangeTrackingEnv(cfg)
        env.reset(seed=0)
        try:
            self.assertEqual(env.guidance.__class__.__name__, "HybridGuidance")
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            self.assertTrue(info["direct_track_mode_effective"])
            self.assertEqual(info["virtual_point_source"], "direct_track")
        finally:
            env.close()


class TestCheckpointRequiredForVppVariants(unittest.TestCase):
    """VPP variants must raise FileNotFoundError when checkpoint is missing and no random flag."""

    def test_missing_checkpoint_raises(self):
        import yaml
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5c_ckpt_guard"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "tmp_config.yaml"
        with open(PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["methods"]["no_prediction"]["checkpoint"] = str(output_dir / "nonexistent.pt")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with self.assertRaises(FileNotFoundError):
            run_candidate_confirmation(
                candidate_points=[("pt20", None)],
                output_dir=str(output_dir / "run"),
                input_geometry_csv=str(PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"),
                config_path=str(config_path),
                episodes_per_point=1,
                eval_seeds=[0],
                dry_run=False,
                allow_random_policy=False,
            )


class TestDryRunWritesAllArtifacts(unittest.TestCase):
    """Dry-run must produce all stable output files."""

    def test_all_artifacts_present(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5c_dryrun_artifacts"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A geometry CSV not found")

        run_candidate_confirmation(
            candidate_points=[("pt20", None), ("pt29", None), ("pt38", None)],
            output_dir=str(output_dir),
            input_geometry_csv=str(csv_path),
            dry_run=True,
        )

        required = [
            "requested_config.yaml",
            "resolved_config.yaml",
            "effective_runtime_flags.json",
            "candidate_raw_episodes.csv",
            "candidate_variant_summary.csv",
            "candidate_seed_summary.csv",
            "candidate_success_matrix.csv",
            "trajectory_terminal_metrics.csv",
            "command_saturation_summary.csv",
            "vpp_vs_pn_failure_diagnosis.md",
        ]
        for fname in required:
            self.assertTrue((output_dir / fname).exists(), f"Missing artifact: {fname}")


class TestSuccessMatrixHasExpectedColumns(unittest.TestCase):
    """candidate_success_matrix.csv must contain required columns."""

    def test_columns(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5c_matrix_columns"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A geometry CSV not found")

        run_candidate_confirmation(
            candidate_points=[("pt20", None)],
            output_dir=str(output_dir),
            input_geometry_csv=str(csv_path),
            dry_run=True,
        )

        with open(output_dir / "candidate_success_matrix.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames

        expected = {"point_id", "variant", "seed_0_success", "seed_1_success", "seed_2_success", "cross_seed_stable"}
        self.assertTrue(expected.issubset(set(header)), f"Missing columns: {expected - set(header)}")


class TestDiagnosisMarkdownPaperSafeWording(unittest.TestCase):
    """vpp_vs_pn_failure_diagnosis.md must use paper-safe wording."""

    def test_no_universal_infeasibility_claim(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5c_papersafe"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Stage 6G.5A geometry CSV not found")

        run_candidate_confirmation(
            candidate_points=[("pt20", None)],
            output_dir=str(output_dir),
            input_geometry_csv=str(csv_path),
            dry_run=True,
        )

        content = (output_dir / "vpp_vs_pn_failure_diagnosis.md").read_text(encoding="utf-8")
        self.assertNotIn("universally infeasible", content.lower())
        self.assertNotIn("impossible", content.lower())


class TestScriptHelpSmoke(unittest.TestCase):
    """Stage 6G.5C script must expose --help."""

    def test_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "run_stage6g5c_candidate_confirmation.py"
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
