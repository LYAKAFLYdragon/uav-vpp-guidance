"""Stage 6G.5D: PN mode-switch & VPP offset mechanism contract tests."""

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "run_stage6g5d_pn_mode_switch_probe",
    str(PROJECT_ROOT / "scripts" / "run_stage6g5d_pn_mode_switch_probe.py"),
)
_runner_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner_mod)
run_mode_switch_probe = _runner_mod.run_mode_switch_probe


class TestModeSwitchGateActivates(unittest.TestCase):
    """Mode-switch gate must activate for pt20/pt29/pt38 (tail-chase, high energy)."""

    def test_gate_active_for_tail_chase(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        # Tail-chase scenario
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            self.assertTrue(info.get("mode_switch_effective", False),
                            f"Gate should activate for tail-chase; reason={info.get('mode_switch_reason')}")
            self.assertEqual(info.get("effective_guidance_mode"), "proportional_navigation")
        finally:
            env.close()

    def test_gate_inactive_for_high_aspect(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        # High aspect scenario
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=90, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            self.assertFalse(info.get("mode_switch_effective", True),
                             "Gate should NOT activate for high aspect angle")
        finally:
            env.close()


class TestModeSwitchPnBypassesVPP(unittest.TestCase):
    """When mode-switch is active, VPP generator must be bypassed."""

    def test_vpp_not_called_when_switch_active(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        called = []
        original = env.virtual_point_generator.action_to_virtual_point

        def _patched(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        env.virtual_point_generator.action_to_virtual_point = _patched
        try:
            env.step(np.zeros(3))
            self.assertEqual(len(called), 0, "VPP generator should be bypassed when mode-switch is active")
        finally:
            env.close()


class TestModeSwitchLatch(unittest.TestCase):
    """Once mode-switch activates, it must stay latched for the episode."""

    def test_latch_persists_after_gate_deactivates(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        # Tail-chase scenario where gate activates on step 1
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            # Step 1: gate should be active
            obs, reward, terminated, truncated, info1 = env.step(np.zeros(3))
            self.assertTrue(info1.get("mode_switch_effective", False),
                            "Gate should activate on step 1")
            # Step 2: even if pre-step geometry would de-activate gate,
            # latch should keep mode-switch effective
            obs, reward, terminated, truncated, info2 = env.step(np.zeros(3))
            self.assertTrue(info2.get("mode_switch_effective", False),
                            "Latch should keep mode-switch active on step 2")
            self.assertEqual(info2.get("effective_guidance_mode"), "proportional_navigation")
            self.assertEqual(info2.get("virtual_point_source"), "direct_track")
            self.assertIn(info2.get("mode_switch_reason"), ("gate_active", "latched"))
        finally:
            env.close()

    def test_latch_resets_on_new_episode(self):
        from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
        from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario

        config_path = PROJECT_ROOT / "config" / "experiment" / "stage6g5_wide_geometry_smoke.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f)

        cfg["guidance"]["mode_switch"] = {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        }
        env = CloseRangeTrackingEnv(cfg)
        scenario = build_geometry_scenario(
            initial_range_m=2000, ego_speed_mps=340, target_speed_mps=120,
            aspect_angle_deg=0, altitude_diff_m=0, base_altitude_m=5000.0,
        )
        env.reset(scenario=scenario, seed=0)
        try:
            env.step(np.zeros(3))  # activate latch
            self.assertTrue(env._mode_switch_latched)
            # Reset should clear latch
            env.reset(scenario=scenario, seed=1)
            self.assertFalse(env._mode_switch_latched)
        finally:
            env.close()


class TestDryRunArtifactsExist(unittest.TestCase):
    """Dry-run must produce all expected output files."""

    def test_all_artifacts_present(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5d_dryrun"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Geometry CSV not found")

        run_mode_switch_probe(
            candidate_points=[("pt20", None)],
            output_dir=str(output_dir),
            input_geometry_csv=str(csv_path),
            dry_run=True,
        )

        required = [
            "requested_config.yaml",
            "resolved_config.yaml",
            "effective_runtime_flags.json",
            "mode_switch_raw_episodes.csv",
            "mode_switch_variant_summary.csv",
            "mode_switch_seed_summary.csv",
            "mode_switch_success_matrix.csv",
            "README_result_block.md",
        ]
        for fname in required:
            self.assertTrue((output_dir / fname).exists(), f"Missing artifact: {fname}")


class TestReadmeResultBlockPaperSafe(unittest.TestCase):
    """README result block must use paper-safe wording."""

    def test_no_universal_claims(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5d_papersafe"
        if output_dir.exists():
            shutil.rmtree(output_dir)

        csv_path = PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"
        if not csv_path.exists():
            self.skipTest("Geometry CSV not found")

        run_mode_switch_probe(
            candidate_points=[("pt20", None)],
            output_dir=str(output_dir),
            input_geometry_csv=str(csv_path),
            dry_run=True,
        )

        content = (output_dir / "README_result_block.md").read_text(encoding="utf-8")
        self.assertNotIn("universally", content.lower())
        self.assertNotIn("impossible", content.lower())
        self.assertIn("paper-safe", content.lower())


class TestBilevelRemainsBlocked(unittest.TestCase):
    """No variant should claim bilevel is unblocked."""

    def test_runner_does_not_claim_bilevel_unblocked(self):
        import yaml
        output_dir = PROJECT_ROOT / "outputs" / "test_stage6g5d_bilevel_block"
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
            run_mode_switch_probe(
                candidate_points=[("pt20", None)],
                output_dir=str(output_dir / "run"),
                input_geometry_csv=str(PROJECT_ROOT / "outputs" / "stage6g5_geometry_smoke_real_seed0" / "geometry_smoke_points.csv"),
                config_path=str(config_path),
                episodes_per_point=1,
                eval_seeds=[0],
                dry_run=False,
                allow_random_policy=False,
            )


class TestScriptHelpSmoke(unittest.TestCase):
    """Stage 6G.5D script must expose --help."""

    def test_script_help(self):
        script_path = PROJECT_ROOT / "scripts" / "run_stage6g5d_pn_mode_switch_probe.py"
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
