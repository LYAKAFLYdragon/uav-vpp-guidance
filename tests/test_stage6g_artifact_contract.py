import subprocess
import sys
from pathlib import Path

import pytest


_REQUIRED_CKPT = Path("outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt")


class TestStage6GArtifactContract:
    """Test that Stage 6G probe generates required artifacts and adheres to format contracts."""

    def test_dry_run_writes_resolved_config_and_manifest(self, tmp_path):
        output_dir = tmp_path / "probe_out"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage6g_guidance_limitation_probe.py",
                "--dry-run",
                "--output-dir", str(output_dir),
                "--guidance-modes", "los_rate",
                "--scenarios", "favorable",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "run_manifest.json").exists()
        with open(run_dir / "run_manifest.json", "r", encoding="utf-8") as f:
            import json
            manifest = json.load(f)
        assert manifest["run_status"] == "dry_run_completed"

    def test_smoke_run_writes_all_required_artifacts(self, tmp_path):
        if not _REQUIRED_CKPT.exists():
            pytest.skip(f"Checkpoint not found: {_REQUIRED_CKPT}")
        output_dir = tmp_path / "probe_out"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage6g_guidance_limitation_probe.py",
                "--smoke",
                "--output-dir", str(output_dir),
                "--guidance-modes", "los_rate",
                "--scenarios", "favorable",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        required = [
            "resolved_config.yaml",
            "run_manifest.json",
            "raw_episodes.csv",
            "scenario_method_summary.csv",
            "pairwise_mcnemar.csv",
            "paper_safe_claims.md",
            "README_result_block.md",
            "run.log",
        ]
        for artifact in required:
            assert (run_dir / artifact).exists(), f"Missing artifact: {artifact}"

    def test_raw_episodes_has_required_columns(self, tmp_path):
        if not _REQUIRED_CKPT.exists():
            pytest.skip(f"Checkpoint not found: {_REQUIRED_CKPT}")
        import csv
        output_dir = tmp_path / "probe_out"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage6g_guidance_limitation_probe.py",
                "--smoke",
                "--output-dir", str(output_dir),
                "--guidance-modes", "los_rate",
                "--scenarios", "favorable",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        run_dir = run_dirs[0]
        with open(run_dir / "raw_episodes.csv", "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        required_cols = [
            "scenario", "method", "guidance_mode_requested", "effective_guidance_mode",
            "training_seed", "evaluation_seed", "episode_index", "success",
            "termination_reason", "capture_time", "miss_distance", "min_range",
            "oob", "crash", "fallback_used", "prediction_error",
        ]
        for col in required_cols:
            assert col in headers, f"Missing column in raw_episodes.csv: {col}"

    def test_scenario_summary_can_aggregate_from_raw(self, tmp_path):
        if not _REQUIRED_CKPT.exists():
            pytest.skip(f"Checkpoint not found: {_REQUIRED_CKPT}")
        output_dir = tmp_path / "probe_out"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage6g_guidance_limitation_probe.py",
                "--smoke",
                "--output-dir", str(output_dir),
                "--guidance-modes", "los_rate",
                "--scenarios", "favorable",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        run_dir = run_dirs[0]
        assert (run_dir / "raw_episodes.csv").exists()
        assert (run_dir / "scenario_method_summary.csv").exists()
