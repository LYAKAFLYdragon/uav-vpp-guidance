"""Tests for train_fixed_gain.py training entry."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestTrainFixedGainDryRun:
    def test_dry_run_exits_zero(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m", "uav_vpp_guidance.training.train_fixed_gain",
                "--config", "config/experiment/fixed_gain_vpp.yaml",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"dry-run failed: {result.stderr}"
        assert "[DRY-RUN]" in result.stdout
        assert "Fixed gains:" in result.stdout

    def test_dry_run_shows_gains(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m", "uav_vpp_guidance.training.train_fixed_gain",
                "--config", "config/experiment/fixed_gain_vpp.yaml",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "k_los" in result.stdout


class TestTrainFixedGainSmoke:
    def test_smoke_completes_quickly(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m", "uav_vpp_guidance.training.train_fixed_gain",
                "--config", "config/experiment/fixed_gain_vpp.yaml",
                "--smoke",
                "--output-dir", "outputs/test_fixed_gain_smoke",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"smoke failed: {result.stderr}"
        assert (PROJECT_ROOT / "outputs" / "test_fixed_gain_smoke" / "checkpoints").exists() or (
            PROJECT_ROOT / "outputs" / "test_fixed_gain_smoke" / "logs"
        ).exists()

    def test_smoke_creates_checkpoint_dir(self):
        output_dir = PROJECT_ROOT / "outputs" / "test_fixed_gain_smoke2"
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)

        result = subprocess.run(
            [
                sys.executable,
                "-m", "uav_vpp_guidance.training.train_fixed_gain",
                "--config", "config/experiment/fixed_gain_vpp.yaml",
                "--smoke",
                "--output-dir", str(output_dir),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"smoke failed: {result.stderr}"
        assert output_dir.exists()
        assert (output_dir / "config_snapshot.yaml").exists()


class TestConsoleEntryPoint:
    def test_entry_point_exists(self):
        result = subprocess.run(
            [sys.executable, "-c", "from uav_vpp_guidance.training.train_fixed_gain import main; print('OK')"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
