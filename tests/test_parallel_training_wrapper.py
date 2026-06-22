"""Smoke tests for the parallel training wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = REPO_ROOT / "scripts" / "run_parallel_training.py"


def test_parallel_training_dry_run():
    """Dry-run should print commands and exit cleanly without writing outputs."""
    output_root = REPO_ROOT / "outputs" / "_test_parallel_training_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--controllers",
            "ppo_pid",
            "--seeds",
            "0",
            "1",
            "--backend",
            "simple",
            "--smoke",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "DRY-RUN:" in combined
    assert "uav_vpp_guidance.training.train_ppo_pid" in combined
    # No summary file should be written in dry-run mode.
    assert not (output_root / "summary.json").exists()


def test_parallel_training_smoke():
    """A real smoke run should produce checkpoints and a summary JSON."""
    output_root = REPO_ROOT / "outputs" / "_test_parallel_training_smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--controllers",
            "ppo_pid",
            "--seeds",
            "0",
            "--backend",
            "simple",
            "--smoke",
            "--output-root",
            str(output_root),
            "--jobs",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary_path = output_root / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["controller"] == "ppo_pid"
    assert summary[0]["seed"] == 0
    assert summary[0]["rc"] == 0
    assert summary[0]["checkpoint"] is not None
    ckpt = Path(summary[0]["checkpoint"])
    assert ckpt.exists()
