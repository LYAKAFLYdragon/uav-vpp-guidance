"""Smoke tests for the formal flight-control comparison mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_flight_control_comparison.py"
SMOKE_RUNNER = REPO_ROOT / "scripts" / "run_flight_control_smoke_test.py"


def test_formal_mode_dry_run():
    """Formal status with multi-seed/multi-task should validate without error."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--run-id",
            "_test_formal_dry_run",
            "--controllers",
            "enhanced_pid",
            "baseline_pid",
            "--tasks",
            "multi_waypoint",
            "sustained_turn",
            "break_turn",
            "--seeds",
            "0",
            "1",
            "--n-episodes",
            "1",
            "--backend",
            "simple",
            "--status",
            "formal",
            "--no-trajectory",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN complete" in result.stdout


def test_smoke_wrapper_dry_run_accepts_objective_command_shape():
    result = subprocess.run(
        [
            sys.executable,
            str(SMOKE_RUNNER),
            "--controllers",
            "enhanced,robust",
            "--configs",
            "bank76_alt_hold_lift_smooth",
            "--seeds",
            "2",
            "--task",
            "multi_waypoint",
            "--backend",
            "simple",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN complete" in result.stdout
