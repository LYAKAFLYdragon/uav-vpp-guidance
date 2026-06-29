from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_head_on_prediction_lookahead_sweep import build_runner_command  # noqa: E402


def test_build_runner_command_targets_fair_jsbsim_prediction_method():
    args = argparse.Namespace(
        config="config/experiment/jsbsim_hrl_comparison.yaml",
        output_root="outputs/jsbsim_hrl_comparison",
        run_prefix="fair_head_on",
        method="prediction_vpp_jsbsim_compare",
        seeds=[480, 481],
        n_episodes=1,
        backend="jsbsim",
        opponent_stage="expert",
        run_status="formal-small",
        device="cuda",
        jsbsim_root="/tmp/jsbsim-root",
        no_trajectory=True,
        dry_run=True,
        allow_missing_checkpoints=True,
    )

    cmd = build_runner_command(args, 0.5)

    assert "--methods" in cmd
    assert cmd[cmd.index("--methods") + 1] == "prediction_vpp_jsbsim_compare"
    assert cmd[cmd.index("--tasks") + 1] == "head_on"
    assert cmd[cmd.index("--prediction-lookahead-time-s") + 1] == "0.5"
    assert cmd[cmd.index("--run-id") + 1] == "fair_head_on_lookahead_0p5"
    assert cmd[cmd.index("--device") + 1] == "cuda"
    assert "--allow-missing-checkpoints" in cmd
    assert "--dry-run" in cmd
