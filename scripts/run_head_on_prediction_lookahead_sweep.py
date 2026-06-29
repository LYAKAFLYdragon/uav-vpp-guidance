#!/usr/bin/env python3
"""Launch head-on lookahead sweeps for the fair JSBSim prediction compare."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_jsbsim_hrl_comparison.py"
DEFAULT_LOOKAHEADS = [0.0, 0.2, 0.5, 1.0]


def _lookahead_tag(value: float) -> str:
    return f"{float(value):.1f}".replace("-", "m").replace(".", "p")


def build_runner_command(args: argparse.Namespace, lookahead_time_s: float) -> List[str]:
    run_id = f"{args.run_prefix}_lookahead_{_lookahead_tag(lookahead_time_s)}"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        args.config,
        "--run-id",
        run_id,
        "--output-root",
        args.output_root,
        "--preset",
        "main",
        "--methods",
        args.method,
        "--tasks",
        "head_on",
        "--backend",
        args.backend,
        "--opponent-stage",
        args.opponent_stage,
        "--run-status",
        args.run_status,
        "--device",
        args.device,
        "--prediction-lookahead-time-s",
        str(float(lookahead_time_s)),
    ]
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.n_episodes is not None:
        cmd.extend(["--n-episodes", str(int(args.n_episodes))])
    if args.jsbsim_root:
        cmd.extend(["--jsbsim-root", args.jsbsim_root])
    if args.allow_missing_checkpoints:
        cmd.append("--allow-missing-checkpoints")
    if args.no_trajectory:
        cmd.append("--no-trajectory")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run head-on-only prediction lookahead sweeps through run_jsbsim_hrl_comparison.py"
    )
    parser.add_argument("--config", default="config/experiment/jsbsim_hrl_comparison.yaml")
    parser.add_argument("--output-root", default="outputs/jsbsim_hrl_comparison")
    parser.add_argument("--run-prefix", default="head_on_prediction_lookahead_sweep")
    parser.add_argument("--method", default="prediction_vpp_jsbsim_compare")
    parser.add_argument("--lookahead-times", nargs="+", type=float, default=DEFAULT_LOOKAHEADS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--backend", choices=["jsbsim", "simple"], default="jsbsim")
    parser.add_argument("--opponent-stage", choices=["none", "expert", "end_to_end", "curriculum"], default="none")
    parser.add_argument("--run-status", choices=["smoke", "formal-small", "formal"], default="formal-small")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jsbsim-root", default=None)
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs = []
    failures = []
    for lookahead_time_s in args.lookahead_times:
        cmd = build_runner_command(args, lookahead_time_s)
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        run_record = {
            "lookahead_time_s": float(lookahead_time_s),
            "run_id": f"{args.run_prefix}_lookahead_{_lookahead_tag(lookahead_time_s)}",
            "command": cmd,
            "returncode": int(result.returncode),
        }
        runs.append(run_record)
        if result.returncode != 0:
            failures.append(
                {
                    **run_record,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

    manifest_path = output_root / f"{args.run_prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "runner": str(RUNNER),
                "method": args.method,
                "task": "head_on",
                "lookahead_times": [float(value) for value in args.lookahead_times],
                "runs": runs,
                "failures": failures,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if failures:
        for failure in failures:
            sys.stderr.write(failure["stderr"])
            sys.stderr.write(failure["stdout"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
