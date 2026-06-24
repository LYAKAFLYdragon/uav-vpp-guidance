#!/usr/bin/env python3
"""Compatibility smoke harness for low-level flight-control validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_flight_control_comparison.py"

CONTROLLER_ALIASES = {
    "enhanced": "enhanced_pid",
    "enhanced_pid": "enhanced_pid",
    "robust": "robust_pid",
    "robust_pid": "robust_pid",
    "baseline": "baseline_pid",
    "baseline_pid": "baseline_pid",
    "gain_scheduled": "gain_scheduled_pid",
    "gain_scheduled_pid": "gain_scheduled_pid",
}

CONFIG_ALIASES = {
    "bank75_alt_hold_lift_smooth": "config/experiment/ablation_mw_bank75_alt_hold_lift_smooth.yaml",
    "bank76_alt_hold_lift_smooth": "config/experiment/ablation_mw_bank76_alt_hold_lift_smooth.yaml",
    "bank80_alt_hold": "config/experiment/ablation_mw_bank80_alt_hold.yaml",
    "bank80_alt_hold_lift_smooth": "config/experiment/ablation_mw_bank80_alt_hold_lift_smooth.yaml",
}

TASK_CONFIG_ARGS = {
    "multi_waypoint": "--config-mw",
    "multi_waypoint_tracking": "--config-mw",
    "sustained_turn": "--config-st",
    "break_turn": "--config-bt",
}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_seeds(value: str) -> list[str]:
    parts = _split_csv(value)
    if len(parts) == 1 and parts[0].isdigit():
        return [str(i) for i in range(int(parts[0]))]
    return parts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short flight-control smoke test via run_flight_control_comparison.py."
    )
    parser.add_argument(
        "--controllers",
        default="enhanced,robust",
        help="Comma-separated controllers, e.g. enhanced,robust.",
    )
    parser.add_argument(
        "--configs",
        default="bank76_alt_hold_lift_smooth",
        help="Named bank config or explicit YAML path. Only one config is run at a time.",
    )
    parser.add_argument(
        "--seeds",
        default="2",
        help="Seed count (2 -> seeds 0,1) or comma-separated seed ids.",
    )
    parser.add_argument(
        "--task",
        default="multi_waypoint",
        choices=sorted(TASK_CONFIG_ARGS),
        help="Task to smoke test.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--backend", choices=["jsbsim", "simple"], default="jsbsim")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="outputs/flight_control_smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--no-check", action="store_true", help="Skip smoke metric checks.")
    return parser.parse_args()


def _normalize_controllers(raw: str) -> list[str]:
    controllers = []
    for name in _split_csv(raw):
        key = name.lower()
        if key not in CONTROLLER_ALIASES:
            raise ValueError(f"Unknown controller '{name}'")
        controllers.append(CONTROLLER_ALIASES[key])
    return controllers


def _resolve_config(raw: str, task: str) -> str:
    configs = _split_csv(raw)
    if len(configs) != 1:
        raise ValueError("--configs currently accepts one config per smoke run")
    config = configs[0]
    path = CONFIG_ALIASES.get(config, config)
    if path != config and not task.startswith("multi_waypoint"):
        raise ValueError(f"Named bank config '{config}' is only defined for multi_waypoint")
    full_path = REPO_ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def _run_smoke(args: argparse.Namespace) -> Path:
    controllers = _normalize_controllers(args.controllers)
    seeds = _parse_seeds(args.seeds)
    task = "multi_waypoint" if args.task == "multi_waypoint_tracking" else args.task
    config_path = _resolve_config(args.configs, task)
    run_id = args.run_id or f"smoke_{task}_{int(time.time())}"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--controllers",
        *controllers,
        "--tasks",
        task,
        "--seeds",
        *seeds,
        "--n-episodes",
        "1",
        "--jobs",
        str(args.jobs),
        "--backend",
        args.backend,
        "--status",
        "smoke",
        "--output-root",
        args.output_root,
        TASK_CONFIG_ARGS[task],
        config_path,
    ]
    if args.no_trajectory:
        cmd.append("--no-trajectory")
    if args.dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return REPO_ROOT / args.output_root / run_id


def _check_multi_waypoint_smoke(run_dir: Path) -> None:
    aggregate_path = run_dir / "aggregate" / "episode_records.json"
    if not aggregate_path.exists():
        raise RuntimeError(f"Aggregate output not found: {aggregate_path}")
    with open(aggregate_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise RuntimeError("Smoke run produced no episodes")

    crash_reasons = {"crash", "stall", "ground_collision", "altitude_violation"}
    crash_count = sum(
        1
        for ep in episodes
        if str(ep.get("termination_reason", "")).lower() in crash_reasons
    )
    mean_wp = sum(float(ep.get("completed_waypoints", 0)) for ep in episodes) / len(episodes)
    if crash_count != 0 or mean_wp < 2.5:
        raise RuntimeError(
            f"Smoke criteria failed: crash={crash_count}/{len(episodes)}, mean_wp={mean_wp:.2f}"
        )
    print(f"Smoke criteria passed: crash=0/{len(episodes)}, mean_wp={mean_wp:.2f}")


def main() -> None:
    args = _parse_args()
    run_dir = _run_smoke(args)
    if args.dry_run or args.no_check:
        return
    task = "multi_waypoint" if args.task == "multi_waypoint_tracking" else args.task
    if task == "multi_waypoint":
        _check_multi_waypoint_smoke(run_dir)


if __name__ == "__main__":
    main()
