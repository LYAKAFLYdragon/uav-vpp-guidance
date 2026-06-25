#!/usr/bin/env python3
"""
Run End-to-End PPO 1M-steps comparison: 3 seeds × 1M steps each.

Compares pre-fix (throttle 0.0/1.0, no anti-windup) vs post-fix
(throttle 0.4/0.9, anti-windup + protection arbitration) configurations.

Usage:
    # Dry-run to validate all commands
    python scripts/run_e2e_ppo_comparison.py --dry-run

    # Run pre-fix baseline (3 seeds)
    python scripts/run_e2e_ppo_comparison.py --group pre-fix --device cuda

    # Run post-fix comparison (3 seeds)
    python scripts/run_e2e_ppo_comparison.py --group post-fix --device cuda

    # Run single seed (for testing)
    python scripts/run_e2e_ppo_comparison.py --group post-fix --device cuda --seeds 0

Output structure:
    outputs/experiments/e2e_ppo_comparison/
    ├── pre_fix/seed_0/
    ├── pre_fix/seed_1/
    ├── pre_fix/seed_2/
    ├── post_fix/seed_0/
    ├── post_fix/seed_1/
    └── post_fix/seed_2/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pre-fix config: original end-to-end PPO without low-level controller fixes
PRE_FIX_CONFIG = "config/experiment/train_end_to_end_ppo.yaml"

# Post-fix config: end-to-end PPO with Enhanced PID + anti-windup + bank76
POST_FIX_CONFIG = "config/experiment/train_end_to_end_ppo_enhanced.yaml"

TOTAL_TIMESTEPS = 1_000_000
DEFAULT_SEEDS = [0, 1, 2]
OUTPUT_ROOT = "outputs/experiments/e2e_ppo_comparison"


def _get_git_info() -> dict:
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT
            ).strip()
        )
        info["dirty"] = (
            len(
                subprocess.check_output(
                    ["git", "status", "--short"], text=True, cwd=REPO_ROOT
                ).strip()
            )
            > 0
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "branch", "--show-current"], text=True, cwd=REPO_ROOT
            ).strip()
        )
    except Exception:
        pass
    return info


def run_one_seed(
    config_path: str,
    output_dir: str,
    seed: int,
    device: str = "cpu",
    dry_run: bool = False,
) -> dict:
    """Run a single training seed."""
    output = Path(REPO_ROOT / output_dir)
    manifest_path = output / "run_manifest.json"

    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.training.train_end_to_end_ppo",
        "--config", str(REPO_ROOT / config_path),
        "--output-dir", str(output),
        "--seed", str(seed),
        "--total-timesteps", str(TOTAL_TIMESTEPS),
    ]
    if device:
        cmd.extend(["--device", device])

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return {"seed": seed, "status": "dry_run", "output_dir": str(output)}

    print(f"\n{'=' * 60}")
    print(f"Training seed={seed}, config={config_path}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start

    status = "completed" if result.returncode == 0 else "failed"

    # Write manifest
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "command_line": " ".join(cmd),
        "config_path": config_path,
        "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS,
        "output_dir": str(output),
        "git_info": _get_git_info(),
        "method": "end_to_end_ppo",
        "status": status,
        "exit_code": result.returncode,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nSeed {seed} {status}! Elapsed: {elapsed:.0f}s")
    return {
        "seed": seed,
        "status": status,
        "elapsed_seconds": elapsed,
        "output_dir": str(output),
        "exit_code": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run E2E PPO 1M-steps comparison (pre-fix vs post-fix)"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["pre-fix", "post-fix", "all"],
        help="Which config group to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=OUTPUT_ROOT,
        help="Root output directory",
    )
    args = parser.parse_args()

    groups = {
        "pre-fix": (PRE_FIX_CONFIG, "pre_fix"),
        "post-fix": (POST_FIX_CONFIG, "post_fix"),
    }

    if args.group == "all":
        targets = list(groups.items())
    else:
        targets = [(args.group, groups[args.group])]

    results = {}
    for group_name, (config_path, subdir) in targets:
        print(f"\n{'#' * 60}")
        print(f"Group: {group_name} ({config_path})")
        print(f"{'#' * 60}")

        group_results = []
        for seed in args.seeds:
            output_dir = os.path.join(args.output_root, subdir, f"seed_{seed}")
            result = run_one_seed(
                config_path=config_path,
                output_dir=output_dir,
                seed=seed,
                device=args.device,
                dry_run=args.dry_run,
            )
            group_results.append(result)

            if result.get("exit_code", 0) != 0 and not args.dry_run:
                print(f"WARNING: Seed {seed} failed. Continuing with remaining seeds...")

        results[group_name] = group_results

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for group_name, group_results in results.items():
        completed = sum(1 for r in group_results if r["status"] == "completed")
        failed = sum(1 for r in group_results if r["status"] == "failed")
        dry = sum(1 for r in group_results if r["status"] == "dry_run")
        print(f"  {group_name}: {completed} completed, {failed} failed, {dry} dry-run")
        for r in group_results:
            elapsed = r.get("elapsed_seconds", 0)
            print(f"    seed={r['seed']}: {r['status']} ({elapsed:.0f}s) → {r['output_dir']}")

    if not args.dry_run:
        summary_path = REPO_ROOT / args.output_root / "comparison_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_timesteps_per_seed": TOTAL_TIMESTEPS,
            "results": results,
            "git_info": _get_git_info(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
