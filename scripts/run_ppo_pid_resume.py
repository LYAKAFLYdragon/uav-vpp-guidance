#!/usr/bin/env python3
"""
Run PPO+PID resume training.

Resumes the interrupted PPO+PID JSBSim training from step_40960.pt
and continues to 200000 total steps.

Usage:
    # Dry-run to validate
    python scripts/run_ppo_pid_resume.py --dry-run

    # Full resume training
    python scripts/run_ppo_pid_resume.py

    # Resume with custom device
    python scripts/run_ppo_pid_resume.py --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Training was interrupted at step 45056; resume from the last saved checkpoint at 40960.
RESUME_CHECKPOINT = "outputs/experiments/ppo_pid_jsbsim/checkpoints/step_40960.pt"
RESUME_STEP = 40960
TOTAL_TIMESTEPS = 200000
CONFIG = "config/experiment/train_ppo_pid_jsbsim.yaml"
OUTPUT_DIR = "outputs/experiments/ppo_pid_jsbsim_resumed"


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


def main():
    parser = argparse.ArgumentParser(description="Resume PPO+PID JSBSim training")
    parser.add_argument("--dry-run", action="store_true", help="Validate and exit")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = REPO_ROOT / RESUME_CHECKPOINT
    config_path = REPO_ROOT / CONFIG

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    remaining = TOTAL_TIMESTEPS - RESUME_STEP
    print("=" * 60)
    print("PPO+PID Training Resume")
    print("=" * 60)
    print(f"  Config:        {CONFIG}")
    print(f"  Checkpoint:    {RESUME_CHECKPOINT}")
    print(f"  Resume step:   {RESUME_STEP}")
    print(f"  Total target:  {TOTAL_TIMESTEPS}")
    print(f"  Remaining:     {remaining}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Device:        {args.device or 'auto'}")

    if args.dry_run:
        ckpt_size = checkpoint_path.stat().st_size / 1024
        print(f"\n[DRY RUN] Checkpoint size: {ckpt_size:.1f} KB")
        print("[DRY RUN] Validation passed. No training started.")
        return

    cmd = [
        sys.executable, "-m",
        "uav_vpp_guidance.training.train_ppo_pid",
        "--config", str(config_path),
        "--output-dir", args.output_dir,
        "--seed", str(args.seed),
        "--resume", str(checkpoint_path),
        "--resume-step", str(RESUME_STEP),
        "--resume-optimizer",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print(f"\nRunning: {' '.join(cmd)}\n")
    start = datetime.now(timezone.utc)

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    if result.returncode == 0:
        manifest = {
            "start_time": start.isoformat(),
            "elapsed_seconds": elapsed,
            "command_line": sys.argv,
            "config_path": CONFIG,
            "checkpoint_path": RESUME_CHECKPOINT,
            "resume_step": RESUME_STEP,
            "total_target": TOTAL_TIMESTEPS,
            "output_dir": args.output_dir,
            "git_info": _get_git_info(),
            "method": "ppo_pid_resumed",
            "status": "completed",
        }
        manifest_path = REPO_ROOT / args.output_dir / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nResume complete! Elapsed: {elapsed:.0f}s")
        print(f"Manifest: {manifest_path}")
    else:
        print(f"\nResume FAILED with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
