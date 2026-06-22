#!/usr/bin/env python3
"""
Launch multiple training runs in parallel.

Maps controller aliases to their training entry points and spawns one process
per (controller, seed) pair.  Useful for reproducing results across seeds or
for training all PPO-based controllers needed by the flight-control comparison.

Examples
--------
    # Smoke-test PPO+PID and PPO baseline across 3 seeds on simple backend
    python scripts/run_parallel_training.py \
        --controllers ppo_pid ppo \
        --seeds 0 1 2 \
        --backend simple \
        --smoke \
        --jobs 4

    # Train PPO+PID only, 5 seeds, JSBSim backend
    python scripts/run_parallel_training.py \
        --controllers ppo_pid \
        --seeds 0 1 2 3 4 \
        --config-ppo-pid config/experiment/train_ppo_pid_jsbsim.yaml \
        --backend jsbsim \
        --jobs 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Windows/PyTorch+NumPy can initialise multiple OpenMP runtimes and abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

CONTROLLER_TRAINERS: Dict[str, Dict[str, str]] = {
    "ppo_pid": {
        "module": "uav_vpp_guidance.training.train_ppo_pid",
        "default_config": "config/experiment/train_ppo_pid_jsbsim.yaml",
    },
    "ppo": {
        "module": "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "default_config": "config/experiment/train_no_prediction_vpp_ppo_jsbsim_compare.yaml",
    },
}


def _controller_config_flag(controller: str) -> str:
    return f"--config-{controller.replace('_', '-')}"


def _build_command(
    controller: str,
    seed: int,
    config_path: str,
    output_dir: Path,
    smoke: bool,
    backend: Optional[str],
    device: Optional[str],
    domain_rand_scale: Optional[float],
) -> List[str]:
    info = CONTROLLER_TRAINERS[controller]
    cmd = [
        sys.executable,
        "-m",
        info["module"],
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if smoke:
        cmd.append("--smoke")
    if backend is not None:
        cmd.extend(["--backend", backend])
    if device is not None:
        cmd.extend(["--device", device])
    if domain_rand_scale is not None:
        cmd.extend(["--domain-rand-scale", str(domain_rand_scale)])
    return cmd


def _run_training(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function executed in a separate process."""
    controller = args["controller"]
    seed = args["seed"]
    cmd = args["cmd"]
    log_path = Path(args["log_path"])
    output_dir = Path(args["output_dir"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Propagate JSBSIM_ROOT if the backend needs it.
    if args.get("backend") == "jsbsim" and "JSBSIM_ROOT" in os.environ:
        env["JSBSIM_ROOT"] = os.environ["JSBSIM_ROOT"]

    logger.info(f"[start] {controller} seed={seed} -> {output_dir}")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=args.get("cwd"),
        )
        rc = proc.wait()
    elapsed = time.time() - start
    logger.info(f"[done ] {controller} seed={seed} rc={rc} in {elapsed:.1f}s")

    # Heuristic checkpoint path produced by the training scripts.
    checkpoint = output_dir / "checkpoints" / "last.pt"
    return {
        "controller": controller,
        "seed": seed,
        "rc": rc,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "elapsed_s": elapsed,
    }


def _resolve_config(controller: str, cli_config: Optional[str], repo_root: Path) -> str:
    if cli_config:
        return cli_config
    default = repo_root / CONTROLLER_TRAINERS[controller]["default_config"]
    if default.exists():
        return str(default)
    raise FileNotFoundError(
        f"No config provided for {controller} and default not found: {default}"
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple training runs in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--controllers",
        type=str,
        nargs="+",
        default=["ppo_pid", "ppo"],
        choices=list(CONTROLLER_TRAINERS.keys()),
        help="Controllers to train",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds (one run per seed/controller pair)",
    )
    parser.add_argument(
        "--config-ppo-pid",
        type=str,
        default=None,
        help="Config for ppo_pid trainer",
    )
    parser.add_argument(
        "--config-ppo",
        type=str,
        default=None,
        help="Config for ppo trainer",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/parallel_training",
        help="Root directory for all runs",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 4)),
        help="Maximum parallel training processes",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass --smoke to every training run",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["simple", "jsbsim"],
        default=None,
        help="Override simulation backend for all runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Override compute device for all runs",
    )
    parser.add_argument(
        "--domain-rand-scale",
        type=float,
        default=None,
        help="Fixed domain-randomization scale (overrides curriculum)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # Map CLI config flags to controllers.
    config_overrides = {
        "ppo_pid": args.config_ppo_pid,
        "ppo": args.config_ppo,
    }

    work = []
    for controller in args.controllers:
        config_path = _resolve_config(controller, config_overrides.get(controller), repo_root)
        for seed in args.seeds:
            run_dir = output_root / controller / f"seed{seed}"
            log_path = run_dir / "training.log"
            cmd = _build_command(
                controller=controller,
                seed=seed,
                config_path=config_path,
                output_dir=run_dir,
                smoke=args.smoke,
                backend=args.backend,
                device=args.device,
                domain_rand_scale=args.domain_rand_scale,
            )
            work.append(
                {
                    "controller": controller,
                    "seed": seed,
                    "cmd": cmd,
                    "log_path": str(log_path),
                    "output_dir": str(run_dir),
                    "backend": args.backend,
                    "cwd": str(repo_root),
                }
            )

    logger.info(f"Scheduling {len(work)} training runs (max {args.jobs} parallel)")
    if args.dry_run:
        for item in work:
            logger.info("DRY-RUN: " + " ".join(item["cmd"]))
        return 0

    results: List[Dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as executor:
        futures = {executor.submit(_run_training, item): item for item in work}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Deterministic order for the summary.
    results.sort(key=lambda r: (r["controller"], r["seed"]))

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    failures = [r for r in results if r["rc"] != 0]
    success = [r for r in results if r["rc"] == 0]

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    for r in results:
        status = "OK" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        ckpt = r["checkpoint"] or "no checkpoint"
        print(f"{r['controller']:12s} seed={r['seed']:2d} {status:12s} {r['elapsed_s']:6.1f}s  {ckpt}")
    print("=" * 70)
    print(f"Success: {len(success)}/{len(results)}  Failures: {len(failures)}")

    if failures:
        logger.error("Failed runs:")
        for r in failures:
            logger.error(f"  {r['controller']} seed={r['seed']} log={r['log_path']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
