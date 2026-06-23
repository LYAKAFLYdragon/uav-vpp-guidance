#!/usr/bin/env python3
"""
Crossing-angle threshold experiment for the flight-control comparison.

Evaluates PPO+PID and PID controllers across a sweep of target-heading angles
(0 deg = tail chase, 90 deg = broadside crossing) under LOS-rate and PN guidance.
Each trial uses a fixed deterministic scenario so the resulting success-rate
curve reveals the geometric feasibility boundary of the aircraft + controller
combination.

Usage (local smoke on simple backend):
    python scripts/run_crossing_angle_threshold.py \
        --run-id crossing_threshold_smoke \
        --backend simple \
        --seeds 0 1 \
        --checkpoint-ppo-pid outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt \
        --checkpoint-ppo outputs/experiments/no_prediction_vpp_ppo_jsbsim_compare/checkpoints/last.pt

Usage (JSBSim F-16, recommended for paper boundary):
    export JSBSIM_ROOT=/path/to/jsbsim_root
    python scripts/run_crossing_angle_threshold.py \
        --run-id crossing_threshold_jsbsim \
        --backend jsbsim \
        --jobs 4 \
        --checkpoint-ppo-pid outputs/experiments/ppo_pid_jsbsim/checkpoints/last.pt \
        --checkpoint-ppo outputs/experiments/no_prediction_vpp_ppo_jsbsim_compare/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as script from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_flight_control_comparison import (
    CONTROLLER_ALIASES,
    _build_adapter,
    _build_env,
    _build_pid_eval_config,
    _build_ppo_eval_config,
    _config_sha256,
    _get_git_commit,
    _run_episode,
)

logger = logging.getLogger(__name__)

DEFAULT_ANGLES = [0.0, 30.0, 45.0, 60.0, 75.0, 90.0]


def _build_crossing_scenario(
    crossing_angle_deg: float,
    initial_range_m: float = 8000.0,
    ego_speed_mps: float = 340.0,
    target_speed_mps: float = 240.0,
    base_altitude_m: float = 5000.0,
) -> Dict[str, Any]:
    """
    Build a deterministic intercept scenario.

    crossing_angle_deg is the target heading relative to the ego heading:
      0 deg  -> tail chase (target ahead, same direction)
      90 deg -> broadside crossing (target ahead, moving perpendicular)
    """
    theta = math.radians(crossing_angle_deg)
    own_init = {
        "position_m": [0.0, 0.0, float(base_altitude_m)],
        "velocity_mps": float(ego_speed_mps),
        "heading_deg": 0.0,
    }
    target_init = {
        "position_m": [
            float(initial_range_m * math.cos(theta)),
            float(initial_range_m * math.sin(theta)),
            float(base_altitude_m),
        ],
        "velocity_mps": float(target_speed_mps),
        "heading_deg": float(crossing_angle_deg),
    }
    return {
        "name": f"crossing_{crossing_angle_deg:02.0f}deg",
        "own_init": own_init,
        "target_init": target_init,
        "metadata": {
            "crossing_angle_deg": crossing_angle_deg,
            "initial_range_m": initial_range_m,
            "ego_speed_mps": ego_speed_mps,
            "target_speed_mps": target_speed_mps,
            "base_altitude_m": base_altitude_m,
        },
    }


def _guidance_mode_canonical(mode: str) -> str:
    """Map user-friendly guidance names to config keys."""
    mode = mode.lower().strip()
    aliases = {
        "pn": "proportional_navigation",
        "proportional_navigation": "proportional_navigation",
        "los_rate": "los_rate",
        "los-rate": "los_rate",
        "hybrid": "hybrid",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown guidance mode: {mode}")
    return aliases[mode]


def _run_single_condition(
    run_id: str,
    controller: str,
    guidance_mode: str,
    crossing_angle_deg: float,
    seed: int,
    base_config: Dict[str, Any],
    task_config: Dict[str, Any],
    checkpoint_ppo_pid: Optional[str],
    checkpoint_ppo: Optional[str],
    checkpoint_apic_pid: Optional[str],
    device: str,
    backend_override: Optional[str],
    save_full: bool,
) -> Dict[str, Any]:
    """Run one episode for a single (controller, guidance, angle, seed) condition."""
    if controller in ("ppo_pid", "apic_pid", "ppo"):
        path = {
            "ppo_pid": checkpoint_ppo_pid,
            "apic_pid": checkpoint_apic_pid,
            "ppo": checkpoint_ppo,
        }[controller]
        config = _build_ppo_eval_config(
            path, base_config, task_config, backend_override=backend_override
        )
    else:
        ll_type = {
            "enhanced_pid": "enhanced_pid",
            "baseline_pid": "baseline_pid",
            "gain_scheduled_pid": "gain_scheduled_pid",
        }[controller]
        config = _build_pid_eval_config(
            base_config, task_config, ll_type, backend_override=backend_override
        )

    # Override guidance mode for this condition.
    if "guidance" not in config:
        config["guidance"] = {}
    config["guidance"]["mode"] = _guidance_mode_canonical(guidance_mode)

    env = _build_env(config)
    adapter = _build_adapter(
        controller, config, checkpoint_ppo_pid, checkpoint_ppo, checkpoint_apic_pid, device=device
    )

    scenario = _build_crossing_scenario(crossing_angle_deg)
    config_sha256 = _config_sha256(config)
    git_commit = _get_git_commit()

    # Use a deterministic episode id derived from the condition.
    ep_seed = int(
        (hash(controller) % 100000)
        + (hash(guidance_mode) % 1000)
        + crossing_angle_deg * 100
        + seed * 10000
    ) % (2**31)

    task_name = scenario["name"]
    ep_json = _run_episode(
        env,
        adapter,
        ep_seed,
        run_id,
        task_name,
        controller,
        episode=seed,
        config=config,
        config_sha256=config_sha256,
        git_commit=git_commit,
        save_full=save_full,
        scenario=scenario,
    )
    # Tag the record with experimental condition.
    ep_json["crossing_angle_deg"] = crossing_angle_deg
    ep_json["guidance_mode"] = guidance_mode
    ep_json["scenario"] = scenario
    env.close()
    return ep_json


def _evaluate_worker(args: Tuple) -> List[Dict[str, Any]]:
    """Worker that runs all seeds for one (controller, guidance, angle) condition."""
    (
        run_id,
        controller,
        guidance_mode,
        crossing_angle_deg,
        seeds,
        base_config,
        task_config,
        checkpoint_ppo_pid,
        checkpoint_ppo,
        checkpoint_apic_pid,
        device,
        backend_override,
        save_full,
    ) = args
    records = []
    for seed in seeds:
        try:
            rec = _run_single_condition(
                run_id,
                controller,
                guidance_mode,
                crossing_angle_deg,
                seed,
                base_config,
                task_config,
                checkpoint_ppo_pid,
                checkpoint_ppo,
                checkpoint_apic_pid,
                device,
                backend_override,
                save_full,
            )
            records.append(rec)
        except Exception as exc:
            logger.exception(
                f"Failed {controller}/{guidance_mode}/{crossing_angle_deg}deg/seed{seed}: {exc}"
            )
    return records


def _aggregate_results(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-condition success rates."""
    rows = []
    for rec in records:
        rows.append(
            {
                "controller": rec.get("controller"),
                "guidance_mode": rec.get("guidance_mode"),
                "crossing_angle_deg": rec.get("crossing_angle_deg"),
                "seed": rec.get("seed"),
                "success": bool(rec.get("success", False)),
                "reason": rec.get("termination_reason", "unknown"),
                "steps": rec.get("steps"),
                "total_time_s": rec.get("total_time_s"),
                "final_range_m": rec.get("final_range_m"),
                "final_altitude_m": rec.get("final_altitude_m"),
                "final_speed_mps": rec.get("final_speed_mps"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _plot_success_rate(df: pd.DataFrame, out_png: Path) -> None:
    """Plot success rate vs crossing angle, one line per controller/guidance."""
    controllers = sorted(df["controller"].unique())
    guidance_modes = sorted(df["guidance_mode"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles = {"los_rate": "-", "pn": "--"}
    colors = {
        "ppo_pid": "C0",
        "ppo": "C1",
        "enhanced_pid": "C2",
        "baseline_pid": "C3",
        "gain_scheduled_pid": "C4",
    }
    labels = {
        "ppo_pid": "PPO+PID",
        "ppo": "PPO",
        "enhanced_pid": "Enhanced PID",
        "baseline_pid": "Baseline PID",
        "gain_scheduled_pid": "GainScheduled PID",
    }

    for controller in controllers:
        for gmode in guidance_modes:
            sub = df[(df["controller"] == controller) & (df["guidance_mode"] == gmode)]
            if sub.empty:
                continue
            grouped = (
                sub.groupby("crossing_angle_deg")
                .agg(success_rate=("success", "mean"), n=("success", "count"))
                .reset_index()
                .sort_values("crossing_angle_deg")
            )
            label = f"{labels.get(controller, controller)} + {gmode.upper()}"
            ax.plot(
                grouped["crossing_angle_deg"],
                grouped["success_rate"],
                label=label,
                color=colors.get(controller, None),
                linestyle=linestyles.get(gmode, "-"),
                marker="o",
                linewidth=2,
            )

    ax.set_xlabel("Crossing angle / deg")
    ax.set_ylabel("Success rate")
    ax.set_title("Crossing-Angle Threshold: Success Rate vs Target Heading")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved success-rate plot to {out_png}")


def _plot_failure_telemetry(df: pd.DataFrame, out_png: Path) -> None:
    """Plot final range vs crossing angle to show divergence."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for controller in sorted(df["controller"].unique()):
        for gmode in sorted(df["guidance_mode"].unique()):
            sub = df[(df["controller"] == controller) & (df["guidance_mode"] == gmode)]
            if sub.empty:
                continue
            grouped = (
                sub.groupby("crossing_angle_deg")["final_range_m"]
                .mean()
                .reset_index()
                .sort_values("crossing_angle_deg")
            )
            ax.plot(
                grouped["crossing_angle_deg"],
                grouped["final_range_m"],
                marker="x",
                label=f"{controller} + {gmode}",
            )
    ax.set_xlabel("Crossing angle / deg")
    ax.set_ylabel("Mean final range / m")
    ax.set_title("Crossing-Angle Threshold: Final Range (Divergence Indicator)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved final-range plot to {out_png}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Crossing-angle threshold experiment")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument(
        "--controllers",
        type=str,
        nargs="+",
        default=["ppo_pid", "enhanced_pid", "baseline_pid"],
        help="Controllers to evaluate",
    )
    parser.add_argument(
        "--guidance-modes",
        type=str,
        nargs="+",
        default=["los_rate", "pn"],
        help="Guidance modes to evaluate",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=DEFAULT_ANGLES,
        help="Crossing angles in degrees",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to evaluate",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--checkpoint-ppo-pid",
        type=str,
        default=None,
        help="Checkpoint path for PPO+PID-Hybrid",
    )
    parser.add_argument(
        "--checkpoint-ppo",
        type=str,
        default=None,
        help="Checkpoint path for PPO-FixedPID",
    )
    parser.add_argument(
        "--checkpoint-apic-pid",
        type=str,
        default=None,
        help="Checkpoint path for APIC-PID",
    )
    parser.add_argument(
        "--config-base",
        type=str,
        default="config/experiment/compare_flight_control_base.yaml",
        help="Base comparison config",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/flight_control_compare",
        help="Root output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for PPO inference",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["simple", "jsbsim"],
        help="Override simulation backend",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable full trajectory saving to reduce disk usage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and exit without running simulations",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Validate backend/data availability.
    if args.backend == "jsbsim" and not os.environ.get("JSBSIM_ROOT"):
        logger.warning(
            "JSBSIM_ROOT is not set. JSBSim backend will likely fall back to simple "
            "or fail. Set JSBSIM_ROOT=/path/to/jsbsim_root for F-16 evaluation."
        )

    run_dir = Path(args.output_root) / args.run_id
    raw_dir = run_dir / "raw"
    figures_dir = run_dir / "figures"
    for d in (run_dir, raw_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "runner.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Loading base config")
    base_config = {}
    from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

    def _load_cfg(path: str) -> dict:
        cfg = load_yaml_config(path)
        includes = cfg.pop("includes", [])
        merged = {}
        cfg_dir = os.path.dirname(path)
        for inc in includes:
            inc_full = os.path.join(cfg_dir, inc)
            if not os.path.exists(inc_full):
                inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc))
            if os.path.exists(inc_full):
                merged = merge_config(merged, _load_cfg(inc_full))
        return merge_config(merged, cfg)

    base_config = _load_cfg(args.config_base)

    # Dummy task config: just enough for _build_*_eval_config.
    task_config = {
        "env": {
            "max_high_level_steps": base_config.get("env", {}).get(
                "max_high_level_steps", 512
            ),
        },
        "task": {"name": "crossing_threshold"},
    }

    # Validate controllers
    unknown = [c for c in args.controllers if c not in CONTROLLER_ALIASES]
    if unknown:
        raise ValueError(f"Unknown controllers: {unknown}")

    if "ppo_pid" in args.controllers and not args.checkpoint_ppo_pid:
        raise ValueError("--checkpoint-ppo-pid required when evaluating ppo_pid")
    if "ppo" in args.controllers and not args.checkpoint_ppo:
        raise ValueError("--checkpoint-ppo required when evaluating ppo")
    if "apic_pid" in args.controllers and not args.checkpoint_apic_pid:
        raise ValueError("--checkpoint-apic-pid required when evaluating apic_pid")

    if args.dry_run:
        logger.info("DRY RUN: validating configs")
        for controller in args.controllers:
            for gmode in args.guidance_modes:
                for angle in args.angles:
                    if controller in ("ppo_pid", "apic_pid", "ppo"):
                        path = {
                            "ppo_pid": args.checkpoint_ppo_pid,
                            "apic_pid": args.checkpoint_apic_pid,
                            "ppo": args.checkpoint_ppo,
                        }[controller]
                        cfg = _build_ppo_eval_config(
                            path, base_config, task_config, backend_override=args.backend
                        )
                    else:
                        ll_type = {
                            "enhanced_pid": "enhanced_pid",
                            "baseline_pid": "baseline_pid",
                            "gain_scheduled_pid": "gain_scheduled_pid",
                        }[controller]
                        cfg = _build_pid_eval_config(
                            base_config, task_config, ll_type, backend_override=args.backend
                        )
                    cfg["guidance"]["mode"] = gmode
                    logger.info(
                        f"  {controller}/{gmode}/{angle}deg: config ok "
                        f"(sha256={_config_sha256(cfg)})"
                    )
        logger.info("DRY RUN complete")
        return

    # Build work list
    work_items = []
    for controller in args.controllers:
        for gmode in args.guidance_modes:
            for angle in args.angles:
                work_items.append(
                    (
                        args.run_id,
                        controller,
                        gmode,
                        angle,
                        args.seeds,
                        base_config,
                        task_config,
                        args.checkpoint_ppo_pid,
                        args.checkpoint_ppo,
                        args.checkpoint_apic_pid,
                        args.device,
                        args.backend,
                        not args.no_trajectory,
                    )
                )

    logger.info(
        f"Total work items: {len(work_items)} "
        f"({len(args.controllers)} controllers x {len(args.guidance_modes)} guidance modes x "
        f"{len(args.angles)} angles x {len(args.seeds)} seeds)"
    )

    all_records: List[Dict[str, Any]] = []
    start_time = time.time()
    max_workers = min(args.jobs, len(work_items))

    if max_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_evaluate_worker, item): item for item in work_items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                    logger.info(
                        f"Finished {item[1]}/{item[2]}/{item[3]}deg ({len(records)} records)"
                    )
                except Exception as exc:
                    logger.exception(f"Failed {item[1]}/{item[2]}/{item[3]}deg: {exc}")
    else:
        for item in work_items:
            try:
                records = _evaluate_worker(item)
                all_records.extend(records)
                logger.info(
                    f"Finished {item[1]}/{item[2]}/{item[3]}deg ({len(records)} records)"
                )
            except Exception as exc:
                logger.exception(f"Failed {item[1]}/{item[2]}/{item[3]}deg: {exc}")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s; {len(all_records)} episodes")

    # Save raw records
    raw_path = raw_dir / "episode_records.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved raw records to {raw_path}")

    # Aggregate and save CSV
    df = _aggregate_results(all_records)
    csv_path = run_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV to {csv_path}")

    # Summary table
    summary = (
        df.groupby(["controller", "guidance_mode", "crossing_angle_deg"])
        .agg(success_rate=("success", "mean"), n=("success", "count"))
        .reset_index()
    )
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved summary CSV to {summary_path}")

    # Plots
    _plot_success_rate(df, figures_dir / "crossing_threshold_success_rate.png")
    _plot_failure_telemetry(df, figures_dir / "crossing_threshold_final_range.png")

    # Manifest
    manifest = {
        "run_id": args.run_id,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "elapsed_seconds": elapsed,
        "git_commit": _get_git_commit(),
        "backend": args.backend or base_config.get("backend", "jsbsim"),
        "controllers": args.controllers,
        "guidance_modes": args.guidance_modes,
        "angles": args.angles,
        "seeds": args.seeds,
        "total_episodes": len(all_records),
        "checkpoint_ppo_pid": args.checkpoint_ppo_pid,
        "checkpoint_ppo": args.checkpoint_ppo,
        "checkpoint_apic_pid": args.checkpoint_apic_pid,
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
