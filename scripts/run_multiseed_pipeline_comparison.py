#!/usr/bin/env python3
"""
Multi-seed full three-stage pipeline comparison: VPP vs No-VPP.

This script runs the complete curriculum-adversarial pipeline for both VPP and
No-VPP across multiple random seeds, logs each run to SwanLab, then performs a
multi-seed evaluation of the final Stage-3 pursuer and aggregates the results.

Usage:
    source activate jsbenv
    python scripts/run_multiseed_pipeline_comparison.py \
        --seeds 0 1 2 \
        --output-root outputs/adversarial_curriculum_pilot \
        --eval-episodes 100 \
        --use-swanlab \
        --swanlab-project uav-vpp-guidance \
        --swanlab-exp multiseed_pipeline_v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.utils.swanlab_logger import SwanLabLogger
from scripts.train_curriculum_adversarial import (
    evaluate_pursuer_checkpoint,
    load_experiment_config,
)


def run_pipeline(mode: str, seed: int, output_dir: str, use_swanlab: bool,
                 swanlab_project: Optional[str], swanlab_exp: Optional[str]) -> Dict[str, Any]:
    """Run one full three-stage pipeline via subprocess."""
    cmd = [
        sys.executable,
        os.path.join(_PROJECT_ROOT, "scripts", "train_curriculum_adversarial.py"),
        "--mode", mode,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if use_swanlab:
        cmd.append("--use-swanlab")
        if swanlab_project:
            cmd.extend(["--swanlab-project", swanlab_project])
        if swanlab_exp:
            cmd.extend(["--swanlab-exp", swanlab_exp])

    print("\n" + "=" * 80)
    print(f"[PIPELINE] mode={mode} seed={seed}")
    print(f"[PIPELINE] cmd={' '.join(cmd)}")
    print("=" * 80)

    start = time.time()
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    elapsed = time.time() - start

    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    status = manifest.get("status", "unknown")
    print(f"[PIPELINE] mode={mode} seed={seed} status={status} elapsed={elapsed:.0f}s")

    return {
        "mode": mode,
        "seed": seed,
        "output_dir": output_dir,
        "manifest": manifest,
        "elapsed_s": elapsed,
        "returncode": result.returncode,
    }


def evaluate_manifest(manifest_path: str, num_episodes: int = 100,
                      eval_seeds: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Evaluate the Stage-3 pursuer from a manifest across multiple eval seeds."""
    if eval_seeds is None:
        eval_seeds = [0, 1, 2, 3, 4]

    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stage3 = manifest.get("stages", {}).get("stage3", {})
    pursuer_ckpt = stage3.get("pursuer_ckpt")
    if not pursuer_ckpt or not os.path.exists(pursuer_ckpt):
        print(f"[EVAL] Stage-3 pursuer checkpoint missing for {manifest_path}")
        return None

    target_ckpt = os.path.join(
        os.path.dirname(os.path.dirname(pursuer_ckpt)),
        "target", "checkpoints", "best.pt"
    )
    if not os.path.exists(target_ckpt):
        print(f"[EVAL] Stage-3 target checkpoint missing for {manifest_path}")
        return None

    # Use the same pursuer config that produced this checkpoint
    mode = "vpp"  # default heuristic
    if "no_vpp" in pursuer_ckpt.lower():
        mode = "no_vpp"
    elif "vpp" in pursuer_ckpt.lower():
        mode = "vpp"

    if mode == "no_vpp":
        config_path = "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml"
    else:
        config_path = "config/adversarial/train_pursuer_v2_pilot_aggressive_stage3.yaml"

    config = load_experiment_config(config_path)
    config["target_checkpoint"] = target_ckpt
    config["pursuer_checkpoint"] = pursuer_ckpt
    # Ensure evaluation uses the smoke_test scenario registry for diversity
    if not config.get("scenario_sampler", {}).get("enabled", False):
        config["scenario_sampler"] = {
            "enabled": True,
            "source": "registry_set",
            "set": "smoke_test",
            "seed": 42,
        }

    print(f"[EVAL] mode={mode} manifest={manifest_path}")
    res = evaluate_pursuer_checkpoint(
        pursuer_ckpt=pursuer_ckpt,
        target_ckpt=target_ckpt,
        config=config,
        num_episodes=num_episodes,
        seeds=eval_seeds,
    )
    res["mode"] = mode
    res["manifest"] = manifest_path
    res["pursuer_ckpt"] = pursuer_ckpt
    res["target_ckpt"] = target_ckpt
    return res


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-mode capture rates across seeds."""
    by_mode: Dict[str, List[float]] = {}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r["success_rate"])

    summary = {}
    for mode, srs in by_mode.items():
        arr = np.array(srs)
        summary[mode] = {
            "num_seeds": len(arr),
            "mean_sr": float(np.mean(arr)),
            "std_sr": float(np.std(arr)),
            "min_sr": float(np.min(arr)),
            "max_sr": float(np.max(arr)),
            "seeds": srs,
        }

    # Paired difference if both modes have the same number of seeds
    if "vpp" in summary and "no_vpp" in summary:
        vpp_srs = summary["vpp"]["seeds"]
        no_vpp_srs = summary["no_vpp"]["seeds"]
        if len(vpp_srs) == len(no_vpp_srs):
            diffs = np.array(vpp_srs) - np.array(no_vpp_srs)
            summary["diff"] = {
                "mean": float(np.mean(diffs)),
                "std": float(np.std(diffs)),
                "values": [float(x) for x in diffs],
            }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed full pipeline comparison: VPP vs No-VPP."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds to run for each mode.",
    )
    parser.add_argument(
        "--output-root", type=str,
        default="outputs/adversarial_curriculum_pilot",
        help="Root output directory for all seeds/modes.",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Total evaluation episodes per final Stage-3 checkpoint.",
    )
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Seeds used during final evaluation rollouts.",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and only evaluate existing manifests.",
    )
    parser.add_argument(
        "--use-swanlab", action="store_true", help="Enable SwanLab logging"
    )
    parser.add_argument(
        "--swanlab-project", type=str, default="uav-vpp-guidance",
        help="SwanLab project name",
    )
    parser.add_argument(
        "--swanlab-exp", type=str, default="multiseed_pipeline_v1",
        help="SwanLab experiment name for the comparison summary",
    )
    parser.add_argument(
        "--mode", type=str, choices=["vpp", "no_vpp", "both"], default="both",
        help="Which mode(s) to run.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    # Top-level SwanLab run for the comparison summary
    swanlab_logger = None
    if args.use_swanlab:
        swanlab_logger = SwanLabLogger(
            project=args.swanlab_project,
            experiment=args.swanlab_exp,
            config={
                "seeds": args.seeds,
                "eval_episodes": args.eval_episodes,
                "eval_seeds": args.eval_seeds,
                "mode": args.mode,
            },
            enabled=True,
        )

    modes = []
    if args.mode in ("vpp", "both"):
        modes.append("vpp")
    if args.mode in ("no_vpp", "both"):
        modes.append("no_vpp")

    pipeline_results: List[Dict[str, Any]] = []
    eval_results: List[Dict[str, Any]] = []

    try:
        if not args.skip_training:
            for seed in args.seeds:
                for mode in modes:
                    output_dir = os.path.join(args.output_root, f"{mode}_full_multiseed_s{seed}")
                    exp_name = f"{args.swanlab_exp}_{mode}_s{seed}" if args.use_swanlab else None
                    res = run_pipeline(
                        mode=mode,
                        seed=seed,
                        output_dir=output_dir,
                        use_swanlab=args.use_swanlab,
                        swanlab_project=args.swanlab_project,
                        swanlab_exp=exp_name,
                    )
                    pipeline_results.append(res)
                    if swanlab_logger is not None:
                        swanlab_logger.log({
                            f"pipeline/{mode}_seed{seed}_status": res["returncode"],
                            f"pipeline/{mode}_seed{seed}_elapsed_s": res["elapsed_s"],
                            f"pipeline/{mode}_seed{seed}_manifest_status": 1.0 if res["manifest"].get("status") == "completed" else 0.0,
                        }, step=seed)

        # Evaluate all manifests found under output_root
        print("\n" + "=" * 80)
        print("[EVAL] Multi-seed evaluation of Stage-3 checkpoints")
        print("=" * 80)

        for seed in args.seeds:
            for mode in modes:
                output_dir = os.path.join(args.output_root, f"{mode}_full_multiseed_s{seed}")
                manifest_path = os.path.join(output_dir, "manifest.json")
                res = evaluate_manifest(
                    manifest_path=manifest_path,
                    num_episodes=args.eval_episodes,
                    eval_seeds=args.eval_seeds,
                )
                if res is not None:
                    eval_results.append(res)
                    if swanlab_logger is not None:
                        swanlab_logger.log({
                            f"eval/{mode}_seed{seed}_capture_rate": res["success_rate"],
                            f"eval/{mode}_seed{seed}_num_episodes": res["num_episodes"],
                        }, step=seed)

        summary = aggregate_results(eval_results)

        print("\n" + "=" * 80)
        print("[SUMMARY] Multi-seed Stage-3 capture rates")
        print("=" * 80)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if swanlab_logger is not None:
            for mode, stats in summary.items():
                if mode in ("vpp", "no_vpp"):
                    swanlab_logger.log({
                        f"summary/{mode}_mean_sr": stats["mean_sr"],
                        f"summary/{mode}_std_sr": stats["std_sr"],
                        f"summary/{mode}_num_seeds": stats["num_seeds"],
                    }, step=0)
            if "diff" in summary:
                swanlab_logger.log({
                    "summary/diff_mean_sr": summary["diff"]["mean"],
                    "summary/diff_std_sr": summary["diff"]["std"],
                }, step=0)

        # Save local JSON
        out_json = os.path.join(args.output_root, f"{args.swanlab_exp}_multiseed_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "args": vars(args),
                "pipeline_results": pipeline_results,
                "eval_results": eval_results,
                "summary": summary,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[SUMMARY] Saved to {out_json}")

    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()


if __name__ == "__main__":
    main()
