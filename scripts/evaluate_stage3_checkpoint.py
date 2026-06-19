#!/usr/bin/env python3
"""
Standalone multi-seed evaluation of a Stage-3 pursuer checkpoint vs its target.

Useful for evaluating checkpoints produced by `run_vpp_stage3_only_fix.py` or any
other ad-hoc training that does not have a train_curriculum_adversarial manifest.

Usage:
    source activate jsbenv
    python scripts/evaluate_stage3_checkpoint.py \
        --pursuer-ckpt outputs/adversarial_curriculum_pilot/vpp_stage3_only_fix/stage3/pursuer/checkpoints/best.pt \
        --target-ckpt outputs/adversarial_curriculum_pilot/vpp_stage3_only_fix/stage3/target/checkpoints/best.pt \
        --config config/adversarial/train_pursuer_v2_pilot_aggressive_stage3.yaml \
        --num-episodes 100 --seeds 0 1 2 3 4 \
        --use-swanlab --swanlab-project uav-vpp-guidance --swanlab-exp vpp_stage3_only_fix_multiseed_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.utils.swanlab_logger import SwanLabLogger
from scripts.train_curriculum_adversarial import evaluate_pursuer_checkpoint, load_experiment_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage-3 pursuer vs target.")
    parser.add_argument("--pursuer-ckpt", type=str, required=True)
    parser.add_argument("--target-ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="uav-vpp-guidance")
    parser.add_argument("--swanlab-exp", type=str, default="stage3_multiseed_eval")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    config["pursuer_checkpoint"] = args.pursuer_ckpt
    config["target_checkpoint"] = args.target_ckpt
    # Ensure evaluation uses the smoke_test scenario registry for diversity
    if not config.get("scenario_sampler", {}).get("enabled", False):
        config["scenario_sampler"] = {
            "enabled": True,
            "source": "registry_set",
            "set": "smoke_test",
            "seed": 42,
        }

    swanlab_logger = None
    if args.use_swanlab:
        swanlab_logger = SwanLabLogger(
            project=args.swanlab_project,
            experiment=args.swanlab_exp,
            config={
                "pursuer_ckpt": args.pursuer_ckpt,
                "target_ckpt": args.target_ckpt,
                "config": args.config,
                "num_episodes": args.num_episodes,
                "seeds": args.seeds,
            },
            enabled=True,
        )

    print(f"[EVAL] pursuer={args.pursuer_ckpt}")
    print(f"[EVAL] target={args.target_ckpt}")
    print(f"[EVAL] num_episodes={args.num_episodes} seeds={args.seeds}")

    try:
        res = evaluate_pursuer_checkpoint(
            pursuer_ckpt=args.pursuer_ckpt,
            target_ckpt=args.target_ckpt,
            config=config,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        res["pursuer_ckpt"] = args.pursuer_ckpt
        res["target_ckpt"] = args.target_ckpt
        res["config"] = args.config

        print(f"[EVAL] success_rate={res['success_rate']:.2%} num_episodes={res['num_episodes']}")

        if swanlab_logger is not None:
            swanlab_logger.log({
                "eval/success_rate": res["success_rate"],
                "eval/num_episodes": res["num_episodes"],
            }, step=0)

        out_json = args.output_json or "outputs/adversarial_pilot/stage3_multiseed_eval.json"
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[EVAL] Saved to {out_json}")

    finally:
        if swanlab_logger is not None:
            swanlab_logger.finish()


if __name__ == "__main__":
    main()
