#!/usr/bin/env python3
"""
Train a feasibility gradient of 5 disadvantage scenarios.

Each scenario is trained independently for 10K steps (seed 0) using the v5
configuration (safety protections enabled, dynamics_aware=false).

Usage:
    source activate jsbenv
    python scripts/train_disadvantage_gradient.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


SCENES = [
    {
        "name": "disadvantage_gradient_1",
        "config": "config/experiment/disadvantage_gradient_1.yaml",
        "label": "S1: ego 240 / tgt 180 / offset 200",
        "difficulty": 1,
    },
    {
        "name": "disadvantage_gradient_2",
        "config": "config/experiment/disadvantage_gradient_2.yaml",
        "label": "S2: ego 230 / tgt 190 / offset 300",
        "difficulty": 2,
    },
    {
        "name": "disadvantage_gradient_3",
        "config": "config/experiment/disadvantage_gradient_3.yaml",
        "label": "S3: ego 220 / tgt 200 / offset 400",
        "difficulty": 3,
    },
    {
        "name": "disadvantage_gradient_4",
        "config": "config/experiment/disadvantage_gradient_4.yaml",
        "label": "S4: ego 210 / tgt 210 / offset 500",
        "difficulty": 4,
    },
    {
        "name": "disadvantage_gradient_5",
        "config": "config/experiment/disadvantage_gradient_5.yaml",
        "label": "S5: ego 200 / tgt 220 / offset 600",
        "difficulty": 5,
    },
]


def train_scene(
    scene: dict,
    seed: int = 0,
    backend: str = "jsbsim",
    device: str = "cpu",
    use_swanlab: bool = False,
    swanlab_project: str = "uav-vpp-guidance",
) -> Path:
    """Train one scene and return the output directory."""
    config_path = os.path.join(_PROJECT_ROOT, scene["config"])

    cmd = [
        sys.executable,
        os.path.join(_PROJECT_ROOT, "scripts", "train_curriculum_ppo.py"),
        "--config", config_path,
        "--backend", backend,
        "--device", device,
        "--seed", str(seed),
        "--total-timesteps", "10000",
    ]
    if use_swanlab:
        cmd.append("--use-swanlab")
        cmd.extend(["--swanlab-project", swanlab_project])
        cmd.extend(["--swanlab-exp", scene["name"]])

    print("\n" + "=" * 80)
    print(f"[TRAIN] {scene['label']}")
    print("=" * 80)
    print(f"cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {scene['name']} with exit code {result.returncode}")

    # Derive output directory from config experiment name
    from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

    def load_config(path: str) -> dict:
        base = load_yaml_config(path)
        includes = base.pop("includes", [])
        merged = {}
        cfg_dir = os.path.dirname(path)
        for inc in includes:
            inc_full = os.path.join(cfg_dir, inc)
            if not os.path.exists(inc_full):
                inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc))
            if os.path.exists(inc_full):
                merged = merge_config(merged, load_yaml_config(inc_full))
        return merge_config(merged, base)

    cfg = load_config(config_path)
    exp_name = cfg.get("experiment", {}).get("name", scene["name"])
    output_dir = Path("outputs/experiments") / exp_name
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train disadvantage feasibility gradient.")
    parser.add_argument("--backend", type=str, default="jsbsim", choices=["simple", "jsbsim"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="uav-vpp-guidance")
    args = parser.parse_args()

    summary = []
    for scene in SCENES:
        output_dir = train_scene(
            scene,
            seed=args.seed,
            backend=args.backend,
            device=args.device,
            use_swanlab=args.use_swanlab,
            swanlab_project=args.swanlab_project,
        )
        checkpoint = output_dir / "checkpoints" / "best.pt"
        summary.append({
            "name": scene["name"],
            "label": scene["label"],
            "difficulty": scene["difficulty"],
            "config": scene["config"],
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint),
        })
        print(f"[TRAIN] {scene['name']} -> {output_dir}")

    # Save summary for evaluation script
    out_root = Path("outputs/disadvantage_gradient")
    out_root.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_root / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[TRAIN] Summary saved to {out_root / 'training_summary.json'}")


if __name__ == "__main__":
    main()
