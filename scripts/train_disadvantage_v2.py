#!/usr/bin/env python3
"""
Train the redesigned front-hemisphere disadvantage scenes (v2).

Each scene is trained independently for 10K steps (seed 0) using the v5
configuration with safety protections enabled.

Usage:
    source activate jsbenv
    python scripts/train_disadvantage_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


SCENES = [
    {
        "name": "disadvantage_mild",
        "config": "config/experiment/disadvantage_mild.yaml",
        "label": "Mild: ATA~22°, ego 240 / tgt 180 / closure~152",
        "difficulty": 1,
    },
    {
        "name": "disadvantage_moderate",
        "config": "config/experiment/disadvantage_moderate.yaml",
        "label": "Moderate: ATA~66°, ego 182 / tgt 225 / closure~200 (domain rand scale 0.10)",
        "difficulty": 2,
    },
    {
        "name": "disadvantage_hard",
        "config": "config/experiment/disadvantage_hard.yaml",
        "label": "Hard: ATA~80°, ego 200 / tgt 200 / closure~35",
        "difficulty": 3,
    },
]


def load_config_with_includes(path: str) -> dict:
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


def train_scene(
    scene: dict,
    seed: int = 0,
    backend: str = "jsbsim",
    device: str = "cpu",
    use_swanlab: bool = False,
    swanlab_project: str = "uav-vpp-guidance",
) -> Path:
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

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {scene['name']} with exit code {result.returncode}")

    cfg = load_config_with_includes(config_path)
    exp_name = cfg.get("experiment", {}).get("name", scene["name"])
    output_dir = Path("outputs/experiments") / exp_name
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train redesigned disadvantage scenes v2.")
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

    out_root = Path("outputs/disadvantage_v2")
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[TRAIN] Summary saved to {out_root / 'training_summary.json'}")


if __name__ == "__main__":
    main()
