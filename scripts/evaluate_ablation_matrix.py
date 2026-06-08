#!/usr/bin/env python3
"""
Evaluate all trained checkpoints in the ablation matrix.

Usage:
    python scripts/evaluate_ablation_matrix.py \
        --manifest outputs/ablation_matrix/manifest.json \
        --eval-seeds 10 \
        --eval-eps 50
"""

import argparse
import json
from pathlib import Path

from run_ablation_matrix import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output_root = Path(args.manifest).parent

    for entry in manifest:
        if entry["status"] != "success" or not entry.get("checkpoint"):
            continue

        method_key = entry["method"]
        target_mode_key = entry["target_mode"]
        seed = entry["seed"]
        checkpoint = Path(entry["checkpoint"])

        print(f"Evaluating {method_key}_{target_mode_key}_s{seed} ...")
        run_evaluation(
            method_key, target_mode_key, seed, checkpoint,
            args.eval_seeds, args.eval_eps, output_root, args.gpu
        )


if __name__ == "__main__":
    main()
