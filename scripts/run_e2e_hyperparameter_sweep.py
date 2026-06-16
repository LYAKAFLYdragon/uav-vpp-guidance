#!/usr/bin/env python3
"""
Fairness hyperparameter sweep for the End-to-End DRL baseline.

Addresses reviewer concern #2: the E2E baseline must be evaluated with the same
observation space, reward function, and a reasonable hyperparameter search
before claiming that hierarchical decomposition is "necessary".

Usage:
    python scripts/run_e2e_hyperparameter_sweep.py \
        --base-config config/experiment/train_end_to_end_ppo.yaml \
        --seeds 5 --steps 200000 --align-reward

The script grid-searches over learning rates and network sizes, writes one
config per (lr, hidden) combination under a temporary directory, and calls
``run_end_to_end_ppo_multiseed.py`` for each.  A consolidated report with
per-seed success rates is written to ``outputs/e2e_fairness_sweep``.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


# Reward weights copied from the hierarchical VPP baseline so the E2E baseline
# is evaluated with an identical reward function.
VPP_REWARD = {
    "w_range": 0.6,
    "w_angle": 0.9,
    "w_energy": 0.0,
    "w_safety": 3.0,
    "w_saturation": 0.5,
    "w_smooth": 0.2,
    "w_turn_rate": 0.5,
    "w_closing": 0.1,
    "w_alive": 0.02,
    "w_overshoot": 1.0,
    "w_boundary": 0.3,
    "boundary_range_m": 6000.0,
    "boundary_alt_min_m": 1000.0,
    "boundary_alt_max_m": 9000.0,
    "ideal_range_min": 700.0,
    "ideal_range_max": 1100.0,
    "terminal_success": 400.0,
    "terminal_failure": -300.0,
    "terminal_crash": -600.0,
    "min_altitude_m": 500.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="E2E baseline fairness sweep")
    parser.add_argument("--base-config", type=str,
                        default="config/experiment/train_end_to_end_ppo.yaml")
    parser.add_argument("--output-root", type=str,
                        default="outputs/e2e_fairness_sweep")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Independent training seeds per configuration")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--learning-rates", type=float, nargs="+",
                        default=[1.0e-4, 3.0e-4, 1.0e-3])
    parser.add_argument("--hidden-sizes", type=str, nargs="+",
                        default=["64,64", "128,128", "256,256"],
                        help="Comma-separated hidden layer sizes")
    parser.add_argument("--align-reward", action="store_true",
                        help="Use the same reward weights as the VPP baseline")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_config(base_path: str, lr: float, hidden: List[int], steps: int,
                 align_reward: bool) -> Dict:
    cfg = load_yaml_config(base_path)
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = Path(base_path).parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    cfg = merge_config(merged, cfg)

    cfg.setdefault("ppo", {})["learning_rate"] = lr
    cfg.setdefault("ppo", {})["total_timesteps"] = steps
    cfg.setdefault("policy", {})["hidden_sizes"] = hidden
    cfg.setdefault("evaluation", {})["domain_rand_scale"] = 0.15

    if align_reward:
        cfg["reward"] = merge_config(cfg.get("reward", {}), VPP_REWARD)

    return cfg


def run_sweep(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    hidden_lists = [list(map(int, h.split(","))) for h in args.hidden_sizes]

    manifest = {
        "base_config": args.base_config,
        "seeds": args.seeds,
        "steps": args.steps,
        "align_reward": args.align_reward,
        "runs": [],
    }
    all_results = []

    for lr in args.learning_rates:
        for hidden in hidden_lists:
            cfg_id = f"lr{lr:.0e}_h{'x'.join(map(str, hidden))}"
            cfg_path = config_root / f"{cfg_id}.yaml"
            cfg = build_config(args.base_config, lr, hidden, args.steps, args.align_reward)
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

            exp_dir = output_root / cfg_id
            print(f"\n{'='*60}")
            print(f"[SWEEP] {cfg_id}: lr={lr}, hidden={hidden}")
            print(f"{'='*60}")

            cmd = [
                sys.executable,
                "scripts/run_end_to_end_ppo_multiseed.py",
                "--config", str(cfg_path),
                "--seeds", str(args.seeds),
                "--output-root", str(output_root),
                "--exp-name", cfg_id,
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            if args.skip_existing:
                cmd.append("--skip-existing")

            start = time.time()
            rc = subprocess.run(cmd, check=False).returncode
            elapsed = time.time() - start

            manifest["runs"].append({
                "id": cfg_id,
                "learning_rate": lr,
                "hidden_sizes": hidden,
                "elapsed_s": elapsed,
                "returncode": rc,
            })

            if rc == 0:
                agg_path = exp_dir / "multiseed_manifest.json"
                if agg_path.exists():
                    with open(agg_path, "r", encoding="utf-8") as f:
                        agg = json.load(f)
                    summary = agg.get("aggregate", {})
                    all_results.append({
                        "config_id": cfg_id,
                        "learning_rate": lr,
                        "hidden_sizes": str(hidden),
                        "success_rate_mean": summary.get("success_rate_mean", np.nan),
                        "success_rate_std": summary.get("success_rate_std", np.nan),
                        "mean_return_mean": summary.get("mean_return_mean", np.nan),
                        "mean_return_std": summary.get("mean_return_std", np.nan),
                    })

    # Write consolidated report
    report_path = output_root / "sweep_summary.csv"
    if all_results:
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[SAVED] {report_path}")

    manifest_path = output_root / "sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {manifest_path}")

    if all_results:
        best = max(all_results, key=lambda r: r["success_rate_mean"])
        print(f"\nBest configuration: {best['config_id']} "
              f"SR={best['success_rate_mean']:.2%} ± {best['success_rate_std']:.2%}")


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
