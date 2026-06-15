#!/usr/bin/env python3
"""
Clamped-VPP ablation.

Addresses reviewer concern #3: the VPP 0% success rate may be an
implementation/training-stability issue (offset-induced OOB) rather than a
fundamental architectural defect. This script trains the VPP layer with a
severely restricted offset box (default ±100 m in all axes) and evaluates
whether the policy can then solve the canonical scenarios.

Usage:
    python scripts/run_vpp_clamped_ablation.py \
        --base-config config/experiment/train_no_prediction_vpp_ppo.yaml \
        --seeds 5 --long-limit 100 --lat-limit 100 --vert-limit 100
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def parse_args():
    parser = argparse.ArgumentParser(description="Clamped-VPP ablation")
    parser.add_argument("--base-config", type=str,
                        default="config/experiment/train_no_prediction_vpp_ppo.yaml")
    parser.add_argument("--output-root", type=str,
                        default="outputs/experiments/vpp_clamped_ablation")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--long-limit", type=float, default=100.0)
    parser.add_argument("--lat-limit", type=float, default=100.0)
    parser.add_argument("--vert-limit", type=float, default=100.0)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_config(base_path: str, limits: Tuple[float, float, float], steps: int) -> dict:
    cfg = load_yaml_config(base_path)
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = Path(base_path).parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    cfg = merge_config(merged, cfg)

    long_lim, lat_lim, vert_lim = limits
    cfg.setdefault("virtual_point", {})["d_long_range"] = [-long_lim, long_lim]
    cfg.setdefault("virtual_point", {})["d_lat_range"] = [-lat_lim, lat_lim]
    cfg.setdefault("virtual_point", {})["d_vert_range"] = [-vert_lim, vert_lim]
    cfg.setdefault("virtual_point", {})["dynamics_aware"] = True
    cfg.setdefault("ppo", {})["total_timesteps"] = steps
    cfg.setdefault("evaluation", {})["domain_rand_scale"] = 0.15
    return cfg


def run_single(config_path: str, seed: int, output_dir: str, device):
    cmd = [
        sys.executable,
        "-m", "uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "--config", config_path,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if device:
        cmd.extend(["--device", device])
    print(f"\n[SEED {seed}] {'='*60}")
    print(" ".join(cmd))
    start = time.time()
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[SEED {seed}] {'OK' if rc == 0 else 'FAILED'} in {time.time()-start:.1f}s")
    return rc


def _read_eval(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def aggregate(seed_dirs):
    numeric = ["success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
               "mean_return", "std_return"]
    summary = {"num_seeds": len(seed_dirs)}
    vals = {k: [] for k in numeric}
    for d in seed_dirs:
        row = _read_eval(os.path.join(d, "logs", "eval_log.csv"))
        for k in numeric:
            try:
                v = float(row.get(k, np.nan))
                if np.isfinite(v):
                    vals[k].append(v)
            except Exception:
                pass
    for k, vs in vals.items():
        if vs:
            summary[f"{k}_mean"] = float(np.mean(vs))
            summary[f"{k}_std"] = float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    limits = (args.long_limit, args.lat_limit, args.vert_limit)
    cfg = build_config(args.base_config, limits, args.steps)
    cfg_path = output_root / "config_clamped.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    seed_dirs = []
    run_results = []
    for seed in range(args.seeds):
        seed_dir = str(output_root / f"seed_{seed}")
        seed_dirs.append(seed_dir)
        if args.skip_existing and (Path(seed_dir) / "checkpoints" / "best.pt").exists():
            print(f"[SEED {seed}] skipping (checkpoint exists)")
            run_results.append({"seed": seed, "status": "skipped"})
            continue
        rc = run_single(str(cfg_path), seed, seed_dir, args.device)
        run_results.append({"seed": seed, "status": "success" if rc == 0 else "failed"})

    summary = aggregate(seed_dirs)
    manifest = {
        "config": str(cfg_path),
        "limits_m": {"long": args.long_limit, "lat": args.lat_limit, "vert": args.vert_limit},
        "seeds": args.seeds,
        "run_results": run_results,
        "aggregate": summary,
    }
    with open(output_root / "clamped_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("CLAMPED VPP SUMMARY")
    print("="*60)
    print(f"Success rate: {summary.get('success_rate_mean', 'N/A')} "
          f"± {summary.get('success_rate_std', 'N/A')}")
    print(f"OOB rate: {summary.get('out_of_bounds_rate_mean', 'N/A')} "
          f"± {summary.get('out_of_bounds_rate_std', 'N/A')}")


if __name__ == "__main__":
    main()
