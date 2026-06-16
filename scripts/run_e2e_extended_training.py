#!/usr/bin/env python3
"""
Extended-training study for the End-to-End DRL baseline.

Addresses reviewer concern #2: even if E2E fails at 200k steps, it must be
evaluated at 500k, 1M, and 2M steps (with otherwise identical hyperparameters)
before concluding that hierarchy is necessary.

Usage:
    python scripts/run_e2e_extended_training.py \
        --config config/experiment/train_end_to_end_ppo.yaml \
        --seeds 5 --align-reward
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


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
    parser = argparse.ArgumentParser(description="E2E extended training study")
    parser.add_argument("--config", type=str,
                        default="config/experiment/train_end_to_end_ppo.yaml")
    parser.add_argument("--output-root", type=str,
                        default="outputs/e2e_extended_training")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+",
                        default=[500000, 1000000, 2000000])
    parser.add_argument("--align-reward", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def prepare_config(config_path: str, steps: int, align_reward: bool) -> Path:
    import yaml
    cfg = load_yaml_config(config_path)
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = Path(config_path).parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    cfg = merge_config(merged, cfg)
    cfg.setdefault("ppo", {})["total_timesteps"] = steps
    cfg.setdefault("evaluation", {})["domain_rand_scale"] = 0.15
    if align_reward:
        cfg["reward"] = merge_config(cfg.get("reward", {}), VPP_REWARD)

    out_dir = Path("outputs/e2e_extended_training/configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e2e_steps_{steps}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return out_path


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    manifest = {"config": args.config, "seeds": args.seeds,
                "align_reward": args.align_reward, "runs": []}

    for steps in args.steps:
        cfg_path = prepare_config(args.config, steps, args.align_reward)
        exp_name = f"e2e_steps_{steps}"
        print(f"\n{'='*60}")
        print(f"[EXTENDED] {steps} steps")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            "scripts/run_end_to_end_ppo_multiseed.py",
            "--config", str(cfg_path),
            "--seeds", str(args.seeds),
            "--output-root", str(output_root),
            "--exp-name", exp_name,
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        if args.skip_existing:
            cmd.append("--skip-existing")

        start = time.time()
        rc = subprocess.run(cmd, check=False).returncode
        elapsed = time.time() - start
        manifest["runs"].append({"steps": steps, "elapsed_s": elapsed, "returncode": rc})

        if rc == 0:
            agg_path = output_root / exp_name / "multiseed_manifest.json"
            if agg_path.exists():
                with open(agg_path, "r", encoding="utf-8") as f:
                    agg = json.load(f)
                summary = agg.get("aggregate", {})
                results.append({
                    "steps": steps,
                    "success_rate_mean": summary.get("success_rate_mean", np.nan),
                    "success_rate_std": summary.get("success_rate_std", np.nan),
                    "mean_return_mean": summary.get("mean_return_mean", np.nan),
                    "mean_return_std": summary.get("mean_return_std", np.nan),
                })

    if results:
        report_path = output_root / "extended_training_summary.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SAVED] {report_path}")

    manifest_path = output_root / "extended_training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {manifest_path}")


if __name__ == "__main__":
    main()
