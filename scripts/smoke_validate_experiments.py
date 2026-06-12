#!/usr/bin/env python3
"""
Smoke validation gate for all training experiments.

Runs a short smoke test for each experiment type, parses the final eval metrics,
and reports PASS/FAIL against configurable thresholds. Use this before launching
large-scale multi-seed training on any machine.

Usage:
    python scripts/smoke_validate_experiments.py
    python scripts/smoke_validate_experiments.py --sr-threshold 0.20 --oob-threshold 0.80
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

EXPERIMENTS = [
    {
        "id": "A1",
        "name": "baseline_vpp_los",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo.yaml",
    },
    {
        "id": "A2",
        "name": "end_to_end_drl",
        "script": "python -m uav_vpp_guidance.training.train_end_to_end_ppo",
        "config": "config/experiment/train_end_to_end_ppo.yaml",
    },
    {
        "id": "A3",
        "name": "no_vpp_ablation",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo.yaml",
        "override": {"virtual_point": {"offset": [0.0, 0.0, 0.0]}},
    },
    {
        "id": "B1",
        "name": "constrained_vpp",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/stage6f5_feasible_geometry_constrained.yaml",
    },
    {
        "id": "B2",
        "name": "curriculum_learning",
        "script": "python scripts/train_curriculum_ppo.py",
        "config": "config/experiment/train_curriculum_ppo.yaml",
    },
    {
        "id": "B3",
        "name": "hybrid_mode_switch",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_hybrid_mode_switch.yaml",
    },
    {
        "id": "E1",
        "name": "domain_randomization_train",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo_domain_rand.yaml",
    },
    {
        "id": "E2",
        "name": "domain_randomization_control",
        "script": "python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo",
        "config": "config/experiment/train_no_prediction_vpp_ppo.yaml",
    },
]


def make_override_config(base_config, overrides):
    """Write a temporary config with overrides applied."""
    import yaml
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def merge(base, over):
        for k, v in over.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                merge(base[k], v)
            else:
                base[k] = v

    merge(cfg, overrides)
    tmp = ROOT / "outputs" / "smoke_validate" / f"config_override_{os.urandom(4).hex()}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp


def run_smoke(exp, device="cpu"):
    """Run a single smoke test and return parsed metrics."""
    outdir = ROOT / "outputs" / "smoke_validate" / exp["name"]
    shutil.rmtree(outdir, ignore_errors=True)

    config_path = ROOT / exp["config"]
    tmp_config = None
    if "override" in exp:
        tmp_config = make_override_config(config_path, exp["override"])
        config_path = tmp_config

    cmd = (
        f"{exp['script']} --config {config_path} --smoke --device {device} --output-dir {outdir}"
    )
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    result = {
        "id": exp["id"],
        "name": exp["name"],
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "eval_sr": None,
        "eval_crash": None,
        "eval_oob": None,
    }

    eval_files = list((outdir / "logs").glob("eval_log.csv")) if (outdir / "logs").exists() else []
    if eval_files:
        with open(eval_files[0], "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            result["eval_sr"] = float(last.get("success_rate", 0))
            result["eval_crash"] = float(last.get("crash_rate", 0))
            result["eval_oob"] = float(last.get("out_of_bounds_rate", 0))

    if tmp_config:
        try:
            os.remove(tmp_config)
        except OSError:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke validation gate for training experiments.")
    parser.add_argument("--sr-threshold", type=float, default=0.20, help="Minimum eval success rate to pass.")
    parser.add_argument("--oob-threshold", type=float, default=0.80, help="Maximum eval OOB rate to pass.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for smoke tests (cpu/cuda).")
    parser.add_argument("--output", type=str, default="outputs/smoke_validate/report.json", help="Report path.")
    args = parser.parse_args()

    report_dir = Path(args.output).parent
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SMOKE VALIDATION GATE")
    print(f"Thresholds: eval SR >= {args.sr_threshold}, eval OOB <= {args.oob_threshold}")
    print("=" * 70)

    results = []
    for exp in EXPERIMENTS:
        print(f"\n[{exp['id']}] {exp['name']}")
        res = run_smoke(exp, device=args.device)
        sr = res["eval_sr"] if res["eval_sr"] is not None else -1.0
        oob = res["eval_oob"] if res["eval_oob"] is not None else 1.0
        rc = res["returncode"]
        passed = (rc == 0) and (sr >= args.sr_threshold) and (oob <= args.oob_threshold)
        res["passed"] = passed
        results.append(res)
        status = "PASS" if passed else "FAIL"
        print(
            f"  {status} | rc={rc} | SR={sr:.3f} | OOB={oob:.3f} | crash={res['eval_crash']:.3f} | {res['elapsed_seconds']:.1f}s"
        )

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_count}/{total} passed")
    print("=" * 70)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to {args.output}")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
