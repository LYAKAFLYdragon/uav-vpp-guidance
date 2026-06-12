#!/usr/bin/env python3
"""
Flyable-baseline smoke gate.

Validates that the default (curriculum + balanced-scenario) config can train a
no-prediction VPP policy that reaches a minimum success rate under the *strict*
success criteria used in final evaluation.  This gate should pass before any
large-scale multi-seed experiment is resumed.

Usage:
    python scripts/smoke_gate.py
    python scripts/smoke_gate.py --budget 20000 --seeds 5 --device cpu
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_CONFIG = ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
DEFAULT_FIXED_GAIN_CONFIG = ROOT / "config" / "experiment" / "fixed_gain_vpp.yaml"
DEFAULT_BUDGET = 10_000
DEFAULT_SEEDS = 3


def load_config(path: Path) -> dict:
    """Load a YAML config, merging includes like the training entry point."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_path = path.parent / inc
        if inc_path.exists():
            with open(inc_path, "r", encoding="utf-8") as f:
                merged = _deep_merge(merged, yaml.safe_load(f))
    return _deep_merge(merged, cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge (base is mutated)."""
    if override is None:
        return base
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def make_temporary_config(base_config: Path, budget: int) -> Path:
    """Create a temporary config with reduced training budget."""
    cfg = load_config(base_config)
    cfg.setdefault("ppo", {})
    cfg["ppo"]["total_timesteps"] = budget
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["eval_interval"] = budget
    cfg["evaluation"]["save_trajectories"] = False
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["save_interval"] = budget

    tmp_dir = ROOT / "outputs" / "smoke_gate" / "configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"smoke_gate_{budget}_{os.urandom(4).hex()}.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


def parse_eval_log(log_path: Path) -> dict:
    """Read the last eval row from eval_log.csv."""
    if not log_path.exists():
        return {}
    with open(log_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "success_rate": float(last.get("success_rate", 0.0)),
        "crash_rate": float(last.get("crash_rate", 0.0)),
        "out_of_bounds_rate": float(last.get("out_of_bounds_rate", 0.0)),
        "timeout_rate": float(last.get("timeout_rate", 0.0)),
        "mean_return": float(last.get("mean_return", 0.0)),
        "step": int(last.get("step", 0)),
    }


def run_fixed_gain_eval(fg_config: Path, episodes: int, seeds: list) -> dict:
    """Run deterministic zero-action fixed-gain evaluation (L1)."""
    outdir = ROOT / "outputs" / "smoke_gate" / "l1_fixed_gain"
    shutil.rmtree(outdir, ignore_errors=True)

    seeds_str = " ".join(str(s) for s in seeds)
    cmd = (
        f"python -m uav_vpp_guidance.training.train_fixed_gain "
        f"--config {fg_config} --eval-only "
        f"--eval-episodes {episodes} --eval-seeds {seeds_str} "
        f"--output-dir {outdir}"
    )

    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    # The eval-only path prints metrics to stdout but does not write eval_log.csv.
    metrics = _parse_eval_stdout(proc.stdout)

    return {
        "stage": "L1_fixed_gain",
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "eval": metrics,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
    }


def _parse_eval_stdout(stdout: str) -> dict:
    """Parse the 'Eval Return: ... | Success: ...' line from train_fixed_gain --eval-only."""
    for line in stdout.splitlines():
        if "Eval Return:" in line and "Success:" in line:
            try:
                parts = line.split("|")
                mean_return = float(parts[0].split(":")[1].split("±")[0].strip())
                success_rate = float(parts[1].split(":")[1].strip().replace("%", "")) / 100.0
                crash_rate = float(parts[2].split(":")[1].strip().replace("%", "")) / 100.0
                oob_rate = float(parts[3].split(":")[1].strip().replace("%", "")) / 100.0
                return {
                    "success_rate": success_rate,
                    "crash_rate": crash_rate,
                    "out_of_bounds_rate": oob_rate,
                    "mean_return": mean_return,
                }
            except Exception:
                pass
    return {}


def run_single_seed(base_config: Path, seed: int, budget: int, device: str) -> dict:
    """Run one smoke-gate seed and return parsed metrics."""
    outdir = ROOT / "outputs" / "smoke_gate" / f"seed_{seed}"
    shutil.rmtree(outdir, ignore_errors=True)

    tmp_config = make_temporary_config(base_config, budget)
    cmd = (
        f"python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo "
        f"--config {tmp_config} --seed {seed} --device {device} --output-dir {outdir}"
    )

    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    try:
        os.remove(tmp_config)
    except OSError:
        pass

    log_path = outdir / "logs" / "eval_log.csv"
    metrics = parse_eval_log(log_path)

    return {
        "seed": seed,
        "budget": budget,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "eval": metrics,
        "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Flyable-baseline smoke gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base experiment config.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Training steps per seed.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of random seeds.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device.")
    parser.add_argument("--sr-threshold", type=float, default=0.25, help="Minimum mean eval success rate.")
    parser.add_argument("--max-crash-rate", type=float, default=0.50, help="Maximum mean eval crash rate.")
    parser.add_argument("--max-oob-rate", type=float, default=0.50, help="Maximum mean eval OOB rate.")
    parser.add_argument("--fixed-gain-config", type=Path, default=DEFAULT_FIXED_GAIN_CONFIG, help="Fixed-gain baseline config.")
    parser.add_argument("--fixed-gain-episodes", type=int, default=30, help="Episodes for L1 fixed-gain eval.")
    parser.add_argument("--fixed-gain-seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds for L1 fixed-gain eval.")
    parser.add_argument("--fixed-gain-sr-threshold", type=float, default=0.50, help="L1 minimum success rate.")
    parser.add_argument("--fixed-gain-max-crash-rate", type=float, default=0.25, help="L1 maximum crash rate.")
    parser.add_argument("--skip-l1", action="store_true", help="Skip L1 fixed-gain evaluation.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "smoke_gate" / "report.json")
    args = parser.parse_args()

    print("=" * 70)
    print("FLYABLE BASELINE SMOKE GATE")
    print(f"Config: {args.config}")
    print(f"Budget: {args.budget} steps/seed  |  Seeds: {args.seeds}  |  Device: {args.device}")
    print(f"L3 pass thresholds: mean SR >= {args.sr_threshold:.0%}, crash <= {args.max_crash_rate:.0%}, OOB <= {args.max_oob_rate:.0%}")
    if not args.skip_l1:
        print(f"L1 pass thresholds: SR >= {args.fixed_gain_sr_threshold:.0%}, crash <= {args.fixed_gain_max_crash_rate:.0%}")
    print("=" * 70)

    l1_result = None
    if not args.skip_l1:
        print("\n[L1] Deterministic fixed-gain (zero-action) evaluation...")
        l1_result = run_fixed_gain_eval(
            args.fixed_gain_config,
            args.fixed_gain_episodes,
            args.fixed_gain_seeds,
        )
        l1_sr = l1_result["eval"].get("success_rate", -1.0)
        l1_crash = l1_result["eval"].get("crash_rate", 1.0)
        l1_oob = l1_result["eval"].get("out_of_bounds_rate", 1.0)
        print(
            f"  rc={l1_result['returncode']} | SR={l1_sr:.3f} | crash={l1_crash:.3f} | "
            f"OOB={l1_oob:.3f} | {l1_result['elapsed_seconds']:.1f}s"
        )
        if l1_result["returncode"] != 0:
            print("  stdout tail:\n", l1_result["stdout_tail"])
            print("  stderr tail:\n", l1_result["stderr_tail"])

    results = []
    for seed in range(args.seeds):
        print(f"\n[Seed {seed}] Starting {args.budget}-step training...")
        res = run_single_seed(args.config, seed, args.budget, args.device)
        sr = res["eval"].get("success_rate", -1.0)
        crash = res["eval"].get("crash_rate", 1.0)
        rc = res["returncode"]
        print(
            f"  rc={rc} | SR={sr:.3f} | crash={crash:.3f} | "
            f"OOB={res['eval'].get('out_of_bounds_rate', -1):.3f} | {res['elapsed_seconds']:.1f}s"
        )
        if rc != 0:
            print("  stdout tail:\n", res["stdout_tail"])
            print("  stderr tail:\n", res["stderr_tail"])
        results.append(res)

    srs = [r["eval"].get("success_rate", 0.0) for r in results if r["returncode"] == 0 and r["eval"]]
    crashes = [r["eval"].get("crash_rate", 1.0) for r in results if r["returncode"] == 0 and r["eval"]]
    oobs = [r["eval"].get("out_of_bounds_rate", 1.0) for r in results if r["returncode"] == 0 and r["eval"]]
    mean_sr = float(sum(srs) / len(srs)) if srs else 0.0
    mean_crash = float(sum(crashes) / len(crashes)) if crashes else 1.0
    mean_oob = float(sum(oobs) / len(oobs)) if oobs else 1.0
    all_ok = all(r["returncode"] == 0 for r in results)

    l3_passed = (
        all_ok
        and mean_sr >= args.sr_threshold
        and mean_crash <= args.max_crash_rate
        and mean_oob <= args.max_oob_rate
    )

    l1_passed = True
    if not args.skip_l1 and l1_result is not None:
        l1_passed = (
            l1_result["returncode"] == 0
            and l1_result["eval"].get("success_rate", 0.0) >= args.fixed_gain_sr_threshold
            and l1_result["eval"].get("crash_rate", 1.0) <= args.fixed_gain_max_crash_rate
        )

    passed = l3_passed and l1_passed

    report = {
        "passed": passed,
        "l1_passed": l1_passed,
        "l3_passed": l3_passed,
        "config": str(args.config),
        "budget": args.budget,
        "seeds": args.seeds,
        "device": args.device,
        "mean_success_rate": mean_sr,
        "mean_crash_rate": mean_crash,
        "mean_oob_rate": mean_oob,
        "sr_threshold": args.sr_threshold,
        "max_crash_rate": args.max_crash_rate,
        "max_oob_rate": args.max_oob_rate,
        "l1_result": l1_result,
        "all_ran_ok": all_ok,
        "per_seed": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    if not args.skip_l1:
        print(
            f"L1 fixed-gain:  SR={l1_result['eval'].get('success_rate', -1):.2%}  "
            f"crash={l1_result['eval'].get('crash_rate', -1):.2%}  "
            f"({'PASS' if l1_passed else 'FAIL'})"
        )
    print(
        f"L3 RL baseline: SR={mean_sr:.2%}  crash={mean_crash:.2%}  OOB={mean_oob:.2%}  "
        f"({'PASS' if l3_passed else 'FAIL'})"
    )
    if passed:
        print("OVERALL PASS")
    else:
        print("OVERALL FAIL — Do not resume large-scale experiments until this gate passes.")
    print(f"Report: {args.output}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
