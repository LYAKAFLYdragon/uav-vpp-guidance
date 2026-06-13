#!/usr/bin/env python3
"""
Run a 5-seed comparison of method-innovation branches.

Algorithms:
  - baseline:      standard PPO
  - cr_ppo:        CR-PPO (complexity regularization)
  - intentional:   Intentional PPO (ICU + IAU + CAIS)
  - intentional_c: Intentional PPO with only ICU
  - intentional_a: Intentional PPO with only IAU

Usage:
    python scripts/run_method_innovation_comparison.py [--seeds 5] [--steps 50000] [--parallel N]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


BASE_CONFIG = Path("config/method_innovation_comparison.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/method_innovation_compare")
ALGORITHMS = {
    "baseline": {
        "algorithm": "ppo",
        "config_overrides": {},
    },
    "cr_ppo": {
        "algorithm": "cr_ppo",
        "config_overrides": {
            "ppo": {
                "complexity_coef": 1.0e-3,
                "cr_n_bins": 8,
            }
        },
    },
    "intentional": {
        "algorithm": "intentional_ppo",
        "config_overrides": {
            "ppo": {
                "use_intentional_critic": True,
                "use_intentional_actor": True,
                "use_combat_aware_eta": True,
                "eta_critic": 0.1,
                "eta_actor": 0.01,
                "iu_eps": 1.0e-8,
                "beta_adv": 0.999,
            },
            "combat_aware": {
                "eta_actor": 0.01,
                "eta_critic": 0.1,
                "range_thresholds_m": [3000.0, 6000.0],
                "terminal_range_m": 1200.0,
                "aspect_threshold_deg": 30.0,
            },
        },
    },
    "intentional_c": {
        "algorithm": "intentional_ppo",
        "config_overrides": {
            "ppo": {
                "use_intentional_critic": True,
                "use_intentional_actor": False,
                "use_combat_aware_eta": False,
                "eta_critic": 0.1,
                "eta_actor": 0.01,
                "iu_eps": 1.0e-8,
                "beta_adv": 0.999,
            }
        },
    },
    "intentional_a": {
        "algorithm": "intentional_ppo",
        "config_overrides": {
            "ppo": {
                "use_intentional_critic": False,
                "use_intentional_actor": True,
                "use_combat_aware_eta": False,
                "eta_critic": 0.1,
                "eta_actor": 0.01,
                "iu_eps": 1.0e-8,
                "beta_adv": 0.999,
            }
        },
    },
}


def _load_config_recursive(path, visited=None):
    """Load a YAML config and recursively resolve its ``includes``."""
    if visited is None:
        visited = set()
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"Cyclic include detected: {path}")
    visited.add(path)

    cfg = load_yaml_config(str(path))
    includes = cfg.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, _load_config_recursive(inc_full, visited))
    return merge_config(merged, cfg)


def build_config(base_path, overrides, total_timesteps=None, seed=0):
    cfg = _load_config_recursive(base_path)
    cfg = merge_config(cfg, overrides)
    if total_timesteps is not None:
        cfg.setdefault("ppo", {})["total_timesteps"] = total_timesteps
    cfg.setdefault("experiment", {})["seed"] = seed
    return cfg


def run_single(base_path, output_root, algo_key, seed, total_timesteps, device, backend):
    spec = ALGORITHMS[algo_key]
    output_dir = Path(output_root) / algo_key / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(base_path, spec["config_overrides"], total_timesteps, seed)
    cfg_path = output_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config", str(cfg_path),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        "--device", device,
        "--backend", backend,
        "--algorithm", spec["algorithm"],
    ]

    print(f"\n[RUN] {algo_key} seed={seed} -> {output_dir}")
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    print(f"[DONE] {algo_key} seed={seed} in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run method-innovation comparison")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--steps", type=int, default=50000, help="Total timesteps per run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"])
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--algos", type=str, default=None, help="Comma-separated algorithm keys to run")
    parser.add_argument("--config", type=str, default=str(BASE_CONFIG))
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT

    algorithms = list(ALGORITHMS.keys())
    if args.algos:
        algorithms = [a.strip() for a in args.algos.split(",") if a.strip() in ALGORITHMS]
        if not algorithms:
            raise ValueError(f"No valid algorithms in {args.algos}")

    seed_list = list(range(args.seeds))
    tasks = [(args.config, str(output_root), algo, seed, args.steps, args.device, args.backend)
             for algo in algorithms for seed in seed_list]

    print(f"Running comparison: {algorithms} x {len(seed_list)} seeds x {args.steps} steps")
    print(f"Output root: {output_root}")

    if args.parallel <= 1:
        failures = 0
        for t in tasks:
            rc = run_single(*t)
            failures += (rc != 0)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        failures = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single, *t): t for t in tasks}
            for future in as_completed(futures):
                rc = future.result()
                failures += (rc != 0)

    print(f"\nAll runs complete. Failures: {failures}/{len(tasks)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
