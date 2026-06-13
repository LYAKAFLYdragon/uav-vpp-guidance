#!/usr/bin/env python3
"""
Hyperparameter tuning sweeps for CR-PPO and Intentional PPO.

Runs a grid search over the hyperparameter spaces recommended in the
method-innovation validation report, using the quasi-realistic backend
(actuator dynamics + terminal boundary layer + potential-based shaping).

Usage:
    python scripts/run_method_innovation_tuning.py --sweep eta --seeds 3 --steps 20000
    python scripts/run_method_innovation_tuning.py --sweep complexity --seeds 3 --steps 20000
    python scripts/run_method_innovation_tuning.py --sweep both --seeds 3 --steps 20000 --parallel 2
"""

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml

from run_method_innovation_comparison import build_config


BASE_CONFIG = Path("config/method_innovation_comparison.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/method_innovation_tuning")


ETA_ACTOR_GRID = [1e-3, 1e-2, 1e-1, 1.0]
ETA_CRITIC_GRID = [1e-2, 1e-1, 1.0, 10.0]
COMPLEXITY_COEF_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
CR_N_BINS_GRID = [4, 8, 16]


def _intentional_overrides(eta_actor, eta_critic):
    return {
        "ppo": {
            "use_intentional_critic": True,
            "use_intentional_actor": True,
            "use_combat_aware_eta": True,
            "eta_critic": eta_critic,
            "eta_actor": eta_actor,
            "iu_eps": 1.0e-8,
            "beta_adv": 0.999,
        },
        "combat_aware": {
            "eta_actor": eta_actor,
            "eta_critic": eta_critic,
            "range_thresholds_m": [3000.0, 6000.0],
            "terminal_range_m": 1200.0,
            "aspect_threshold_deg": 30.0,
        },
    }


def _cr_ppo_overrides(complexity_coef, cr_n_bins):
    return {
        "ppo": {
            "complexity_coef": complexity_coef,
            "cr_n_bins": cr_n_bins,
        }
    }


def _make_tasks(sweep):
    tasks = []
    if sweep in ("eta", "both"):
        for eta_actor in ETA_ACTOR_GRID:
            for eta_critic in ETA_CRITIC_GRID:
                algo_name = f"intentional_etaA{eta_actor:.0e}_etaC{eta_critic:.0e}"
                tasks.append((
                    algo_name,
                    "intentional_ppo",
                    _intentional_overrides(eta_actor, eta_critic),
                ))
    if sweep in ("complexity", "both"):
        for complexity_coef in COMPLEXITY_COEF_GRID:
            for cr_n_bins in CR_N_BINS_GRID:
                algo_name = f"cr_ppo_c{complexity_coef:.0e}_bins{cr_n_bins}"
                tasks.append((
                    algo_name,
                    "cr_ppo",
                    _cr_ppo_overrides(complexity_coef, cr_n_bins),
                ))
    return tasks


def run_single(base_config, output_root, algo_name, algorithm, overrides, seed, total_timesteps, device, backend):
    output_dir = Path(output_root) / algo_name / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(base_config, overrides, total_timesteps, seed)
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
        "--algorithm", algorithm,
    ]
    print(f"\n[RUN] {algo_name} seed={seed} -> {output_dir}")
    start = __import__("time").time()
    rc = subprocess.run(cmd, cwd=Path(__file__).parent.parent).returncode
    elapsed = __import__("time").time() - start
    print(f"[DONE] {algo_name} seed={seed} in {elapsed:.1f}s (exit={rc})")
    return rc


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for method innovations")
    parser.add_argument("--sweep", type=str, default="eta", choices=["eta", "complexity", "both"])
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds per config")
    parser.add_argument("--steps", type=int, default=20000, help="Total timesteps per run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"])
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=str(BASE_CONFIG))
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a single config/seed for a quick sanity check")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    tasks = _make_tasks(args.sweep)

    if args.smoke:
        tasks = tasks[:1]
        args.seeds = 1
        args.steps = 512
        print("[SMOKE] Running a single tuning task")

    seed_list = list(range(args.seeds))
    job_list = [
        (args.config, str(output_root), name, algo, over, seed, args.steps, args.device, args.backend)
        for name, algo, over in tasks
        for seed in seed_list
    ]

    print(f"Sweep: {args.sweep} | Tasks: {len(tasks)} | Seeds: {len(seed_list)} | Steps: {args.steps}")
    print(f"Output root: {output_root}")

    if args.parallel <= 1:
        failures = 0
        for job in job_list:
            failures += (run_single(*job) != 0)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        failures = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single, *job): job for job in job_list}
            for future in as_completed(futures):
                failures += (future.result() != 0)

    print(f"\nAll tuning runs complete. Failures: {failures}/{len(job_list)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
