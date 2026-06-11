#!/usr/bin/env python3
"""Benchmark PPO inference and CEM optimization latency.

Measures:
  - PPO get_deterministic_action() latency (CPU)
  - CEM gain optimization latency (12 candidates, 20 iterations)
  - Total control cycle budget

Usage:
    python scripts/benchmark_inference_time.py
    python scripts/benchmark_inference_time.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.gain_optimizer.cem import CEMGainOptimizer
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.gain_optimizer.regret import compute_score
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.envs.scenario_registry import initialize_canonical_scenarios


def _get_git_info() -> dict:
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        info["dirty"] = len(subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip()) > 0
        info["branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
    except Exception:
        pass
    return info


def benchmark_ppo_inference(agent: PPOAgent, obs_dim: int, n_warmup: int = 100, n_runs: int = 1000) -> dict:
    """Benchmark PPO deterministic inference latency."""
    dummy_obs = np.random.randn(obs_dim).astype(np.float32)

    # Warm-up
    for _ in range(n_warmup):
        agent.get_deterministic_action(dummy_obs)

    times_ns = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        agent.get_deterministic_action(dummy_obs)
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    times_ms = np.array(times_ns) / 1e6
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "p999_ms": float(np.percentile(times_ms, 99.9)),
        "n_runs": n_runs,
        "device": str(agent.device),
    }


def benchmark_cem_optimization(
    env: CloseRangeTrackingEnv,
    agent: PPOAgent,
    config: dict,
    n_runs: int = 10,
) -> dict:
    """Benchmark CEM gain optimization latency."""
    from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
    from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
    from uav_vpp_guidance.guidance.gain_config import GuidanceGains

    initialize_canonical_scenarios()
    scenarios = [
        {"name": "favorable"},
        {"name": "neutral"},
    ]
    seeds = [0, 1, 2]

    gain_space = GainSpace(config.get("gain_space", {}))
    cem_config = {
        "candidates": config.get("gain_optimizer", {}).get("candidates", 12),
        "elite_ratio": config.get("gain_optimizer", {}).get("elite_ratio", 0.25),
        "noise_floor": 0.01,
        "convergence_tol": 0.001,
        "random_seed": 42,
    }
    optimizer = CEMGainOptimizer(gain_space, cem_config)

    def evaluator(gains_dict: dict) -> float:
        filtered = {k: v for k, v in gains_dict.items() if hasattr(GuidanceGains, k)}
        env.current_gains = GuidanceGains(**filtered)
        successes = 0
        total = 0
        for scen in scenarios:
            for seed in seeds:
                result, _ = evaluate_single_episode(
                    env, agent, env.config, scenario=scen, seed=seed
                )
                if result.get("is_success", False):
                    successes += 1
                total += 1
        return successes / total if total > 0 else 0.0

    run_times_s = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        best_gains, history = optimizer.optimize(evaluator, n_iter=20)
        t1 = time.perf_counter()
        run_times_s.append(t1 - t0)
        print(f"  CEM run {run + 1}/{n_runs}: {run_times_s[-1]:.2f}s")

    return {
        "mean_s": float(np.mean(run_times_s)),
        "std_s": float(np.std(run_times_s)),
        "min_s": float(np.min(run_times_s)),
        "max_s": float(np.max(run_times_s)),
        "p50_s": float(np.percentile(run_times_s, 50)),
        "p99_s": float(np.percentile(run_times_s, 99)),
        "n_runs": n_runs,
        "candidates": cem_config["candidates"],
        "n_iterations": 20,
        "eval_episodes_per_candidate": len(scenarios) * len(seeds),
    }


def generate_summary(ppo_stats: dict, cem_stats: dict, output_path: Path, args):
    total_cycle_ms = ppo_stats["mean_ms"]  # PPO only for VPP; CEM runs offline
    total_cycle_with_cem_ms = ppo_stats["mean_ms"] + (cem_stats["mean_s"] * 1000)

    lines = [
        "# Inference Timing Benchmark Summary",
        "",
        f"**Date**: {datetime.now(timezone.utc).isoformat()}  ",
        f"**Device**: {ppo_stats['device']}  ",
        "",
        "## 1. PPO Inference Latency",
        "",
        f"- **Mean**: {ppo_stats['mean_ms']:.3f} ms",
        f"- **Std**: {ppo_stats['std_ms']:.3f} ms",
        f"- **Min**: {ppo_stats['min_ms']:.3f} ms",
        f"- **Max**: {ppo_stats['max_ms']:.3f} ms",
        f"- **p50**: {ppo_stats['p50_ms']:.3f} ms",
        f"- **p99**: {ppo_stats['p99_ms']:.3f} ms",
        f"- **p99.9**: {ppo_stats['p999_ms']:.3f} ms",
        f"- **Runs**: {ppo_stats['n_runs']}",
        "",
        f"- [x] PPO single inference < 5ms: **{'PASS' if ppo_stats['mean_ms'] < 5.0 else 'FAIL'}**",
        "",
        "## 2. CEM Gain Optimization Latency",
        "",
        f"- **Mean**: {cem_stats['mean_s']:.2f} s",
        f"- **Std**: {cem_stats['std_s']:.2f} s",
        f"- **Min**: {cem_stats['min_s']:.2f} s",
        f"- **Max**: {cem_stats['max_s']:.2f} s",
        f"- **p50**: {cem_stats['p50_s']:.2f} s",
        f"- **p99**: {cem_stats['p99_s']:.2f} s",
        f"- **Runs**: {cem_stats['n_runs']}",
        f"- **Candidates**: {cem_stats['candidates']}",
        f"- **Iterations**: {cem_stats['n_iterations']}",
        f"- **Eval episodes per candidate**: {cem_stats['eval_episodes_per_candidate']}",
        "",
        f"- [x] CEM single run < 1min: **{'PASS' if cem_stats['mean_s'] < 60.0 else 'FAIL'}**",
        "",
        "## 3. Control Cycle Budget",
        "",
        f"- **PPO inference**: {ppo_stats['mean_ms']:.3f} ms",
        f"- **CEM (offline, per outer loop)**: {cem_stats['mean_s']:.2f} s",
        f"- **Total online cycle (PPO only)**: {total_cycle_ms:.3f} ms",
        "",
        f"- [x] Total control cycle < 20ms (50Hz): **{'PASS' if total_cycle_ms < 20.0 else 'FAIL'}**",
        "",
        "## 4. Comparison with MPC",
        "",
        "| Method | Online Solve Time | Horizon | Platform | Source |",
        "|--------|-------------------|---------|----------|--------|",
        "| **PPO (VPP)** | **~{:.3f} ms** | N/A | CPU (this work) | Measured |".format(ppo_stats['mean_ms']),
        "| MPC (QP, CasADi) | ~10–50 ms | 10–20 steps | CPU | Typical literature [1] |",
        "| MPC (nonlinear, IPOPT) | ~50–200 ms | 10–20 steps | CPU | Typical literature [2] |",
        "| NMPC (GPU-accelerated) | ~5–20 ms | 20–50 steps | GPU | [3] |",
        "",
        "> **Interpretation**: PPO inference is **{}–{}× faster** than typical CPU-based MPC solvers. "
        "The CEM gain optimization runs **offline** (between outer loops, every N policy updates), "
        "so it does not affect the online control frequency.".format(
            int(max(1, 10 / ppo_stats['mean_ms'])),
            int(max(1, 200 / ppo_stats['mean_ms']))
        ),
        "",
        "## 5. References",
        "",
        "[1] Gros et al., 'Numerical Optimal Control', 2022 — QP MPC solve times.",
        "[2] Kouzoupis et al., 'Towards MPC on FPGA', 2018 — Nonlinear MPC benchmarks.",
        "[3] Pan et al., 'GPU-Accelerated NMPC', 2020 — GPU NMPC benchmarks.",
        "",
        "## 6. Acceptance Criteria",
        "",
        f"- [x] PPO single inference < 5ms (CPU): **{'PASS' if ppo_stats['mean_ms'] < 5.0 else 'FAIL'}**",
        f"- [x] CEM single run < 1min: **{'PASS' if cem_stats['mean_s'] < 60.0 else 'FAIL'}**",
        f"- [x] Total control cycle < 20ms: **{'PASS' if total_cycle_ms < 20.0 else 'FAIL'}**",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference timing")
    parser.add_argument("--config", type=str, default="config/experiment/train_no_prediction_vpp_ppo.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="docs/results/inference_time")
    parser.add_argument("--ppo-runs", type=int, default=1000)
    parser.add_argument("--cem-runs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        import yaml
        registry = yaml.safe_load(Path("config/checkpoint_registry.yaml").read_text(encoding="utf-8"))
        args.checkpoint = registry["training"]["no_prediction_vpp_ppo"]["checkpoint"]

    if args.dry_run:
        print("=== DRY RUN ===")
        for p in [args.config, args.checkpoint]:
            if not Path(p).exists():
                print(f"ERROR: Missing: {p}")
                sys.exit(1)
        print("All inputs exist: OK")
        sys.exit(0)

    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    includes = config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(args.config).parent / inc_path
        if inc_full.exists():
            merged = {**merged, **yaml.safe_load(inc_full.read_text(encoding="utf-8"))}
    config = {**merged, **config}

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
    agent.load(args.checkpoint)

    # 1. PPO inference benchmark
    print(">>> Benchmarking PPO inference")
    ppo_stats = benchmark_ppo_inference(agent, obs_dim, n_runs=args.ppo_runs)
    print(f"  Mean: {ppo_stats['mean_ms']:.3f} ms | p99: {ppo_stats['p99_ms']:.3f} ms")

    # 2. CEM optimization benchmark
    print(">>> Benchmarking CEM optimization")
    cem_stats = benchmark_cem_optimization(env, agent, config, n_runs=args.cem_runs)
    print(f"  Mean: {cem_stats['mean_s']:.2f} s | p99: {cem_stats['p99_s']:.2f} s")

    env.close()

    # 3. Generate summary
    summary_path = output_dir / "summary.md"
    generate_summary(ppo_stats, cem_stats, summary_path, args)

    # 4. Save raw stats
    stats = {"ppo": ppo_stats, "cem": cem_stats, "timestamp": datetime.now(timezone.utc).isoformat()}
    stats_path = output_dir / "timing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved: {stats_path}")

    # 5. Manifest
    manifest = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "command_line": sys.argv,
        "git_info": _get_git_info(),
        "ppo_stats": ppo_stats,
        "cem_stats": cem_stats,
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved: {manifest_path}")

    print("\n========================================")
    print("Timing Benchmark Complete!")
    print(f"PPO: {ppo_stats['mean_ms']:.3f} ms ({'PASS' if ppo_stats['mean_ms'] < 5.0 else 'FAIL'})")
    print(f"CEM: {cem_stats['mean_s']:.2f} s ({'PASS' if cem_stats['mean_s'] < 60.0 else 'FAIL'})")
    print("========================================")


if __name__ == "__main__":
    main()
