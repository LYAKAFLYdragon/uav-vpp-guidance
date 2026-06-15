#!/usr/bin/env python3
"""Pilot experiment: VPP vs No-VPP vs E2E against a maneuvering target.

Usage (quick smoke):
    python scripts/run_maneuver_target_pilot.py --smoke

Usage (full pilot, ~1-2 h per method on CPU):
    python scripts/run_maneuver_target_pilot.py --timesteps 10000 --episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


METHODS = {
    "vpp": {
        "config": "config/experiment/maneuver_target_vpp_pilot.yaml",
        "name": "VPP",
    },
    "no_vpp": {
        "config": "config/experiment/maneuver_target_no_vpp_pilot.yaml",
        "name": "No-VPP",
    },
    "e2e": {
        "config": "config/experiment/maneuver_target_e2e_pilot.yaml",
        "name": "E2E",
    },
}


def load_experiment_config(config_path: str) -> dict:
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, base_config)


def train_method(key: str, output_dir: str, device: str, smoke: bool, timesteps: int | None) -> Path:
    cfg_path = METHODS[key]["config"]
    print(f"\n{'='*60}")
    print(f"Training {METHODS[key]['name']} -> {output_dir}")
    print(f"{'='*60}")
    cmd = [
        sys.executable,
        "scripts/train_curriculum_ppo.py",
        "--config", cfg_path,
        "--output-dir", output_dir,
        "--backend", "jsbsim",
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")
    if timesteps is not None:
        cmd.extend(["--total-timesteps", str(timesteps)])
    subprocess.run(cmd, check=True)
    return Path(output_dir) / "checkpoints" / "best.pt"


def evaluate_method(key: str, checkpoint: Path, episodes: int) -> dict:
    cfg_path = METHODS[key]["config"]
    config = load_experiment_config(cfg_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(str(checkpoint))

    scenarios = config.get("scenarios", {})
    per_scenario = {}
    all_records = []

    for scen_name, scenario in scenarios.items():
        records = []
        for ep in range(episodes):
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scenario,
                seed=ep,
                save_trajectory=False,
                method_name=key,
            )
            records.append(result)
        per_scenario[scen_name] = _summarize(records)
        all_records.extend(records)

    env.close()
    return {"method": METHODS[key]["name"], "per_scenario": per_scenario, "overall": _summarize(all_records)}


def _summarize(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {}
    successes = sum(1 for r in records if r.get("is_success"))
    crashes = sum(1 for r in records if r.get("is_crash"))
    timeouts = sum(1 for r in records if r.get("reason") == "timeout")
    returns = [r.get("return", np.nan) for r in records]
    min_ranges = [r.get("min_range_m", np.nan) for r in records]
    final_ranges = [r.get("final_range_m", np.nan) for r in records]
    return {
        "n": n,
        "success_rate": successes / n,
        "crash_rate": crashes / n,
        "timeout_rate": timeouts / n,
        "mean_return": float(np.nanmean(returns)),
        "std_return": float(np.nanstd(returns)),
        "mean_min_range_m": float(np.nanmean(min_ranges)),
        "mean_final_range_m": float(np.nanmean(final_ranges)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None, help="Override ppo.total_timesteps (e.g. 50000)")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scenario for eval")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (512 steps)")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate existing checkpoints")
    args = parser.parse_args()

    output_root = Path("outputs/experiments")
    results = {}

    for key, info in METHODS.items():
        exp_name = f"maneuver_target_{key}_pilot_s0"
        output_dir = str(output_root / exp_name)
        checkpoint = Path(output_dir) / "checkpoints" / "best.pt"

        if not args.skip_train and (args.retrain or not checkpoint.exists()):
            train_method(key, output_dir, args.device, args.smoke, args.timesteps)
        else:
            if args.skip_train and not checkpoint.exists():
                print(f"WARNING: checkpoint not found for {key}, skipping eval")
                continue
            print(f"\nUsing existing checkpoint for {info['name']}: {checkpoint}")

        print(f"\nEvaluating {info['name']} ...")
        results[key] = evaluate_method(key, checkpoint, args.episodes)

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Method':<12} {'Scenario':<14} {'N':>4} {'Success%':>10} {'Crash%':>8} {'Timeout%':>10} {'MeanReturn':>12} {'MeanMinRng':>12} {'MeanFinRng':>12}")
    print("-" * 100)
    for key, res in results.items():
        for scen, stats in res["per_scenario"].items():
            print(
                f"{res['method']:<12} {scen:<14} {stats['n']:>4} "
                f"{stats['success_rate']*100:>9.1f}% {stats['crash_rate']*100:>7.1f}% "
                f"{stats['timeout_rate']*100:>9.1f}% {stats['mean_return']:>11.1f} "
                f"{stats['mean_min_range_m']:>11.0f} {stats['mean_final_range_m']:>11.0f}"
            )
        ov = res["overall"]
        print(
            f"{res['method']:<12} {'OVERALL':<14} {ov['n']:>4} "
            f"{ov['success_rate']*100:>9.1f}% {ov['crash_rate']*100:>7.1f}% "
            f"{ov['timeout_rate']*100:>9.1f}% {ov['mean_return']:>11.1f} "
            f"{ov['mean_min_range_m']:>11.0f} {ov['mean_final_range_m']:>11.0f}"
        )
        print("-" * 100)

    # Save results
    out_dir = Path("outputs/maneuver_target_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
