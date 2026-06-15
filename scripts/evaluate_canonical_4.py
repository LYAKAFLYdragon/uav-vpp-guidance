#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the four canonical scenarios.

Usage:
    python scripts/evaluate_canonical_4.py \
        --checkpoint outputs/experiments/p0p1_fixed_no_vpp_s0/checkpoints/best.pt \
        --method no_vpp --output results.json

Methods:
    vpp      - use the policy to output VPP offsets (default)
    no_vpp   - zero-offset direct tracking
    e2e      - end-to-end policy outputs low-level commands
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_experiment_config(config_path):
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def make_env(method, backend="simple"):
    # Use the canonical no-prediction config as the base.
    config = load_experiment_config(
        PROJECT_ROOT / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml"
    )
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"

    if method == "e2e":
        config["end_to_end"] = {"enabled": True}
        config["virtual_point"]["enabled"] = False
    elif method == "no_vpp":
        config["virtual_point"]["enabled"] = True
        config["virtual_point"]["mode"] = "zero_offset"
    else:  # vpp
        config["virtual_point"]["enabled"] = True
        config["virtual_point"].pop("mode", None)
    return CloseRangeTrackingEnv(config)


def evaluate_checkpoint(checkpoint_path, method, n_eps=30, backend="simple"):
    env = make_env(method, backend=backend)
    obs = env.reset(seed=0, scenario="favorable")
    obs_dim = obs["observation_vector"].shape[0]
    act_dim = 3

    agent = None
    if method != "no_vpp":
        agent = PPOAgent(obs_dim, act_dim, env.config)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        agent.network.load_state_dict(ckpt["network_state_dict"])
        agent.network.eval()

    results = {}
    for scenario in ["favorable", "neutral", "disadvantage", "challenging"]:
        stats = {"success": 0, "crash": 0, "oob": 0, "timeout": 0, "total": 0}
        for ep in range(n_eps):
            obs = env.reset(seed=ep + 1, scenario=scenario)
            done = False
            while not done:
                if agent is None:
                    action = np.zeros(act_dim, dtype=np.float32)
                else:
                    action = agent.select_action(
                        obs["observation_vector"], deterministic=True, store=False
                    )[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            term = info.get("termination_info", {})
            reason = term.get("reason", "unknown")
            stats["total"] += 1
            if reason == "success":
                stats["success"] += 1
            elif reason == "crash":
                stats["crash"] += 1
            elif reason == "out_of_bounds":
                stats["oob"] += 1
            elif reason == "timeout":
                stats["timeout"] += 1
            else:
                stats["timeout"] += 1

        results[scenario] = {
            k: (v / stats["total"] * 100 if k != "total" else v)
            for k, v in stats.items()
        }

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on canonical 4 scenes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default="vpp",
                        choices=["vpp", "no_vpp", "e2e"])
    parser.add_argument("--n-eps", type=int, default=30)
    parser.add_argument("--backend", type=str, default="simple",
                        choices=["simple", "jsbsim"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.method != "no_vpp" and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    results = evaluate_checkpoint(
        args.checkpoint, args.method, n_eps=args.n_eps, backend=args.backend
    )

    summary = {
        "checkpoint": args.checkpoint,
        "method": args.method,
        "n_eps": args.n_eps,
        "backend": args.backend,
        "per_scenario": results,
    }

    # Print table
    print(f"\n=== Canonical 4 evaluation: {args.method} ===")
    print(f"{'Scenario':<14} {'Success':>8} {'Crash':>8} {'OOB':>8} {'Timeout':>8}")
    overall_success = []
    for scen, stat in results.items():
        print(
            f"{scen:<14} {stat['success']:>7.1f}% {stat['crash']:>7.1f}% "
            f"{stat['oob']:>7.1f}% {stat['timeout']:>7.1f}%"
        )
        overall_success.append(stat["success"])
    summary["overall_success"] = float(np.mean(overall_success))
    print(f"\nOverall success: {summary['overall_success']:.1f}%")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
