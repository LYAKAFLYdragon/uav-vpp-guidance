#!/usr/bin/env python3
"""
Stage 6H.0-R: Replay historical Stage 6F.5A scenarios with current code + checkpoint.

Goal: Determine if the same scenarios that succeeded in 6F.5A still succeed
with the current codebase and available checkpoint.

This answers the regression question: did code changes or checkpoint mismatch
cause the 6H.0 baseline search to find zero candidates?

Outputs:
    docs/results/stage6h0r_replay_stage6f_results.json
    docs/results/stage6h0r_replay_stage6f_results.md
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_checkpoint(config):
    """Resolve checkpoint path, accounting for _seed0 variant."""
    methods = config.get("methods", {})
    for name, method in methods.items():
        ckpt = method.get("checkpoint")
        if not ckpt:
            continue
        ckpt_path = PROJECT_ROOT / ckpt
        if ckpt_path.exists():
            return str(ckpt_path), name
        # Try _seed0 variant
        parts = ckpt.split("/")
        exp_name = parts[2]  # e.g. no_prediction_vpp_ppo
        seed0_name = f"{exp_name}_seed0"
        parts[2] = seed0_name
        alt_path = PROJECT_ROOT / "/".join(parts)
        if alt_path.exists():
            return str(alt_path), name
    return None, None


def run_scenario_episode(scenario, full_config, checkpoint_path, seed=0):
    """Run a single episode for a scenario using current CloseRangeTrackingEnv."""
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    from uav_vpp_guidance.agents.ppo_agent import PPOAgent
    from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
    from uav_vpp_guidance.utils.config import merge_config

    # Build method-specific config (no_prediction baseline)
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))

    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load(checkpoint_path)

    try:
        result, _ = evaluate_single_episode(
            env, agent, method_config, scenario=scenario, seed=seed,
            save_trajectory=False, method_name="no_prediction",
        )
    finally:
        env.close()

    return {
        "success": bool(result.get("is_success", False)),
        "failure_reason": result.get("reason", "unknown"),
        "final_range_m": float(result.get("final_range_m", float("nan"))),
        "final_ata_deg": float(result.get("final_ata_deg", float("nan"))),
        "min_range_m": float(result.get("min_range_m", float("nan"))),
        "steps": int(result.get("length", 0)),
    }


def run_episodes_for_scenario(scenario_config, full_config, checkpoint_path, n_eps=5, base_seed=1000):
    """Run multiple episodes for one scenario."""
    results = []
    for i in range(n_eps):
        r = run_scenario_episode(scenario_config, full_config, checkpoint_path, seed=base_seed + i)
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eps", type=int, default=5, help="Episodes per scenario")
    parser.add_argument("--output-dir", type=str, default="docs/results")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--method", type=str, default="no_prediction", help="Method name to replay")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    env_kwargs = config.get("env", {})

    # Resolve checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        method_name = args.method
    else:
        methods = config.get("methods", {})
        if args.method not in methods:
            print(f"Method {args.method} not found in config. Available: {list(methods.keys())}")
            sys.exit(1)
        ckpt = methods[args.method].get("checkpoint")
        checkpoint_path = None
        if ckpt:
            ckpt_p = PROJECT_ROOT / ckpt
            if ckpt_p.exists():
                checkpoint_path = str(ckpt_p)
            else:
                parts = ckpt.split("/")
                parts[2] = f"{parts[2]}_seed0"
                alt_p = PROJECT_ROOT / "/".join(parts)
                if alt_p.exists():
                    checkpoint_path = str(alt_p)
        method_name = args.method

    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    results = {
        "replay_date": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "method": method_name,
        "n_eps_per_scenario": args.n_eps,
        "env_config": env_kwargs,
        "scenarios": {},
    }

    for scen_name, scen_config in config.get("scenarios", {}).items():
        print(f"Replaying scenario: {scen_name} ...")
        eps = run_episodes_for_scenario(
            scen_config, config, checkpoint_path, n_eps=args.n_eps
        )
        successes = [e["success"] for e in eps]
        results["scenarios"][scen_name] = {
            "episodes": eps,
            "success_rate": float(np.mean(successes)),
            "mean_final_range": float(np.nanmean([e["final_range_m"] for e in eps])),
            "mean_final_ata": float(np.nanmean([e["final_ata_deg"] for e in eps])),
        }
        print(f"  Success rate: {results['scenarios'][scen_name]['success_rate']:.2%}")

    # Save JSON
    json_path = output_dir / "stage6h0r_replay_stage6f_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Save Markdown
    md_path = output_dir / "stage6h0r_replay_stage6f_results.md"
    lines = [
        "# Stage 6H.0-R: Replay Stage 6F Scenarios",
        "",
        f"**Replay date**: {results['replay_date']}",
        f"**Checkpoint**: `{results['checkpoint_path']}`",
        f"**Method**: {results['method']}",
        f"**Episodes per scenario**: {results['n_eps_per_scenario']}",
        "",
        "## Results",
        "",
        "| Scenario | Success Rate | Mean Final Range (m) | Mean Final ATA (°) | Status |",
        "|---|---|---|---|---|",
    ]
    for scen_name, info in results["scenarios"].items():
        sr = info["success_rate"]
        status = "✅ Acceptable" if sr >= 0.6 else ("⚠️ Marginal" if sr >= 0.4 else "❌ Failing")
        lines.append(
            f"| {scen_name} | {sr:.1%} | {info['mean_final_range']:.1f} | "
            f"{info['mean_final_ata']:.1f} | {status} |"
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
    ])

    all_sr = [info["success_rate"] for info in results["scenarios"].values()]
    if min(all_sr) >= 0.6:
        lines.append(
            "**All scenarios remain feasible** with the current code + checkpoint.\n"
            "The zero-candidate result in 6H.0 baseline search is due to **search space mismatch**,\n"
            "not regression. Proceed to expand search or run threshold optimization on replayed successes."
        )
    elif max(all_sr) >= 0.6:
        lines.append(
            "**Partial feasibility**: some scenarios succeed, others fail.\n"
            "This suggests either checkpoint mismatch or code changes selectively broke certain geometries."
        )
    else:
        lines.append(
            "**No scenarios succeed** with the current code + checkpoint.\n"
            "This indicates a genuine regression or checkpoint mismatch.\n"
            "Cannot proceed to threshold optimization until root cause is resolved."
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReplay results JSON: {json_path}")
    print(f"Replay results MD: {md_path}")
    print(f"Overall: min={min(all_sr):.1%}, max={max(all_sr):.1%}, mean={np.mean(all_sr):.1%}")


if __name__ == "__main__":
    main()
