#!/usr/bin/env python3
"""
Stage 6H.0-R: Multi-checkpoint regression baseline probe.

Tests ALL available checkpoints against Stage 6F non-tail-chase scenarios
(neutral, disadvantage, challenging) to find any checkpoint that can serve
as a regression baseline.

This is NOT a grid search — it focuses on the exact geometries that were
known to work in Stage 6F, using every checkpoint variant available.

Usage:
    python scripts/find_stage6h0_regression_baseline_multi_checkpoint.py
"""

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.utils.config import merge_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All checkpoints to test, grouped by method
CHECKPOINTS = {
    "no_prediction": [
        ("outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt", "seed0"),
        ("outputs/experiments/no_prediction_vpp_ppo_seed1/checkpoints/best.pt", "seed1"),
        ("outputs/experiments/no_prediction_vpp_ppo_20k/checkpoints/best.pt", "20k"),
        ("outputs/experiments/no_prediction_vpp_ppo_acceptance/checkpoints/best.pt", "acceptance"),
        ("outputs/experiments/no_prediction_vpp_ppo_backup_20260604_102227/checkpoints/best.pt", "backup"),
        ("outputs/experiments/no_prediction_vpp_ppo_quick/checkpoints/best.pt", "quick"),
        ("outputs/audit_no_pred_final/checkpoints/best.pt", "audit_final"),
    ],
    "cv_prediction": [
        ("outputs/experiments/vpp_ppo_cv_prediction_seed0/checkpoints/best.pt", "seed0"),
        ("outputs/experiments/vpp_ppo_cv_prediction_seed1/checkpoints/best.pt", "seed1"),
        ("outputs/experiments/vpp_ppo_cv_prediction_seed2/checkpoints/best.pt", "seed2"),
        ("outputs/audit_cv_final/checkpoints/best.pt", "audit_final"),
    ],
    "ca_prediction": [
        ("outputs/experiments/vpp_ppo_ca_prediction_seed0/checkpoints/best.pt", "seed0"),
        ("outputs/experiments/vpp_ppo_ca_prediction_seed1/checkpoints/best.pt", "seed1"),
        ("outputs/experiments/vpp_ppo_ca_prediction_seed2/checkpoints/best.pt", "seed2"),
        ("outputs/audit_ca_final/checkpoints/best.pt", "audit_final"),
    ],
    "gru_frozen": [
        ("outputs/experiments/vpp_ppo_gru_frozen_seed0/checkpoints/best.pt", "seed0"),
    ],
}

# Stage 6F non-tail-chase scenarios
SCENARIOS = {
    "neutral": {
        "name": "neutral",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
        "target_init": {"position_m": [2000.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 180.0},
    },
    "disadvantage": {
        "name": "disadvantage",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 0.0},
        "target_init": {"position_m": [-600.0, 400.0, 5000.0], "velocity_mps": 220.0, "heading_deg": 30.0},
    },
    "challenging": {
        "name": "challenging",
        "own_init": {"position_m": [0.0, 0.0, 5000.0], "velocity_mps": 200.0, "heading_deg": 45.0},
        "target_init": {"position_m": [1500.0, 1500.0, 5200.0], "velocity_mps": 210.0, "heading_deg": 225.0},
    },
}


def _run_episode(env, agent, method_config, scenario, seed):
    try:
        result, _ = evaluate_single_episode(
            env, agent, method_config, scenario=scenario, seed=seed,
            save_trajectory=False, method_name="no_prediction",
        )
        return {
            "success": bool(result.get("is_success", False)),
            "reason": result.get("reason", "unknown"),
            "final_range_m": float(result.get("final_range_m", float("nan"))),
        }
    except Exception as exc:
        return {"success": False, "reason": f"exception:{exc}", "final_range_m": float("nan")}


def main():
    config_path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_dir = PROJECT_ROOT / "outputs" / "stage6h0_multi_ckpt_baseline_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "probe_date": datetime.now().isoformat(),
        "scenarios_tested": list(SCENARIOS.keys()),
        "methods_tested": {},
    }

    # We only need non-tail-chase; favorable is tail-chase
    non_tail_scenarios = {k: v for k, v in SCENARIOS.items() if k != "favorable"}

    for method_name, ckpt_list in CHECKPOINTS.items():
        method_override = full_config.get("methods", {}).get(method_name, {})
        if not method_override:
            print(f"Skipping {method_name}: not in config")
            continue

        for ckpt_path, ckpt_label in ckpt_list:
            ckpt_full = PROJECT_ROOT / ckpt_path
            if not ckpt_full.exists():
                print(f"  SKIP missing: {ckpt_path}")
                continue

            print(f"\nTesting {method_name}/{ckpt_label} ({ckpt_path}) ...")
            method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))
            env = CloseRangeTrackingEnv(method_config)
            sample_obs = env.reset(seed=0)
            obs_dim = int(sample_obs["observation_vector"].shape[0])
            action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
            agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")
            agent.load(str(ckpt_full))

            key = f"{method_name}/{ckpt_label}"
            results["methods_tested"][key] = {}

            for scen_name, scen_config in non_tail_scenarios.items():
                eps = []
                for seed in range(3):
                    r = _run_episode(env, agent, method_config, scen_config, seed=1000 + seed)
                    eps.append(r)
                    status = "OK" if r["success"] else r["reason"]
                    print(f"  {scen_name} seed={seed} | {status}")

                sr = float(np.mean([e["success"] for e in eps]))
                results["methods_tested"][key][scen_name] = {
                    "success_rate": sr,
                    "episodes": eps,
                }
                print(f"  -> {scen_name} SR={sr:.1%}")

            env.close()

    # Save JSON
    json_path = output_dir / "multi_ckpt_baseline_probe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Build Markdown summary
    md_lines = [
        "# Stage 6H.0-R: Multi-Checkpoint Baseline Probe",
        "",
        f"**Date**: {results['probe_date']}",
        "**Goal**: Find ANY checkpoint that succeeds on non-tail-chase Stage 6F scenarios",
        "",
        "## Results",
        "",
        "| Checkpoint | Scenario | Success Rate | Status |",
        "|---|---|---|---|",
    ]

    found_any = False
    for key, scen_results in results["methods_tested"].items():
        for scen_name, info in scen_results.items():
            sr = info["success_rate"]
            status = "✅ Candidate" if sr >= 0.6 else "❌ Failing"
            if sr >= 0.6:
                found_any = True
            md_lines.append(f"| {key} | {scen_name} | {sr:.1%} | {status} |")

    md_lines.extend([
        "",
        "## Conclusion",
        "",
    ])

    if found_any:
        md_lines.append(
            "**At least one checkpoint succeeded on a non-tail-chase scenario.**\n"
            "These can be used to construct a regression baseline for 6H.0-lite."
        )
    else:
        md_lines.append(
            "**No checkpoint succeeded on any non-tail-chase scenario.**\n"
            "This indicates a genuine limitation: the available VPP checkpoints\n"
            "do not generalize to non-tail-chase geometries in the simple backend.\n"
            "The 6H.0-lite threshold search remains blocked until:\n"
            "(a) a new checkpoint is trained on diverse geometries, or\n"
            "(b) the cause of VPP failure on non-tail-chase is resolved."
        )

    md_path = output_dir / "multi_ckpt_baseline_probe.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nResults saved to {json_path} and {md_path}")
    if found_any:
        print("FOUND at least one candidate checkpoint!")
    else:
        print("No candidate checkpoint found.")


if __name__ == "__main__":
    main()
