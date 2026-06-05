#!/usr/bin/env python3
"""
Stage 6G.5A: Wide Geometry Sweep Smoke Runner.

Samples 30–50 points from a 324-combo geometry grid to test whether any
tail-chase / stern-conversion configuration is feasible before committing
to full sweep or bilevel gain optimization.

Scope:
    no_prediction baseline only. Predictor-policy feasibility requires Stage 6G.5B.

Usage:
    python scripts/run_stage6g5_geometry_smoke.py --dry-run
    python scripts/run_stage6g5_geometry_smoke.py --sample-size 40 --sampling-method random
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.geometry_scenario import (
    build_geometry_scenario,
    compute_geometry_metadata,
    build_full_grid as _build_full_grid,
    sample_grid as _sample_grid,
)


# ------------------------------------------------------------------
# Dry-run / runner core
# ------------------------------------------------------------------

def run_geometry_smoke(
    config_path: str,
    output_dir: str,
    sample_size: int = 40,
    sampling_method: str = "random",
    seed: int = 0,
    episodes_per_point: int = 3,
    eval_seeds=None,
    dry_run: bool = False,
    allow_random_policy: bool = False,
):
    if eval_seeds is None:
        eval_seeds = [0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base config
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    grid_def = base_config.get("geometry_grid", {})
    if not grid_def:
        raise ValueError("Config missing 'geometry_grid'")

    # Sample points
    set_seed(seed)
    sampled_points = _sample_grid(grid_def, sample_size, sampling_method, seed)

    # Build metadata for each point
    points_with_meta = []
    for pt in sampled_points:
        meta = compute_geometry_metadata(pt)
        entry = {**pt, **meta}
        points_with_meta.append(entry)

    method_name = "no_prediction"
    method_override = base_config.get("methods", {}).get(method_name, {})
    ckpt = method_override.get("checkpoint")

    # Determine policy type for audit trail
    if dry_run:
        policy_type = "dry_run (no policy loaded)"
    elif ckpt and os.path.exists(ckpt):
        policy_type = "loaded_checkpoint"
    elif allow_random_policy:
        policy_type = "random_policy (explicitly allowed)"
    else:
        policy_type = "missing_checkpoint"

    # Save plan / points
    plan = {
        "experiment_name": base_config.get("experiment", {}).get("name", "stage6g5_wide_geometry_smoke"),
        "config_path": str(config_path),
        "sample_size": sample_size,
        "sampling_method": sampling_method,
        "seed": seed,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "total_grid_size": len(_build_full_grid(grid_def)),
        "sampled_points_count": len(sampled_points),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
        "policy_type": policy_type,
        "loaded_policy_checkpoint_path": str(ckpt) if ckpt else None,
        "methods_evaluated": [method_name],
        "scope_note": (
            "Stage 6G.5A smoke tests baseline geometric feasibility only; "
            "predictor-policy feasibility requires Stage 6G.5B."
        ),
    }

    plan_path = output_path / "geometry_smoke_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved plan: {plan_path}")

    points_csv_path = output_path / "geometry_smoke_points.csv"
    with open(points_csv_path, "w", newline="", encoding="utf-8") as f:
        if points_with_meta:
            writer = csv.DictWriter(f, fieldnames=list(points_with_meta[0].keys()))
            writer.writeheader()
            writer.writerows(points_with_meta)
    print(f"  Saved points CSV: {points_csv_path}")

    # Save resolved config
    resolved_config = copy.deepcopy(base_config)
    resolved_config["experiment"]["sample_size"] = sample_size
    resolved_config["experiment"]["sampling_method"] = sampling_method
    resolved_config["experiment"]["seed"] = seed
    resolved_config["experiment"]["episodes_per_point"] = episodes_per_point
    resolved_config["experiment"]["eval_seeds"] = eval_seeds
    resolved_config["experiment"]["dry_run"] = dry_run
    resolved_config["experiment"]["allow_random_policy"] = allow_random_policy

    resolved_path = output_path / "resolved_config.yaml"
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved resolved config: {resolved_path}")

    if dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would evaluate {len(sampled_points)} geometry points")
        print(f"Episodes per point: {episodes_per_point}")
        print(f"Eval seeds: {eval_seeds}")
        print("No simulation executed.")

        # Write stable output files even in dry-run
        _write_empty_csv(output_path / "geometry_smoke_summary.csv", ["params", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "feasible_candidates.csv", ["params", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "failed_points.csv", ["params", "success_rate", "success_count", "total"])

        summary_md = _render_summary_md(plan, points_with_meta, evaluated_count=0, successes=[])
        md_path = output_path / "geometry_smoke_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"  Saved dry-run summary: {md_path}")
        return plan

    # ------------------------------------------------------------------
    # Real execution (not required for dry-run validation)
    # ------------------------------------------------------------------
    print("\n=== Stage 6G.5A: Geometry Smoke Execution ===")

    # Policy loading contract: fail fast unless --allow-random-policy
    if not (ckpt and os.path.exists(ckpt)) and not allow_random_policy:
        raise FileNotFoundError(
            f"Checkpoint missing for method '{method_name}': {ckpt}\n"
            "Use --allow-random-policy to proceed with a random policy (not recommended for real smoke)."
        )

    all_episodes = []

    # Import heavy deps only when needed
    from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
        evaluate_single_episode,
        load_experiment_config,
    )
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    from uav_vpp_guidance.agents.ppo_agent import PPOAgent
    from uav_vpp_guidance.utils.config import merge_config

    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    method_config = merge_config(copy.deepcopy(config), copy.deepcopy(method_override))
    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")

    if ckpt and os.path.exists(ckpt):
        agent.load(ckpt)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  WARNING: checkpoint missing ({ckpt}), using random policy (explicitly allowed)")

    for pt_idx, pt in enumerate(points_with_meta):
        scenario = build_geometry_scenario(
            pt["initial_range_m"],
            pt["ego_speed_mps"],
            pt["target_speed_mps"],
            pt["aspect_angle_deg"],
            pt["altitude_diff_m"],
            base_altitude_m=5000.0,
        )
        scenario["name"] = "_".join(f"{k}={v}" for k, v in pt.items() if k in {
            "initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"
        })

        for ev_seed in eval_seeds:
            for ep_idx in range(episodes_per_point):
                episode_seed = ev_seed * 100000 + pt_idx * 1000 + ep_idx
                set_seed(episode_seed)
                try:
                    ep_result, _ = evaluate_single_episode(
                        env, agent, method_config, scenario=scenario, seed=episode_seed,
                        save_trajectory=False, method_name=method_name,
                    )
                    ep_result["point_index"] = pt_idx
                    ep_result["episode_index"] = ep_idx
                    ep_result["evaluation_seed"] = ev_seed
                    ep_result["episode_seed"] = episode_seed
                    ep_result["geometry_params"] = pt
                    all_episodes.append(ep_result)
                    status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                    print(f"  {scenario['name']} | ev={ev_seed} ep={ep_idx} | {status}")
                except Exception as exc:
                    print(f"  {scenario['name']} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                    all_episodes.append({
                        "point_index": pt_idx,
                        "episode_index": ep_idx,
                        "evaluation_seed": ev_seed,
                        "episode_seed": episode_seed,
                        "geometry_params": pt,
                        "is_success": False,
                        "reason": f"exception:{exc}",
                    })

    env.close()

    # Aggregate
    success_by_point = {}
    for ep in all_episodes:
        pt = ep["geometry_params"]
        key = "_".join(f"{k}={v}" for k, v in pt.items() if k in {
            "initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"
        })
        success_by_point.setdefault(key, {"success": 0, "total": 0, "params": pt})
        success_by_point[key]["total"] += 1
        if ep.get("is_success"):
            success_by_point[key]["success"] += 1

    evaluated_count = len(all_episodes)
    successes = [
        {
            "params": v["params"],
            "success_rate": v["success"] / v["total"],
            "success_count": v["success"],
            "total": v["total"],
        }
        for v in success_by_point.values()
    ]

    # Save summary CSV
    summary_csv_path = output_path / "geometry_smoke_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        if successes:
            writer = csv.DictWriter(f, fieldnames=list(successes[0].keys()))
            writer.writeheader()
            writer.writerows(successes)
    print(f"  Saved summary CSV: {summary_csv_path}")

    # Feasible candidates (>20% success)
    feasible = [s for s in successes if s["success_rate"] > 0.20]
    feasible_path = output_path / "feasible_candidates.csv"
    with open(feasible_path, "w", newline="", encoding="utf-8") as f:
        if feasible:
            writer = csv.DictWriter(f, fieldnames=list(feasible[0].keys()))
            writer.writeheader()
            writer.writerows(feasible)
    print(f"  Saved feasible candidates: {feasible_path} ({len(feasible)} candidates)")

    # Failed points
    failed = [s for s in successes if s["success_rate"] <= 0.20]
    failed_path = output_path / "failed_points.csv"
    with open(failed_path, "w", newline="", encoding="utf-8") as f:
        if failed:
            writer = csv.DictWriter(f, fieldnames=list(failed[0].keys()))
            writer.writeheader()
            writer.writerows(failed)
    print(f"  Saved failed points: {failed_path} ({len(failed)} points)")

    # Summary markdown
    summary_md = _render_summary_md(plan, points_with_meta, evaluated_count, successes)
    md_path = output_path / "geometry_smoke_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  Saved summary markdown: {md_path}")

    return plan


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_empty_csv(path: Path, fieldnames: list):
    """Write a CSV with headers only (stable output for dry-run)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


# ------------------------------------------------------------------
# Markdown summary renderer
# ------------------------------------------------------------------

def _render_summary_md(plan, points_with_meta, evaluated_count, successes):
    best = max(successes, key=lambda x: x["success_rate"]) if successes else None
    any_success = any(s["success_rate"] > 0.20 for s in successes)

    # Bilevel unblocking rule:
    # - Must have at least one geometry with >20% success
    # - And that geometry must have gain-sensitive failure modes (approximated by
    #   closure_rate being positive and not absurdly high, indicating geometry
    #   is feasible but guidance may need tuning).
    bilevel_unblocked = False
    if any_success and best is not None:
        best_params = best["params"]
        closure = compute_geometry_metadata(best_params)["closure_rate_mps"]
        # Gain-sensitive if closure is moderate (not trivially easy, not impossible)
        if 20.0 < closure < 250.0:
            bilevel_unblocked = True

    lines = [
        "# Stage 6G.5A Geometry Smoke Summary",
        "",
        f"- **Experiment**: {plan['experiment_name']}",
        f"- **Timestamp**: {plan['timestamp']}",
        f"- **Total grid size**: {plan['total_grid_size']}",
        f"- **Sampled points**: {plan['sampled_points_count']}",
        f"- **Sampling method**: {plan['sampling_method']}",
        f"- **Evaluated episodes**: {evaluated_count}",
        f"- **Methods evaluated**: {plan['methods_evaluated']}",
        f"- **Scope note**: {plan['scope_note']}",
        f"- **Any success >20%**: {any_success}",
        f"- **Best success rate**: {best['success_rate']*100:.1f}%" if best else "- **Best success rate**: N/A",
        f"- **Best geometry**: {best['params']}" if best else "- **Best geometry**: N/A",
        f"- **Bilevel unblocked candidate**: {bilevel_unblocked}",
        "",
        "> **Note**: `bilevel_unblocked_candidate` is `true` only when a geometry combo",
        "> shows >20% success *and* the closure rate suggests the failure mode is",
        "> gain-sensitive rather than geometrically infeasible.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5A Wide Geometry Sweep Smoke Runner")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--sampling-method", type=str, default="random", choices=["random", "latin_hypercube"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-point", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dry-run", action="store_true", help="Print plan and save metadata without running episodes")
    parser.add_argument("--allow-random-policy", action="store_true", help="Allow random policy when checkpoint is missing (not recommended for real smoke)")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5_geometry_smoke")
    args = parser.parse_args()

    run_geometry_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        sampling_method=args.sampling_method,
        seed=args.seed,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
