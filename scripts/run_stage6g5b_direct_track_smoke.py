#!/usr/bin/env python3
"""
Stage 6G.5B: Direct-Track / Pure-PN Control Feasibility Probe.

Evaluates whether bypassing the VPP layer (policy → offset) improves
success rate in tail-chase geometries where Stage 6G.5A found 0% success.

Variants:
    vpp_trained_ppo      — Baseline: VPP with trained PPO checkpoint.
    direct_target_los    — direct_track_mode=true, LOS-rate guidance on target.
    pure_pn_no_vpp       — direct_track_mode=true, proportional navigation on target.

Usage:
    python scripts/run_stage6g5b_direct_track_smoke.py \
        --input-geometry outputs/stage6g5_geometry_smoke_real_seed0/geometry_smoke_points.csv \
        --dry-run \
        --output-dir outputs/stage6g5b_direct_track_smoke_dryrun
"""

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
    load_experiment_config,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config


VARIANTS = {
    "vpp_trained_ppo": {
        "description": "Baseline VPP with trained PPO checkpoint",
        "direct_track_mode": False,
        "guidance_mode": "los_rate",
        "use_vpp": True,
    },
    "direct_target_los": {
        "description": "Direct target tracking with LOS-rate guidance (no VPP offset)",
        "direct_track_mode": True,
        "guidance_mode": "los_rate",
        "use_vpp": False,
    },
    "pure_pn_no_vpp": {
        "description": "Pure proportional navigation on target (no VPP offset)",
        "direct_track_mode": True,
        "guidance_mode": "proportional_navigation",
        "use_vpp": False,
    },
}


def _load_geometry_points(csv_path: str):
    """Load sampled geometry points from Stage 6G.5A CSV."""
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = {}
            for k, v in row.items():
                # Only keep raw geometry parameters
                if k in {"initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"}:
                    pt[k] = float(v) if "." in v else int(v)
                elif k in {"closure_rate_mps", "range_rate_mps", "estimated_time_to_capture_s"}:
                    pt[k] = float(v)
                elif k == "expected_feasible_flag":
                    pt[k] = v.lower() == "true"
            points.append(pt)
    return points


def _make_env_agent(config, method_override, variant_info, allow_random_policy=False):
    """Create env + agent for a variant."""
    method_config = merge_config(copy.deepcopy(config), copy.deepcopy(method_override))
    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(method_config.get("policy", {}).get("action_dim", 3))
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device="cpu")

    ckpt = method_override.get("checkpoint")
    ckpt_exists = ckpt and os.path.exists(ckpt)
    if ckpt_exists:
        agent.load(ckpt)
    elif allow_random_policy:
        print(f"  WARNING: checkpoint missing ({ckpt}), using random policy")
    else:
        raise FileNotFoundError(
            f"Checkpoint missing: {ckpt}\nUse --allow-random-policy to proceed with random policy."
        )

    policy_type = "trained_ppo" if ckpt_exists else ("random_policy" if allow_random_policy else "missing_checkpoint")
    return env, agent, method_config, policy_type, ckpt


def run_direct_track_smoke(
    input_geometry_csv: str,
    output_dir: str,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
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

    # Load geometry points from Stage 6G.5A
    points = _load_geometry_points(input_geometry_csv)
    print(f"Loaded {len(points)} geometry points from {input_geometry_csv}")

    # Save requested / resolved configs
    requested_config = {
        "experiment_name": "stage6g5b_direct_track_smoke",
        "input_geometry_csv": input_geometry_csv,
        "config_path": config_path,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "variants": VARIANTS,
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
    }
    with open(output_path / "requested_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(requested_config, f, default_flow_style=False, sort_keys=False)

    resolved_config = copy.deepcopy(base_config)
    resolved_config["experiment"] = resolved_config.get("experiment", {})
    resolved_config["experiment"]["name"] = "stage6g5b_direct_track_smoke"
    resolved_config["experiment"]["episodes_per_point"] = episodes_per_point
    resolved_config["experiment"]["eval_seeds"] = eval_seeds
    resolved_config["experiment"]["dry_run"] = dry_run
    resolved_config["experiment"]["allow_random_policy"] = allow_random_policy
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(points)} points × {len(VARIANTS)} variants")
        print(f"Episodes per point: {episodes_per_point}")
        print("No simulation executed.")
        # Write stable empty artifacts
        _write_empty_csv(output_path / "raw_episodes.csv", ["variant", "point_index", "episode_index", "evaluation_seed", "episode_seed", "is_success", "reason"])
        _write_empty_csv(output_path / "geometry_method_summary.csv", ["variant", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "feasible_candidates.csv", ["variant", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "direct_track_vs_vpp_comparison.csv", ["variant", "description", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp"])
        _write_empty_csv(output_path / "effective_runtime_flags.json", ["variant", "direct_track_mode", "guidance_mode", "use_vpp"])
        _write_readme_result_block(output_path / "README_result_block.md", {}, dry_run=True)
        return requested_config

    # ------------------------------------------------------------------
    # Real execution
    # ------------------------------------------------------------------
    print("\n=== Stage 6G.5B: Direct-Track Smoke Execution ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    method_name = "no_prediction"
    method_override = config.get("methods", {}).get(method_name, {})

    all_episodes = []
    summary_by_variant = {}
    effective_flags = {}

    for variant_name, variant_info in VARIANTS.items():
        print(f"\n--- Variant: {variant_name} ---")
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = variant_info["direct_track_mode"]
        variant_config["guidance"]["mode"] = variant_info["guidance_mode"]

        env, agent, method_config, policy_type, ckpt = _make_env_agent(
            variant_config, method_override, variant_info, allow_random_policy=allow_random_policy
        )

        ckpt_exists = ckpt and os.path.exists(ckpt)
        effective_flags[variant_name] = {
            "direct_track_mode": variant_info["direct_track_mode"],
            "guidance_mode": variant_info["guidance_mode"],
            "use_vpp": variant_info["use_vpp"],
            "policy_type": policy_type,
            "checkpoint_path": str(ckpt) if ckpt else None,
            "checkpoint_exists": ckpt_exists,
            "checkpoint_size_bytes": os.path.getsize(ckpt) if ckpt_exists else None,
            "allow_random_policy": allow_random_policy,
        }

        variant_success = 0
        variant_total = 0

        for pt_idx, pt in enumerate(points):
            scenario = build_geometry_scenario(
                pt["initial_range_m"],
                pt["ego_speed_mps"],
                pt["target_speed_mps"],
                pt["aspect_angle_deg"],
                pt["altitude_diff_m"],
                base_altitude_m=5000.0,
            )
            scenario["name"] = f"pt{pt_idx}_" + "_".join(f"{k}={v}" for k, v in pt.items() if k in {
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
                        ep_result["variant"] = variant_name
                        ep_result["point_index"] = pt_idx
                        ep_result["episode_index"] = ep_idx
                        ep_result["evaluation_seed"] = ev_seed
                        ep_result["episode_seed"] = episode_seed
                        ep_result["geometry_params"] = pt
                        all_episodes.append(ep_result)
                        status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                        print(f"  {scenario['name']} | {variant_name} | ev={ev_seed} ep={ep_idx} | {status}")
                    except Exception as exc:
                        print(f"  {scenario['name']} | {variant_name} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                        all_episodes.append({
                            "variant": variant_name,
                            "point_index": pt_idx,
                            "episode_index": ep_idx,
                            "evaluation_seed": ev_seed,
                            "episode_seed": episode_seed,
                            "geometry_params": pt,
                            "is_success": False,
                            "reason": f"exception:{exc}",
                        })
                    variant_total += 1
                    if all_episodes[-1].get("is_success"):
                        variant_success += 1

        env.close()
        summary_by_variant[variant_name] = {
            "success_rate": variant_success / variant_total if variant_total > 0 else 0.0,
            "success_count": variant_success,
            "total": variant_total,
        }

    # Save effective runtime flags
    with open(output_path / "effective_runtime_flags.json", "w", encoding="utf-8") as f:
        json.dump(effective_flags, f, indent=2, ensure_ascii=False, default=str)

    # Save raw episodes
    _write_csv_from_dicts(output_path / "raw_episodes.csv", all_episodes,
                          ["variant", "point_index", "episode_index", "evaluation_seed", "episode_seed", "is_success", "reason"])

    # Save geometry-method summary
    summary_rows = []
    for vname, s in summary_by_variant.items():
        row = {"variant": vname, **s}
        row.update(effective_flags[vname])
        summary_rows.append(row)
    _write_csv_from_dicts(output_path / "geometry_method_summary.csv", summary_rows,
                          ["variant", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp"])

    # Save feasible candidates
    feasible = [r for r in summary_rows if r["success_rate"] > 0.20]
    _write_csv_from_dicts(output_path / "feasible_candidates.csv", feasible,
                          ["variant", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp"])
    print(f"\nFeasible candidates: {len(feasible)}")

    # Save direct-track vs VPP comparison
    _write_csv_from_dicts(output_path / "direct_track_vs_vpp_comparison.csv", summary_rows,
                          ["variant", "description", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp", "checkpoint_exists"])

    # Save README result block
    _write_readme_result_block(output_path / "README_result_block.md", summary_by_variant, dry_run=False)

    return requested_config


def _write_empty_csv(path: Path, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _write_csv_from_dicts(path: Path, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_readme_result_block(path: Path, summary_by_variant: dict, dry_run: bool):
    lines = ["## Stage 6G.5B Direct-Track / Pure-PN Probe Results", ""]
    if dry_run:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    else:
        lines.append("| Variant | Success Rate | Success / Total |")
        lines.append("|---|---|---|")
        for vname, s in summary_by_variant.items():
            rate = s.get("success_rate", 0.0)
            lines.append(f"| {vname} | {rate*100:.1f}% | {s['success_count']}/{s['total']} |")
        lines.append("")
        lines.append("> **Paper-safe note**: No feasible candidates were found in the tested Stage 6G.5A 40-point geometry sample.")
        lines.append("> Direct-track probe evaluates whether the VPP layer contributes to the observed failure.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5B Direct-Track / Pure-PN Control Feasibility Probe")
    parser.add_argument("--input-geometry", type=str, required=True, help="Path to Stage 6G.5A geometry_smoke_points.csv")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--episodes-per-point", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5b_direct_track_smoke")
    args = parser.parse_args()

    run_direct_track_smoke(
        input_geometry_csv=args.input_geometry,
        output_dir=args.output_dir,
        config_path=args.config,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
