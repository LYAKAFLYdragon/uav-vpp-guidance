#!/usr/bin/env python3
"""
Stage 6G.5C: Pure-PN Candidate Confirmation & VPP Failure Diagnosis.

Confirms whether the 3 pure-PN-successful geometries from Stage 6G.5B
(pt20, pt29, pt38) are cross-seed stable, and diagnoses why VPP+LOS
and direct-target LOS fail while pure PN succeeds.

Variants:
    vpp_trained_ppo_los     — VPP policy + LOS-rate guidance (baseline)
    direct_target_los       — no VPP + LOS-rate guidance
    pure_pn_no_vpp          — no VPP + proportional navigation
    hybrid_no_vpp           — no VPP + hybrid guidance
    vpp_policy_pn_guidance  — VPP policy + proportional navigation
    oracle_anchor_pn        — (dry-run only) oracle future anchor + PN
    rule_based_vpp_pn       — (dry-run only) rule-based VPP + PN

Usage:
    python scripts/run_stage6g5c_candidate_confirmation.py \
        --candidate-points pt20 pt29 pt38 \
        --eval-seeds 0 1 2 \
        --episodes-per-point 10 \
        --output-dir outputs/stage6g5c_candidate_confirmation_seed012
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
from uav_vpp_guidance.utils.geometry_scenario import build_geometry_scenario
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
    load_experiment_config,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.utils.config import merge_config


VARIANTS = {
    "vpp_trained_ppo_los": {
        "description": "Baseline VPP policy with LOS-rate guidance",
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
    "hybrid_no_vpp": {
        "description": "Hybrid guidance on direct target (no VPP offset)",
        "direct_track_mode": True,
        "guidance_mode": "hybrid",
        "use_vpp": False,
    },
    "vpp_policy_pn_guidance": {
        "description": "VPP policy with proportional navigation guidance",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
    },
    "oracle_anchor_pn": {
        "description": "Oracle future anchor + proportional navigation (dry-run only)",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
        "anchor_mode": "oracle_future_position",
        "dry_run_only": True,
    },
    "rule_based_vpp_pn": {
        "description": "Rule-based VPP pursuit + proportional navigation (dry-run only)",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
        "anchor_mode": "rule_based_pursuit",
        "dry_run_only": True,
    },
}


def _load_geometry_points(csv_path: str):
    """Load sampled geometry points from Stage 6G.5B CSV."""
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = {}
            for k, v in row.items():
                if k in {"initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"}:
                    pt[k] = float(v) if "." in v else int(v)
                elif k in {"closure_rate_mps", "range_rate_mps", "estimated_time_to_capture_s"}:
                    pt[k] = float(v)
                elif k == "expected_feasible_flag":
                    pt[k] = v.lower() == "true"
            points.append(pt)
    return points


def _select_candidate_points(all_points, candidate_ids):
    """Select specific candidate points by index."""
    selected = []
    for cid in candidate_ids:
        if cid.startswith("pt"):
            idx = int(cid[2:])
            if 0 <= idx < len(all_points):
                selected.append((cid, all_points[idx]))
            else:
                raise ValueError(f"Candidate point {cid} out of range (0-{len(all_points)-1})")
        else:
            raise ValueError(f"Invalid candidate id: {cid}, expected pt<N>")
    return selected


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


def run_candidate_confirmation(
    candidate_points,
    output_dir: str,
    input_geometry_csv: str = None,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    episodes_per_point: int = 10,
    eval_seeds=None,
    dry_run: bool = False,
    allow_random_policy: bool = False,
):
    if eval_seeds is None:
        eval_seeds = [0, 1, 2]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    if input_geometry_csv:
        all_points = _load_geometry_points(input_geometry_csv)
        candidate_ids = [cid for cid, _ in candidate_points]
        selected = _select_candidate_points(all_points, candidate_ids)
        if len(selected) != len(candidate_points):
            raise ValueError("Some candidate points not found in geometry CSV")
        points = selected
    else:
        points = candidate_points

    requested_config = {
        "experiment_name": "stage6g5c_candidate_confirmation",
        "input_geometry_csv": input_geometry_csv,
        "config_path": config_path,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "candidate_points": [cid for cid, _ in points],
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
    }
    with open(output_path / "requested_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(requested_config, f, default_flow_style=False, sort_keys=False)

    resolved_config = copy.deepcopy(base_config)
    resolved_config["experiment"] = resolved_config.get("experiment", {})
    resolved_config["experiment"]["name"] = "stage6g5c_candidate_confirmation"
    resolved_config["experiment"]["episodes_per_point"] = episodes_per_point
    resolved_config["experiment"]["eval_seeds"] = eval_seeds
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(points)} candidate points × {len([v for v in VARIANTS.values() if not v.get('dry_run_only')])} variants")
        print(f"Episodes per point: {episodes_per_point}")
        print(f"Eval seeds: {eval_seeds}")
        print("No simulation executed.")
        _write_empty_csv(output_path / "candidate_raw_episodes.csv", [
            "variant", "point_id", "episode_index", "evaluation_seed", "episode_seed",
            "is_success", "reason", "capture_time_s", "min_range_m", "final_range_m", "final_ata_deg"
        ])
        _write_empty_csv(output_path / "candidate_variant_summary.csv", [
            "variant", "success_rate", "success_count", "total", "crash_rate", "out_of_bounds_rate",
            "mean_capture_time_s", "mean_min_range_m", "mean_final_range_m", "mean_final_ata_deg"
        ])
        _write_empty_csv(output_path / "candidate_seed_summary.csv", [
            "variant", "point_id", "eval_seed", "success_rate", "success_count", "total"
        ])
        _write_empty_csv(output_path / "candidate_success_matrix.csv", [
            "point_id", "variant", "seed_0_success", "seed_1_success", "seed_2_success", "cross_seed_stable"
        ])
        _write_empty_csv(output_path / "trajectory_terminal_metrics.csv", [
            "variant", "point_id", "episode_seed", "min_range_m", "final_range_m", "final_ata_deg",
            "capture_time_s", "altitude_loss_rate", "energy_proxy", "nz_saturation_rate", "roll_rate_saturation_rate"
        ])
        _write_empty_csv(output_path / "command_saturation_summary.csv", [
            "variant", "point_id", "nz_cmd_max", "nz_cmd_mean", "nz_cmd_saturation_rate",
            "roll_rate_cmd_max", "roll_rate_cmd_mean", "roll_rate_cmd_saturation_rate",
            "throttle_cmd_max", "throttle_cmd_mean", "throttle_cmd_saturation_rate"
        ])
        with open(output_path / "effective_runtime_flags.json", "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        _write_diagnosis_md(output_path / "vpp_vs_pn_failure_diagnosis.md", {}, dry_run=True)
        return requested_config

    # Real execution
    print("\n=== Stage 6G.5C: Candidate Confirmation & VPP Failure Diagnosis ===")
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
        if variant_info.get("dry_run_only") and not dry_run:
            print(f"\n--- Variant: {variant_name} (skipped: dry-run only) ---")
            continue

        print(f"\n--- Variant: {variant_name} ---")
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = variant_info["direct_track_mode"]
        variant_config["guidance"]["mode"] = variant_info["guidance_mode"]
        if "anchor_mode" in variant_info:
            variant_config.setdefault("virtual_point", {})
            variant_config["virtual_point"]["anchor_mode"] = variant_info["anchor_mode"]

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
            "allow_random_policy": allow_random_policy,
        }

        variant_success = 0
        variant_total = 0

        for point_id, pt in points:
            scenario = build_geometry_scenario(
                pt["initial_range_m"],
                pt["ego_speed_mps"],
                pt["target_speed_mps"],
                pt["aspect_angle_deg"],
                pt["altitude_diff_m"],
                base_altitude_m=5000.0,
            )
            scenario["name"] = point_id

            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + int(point_id[2:]) * 1000 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, trajectory = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name=method_name,
                        )
                        ep_result["variant"] = variant_name
                        ep_result["point_id"] = point_id
                        ep_result["episode_index"] = ep_idx
                        ep_result["evaluation_seed"] = ev_seed
                        ep_result["episode_seed"] = episode_seed
                        ep_result["geometry_params"] = pt
                        all_episodes.append(ep_result)
                        status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                        print(f"  {point_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | {status}")
                    except Exception as exc:
                        print(f"  {point_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                        all_episodes.append({
                            "variant": variant_name,
                            "point_id": point_id,
                            "episode_index": ep_idx,
                            "evaluation_seed": ev_seed,
                            "episode_seed": episode_seed,
                            "geometry_params": pt,
                            "is_success": False,
                            "reason": f"exception:{exc}",
                            "capture_time_s": np.nan,
                            "min_range_m": np.nan,
                            "final_range_m": np.nan,
                            "final_ata_deg": np.nan,
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
    _write_csv_from_dicts(output_path / "candidate_raw_episodes.csv", all_episodes, [
        "variant", "point_id", "episode_index", "evaluation_seed", "episode_seed",
        "is_success", "reason", "capture_time_s", "min_range_m", "final_range_m", "final_ata_deg"
    ])

    # Aggregate by variant, point, seed
    variant_summary, seed_summary, success_matrix, terminal_metrics, sat_summary = _aggregate_by_all_dimensions(
        all_episodes, points, eval_seeds
    )

    _write_csv_from_dicts(output_path / "candidate_variant_summary.csv", variant_summary, [
        "variant", "success_rate", "success_count", "total", "crash_rate", "out_of_bounds_rate",
        "mean_capture_time_s", "mean_min_range_m", "mean_final_range_m", "mean_final_ata_deg",
        "mean_nz_saturation_rate", "mean_roll_rate_saturation_rate", "mean_throttle_saturation_rate",
        "mean_altitude_loss_rate", "mean_energy_proxy"
    ])

    _write_csv_from_dicts(output_path / "candidate_seed_summary.csv", seed_summary, [
        "variant", "point_id", "eval_seed", "success_rate", "success_count", "total"
    ])

    _write_csv_from_dicts(output_path / "candidate_success_matrix.csv", success_matrix, [
        "point_id", "variant", "seed_0_success", "seed_1_success", "seed_2_success", "cross_seed_stable"
    ])

    _write_csv_from_dicts(output_path / "trajectory_terminal_metrics.csv", terminal_metrics, [
        "variant", "point_id", "episode_seed", "min_range_m", "final_range_m", "final_ata_deg",
        "capture_time_s", "altitude_loss_rate", "energy_proxy",
        "nz_saturation_rate", "roll_rate_saturation_rate", "throttle_saturation_rate"
    ])

    _write_csv_from_dicts(output_path / "command_saturation_summary.csv", sat_summary, [
        "variant", "point_id", "nz_cmd_max", "nz_cmd_mean", "nz_cmd_saturation_rate",
        "roll_rate_cmd_max", "roll_rate_cmd_mean", "roll_rate_cmd_saturation_rate",
        "throttle_cmd_max", "throttle_cmd_mean", "throttle_cmd_saturation_rate"
    ])

    _write_diagnosis_md(output_path / "vpp_vs_pn_failure_diagnosis.md", {
        "variant_summary": variant_summary,
        "success_matrix": success_matrix,
        "seed_summary": seed_summary,
    }, dry_run=False)

    return requested_config


def _aggregate_by_all_dimensions(all_episodes, points, eval_seeds):
    """Aggregate episodes by variant, point, and seed."""
    from collections import defaultdict

    # Group by variant
    by_variant = defaultdict(list)
    by_variant_point = defaultdict(lambda: defaultdict(list))
    by_variant_point_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ep in all_episodes:
        v = ep.get("variant", "unknown")
        p = ep.get("point_id", "unknown")
        s = ep.get("evaluation_seed", 0)
        by_variant[v].append(ep)
        by_variant_point[v][p].append(ep)
        by_variant_point_seed[v][p][s].append(ep)

    def _safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else np.nan

    # Variant summary
    variant_summary = []
    for v, eps in by_variant.items():
        n = len(eps)
        succ = sum(1 for e in eps if e.get("is_success"))
        crashes = sum(1 for e in eps if e.get("is_crash"))
        oob = sum(1 for e in eps if e.get("is_out_of_bounds"))
        lengths = [e.get("length", 0) for e in eps if e.get("is_success")]
        variant_summary.append({
            "variant": v,
            "success_rate": succ / n if n else 0.0,
            "success_count": succ,
            "total": n,
            "crash_rate": crashes / n if n else 0.0,
            "out_of_bounds_rate": oob / n if n else 0.0,
            "mean_capture_time_s": _safe_mean([e.get("length", np.nan) * 0.2 for e in eps if e.get("is_success")]),
            "mean_min_range_m": _safe_mean([e.get("min_range_m", np.nan) for e in eps]),
            "mean_final_range_m": _safe_mean([e.get("final_range_m", np.nan) for e in eps]),
            "mean_final_ata_deg": _safe_mean([e.get("final_ata_deg", np.nan) for e in eps]),
            "mean_nz_saturation_rate": _safe_mean([e.get("nz_cmd_saturation_rate", np.nan) for e in eps]),
            "mean_roll_rate_saturation_rate": _safe_mean([e.get("roll_rate_cmd_saturation_rate", np.nan) for e in eps]),
            "mean_throttle_saturation_rate": _safe_mean([e.get("throttle_cmd_saturation_rate", np.nan) for e in eps]),
            "mean_altitude_loss_rate": _safe_mean([e.get("altitude_loss_rate", np.nan) for e in eps]),
            "mean_energy_proxy": _safe_mean([e.get("energy_proxy", np.nan) for e in eps]),
        })

    # Seed summary
    seed_summary = []
    for v, pdict in by_variant_point_seed.items():
        for p, sdict in pdict.items():
            for s, eps in sdict.items():
                n = len(eps)
                succ = sum(1 for e in eps if e.get("is_success"))
                seed_summary.append({
                    "variant": v,
                    "point_id": p,
                    "eval_seed": s,
                    "success_rate": succ / n if n else 0.0,
                    "success_count": succ,
                    "total": n,
                })

    # Success matrix (cross-seed stability)
    success_matrix = []
    for v, pdict in by_variant_point.items():
        for p, eps in pdict.items():
            seed_success = {}
            for s in eval_seeds:
                seed_eps = [e for e in eps if e.get("evaluation_seed") == s]
                seed_success[s] = sum(1 for e in seed_eps if e.get("is_success")) / max(1, len(seed_eps))
            stable = all(seed_success.get(s, 0) > 0.5 for s in eval_seeds)
            success_matrix.append({
                "point_id": p,
                "variant": v,
                "seed_0_success": seed_success.get(0, np.nan),
                "seed_1_success": seed_success.get(1, np.nan),
                "seed_2_success": seed_success.get(2, np.nan),
                "cross_seed_stable": stable,
            })

    # Terminal metrics per episode
    terminal_metrics = []
    for ep in all_episodes:
        terminal_metrics.append({
            "variant": ep.get("variant"),
            "point_id": ep.get("point_id"),
            "episode_seed": ep.get("episode_seed"),
            "min_range_m": ep.get("min_range_m", np.nan),
            "final_range_m": ep.get("final_range_m", np.nan),
            "final_ata_deg": ep.get("final_ata_deg", np.nan),
            "capture_time_s": ep.get("length", 0) * 0.2 if ep.get("is_success") else np.nan,
            "altitude_loss_rate": ep.get("altitude_loss_rate", np.nan),
            "energy_proxy": ep.get("energy_proxy", np.nan),
            "nz_saturation_rate": ep.get("nz_cmd_saturation_rate", np.nan),
            "roll_rate_saturation_rate": ep.get("roll_rate_cmd_saturation_rate", np.nan),
            "throttle_saturation_rate": ep.get("throttle_cmd_saturation_rate", np.nan),
        })

    # Command saturation summary per variant-point
    sat_summary = []
    for v, pdict in by_variant_point.items():
        for p, eps in pdict.items():
            sat_summary.append({
                "variant": v,
                "point_id": p,
                "nz_cmd_max": _safe_mean([e.get("nz_cmd_max", np.nan) for e in eps]),
                "nz_cmd_mean": _safe_mean([e.get("nz_cmd_mean", np.nan) for e in eps]),
                "nz_cmd_saturation_rate": _safe_mean([e.get("nz_cmd_saturation_rate", np.nan) for e in eps]),
                "roll_rate_cmd_max": _safe_mean([e.get("roll_rate_cmd_max", np.nan) for e in eps]),
                "roll_rate_cmd_mean": _safe_mean([e.get("roll_rate_cmd_mean", np.nan) for e in eps]),
                "roll_rate_cmd_saturation_rate": _safe_mean([e.get("roll_rate_cmd_saturation_rate", np.nan) for e in eps]),
                "throttle_cmd_max": _safe_mean([e.get("throttle_cmd_max", np.nan) for e in eps]),
                "throttle_cmd_mean": _safe_mean([e.get("throttle_cmd_mean", np.nan) for e in eps]),
                "throttle_cmd_saturation_rate": _safe_mean([e.get("throttle_cmd_saturation_rate", np.nan) for e in eps]),
            })

    return variant_summary, seed_summary, success_matrix, terminal_metrics, sat_summary


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


def _write_diagnosis_md(path: Path, data: dict, dry_run: bool):
    lines = ["# Stage 6G.5C VPP vs PN Failure Diagnosis", ""]
    if dry_run:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    else:
        lines.append("## Variant Summary")
        lines.append("")
        lines.append("| Variant | Success Rate | Success / Total | Crash Rate | OOB Rate | Mean Min Range (m) |")
        lines.append("|---|---|---|---|---|---|")
        for row in data.get("variant_summary", []):
            lines.append(
                f"| {row['variant']} | {row['success_rate']*100:.1f}% | {row['success_count']}/{row['total']} | "
                f"{row.get('crash_rate', 0)*100:.1f}% | {row.get('out_of_bounds_rate', 0)*100:.1f}% | "
                f"{row.get('mean_min_range_m', 'N/A'):.1f} |"
            )
        lines.append("")
        lines.append("## Cross-Seed Stability")
        lines.append("")
        lines.append("| Point | Variant | Seed 0 | Seed 1 | Seed 2 | Stable? |")
        lines.append("|---|---|---|---|---|---|")
        for row in data.get("success_matrix", []):
            s0 = f"{row.get('seed_0_success', 0)*100:.0f}%"
            s1 = f"{row.get('seed_1_success', 0)*100:.0f}%"
            s2 = f"{row.get('seed_2_success', 0)*100:.0f}%"
            stable = "Yes" if row.get("cross_seed_stable") else "No"
            lines.append(f"| {row['point_id']} | {row['variant']} | {s0} | {s1} | {s2} | {stable} |")
        lines.append("")
        lines.append(
            "> **Paper-safe note**: Results are limited to the 3 candidate geometries "
            "(pt20, pt29, pt38) tested under cross-seed evaluation. "
            "No universal claims about tail-chase feasibility are made."
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5C Pure-PN Candidate Confirmation & VPP Failure Diagnosis")
    parser.add_argument("--candidate-points", type=str, nargs="+", default=["pt20", "pt29", "pt38"],
                        help="Candidate point IDs from Stage 6G.5B (e.g., pt20 pt29 pt38)")
    parser.add_argument("--input-geometry", type=str,
                        default="outputs/stage6g5_geometry_smoke_real_seed0/geometry_smoke_points.csv",
                        help="Path to Stage 6G.5A geometry_smoke_points.csv")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--episodes-per-point", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5c_candidate_confirmation")
    args = parser.parse_args()

    run_candidate_confirmation(
        candidate_points=[(cid, None) for cid in args.candidate_points],
        output_dir=args.output_dir,
        input_geometry_csv=args.input_geometry,
        config_path=args.config,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
