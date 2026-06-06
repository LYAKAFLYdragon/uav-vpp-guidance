#!/usr/bin/env python3
"""
Stage 6G.5D: PN Mode-Switch & VPP Offset Mechanism Probe.

Tests whether a geometry-triggered mode-switch to pure PN can rescue
tail-chase candidates where VPP-based variants fail.

Variants:
    pure_pn_no_vpp              — Baseline pure PN (no VPP).
    vpp_policy_pn_guidance      — VPP + PN reference (expected 0%).
    mode_switch_pn_no_vpp       — Gate active → pure PN; inactive → base guidance.
    mode_switch_vpp_elsewhere   — Gate active → pure PN; elsewhere → VPP+LOS/hybrid.
    vpp_offset_clipped_pn       — VPP + PN with offset norm/vertical clipping.

Usage:
    python scripts/run_stage6g5d_pn_mode_switch_probe.py \
        --candidate-points pt20 pt29 pt38 \
        --eval-seeds 0 1 2 \
        --episodes-per-point 10 \
        --output-dir outputs/stage6g5d_pn_mode_switch_seed012
"""

import argparse
import copy
import csv
import json
import os
import sys
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
    "pure_pn_no_vpp": {
        "description": "Baseline pure PN without VPP",
        "direct_track_mode": True,
        "guidance_mode": "proportional_navigation",
        "use_vpp": False,
    },
    "vpp_policy_pn_guidance": {
        "description": "VPP policy + PN guidance (reference failure)",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
    },
    "mode_switch_pn_no_vpp": {
        "description": "Mode-switch to pure PN in tail-chase gate",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
        "mode_switch": {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        },
    },
    "mode_switch_vpp_elsewhere": {
        "description": "Mode-switch to pure PN in gate; VPP+LOS elsewhere",
        "direct_track_mode": False,
        "guidance_mode": "los_rate",
        "use_vpp": True,
        "mode_switch": {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
        },
    },
}


def _load_geometry_points(csv_path: str):
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt = {}
            for k, v in row.items():
                if k in {"initial_range_m", "ego_speed_mps", "target_speed_mps", "aspect_angle_deg", "altitude_diff_m"}:
                    pt[k] = float(v) if "." in v else int(v)
            points.append(pt)
    return points


def _select_candidate_points(all_points, candidate_ids):
    selected = []
    for cid in candidate_ids:
        idx = int(cid[2:])
        if 0 <= idx < len(all_points):
            selected.append((cid, all_points[idx]))
        else:
            raise ValueError(f"Candidate point {cid} out of range")
    return selected


def _add_neighbors(selected, all_points):
    """Add neighboring failure points for contrast."""
    # Try to find: same range but lower ego speed, same ego but different aspect, same ego/range different alt
    neighbors = []
    seen = {int(cid[2:]) for cid, _ in selected}
    for cid, pt in selected:
        r, ego, aspect, alt = pt["initial_range_m"], pt["ego_speed_mps"], pt["aspect_angle_deg"], pt["altitude_diff_m"]
        for i, candidate in enumerate(all_points):
            if i in seen:
                continue
            c_r, c_ego, c_aspect, c_alt = candidate["initial_range_m"], candidate["ego_speed_mps"], candidate["aspect_angle_deg"], candidate["altitude_diff_m"]
            # Same range, lower ego speed
            if c_r == r and c_ego < ego and c_aspect == aspect:
                neighbors.append((f"pt{i}", candidate))
                seen.add(i)
                break
        for i, candidate in enumerate(all_points):
            if i in seen:
                continue
            c_r, c_ego, c_aspect, c_alt = candidate["initial_range_m"], candidate["ego_speed_mps"], candidate["aspect_angle_deg"], candidate["altitude_diff_m"]
            # Same ego speed, different aspect
            if c_r == r and c_ego == ego and c_aspect != aspect:
                neighbors.append((f"pt{i}", candidate))
                seen.add(i)
                break
        for i, candidate in enumerate(all_points):
            if i in seen:
                continue
            c_r, c_ego, c_aspect, c_alt = candidate["initial_range_m"], candidate["ego_speed_mps"], candidate["aspect_angle_deg"], candidate["altitude_diff_m"]
            # Same ego/range, different altitude diff
            if c_r == r and c_ego == ego and c_aspect == aspect and c_alt != alt:
                neighbors.append((f"pt{i}", candidate))
                seen.add(i)
                break
    return selected + neighbors


def _make_env_agent(config, method_override, allow_random_policy=False):
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
        pass
    else:
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")

    policy_type = "trained_ppo" if ckpt_exists else ("random_policy" if allow_random_policy else "missing_checkpoint")
    return env, agent, method_config, policy_type, ckpt


def run_mode_switch_probe(
    candidate_points,
    output_dir: str,
    input_geometry_csv: str = None,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    episodes_per_point: int = 10,
    eval_seeds=None,
    dry_run: bool = False,
    allow_random_policy: bool = False,
    include_neighbor_failures: bool = False,
):
    if eval_seeds is None:
        eval_seeds = [0, 1, 2]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    all_points = _load_geometry_points(input_geometry_csv) if input_geometry_csv else []
    if all_points:
        selected = _select_candidate_points(all_points, [cid for cid, _ in candidate_points])
        if include_neighbor_failures:
            selected = _add_neighbors(selected, all_points)
    else:
        selected = candidate_points

    requested_config = {
        "experiment_name": "stage6g5d_pn_mode_switch_probe",
        "input_geometry_csv": input_geometry_csv,
        "config_path": config_path,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "candidate_points": [cid for cid, _ in selected],
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
    }
    with open(output_path / "requested_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(requested_config, f, default_flow_style=False, sort_keys=False)

    resolved_config = copy.deepcopy(base_config)
    resolved_config["experiment"] = resolved_config.get("experiment", {})
    resolved_config["experiment"]["name"] = "stage6g5d_pn_mode_switch_probe"
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(selected)} points × {len(VARIANTS)} variants")
        print(f"Episodes per point: {episodes_per_point}")
        print("No simulation executed.")
        _write_empty_csv(output_path / "mode_switch_raw_episodes.csv", ["variant", "point_id", "episode_index", "is_success", "reason"])
        _write_empty_csv(output_path / "mode_switch_variant_summary.csv", ["variant", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "mode_switch_seed_summary.csv", ["variant", "point_id", "eval_seed", "success_rate", "success_count", "total"])
        _write_empty_csv(output_path / "mode_switch_success_matrix.csv", ["point_id", "variant", "seed_0_success", "seed_1_success", "seed_2_success", "cross_seed_stable"])
        with open(output_path / "effective_runtime_flags.json", "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        _write_readme_result_block(output_path / "README_result_block.md", {})
        return requested_config

    print("\n=== Stage 6G.5D: PN Mode-Switch & VPP Offset Mechanism Probe ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    method_override = config.get("methods", {}).get("no_prediction", {})

    all_episodes = []
    summary_by_variant = {}
    effective_flags = {}

    for variant_name, variant_info in VARIANTS.items():
        print(f"\n--- Variant: {variant_name} ---")
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = variant_info["direct_track_mode"]
        variant_config["guidance"]["mode"] = variant_info["guidance_mode"]
        if "mode_switch" in variant_info:
            variant_config["guidance"]["mode_switch"] = variant_info["mode_switch"]

        env, agent, method_config, policy_type, ckpt = _make_env_agent(
            variant_config, method_override, allow_random_policy=allow_random_policy
        )

        ckpt_exists = ckpt and os.path.exists(ckpt)
        effective_flags[variant_name] = {
            "direct_track_mode": variant_info["direct_track_mode"],
            "guidance_mode": variant_info["guidance_mode"],
            "use_vpp": variant_info["use_vpp"],
            "policy_type": policy_type,
            "checkpoint_exists": ckpt_exists,
        }

        variant_success = 0
        variant_total = 0

        for point_id, pt in selected:
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            scenario["name"] = point_id

            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + int(point_id[2:]) * 1000 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, _ = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name="no_prediction",
                        )
                        ep_result["variant"] = variant_name
                        ep_result["point_id"] = point_id
                        ep_result["episode_index"] = ep_idx
                        ep_result["evaluation_seed"] = ev_seed
                        ep_result["episode_seed"] = episode_seed
                        all_episodes.append(ep_result)
                        status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                        print(f"  {point_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | {status}")
                    except Exception as exc:
                        print(f"  {point_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                        all_episodes.append({
                            "variant": variant_name, "point_id": point_id,
                            "episode_index": ep_idx, "evaluation_seed": ev_seed,
                            "episode_seed": episode_seed, "is_success": False,
                            "reason": f"exception:{exc}",
                        })
                    variant_total += 1
                    if all_episodes[-1].get("is_success"):
                        variant_success += 1

        env.close()
        summary_by_variant[variant_name] = {
            "success_rate": variant_success / variant_total if variant_total else 0.0,
            "success_count": variant_success,
            "total": variant_total,
        }

    # Save outputs
    with open(output_path / "effective_runtime_flags.json", "w", encoding="utf-8") as f:
        json.dump(effective_flags, f, indent=2, default=str)

    _write_csv_from_dicts(output_path / "mode_switch_raw_episodes.csv", all_episodes,
                          ["variant", "point_id", "episode_index", "evaluation_seed", "episode_seed",
                           "is_success", "reason", "min_range_m", "final_range_m", "final_ata_deg"])

    summary_rows = []
    for vname, s in summary_by_variant.items():
        row = {"variant": vname, **s}
        row.update(effective_flags[vname])
        summary_rows.append(row)
    _write_csv_from_dicts(output_path / "mode_switch_variant_summary.csv", summary_rows,
                          ["variant", "success_rate", "success_count", "total", "policy_type", "guidance_mode", "use_vpp"])

    # Seed summary and success matrix
    seed_summary = []
    success_matrix = []
    for vname in VARIANTS:
        for point_id, _ in selected:
            seed_success = {}
            for s in eval_seeds:
                eps = [e for e in all_episodes if e.get("variant") == vname and e.get("point_id") == point_id and e.get("evaluation_seed") == s]
                seed_success[s] = sum(1 for e in eps if e.get("is_success")) / max(1, len(eps))
                seed_summary.append({
                    "variant": vname, "point_id": point_id, "eval_seed": s,
                    "success_rate": seed_success[s], "success_count": sum(1 for e in eps if e.get("is_success")), "total": len(eps),
                })
            stable = all(seed_success.get(s, 0) > 0.5 for s in eval_seeds)
            success_matrix.append({
                "point_id": point_id, "variant": vname,
                "seed_0_success": seed_success.get(0, np.nan),
                "seed_1_success": seed_success.get(1, np.nan),
                "seed_2_success": seed_success.get(2, np.nan),
                "cross_seed_stable": stable,
            })

    _write_csv_from_dicts(output_path / "mode_switch_seed_summary.csv", seed_summary,
                          ["variant", "point_id", "eval_seed", "success_rate", "success_count", "total"])
    _write_csv_from_dicts(output_path / "mode_switch_success_matrix.csv", success_matrix,
                          ["point_id", "variant", "seed_0_success", "seed_1_success", "seed_2_success", "cross_seed_stable"])

    _write_readme_result_block(output_path / "README_result_block.md", summary_by_variant)
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


def _write_readme_result_block(path: Path, summary_by_variant: dict):
    lines = ["## Stage 6G.5D PN Mode-Switch Probe Results", ""]
    if not summary_by_variant:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    else:
        lines.append("| Variant | Success Rate | Success / Total |")
        lines.append("|---|---|---|")
        for vname, s in summary_by_variant.items():
            lines.append(f"| {vname} | {s['success_rate']*100:.1f}% | {s['success_count']}/{s['total']} |")
    lines.append("")
    lines.append(
        "> **Paper-safe note**: Results limited to tested candidate geometries. "
        "No universal claims about mode-switch efficacy are made."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5D PN Mode-Switch & VPP Offset Mechanism Probe")
    parser.add_argument("--candidate-points", type=str, nargs="+", default=["pt20", "pt29", "pt38"])
    parser.add_argument("--input-geometry", type=str,
                        default="outputs/stage6g5_geometry_smoke_real_seed0/geometry_smoke_points.csv")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--episodes-per-point", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--include-neighbor-failures", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5d_pn_mode_switch")
    args = parser.parse_args()

    run_mode_switch_probe(
        candidate_points=[(cid, None) for cid in args.candidate_points],
        output_dir=args.output_dir,
        input_geometry_csv=args.input_geometry,
        config_path=args.config,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
        include_neighbor_failures=args.include_neighbor_failures,
    )


if __name__ == "__main__":
    main()
