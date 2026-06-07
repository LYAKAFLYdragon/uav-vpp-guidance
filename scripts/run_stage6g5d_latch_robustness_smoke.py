#!/usr/bin/env python3
"""
Stage 6G.5D-R: Latch Robustness Smoke Test.

Small-scale smoke evaluating mode-switch latch behavior across:
  - Candidate success geometries (pt20/pt29/pt38)
  - Near-threshold geometries (aspect, range sweeps)
  - Negative controls (high aspect, low speed, long range)
  - Feasible non-tail-chase regression (Stage 6F favorable geometry)

Variants:
  pure_pn_no_vpp            — Baseline pure PN.
  mode_switch_latched_episode — Default episode latch (hold-for-episode).
  mode_switch_hysteresis    — Hysteresis exit policy (WIP placeholder).
  mode_switch_min_hold      — Minimum hold time exit policy (WIP placeholder).
  vpp_policy_los            — VPP + LOS baseline (expected fail on tail-chase).
  vpp_policy_pn_guidance    — VPP + PN baseline (expected fail on tail-chase).

Usage:
    python scripts/run_stage6g5d_latch_robustness_smoke.py \
        --output-dir outputs/stage6g5d_latch_robustness_smoke
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


SCENARIOS = {
    # --- Candidate successes ---
    "candidate_pt20": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": -500},
    "candidate_pt29": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 200, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    "candidate_pt38": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 500},
    # --- Near-threshold ---
    "near_aspect_10": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 10, "altitude_diff_m": 0},
    "near_aspect_15": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 15, "altitude_diff_m": 0},
    "near_aspect_20": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 20, "altitude_diff_m": 0},
    "near_range_1800": {"initial_range_m": 1800, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    "near_range_2400": {"initial_range_m": 2400, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    # --- Negative controls ---
    "neg_aspect_60": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 60, "altitude_diff_m": 0},
    "neg_aspect_90": {"initial_range_m": 2000, "ego_speed_mps": 340, "target_speed_mps": 120, "aspect_angle_deg": 90, "altitude_diff_m": 0},
    "neg_low_ego": {"initial_range_m": 2000, "ego_speed_mps": 150, "target_speed_mps": 120, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    "neg_low_closing": {"initial_range_m": 2000, "ego_speed_mps": 150, "target_speed_mps": 140, "aspect_angle_deg": 0, "altitude_diff_m": 0},
    # --- Feasible non-tail-chase regression (Stage 6F favorable) ---
    "regression_favorable": {"initial_range_m": 2000, "ego_speed_mps": 220, "target_speed_mps": 180, "aspect_angle_deg": 0, "altitude_diff_m": 0},
}

VARIANTS = {
    "pure_pn_no_vpp": {
        "description": "Baseline pure PN without VPP",
        "direct_track_mode": True,
        "guidance_mode": "proportional_navigation",
        "use_vpp": False,
    },
    "mode_switch_latched_episode": {
        "description": "Episode latch (default hold-for-episode)",
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
    "mode_switch_hysteresis": {
        "description": "Hysteresis exit policy (placeholder)",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
        "mode_switch": {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
            "exit_policy": "hysteresis",
            "exit_aspect_threshold_deg": 45.0,
            "exit_range_threshold_m": 500.0,
        },
    },
    "mode_switch_min_hold": {
        "description": "Minimum hold time exit policy (placeholder)",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
        "mode_switch": {
            "enabled": True,
            "aspect_threshold_deg": 15.0,
            "range_threshold_m": 3000.0,
            "closing_speed_threshold_mps": 100.0,
            "exit_policy": "min_hold",
            "min_hold_time_s": 2.0,
        },
    },
    "vpp_policy_los": {
        "description": "VPP policy + LOS guidance baseline",
        "direct_track_mode": False,
        "guidance_mode": "los_rate",
        "use_vpp": True,
    },
    "vpp_policy_pn_guidance": {
        "description": "VPP policy + PN guidance baseline",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
    },
}


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


def run_robustness_smoke(
    output_dir: str,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    episodes_per_point: int = 5,
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

    requested_config = {
        "experiment_name": "stage6g5d_latch_robustness_smoke",
        "scenarios": list(SCENARIOS.keys()),
        "variants": list(VARIANTS.keys()),
        "config_path": config_path,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
    }
    with open(output_path / "requested_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(requested_config, f, default_flow_style=False, sort_keys=False)

    resolved_config = copy.deepcopy(base_config)
    resolved_config["experiment"] = resolved_config.get("experiment", {})
    resolved_config["experiment"]["name"] = "stage6g5d_latch_robustness_smoke"
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(SCENARIOS)} scenarios × {len(VARIANTS)} variants")
        print("No simulation executed.")
        _write_empty_csv(output_path / "robustness_raw_episodes.csv",
                         ["variant", "scenario", "episode_index", "is_success", "reason"])
        _write_readme_result_block(output_path / "README_result_block.md", {})
        return requested_config

    print("\n=== Stage 6G.5D-R: Latch Robustness Smoke ===")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    method_override = config.get("methods", {}).get("no_prediction", {})

    all_episodes = []
    summary_by_variant_scenario = {}
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

        for scenario_name, pt in SCENARIOS.items():
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            scenario["name"] = scenario_name

            scenario_success = 0
            scenario_total = 0

            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + hash(scenario_name) % 1000 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, _ = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name="no_prediction",
                        )
                        ep_result["variant"] = variant_name
                        ep_result["scenario"] = scenario_name
                        ep_result["episode_index"] = ep_idx
                        ep_result["evaluation_seed"] = ev_seed
                        ep_result["episode_seed"] = episode_seed
                        all_episodes.append(ep_result)
                        status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                        print(f"  {scenario_name} | {variant_name} | ev={ev_seed} ep={ep_idx} | {status}")
                    except Exception as exc:
                        print(f"  {scenario_name} | {variant_name} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                        all_episodes.append({
                            "variant": variant_name, "scenario": scenario_name,
                            "episode_index": ep_idx, "evaluation_seed": ev_seed,
                            "episode_seed": episode_seed, "is_success": False,
                            "reason": f"exception:{exc}",
                        })
                    scenario_total += 1
                    if all_episodes[-1].get("is_success"):
                        scenario_success += 1

            summary_by_variant_scenario[(variant_name, scenario_name)] = {
                "success_rate": scenario_success / scenario_total if scenario_total else 0.0,
                "success_count": scenario_success,
                "total": scenario_total,
            }

        env.close()

    # Save outputs
    with open(output_path / "effective_runtime_flags.json", "w", encoding="utf-8") as f:
        json.dump(effective_flags, f, indent=2, default=str)

    _write_csv_from_dicts(output_path / "robustness_raw_episodes.csv", all_episodes,
                          ["variant", "scenario", "episode_index", "evaluation_seed", "episode_seed",
                           "is_success", "reason", "min_range_m", "final_range_m", "final_ata_deg"])

    summary_rows = []
    for (vname, sname), s in summary_by_variant_scenario.items():
        row = {"variant": vname, "scenario": sname, **s}
        row.update(effective_flags[vname])
        summary_rows.append(row)
    _write_csv_from_dicts(output_path / "robustness_summary.csv", summary_rows,
                          ["variant", "scenario", "success_rate", "success_count", "total",
                           "policy_type", "guidance_mode", "use_vpp"])

    _write_readme_result_block(output_path / "README_result_block.md", summary_by_variant_scenario)
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


def _write_readme_result_block(path: Path, summary: dict):
    lines = ["## Stage 6G.5D-R Latch Robustness Smoke Results", ""]
    if not summary:
        lines.append("*Status: dry-run only. No real episodes executed.*")
    else:
        lines.append("| Variant | Scenario | Success Rate | Success / Total |")
        lines.append("|---|---|---|---|")
        for (vname, sname), s in summary.items():
            lines.append(f"| {vname} | {sname} | {s['success_rate']*100:.1f}% | {s['success_count']}/{s['total']} |")
    lines.append("")
    lines.append(
        "> **Paper-safe note**: Results limited to tested scenarios. "
        "No universal claims about latch robustness are made. "
        "Negative controls and near-threshold sweeps are exploratory."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5D-R Latch Robustness Smoke")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--episodes-per-point", type=int, default=5)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5d_latch_robustness_smoke")
    args = parser.parse_args()

    run_robustness_smoke(
        output_dir=args.output_dir,
        config_path=args.config,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
