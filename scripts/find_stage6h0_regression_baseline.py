#!/usr/bin/env python3
"""
Stage 6H.0-lite Preflight: Find VPP regression baseline.

Searches a geometry grid for cases where VPP+LOS or VPP+PN succeed.
These become the regression baseline for verifying that mode-switch
threshold search does not harm feasible geometries where VPP already works.

Post-6G.5D insight: VPP+LOS succeeds in crossing/head-on geometries
(aspect ~180deg) but fails in tail-chase (aspect ~0deg). The search
space now includes 180deg to capture these successes.

Search space:
    aspect_angle_deg:   [0, 30, 45, 60, 90, 180]
    initial_range_m:    [1200, 1600, 2000]
    ego_speed_mps:      [220, 280, 340]
    target_speed_mps:   [120, 160, 200]
    altitude_diff_m:    [-500, 0, 500]

Variants:
    vpp_policy_los          — VPP + LOS guidance
    vpp_policy_pn_guidance  — VPP + PN guidance

Candidate rule:
    success_rate >= 0.80
    mean virtual_point_source == "vpp_policy" (verifies VPP is actually used)

Usage:
    python scripts/find_stage6h0_regression_baseline.py \
        --output-dir outputs/stage6h0_regression_baseline_search
"""

import argparse
import copy
import csv
import json
import os
import sys
from itertools import product
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


GEOMETRY_GRID = {
    "aspect_angle_deg": [0, 30, 45, 60, 90, 180],
    "initial_range_m": [1200, 1600, 2000],
    "ego_speed_mps": [220, 280, 340],
    "target_speed_mps": [120, 160, 200],
    "altitude_diff_m": [-500, 0, 500],
}

VARIANTS = {
    "vpp_policy_los": {
        "description": "VPP policy + LOS guidance",
        "direct_track_mode": False,
        "guidance_mode": "los_rate",
        "use_vpp": True,
    },
    "vpp_policy_pn_guidance": {
        "description": "VPP policy + PN guidance",
        "direct_track_mode": False,
        "guidance_mode": "proportional_navigation",
        "use_vpp": True,
    },
}

SUCCESS_RATE_THRESHOLD = 0.80
MIN_ASPECT_DEG = 0.0


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


def _build_scenarios():
    keys = list(GEOMETRY_GRID.keys())
    scenarios = []
    for values in product(*GEOMETRY_GRID.values()):
        pt = dict(zip(keys, values))
        pt["base_altitude_m"] = 5000.0
        scenarios.append(pt)
    return scenarios


def run_baseline_search(
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

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    requested_config = {
        "experiment_name": "stage6h0_regression_baseline_search",
        "geometry_grid": GEOMETRY_GRID,
        "variants": list(VARIANTS.keys()),
        "config_path": config_path,
        "episodes_per_point": episodes_per_point,
        "eval_seeds": eval_seeds,
        "dry_run": dry_run,
        "allow_random_policy": allow_random_policy,
        "success_rate_threshold": SUCCESS_RATE_THRESHOLD,
        "min_aspect_deg": MIN_ASPECT_DEG,
    }
    with open(output_path / "requested_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(requested_config, f, default_flow_style=False, sort_keys=False)

    scenarios = _build_scenarios()
    total_scenarios = len(scenarios) * len(VARIANTS)

    if dry_run:
        print("\n=== DRY RUN ===")
        print(f"Would evaluate {len(scenarios)} geometries × {len(VARIANTS)} variants = {total_scenarios} cells")
        print("No simulation executed.")
        _write_empty_csv(output_path / "regression_baseline_candidates.csv",
                         ["variant", "scenario_id", "aspect_angle_deg", "initial_range_m",
                          "ego_speed_mps", "target_speed_mps", "altitude_diff_m",
                          "success_rate", "success_count", "total"])
        _write_summary_md(output_path / "regression_baseline_summary.md", [], scenarios)
        return requested_config

    print(f"\n=== Stage 6H.0 Preflight: Regression Baseline Search ===")
    print(f"Grid: {len(scenarios)} geometries × {len(VARIANTS)} variants × {len(eval_seeds)} seeds × {episodes_per_point} episodes")
    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False

    method_override = config.get("methods", {}).get("no_prediction", {})
    all_episodes = []
    cell_results = []

    for variant_name, variant_info in VARIANTS.items():
        print(f"\n--- Variant: {variant_name} ---")
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = variant_info["direct_track_mode"]
        variant_config["guidance"]["mode"] = variant_info["guidance_mode"]

        env, agent, method_config, policy_type, ckpt = _make_env_agent(
            variant_config, method_override, allow_random_policy=allow_random_policy
        )

        for idx, pt in enumerate(scenarios):
            scenario_id = f"a{int(pt['aspect_angle_deg'])}_r{int(pt['initial_range_m'])}_e{int(pt['ego_speed_mps'])}_t{int(pt['target_speed_mps'])}_h{int(pt['altitude_diff_m'])}"
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            scenario["name"] = scenario_id

            cell_success = 0
            cell_total = 0
            vpp_source_count = 0
            step_count = 0

            for ev_seed in eval_seeds:
                for ep_idx in range(episodes_per_point):
                    episode_seed = ev_seed * 100000 + idx * 100 + ep_idx
                    set_seed(episode_seed)
                    try:
                        ep_result, traj = evaluate_single_episode(
                            env, agent, method_config, scenario=scenario, seed=episode_seed,
                            save_trajectory=False, method_name="no_prediction",
                        )
                        ep_result["variant"] = variant_name
                        ep_result["scenario_id"] = scenario_id
                        ep_result["episode_index"] = ep_idx
                        ep_result["evaluation_seed"] = ev_seed
                        ep_result["episode_seed"] = episode_seed
                        all_episodes.append(ep_result)
                        status = "SUCCESS" if ep_result["is_success"] else ep_result.get("reason", "FAIL")
                        print(f"  {scenario_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | {status}")

                        # Count VPP source from trajectory if available
                        if traj and len(traj) > 0:
                            for step in traj:
                                info = step.get("info", {})
                                src = info.get("virtual_point_source")
                                if src == "vpp_policy":
                                    vpp_source_count += 1
                                if src is not None:
                                    step_count += 1
                    except Exception as exc:
                        print(f"  {scenario_id} | {variant_name} | ev={ev_seed} ep={ep_idx} | EXCEPTION: {exc}")
                        all_episodes.append({
                            "variant": variant_name, "scenario_id": scenario_id,
                            "episode_index": ep_idx, "evaluation_seed": ev_seed,
                            "episode_seed": episode_seed, "is_success": False,
                            "reason": f"exception:{exc}",
                        })
                    cell_total += 1
                    if all_episodes[-1].get("is_success"):
                        cell_success += 1

            success_rate = cell_success / cell_total if cell_total else 0.0
            vpp_fraction = vpp_source_count / max(1, step_count)

            cell_results.append({
                "variant": variant_name,
                "scenario_id": scenario_id,
                "aspect_angle_deg": pt["aspect_angle_deg"],
                "initial_range_m": pt["initial_range_m"],
                "ego_speed_mps": pt["ego_speed_mps"],
                "target_speed_mps": pt["target_speed_mps"],
                "altitude_diff_m": pt["altitude_diff_m"],
                "success_rate": success_rate,
                "success_count": cell_success,
                "total": cell_total,
                "vpp_fraction": vpp_fraction,
                "is_candidate": (success_rate >= SUCCESS_RATE_THRESHOLD
                                 and pt["aspect_angle_deg"] >= MIN_ASPECT_DEG
                                 and vpp_fraction >= 0.5),
            })

        env.close()

    # Save outputs
    _write_csv_from_dicts(output_path / "regression_baseline_candidates.csv", cell_results,
                          ["variant", "scenario_id", "aspect_angle_deg", "initial_range_m",
                           "ego_speed_mps", "target_speed_mps", "altitude_diff_m",
                           "success_rate", "success_count", "total", "vpp_fraction", "is_candidate"])

    _write_summary_md(output_path / "regression_baseline_summary.md", cell_results, scenarios)
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


def _write_summary_md(path: Path, cell_results: list, scenarios: list):
    candidates = [c for c in cell_results if c.get("is_candidate")]
    lines = ["# Stage 6H.0 Preflight: Regression Baseline Search Summary", ""]
    lines.append(f"**Total geometries evaluated**: {len(scenarios)} × 2 variants = {len(scenarios)*2} cells")
    lines.append(f"**Candidate threshold**: success_rate ≥ {SUCCESS_RATE_THRESHOLD}, VPP fraction ≥ 0.5 (aspect unrestricted)")
    lines.append("")

    if not candidates:
        lines.append("## ⚠️ No candidate geometries found.")
        lines.append("")
        lines.append("All tested non-tail-chase geometries failed under VPP+LOS or VPP+PN in the simple backend.")
        lines.append("Implications:")
        lines.append("- 6H.0-lite threshold search cannot proceed with a validated regression baseline.")
        lines.append("- Options: (a) expand search grid, (b) relax success threshold, (c) investigate why VPP fails on all non-tail-chase geometries.")
        lines.append("")
        lines.append("**Next step**: Do NOT enter full 6H.0-lite threshold search until a regression baseline is confirmed.")
    else:
        lines.append(f"## ✅ {len(candidates)} candidate geometries found")
        lines.append("")
        lines.append("| Variant | Scenario | Aspect | Range | Ego | Target | Alt | Success Rate | VPP Fraction |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in candidates:
            lines.append(
                f"| {c['variant']} | {c['scenario_id']} | {c['aspect_angle_deg']}° | {c['initial_range_m']}m | "
                f"{c['ego_speed_mps']}m/s | {c['target_speed_mps']}m/s | {c['altitude_diff_m']}m | "
                f"{c['success_rate']*100:.1f}% | {c['vpp_fraction']*100:.1f}% |"
            )
        lines.append("")
        lines.append("These geometries can be used as regression baselines for 6H.0-lite threshold search.")

    lines.append("")
    lines.append("> **Paper-safe note**: Results limited to tested grid. No universal claims about VPP feasibility are made."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6H.0 Preflight: Regression Baseline Search")
    parser.add_argument("--config", type=str, default="config/experiment/stage6g5_wide_geometry_smoke.yaml")
    parser.add_argument("--episodes-per-point", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-random-policy", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6h0_regression_baseline_search")
    args = parser.parse_args()

    run_baseline_search(
        output_dir=args.output_dir,
        config_path=args.config,
        episodes_per_point=args.episodes_per_point,
        eval_seeds=args.eval_seeds,
        dry_run=args.dry_run,
        allow_random_policy=args.allow_random_policy,
    )


if __name__ == "__main__":
    main()
