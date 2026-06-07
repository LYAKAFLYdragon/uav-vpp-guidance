#!/usr/bin/env python3
"""
Stage 6G.5D Task B: VPP Offset Mechanism Diagnostics.

Re-runs a small representative subset of Stage 6G.5C episodes with
per-step trajectory saving to extract VPP offset geometry and compare
against pure-PN successful trajectories.

Outputs:
    vpp_offset_distribution.csv
    vpp_anchor_geometry_by_episode.csv
    pn_success_vs_vpp_failure_diagnostics.csv
    stage6g5d_vpp_offset_mechanism.md
"""

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np

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
    return env, agent, method_config


def run_diagnostics(
    input_geometry_csv: str,
    output_dir: str,
    config_path: str = "config/experiment/stage6g5_wide_geometry_smoke.yaml",
    candidate_points: tuple = ("pt20", "pt29", "pt38"),
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_points = _load_geometry_points(input_geometry_csv)
    selected = [(int(cid[2:]), all_points[int(cid[2:])]) for cid in candidate_points]

    config = load_experiment_config(config_path)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    method_override = config.get("methods", {}).get("no_prediction", {})

    variants = {
        "pure_pn_no_vpp": {"direct_track_mode": True, "guidance_mode": "proportional_navigation"},
        "vpp_trained_ppo_los": {"direct_track_mode": False, "guidance_mode": "los_rate"},
        "vpp_policy_pn_guidance": {"direct_track_mode": False, "guidance_mode": "proportional_navigation"},
    }

    offset_records = []
    anchor_records = []
    diagnostic_rows = []

    for variant_name, variant_flags in variants.items():
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("guidance", {})
        variant_config["guidance"]["direct_track_mode"] = variant_flags["direct_track_mode"]
        variant_config["guidance"]["mode"] = variant_flags["guidance_mode"]

        env, agent, method_config = _make_env_agent(variant_config, method_override)

        for pt_idx, pt in selected:
            scenario = build_geometry_scenario(
                pt["initial_range_m"], pt["ego_speed_mps"], pt["target_speed_mps"],
                pt["aspect_angle_deg"], pt["altitude_diff_m"], base_altitude_m=5000.0,
            )
            scenario["name"] = f"pt{pt_idx}"
            episode_seed = pt_idx * 1000
            set_seed(episode_seed)
            ep_result, trajectory = evaluate_single_episode(
                env, agent, method_config, scenario=scenario, seed=episode_seed,
                save_trajectory=True, method_name="no_prediction",
            )

            # Analyze trajectory
            for step_data in trajectory:
                vx = step_data.get("virtual_x", np.nan)
                vy = step_data.get("virtual_y", np.nan)
                vz = step_data.get("virtual_z", np.nan)
                tx = step_data.get("target_x", np.nan)
                ty = step_data.get("target_y", np.nan)
                tz = step_data.get("target_z", np.nan)
                ego_x = step_data.get("ego_x", np.nan)
                ego_y = step_data.get("ego_y", np.nan)
                ego_z = step_data.get("ego_z", np.nan)

                offset = np.array([vx - tx, vy - ty, vz - tz]) if all(np.isfinite([vx, vy, vz, tx, ty, tz])) else np.full(3, np.nan)
                offset_norm = float(np.linalg.norm(offset)) if np.isfinite(offset).all() else np.nan

                offset_records.append({
                    "variant": variant_name,
                    "point_id": f"pt{pt_idx}",
                    "step": step_data["step"],
                    "offset_norm_m": offset_norm,
                    "offset_long_m": float(offset[0]) if np.isfinite(offset[0]) else np.nan,
                    "offset_lat_m": float(offset[1]) if np.isfinite(offset[1]) else np.nan,
                    "offset_vert_m": float(offset[2]) if np.isfinite(offset[2]) else np.nan,
                    "virtual_x": vx,
                    "virtual_y": vy,
                    "virtual_z": vz,
                })

                # Anchor geometry relative to target velocity and LOS
                los_vec = np.array([tx - ego_x, ty - ego_y, tz - ego_z])
                los_norm = np.linalg.norm(los_vec) if np.isfinite(los_vec).all() else np.nan
                anchor_records.append({
                    "variant": variant_name,
                    "point_id": f"pt{pt_idx}",
                    "step": step_data["step"],
                    "range_m": step_data.get("range_m", np.nan),
                    "ata_deg": step_data.get("ata_deg", np.nan),
                    "aspect_deg": step_data.get("aspect_deg", np.nan),
                    "offset_norm_m": offset_norm,
                    "los_norm_m": los_norm,
                    "altitude_m": step_data.get("altitude_m", np.nan),
                    "nz_cmd": step_data.get("nz_cmd", np.nan),
                    "roll_rate_cmd": step_data.get("roll_rate_cmd", np.nan),
                })

            # Episode-level diagnostic
            if trajectory:
                offset_norms = [r["offset_norm_m"] for r in offset_records if r["variant"] == variant_name and r["point_id"] == f"pt{pt_idx}" and np.isfinite(r["offset_norm_m"])]
                diagnostic_rows.append({
                    "variant": variant_name,
                    "point_id": f"pt{pt_idx}",
                    "is_success": ep_result.get("is_success", False),
                    "reason": ep_result.get("reason", ""),
                    "min_range_m": ep_result.get("min_range_m", np.nan),
                    "final_range_m": ep_result.get("final_range_m", np.nan),
                    "final_ata_deg": ep_result.get("final_ata_deg", np.nan),
                    "mean_offset_norm_m": float(np.mean(offset_norms)) if offset_norms else np.nan,
                    "max_offset_norm_m": float(np.max(offset_norms)) if offset_norms else np.nan,
                    "altitude_loss_rate": ep_result.get("altitude_loss_rate", np.nan),
                    "nz_saturation_rate": ep_result.get("nz_cmd_saturation_rate", np.nan),
                    "roll_rate_saturation_rate": ep_result.get("roll_rate_cmd_saturation_rate", np.nan),
                })

        env.close()

    # Write outputs
    _write_csv(output_path / "vpp_offset_distribution.csv", offset_records)
    _write_csv(output_path / "vpp_anchor_geometry_by_episode.csv", anchor_records)
    _write_csv(output_path / "pn_success_vs_vpp_failure_diagnostics.csv", diagnostic_rows)
    _write_md(output_path / "stage6g5d_vpp_offset_mechanism.md", diagnostic_rows)

    print(f"Diagnostics written to {output_dir}")


def _write_csv(path: Path, rows: list):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_md(path: Path, rows: list):
    lines = ["# Stage 6G.5D VPP Offset Mechanism Diagnostics", ""]
    lines.append("## Episode-Level Diagnostic Summary")
    lines.append("")
    lines.append("| Variant | Point | Success | Min Range (m) | Final Range (m) | Mean Offset (m) | Max Offset (m) | Reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['point_id']} | {r['is_success']} | "
            f"{r['min_range_m']:.1f} | {r['final_range_m']:.1f} | "
            f"{r['mean_offset_norm_m']:.1f} | {r['max_offset_norm_m']:.1f} | {r['reason']} |"
        )
    lines.append("")
    lines.append(
        "> **Paper-safe note**: Diagnostic results are limited to one representative episode per point-variant. "
        "No universal claims about VPP offset behavior are made."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 6G.5D VPP Offset Mechanism Diagnostics")
    parser.add_argument("--input-geometry", type=str,
                        default="outputs/stage6g5_geometry_smoke_real_seed0/geometry_smoke_points.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g5d_vpp_offset_diagnostics")
    parser.add_argument("--candidate-points", type=str, nargs="+", default=["pt20", "pt29", "pt38"])
    args = parser.parse_args()

    run_diagnostics(
        input_geometry_csv=args.input_geometry,
        output_dir=args.output_dir,
        candidate_points=tuple(args.candidate_points),
    )


if __name__ == "__main__":
    main()
