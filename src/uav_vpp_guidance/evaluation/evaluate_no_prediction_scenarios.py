"""
No-Prediction VPP Baseline — Scenario-based Batch Evaluation.

Supports both SimplePointMass and JSBSim backends via --backend flag.

Usage:
    python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios \
        --config config/experiment/no_prediction_vpp_scenarios.yaml \
        --backend simple \
        --episodes 5 --seeds 0 1 --save-trajectories

    python -m uav_vpp_guidance.evaluation.evaluate_no_prediction_scenarios \
        --config config/experiment/no_prediction_vpp_jsbsim.yaml \
        --backend jsbsim \
        --episodes 2 --seeds 0 --save-trajectories

Outputs:
    outputs/tables/no_prediction_vpp/{backend}/scenario_metrics.json
    outputs/tables/no_prediction_vpp/{backend}/scenario_metrics.csv
    outputs/trajectories/no_prediction_vpp/{backend}/{scenario}/seed_{seed}/episode_{episode}.csv
"""

import argparse
import csv
import json
import os
import numpy as np
from typing import List, Dict, Any

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy


def compute_ego_score(rel_state: dict) -> float:
    """Compute ego geometric advantage score. Higher = better for ego."""
    range_m = rel_state.get("range_m", 5000.0)
    ata_deg = np.rad2deg(rel_state.get("ata_rad", np.pi))
    aa_deg = np.rad2deg(rel_state.get("aa_rad", np.pi))
    range_score = max(0.0, 1.0 - abs(range_m - 900.0) / 3000.0)
    ata_score = max(0.0, 1.0 - ata_deg / 180.0)
    aa_score = max(0.0, 1.0 - aa_deg / 180.0)
    return (range_score + ata_score + aa_score) / 3.0


def compute_target_score(rel_state: dict) -> float:
    """Compute target escape score. Higher = better for target."""
    range_m = rel_state.get("range_m", 5000.0)
    ata_deg = np.rad2deg(rel_state.get("ata_rad", np.pi))
    range_score = min(1.0, range_m / 4000.0)
    ata_score = min(1.0, ata_deg / 180.0)
    return (range_score + ata_score) / 2.0


def evaluate_scenario(
    env: CloseRangeTrackingEnv,
    scenario_name: str,
    scenario_cfg: dict,
    num_episodes: int = 10,
    seeds: List[int] = None,
    policy=None,
    save_trajectories: bool = False,
    output_root: str = "outputs",
    backend: str = "simple",
) -> Dict[str, Any]:
    """
    Evaluate a single scenario across multiple seeds and episodes.

    Returns:
        dict: Aggregated metrics for the scenario.
    """
    if seeds is None:
        seeds = [0]

    all_episodes = []
    per_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        seed_episodes = []

        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)

            obs = env.reset(scenario=scenario_cfg, seed=ep_seed)
            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            min_ata_deg = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"

            instant_success_count = 0
            advantage_steps = 0
            time_to_first_advantage = None
            ego_scores = []
            target_scores = []

            # Control metrics
            nz_cmds = []
            roll_rate_cmds = []
            throttle_cmds = []
            saturation_flags = []
            altitudes = []
            speeds = []

            trajectory = []

            for step in range(env.max_steps):
                if policy is not None:
                    rel_state = obs.get("relative_state", {})
                    own_state = obs.get("own_state", {})
                    target_state = obs.get("target_state", {})
                    action = policy.get_action(own_state, target_state, rel_state)
                else:
                    action = rng.uniform(-1.0, 1.0, size=3).astype(np.float64)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = np.rad2deg(rel_state.get("ata_rad", 0.0))
                aa_deg = np.rad2deg(rel_state.get("aa_rad", 0.0))
                min_range = min(min_range, range_m)
                min_ata_deg = min(min_ata_deg, ata_deg)
                final_range = range_m
                final_ata = rel_state.get("ata_rad", 0.0)

                if range_m <= 900.0 and ata_deg <= 25.0:
                    instant_success_count += 1

                ego_s = compute_ego_score(rel_state)
                target_s = compute_target_score(rel_state)
                ego_scores.append(ego_s)
                target_scores.append(target_s)

                if ego_s > target_s:
                    advantage_steps += 1
                    if time_to_first_advantage is None:
                        time_to_first_advantage = step

                # Control metrics
                nz_cmds.append(abs(info.get("nz_cmd", 0.0)))
                roll_rate_cmds.append(abs(info.get("roll_rate_cmd", 0.0)))
                throttle_cmds.append(info.get("throttle_cmd", 0.5))
                saturation_flags.append(info.get("saturation_flag", False))

                # Altitude and speed
                own_s = info.get("own_state", {})
                altitudes.append(own_s.get("altitude_m", np.nan))
                speeds.append(own_s.get("speed_mps", own_s.get("vt_mps", np.nan)))

                # Trajectory recording
                target_s_state = info.get("target_state", {})
                vp = info.get("virtual_point", {})
                vp_pos = vp.get("position", np.full(3, np.nan))
                if not isinstance(vp_pos, np.ndarray):
                    vp_pos = np.array(vp_pos) if hasattr(vp_pos, "__len__") else np.full(3, np.nan)

                own_pos = own_s.get("position_m", own_s.get("position_neu", np.full(3, np.nan)))
                target_pos = target_s_state.get("position_m", target_s_state.get("position_neu", np.full(3, np.nan)))
                own_vel = own_s.get("velocity_vector_mps", own_s.get("velocity_ned", np.full(3, np.nan)))
                target_vel = target_s_state.get("velocity_vector_mps", target_s_state.get("velocity_ned", np.full(3, np.nan)))

                # Attitude
                own_att = own_s.get("attitude_rpy", np.full(3, np.nan))
                target_att = target_s_state.get("attitude_rpy", np.full(3, np.nan))
                if not isinstance(own_att, np.ndarray):
                    own_att = np.array(own_att) if hasattr(own_att, "__len__") else np.full(3, np.nan)
                if not isinstance(target_att, np.ndarray):
                    target_att = np.array(target_att) if hasattr(target_att, "__len__") else np.full(3, np.nan)

                trajectory.append({
                    "step": step,
                    "time": step * env.env_config.get("high_level_dt", 0.2),
                    "backend": backend,
                    "ego_x": float(own_pos[0]) if len(own_pos) > 0 else np.nan,
                    "ego_y": float(own_pos[1]) if len(own_pos) > 1 else np.nan,
                    "ego_z": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
                    "ego_vx": float(own_vel[0]) if len(own_vel) > 0 else np.nan,
                    "ego_vy": float(own_vel[1]) if len(own_vel) > 1 else np.nan,
                    "ego_vz": float(own_vel[2]) if len(own_vel) > 2 else np.nan,
                    "ego_speed": float(own_s.get("speed_mps", own_s.get("vt_mps", np.nan))),
                    "ego_roll": float(own_att[0]) if len(own_att) > 0 else np.nan,
                    "ego_pitch": float(own_att[1]) if len(own_att) > 1 else np.nan,
                    "ego_yaw": float(own_att[2]) if len(own_att) > 2 else np.nan,
                    "target_x": float(target_pos[0]) if len(target_pos) > 0 else np.nan,
                    "target_y": float(target_pos[1]) if len(target_pos) > 1 else np.nan,
                    "target_z": float(target_pos[2]) if len(target_pos) > 2 else np.nan,
                    "target_vx": float(target_vel[0]) if len(target_vel) > 0 else np.nan,
                    "target_vy": float(target_vel[1]) if len(target_vel) > 1 else np.nan,
                    "target_vz": float(target_vel[2]) if len(target_vel) > 2 else np.nan,
                    "target_speed": float(target_s_state.get("speed_mps", target_s_state.get("vt_mps", np.nan))),
                    "range_m": range_m,
                    "ata_deg": ata_deg,
                    "aspect_deg": aa_deg,
                    "los_rate": rel_state.get("range_rate_mps", np.nan),
                    "virtual_x": float(vp_pos[0]) if len(vp_pos) > 0 else np.nan,
                    "virtual_y": float(vp_pos[1]) if len(vp_pos) > 1 else np.nan,
                    "virtual_z": float(vp_pos[2]) if len(vp_pos) > 2 else np.nan,
                    "nz_cmd": info.get("nz_cmd", np.nan),
                    "roll_rate_cmd": info.get("roll_rate_cmd", np.nan),
                    "throttle_cmd": info.get("throttle_cmd", np.nan),
                    "elevator_cmd": info.get("elevator_cmd", np.nan),
                    "aileron_cmd": info.get("aileron_cmd", np.nan),
                    "rudder_cmd": info.get("rudder_cmd", np.nan),
                    "throttle_actual": info.get("throttle_actual", np.nan),
                    "saturation_flag": info.get("saturation_flag", False),
                    "ego_score": ego_s,
                    "target_score": target_s,
                    "done": terminated or truncated,
                    "termination_reason": info.get("reason", ""),
                })

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            mean_ego_score = float(np.mean(ego_scores)) if ego_scores else 0.0
            mean_target_score = float(np.mean(target_scores)) if target_scores else 0.0
            score_win = mean_ego_score > mean_target_score

            ep_result = {
                "episode": ep,
                "seed": seed,
                "scenario": scenario_name,
                "return": ep_reward,
                "length": ep_length,
                "min_range_m": min_range,
                "min_ata_deg": min_ata_deg,
                "final_range_m": final_range,
                "final_ata_deg": float(np.rad2deg(final_ata)),
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
                "instant_success_steps": instant_success_count,
                "instant_success_rate": instant_success_count / max(1, ep_length),
                "score_win": score_win,
                "mean_ego_score": mean_ego_score,
                "mean_target_score": mean_target_score,
                "time_to_first_advantage": time_to_first_advantage,
                "advantage_hold_steps": advantage_steps,
                "advantage_hold_time": advantage_steps * env.env_config.get("high_level_dt", 0.2),
                # Control metrics
                "mean_abs_nz_cmd": float(np.mean(nz_cmds)) if nz_cmds else None,
                "max_abs_nz_cmd": float(np.max(nz_cmds)) if nz_cmds else None,
                "mean_abs_roll_rate_cmd": float(np.mean(roll_rate_cmds)) if roll_rate_cmds else None,
                "max_abs_roll_rate_cmd": float(np.max(roll_rate_cmds)) if roll_rate_cmds else None,
                "mean_throttle_cmd": float(np.mean(throttle_cmds)) if throttle_cmds else None,
                "min_altitude": float(np.min(altitudes)) if altitudes else None,
                "max_altitude_loss": float(np.max(altitudes) - np.min(altitudes)) if altitudes else None,
                "mean_speed": float(np.mean(speeds)) if speeds else None,
                "min_speed": float(np.min(speeds)) if speeds else None,
                "max_speed": float(np.max(speeds)) if speeds else None,
                "control_saturation_rate": sum(saturation_flags) / max(1, len(saturation_flags)),
            }
            seed_episodes.append(ep_result)
            all_episodes.append(ep_result)

            if save_trajectories:
                traj_dir = os.path.join(
                    output_root, "trajectories", "no_prediction_vpp",
                    backend, scenario_name, f"seed_{seed}"
                )
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"episode_{ep}.csv")
                if trajectory:
                    with open(traj_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                        writer.writeheader()
                        writer.writerows(trajectory)

        per_seed_results[f"seed_{seed}"] = seed_episodes

    returns = [ep["return"] for ep in all_episodes]
    lengths = [ep["length"] for ep in all_episodes]
    min_ranges = [ep["min_range_m"] for ep in all_episodes]
    min_atas = [ep["min_ata_deg"] for ep in all_episodes]
    final_ranges = [ep["final_range_m"] for ep in all_episodes]
    final_atas = [ep["final_ata_deg"] for ep in all_episodes]
    tta = [ep["time_to_first_advantage"] for ep in all_episodes if ep["time_to_first_advantage"] is not None]
    adv_times = [ep["advantage_hold_time"] for ep in all_episodes]

    # Aggregate control metrics
    all_nz = [e["mean_abs_nz_cmd"] for e in all_episodes if e["mean_abs_nz_cmd"] is not None]
    all_roll = [e["mean_abs_roll_rate_cmd"] for e in all_episodes if e["mean_abs_roll_rate_cmd"] is not None]
    all_sat = [e["control_saturation_rate"] for e in all_episodes if e["control_saturation_rate"] is not None]
    all_alt = [e["min_altitude"] for e in all_episodes if e["min_altitude"] is not None]
    all_speed = [e["mean_speed"] for e in all_episodes if e["mean_speed"] is not None]

    metrics = {
        "scenario": scenario_name,
        "backend": backend,
        "num_episodes": len(all_episodes),
        "num_seeds": len(seeds),
        "episode_return": float(np.mean(returns)) if returns else None,
        "success_rate": sum(1 for ep in all_episodes if ep["is_success"]) / max(1, len(all_episodes)),
        "instant_success_rate": float(np.mean([ep["instant_success_rate"] for ep in all_episodes])),
        "score_win_rate": sum(1 for ep in all_episodes if ep["score_win"]) / max(1, len(all_episodes)),
        "failure_rate": sum(1 for ep in all_episodes if not ep["is_success"]) / max(1, len(all_episodes)),
        "crash_rate": sum(1 for ep in all_episodes if ep["is_crash"]) / max(1, len(all_episodes)),
        "out_of_bounds_rate": sum(1 for ep in all_episodes if ep["is_out_of_bounds"]) / max(1, len(all_episodes)),
        "timeout_rate": sum(1 for ep in all_episodes if ep["is_timeout"]) / max(1, len(all_episodes)),
        "simultaneous_rate": 0.0,
        "mean_final_range": float(np.mean(final_ranges)) if final_ranges else None,
        "mean_final_ata_deg": float(np.mean(final_atas)) if final_atas else None,
        "mean_min_range": float(np.mean(min_ranges)) if min_ranges else None,
        "mean_min_ata_deg": float(np.mean(min_atas)) if min_atas else None,
        "mean_time_to_first_advantage": float(np.mean(tta)) if tta else None,
        "mean_advantage_hold_time": float(np.mean(adv_times)) if adv_times else None,
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
        "mean_score_ego": float(np.mean([ep["mean_ego_score"] for ep in all_episodes])),
        "mean_score_target": float(np.mean([ep["mean_target_score"] for ep in all_episodes])),
        # Extended control metrics
        "mean_abs_nz_cmd": float(np.mean(all_nz)) if all_nz else None,
        "max_abs_nz_cmd": float(np.max([e["max_abs_nz_cmd"] for e in all_episodes if e["max_abs_nz_cmd"] is not None])) if all_episodes else None,
        "mean_abs_roll_rate_cmd": float(np.mean(all_roll)) if all_roll else None,
        "max_abs_roll_rate_cmd": float(np.max([e["max_abs_roll_rate_cmd"] for e in all_episodes if e["max_abs_roll_rate_cmd"] is not None])) if all_episodes else None,
        "mean_throttle_cmd": float(np.mean([e["mean_throttle_cmd"] for e in all_episodes if e["mean_throttle_cmd"] is not None])) if all_episodes else None,
        "min_altitude": float(np.min(all_alt)) if all_alt else None,
        "max_altitude_loss": float(np.mean([e["max_altitude_loss"] for e in all_episodes if e["max_altitude_loss"] is not None])) if all_episodes else None,
        "mean_speed": float(np.mean(all_speed)) if all_speed else None,
        "min_speed": float(np.min([e["min_speed"] for e in all_episodes if e["min_speed"] is not None])) if all_episodes else None,
        "max_speed": float(np.max([e["max_speed"] for e in all_episodes if e["max_speed"] is not None])) if all_episodes else None,
        "control_saturation_rate": float(np.mean(all_sat)) if all_sat else None,
        "episodes": all_episodes,
        "per_seed": per_seed_results,
    }
    return metrics


def save_metrics_csv(metrics_list: List[dict], csv_path: str):
    """Save scenario metrics to CSV."""
    if not metrics_list:
        return
    scalar_keys = [
        "scenario", "backend", "num_episodes", "num_seeds", "episode_return",
        "success_rate", "instant_success_rate", "score_win_rate",
        "failure_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        "simultaneous_rate", "mean_final_range", "mean_final_ata_deg",
        "mean_min_range", "mean_min_ata_deg", "mean_time_to_first_advantage",
        "mean_advantage_hold_time", "mean_episode_length",
        "mean_score_ego", "mean_score_target",
        "mean_abs_nz_cmd", "max_abs_nz_cmd",
        "mean_abs_roll_rate_cmd", "max_abs_roll_rate_cmd",
        "mean_throttle_cmd", "min_altitude", "max_altitude_loss",
        "mean_speed", "min_speed", "max_speed", "control_saturation_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for m in metrics_list:
            row = {k: m.get(k, "") for k in scalar_keys}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Scenario-based Evaluation for No-Prediction VPP")
    parser.add_argument("--config", type=str, required=True, help="Path to scenario config YAML")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed per scenario")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--rule-mode", type=str, default=None, choices=["pure_pursuit", "lag_pursuit", "lead_pursuit"])
    parser.add_argument("--save-trajectories", action="store_true", help="Save per-episode trajectory CSVs")
    args = parser.parse_args()

    base_config = load_yaml_config(args.config)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(args.config), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    config = merge_config(merged, base_config)

    # Override backend from CLI if provided
    config["backend"] = args.backend

    scenarios = config.get("scenarios", {})
    if not scenarios:
        print("No scenarios found in config.")
        return

    env = CloseRangeTrackingEnv(config)

    policy = None
    if args.rule_mode is not None:
        policy = RuleBasedPursuitPolicy(mode=args.rule_mode)
        print(f"Using rule-based policy: {args.rule_mode}")
    else:
        print("Using random policy")

    print(f"Backend: {args.backend}")

    all_scenario_metrics = []
    for scenario_name, scenario_cfg in scenarios.items():
        print(f"\n=== Evaluating scenario: {scenario_name} ===")
        metrics = evaluate_scenario(
            env, scenario_name, scenario_cfg,
            num_episodes=args.episodes,
            seeds=args.seeds,
            policy=policy,
            save_trajectories=args.save_trajectories,
            output_root=config.get("experiment", {}).get("output_root", "outputs"),
            backend=args.backend,
        )
        all_scenario_metrics.append(metrics)

        print(f"  Episodes: {metrics['num_episodes']}, Seeds: {metrics['num_seeds']}")
        print(f"  Success rate: {metrics['success_rate']:.3f}")
        print(f"  Score win rate: {metrics['score_win_rate']:.3f}")
        print(f"  Crash rate: {metrics['crash_rate']:.3f}")
        print(f"  OOB rate: {metrics['out_of_bounds_rate']:.3f}")
        print(f"  Timeout rate: {metrics['timeout_rate']:.3f}")
        print(f"  Mean return: {metrics['episode_return']:.2f}")

    env.close()

    tables_dir = os.path.join("outputs", "tables", "no_prediction_vpp", args.backend)
    os.makedirs(tables_dir, exist_ok=True)

    json_path = os.path.join(tables_dir, "scenario_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_scenario_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nScenario metrics JSON saved to: {json_path}")

    csv_path = os.path.join(tables_dir, "scenario_metrics.csv")
    save_metrics_csv(all_scenario_metrics, csv_path)
    print(f"Scenario metrics CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
