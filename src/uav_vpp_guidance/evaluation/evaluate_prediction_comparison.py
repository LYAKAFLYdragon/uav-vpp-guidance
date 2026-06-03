"""
Prediction comparison evaluation runner.

Compares No-Prediction, CV-Prediction, and CA-Prediction policies
on the same scenarios and seeds.

Supports:
- Random scenario sampling (default)
- Fixed scenario evaluation (--scenarios favorable neutral ...)
- Loading trained checkpoints (--checkpoint)
- Per-scenario breakdown statistics

Usage:
    # Evaluate with random scenarios (no checkpoint, random policy)
    python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --backend simple --episodes 10 --seeds 0 1 2

    # Evaluate fixed scenarios with checkpoint
    python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --checkpoint outputs/experiments/vpp_ppo_cv_prediction/checkpoints/best.pt \
        --backend simple --episodes 10 --seeds 0 1 2 \
        --scenarios favorable neutral disadvantage challenging \
        --save-trajectories
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def load_experiment_config(config_path):
    """Load and merge experiment configuration with includes."""
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def sample_scenario(config, rng):
    """Sample a random scenario from config."""
    scenarios = config.get("scenarios", {})
    if not scenarios:
        return None
    name = rng.choice(list(scenarios.keys()))
    return scenarios[name]


def evaluate_single_episode(env, agent, config, scenario=None, seed=0, save_trajectory=False, method_name=""):
    """Evaluate a single episode and return metrics + trajectory."""
    obs = env.reset(scenario=scenario, seed=seed)
    ep_reward = 0.0
    ep_length = 0
    min_range = float("inf")
    min_ata_deg = float("inf")
    final_range = 0.0
    final_ata = 0.0
    reason = "timeout"
    trajectory = []
    prediction_enabled_count = 0
    prediction_valid_count = 0
    prediction_fallback_count = 0
    prediction_errors = []
    virtual_point_shifts = []
    anchor_shifts = []
    time_to_first_advantage = None
    advantage_hold_steps = 0
    ego_score_sum = 0.0
    target_score_sum = 0.0

    # 离线 prediction error 对齐记录
    # 每个元素: (step, predicted_target_pos, true_target_pos)
    prediction_records = []

    # 预测时间窗口（用于离线对齐）
    lookahead_time_s = env.config.get("trajectory_prediction", {}).get("prediction", {}).get("lookahead_time_s", 1.0)
    high_level_dt = env.env_config.get("high_level_dt", 0.2)
    horizon_steps = max(1, int(round(lookahead_time_s / high_level_dt)))

    for step in range(env.max_steps):
        obs_vec = obs["observation_vector"]
        action = agent.get_deterministic_action(obs_vec)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1

        rel_state = obs.get("relative_state", {})
        range_m = rel_state.get("range_m", 0.0)
        ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
        aa_deg = float(np.rad2deg(rel_state.get("aa_rad", 0.0)))
        min_range = min(min_range, range_m)
        min_ata_deg = min(min_ata_deg, ata_deg)
        final_range = range_m
        final_ata = ata_deg

        ego_score = info.get("ego_score", 0.0)
        target_score = info.get("target_score", 0.0)
        ego_score_sum += ego_score
        target_score_sum += target_score
        if ego_score > target_score:
            if time_to_first_advantage is None:
                time_to_first_advantage = step * env.env_config.get("high_level_dt", 0.2)
            advantage_hold_steps += 1

        if info.get("prediction_enabled"):
            prediction_enabled_count += 1
            if info.get("prediction_valid"):
                prediction_valid_count += 1
            if info.get("prediction_fallback_reason"):
                prediction_fallback_count += 1

        pred_error = info.get("prediction_error_m", np.nan)
        if np.isfinite(pred_error):
            prediction_errors.append(pred_error)

        target_pos = info.get("target_state", {}).get("position_m")
        if target_pos is None:
            target_pos = info.get("target_state", {}).get("position_neu")
        vp_pos = info.get("virtual_point", {}).get("position")
        if target_pos is not None and vp_pos is not None:
            virtual_point_shifts.append(float(np.linalg.norm(np.asarray(vp_pos) - np.asarray(target_pos))))

        pred_target_pos = info.get("predicted_target_position")
        if target_pos is not None and pred_target_pos is not None:
            anchor_shifts.append(float(np.linalg.norm(np.asarray(pred_target_pos) - np.asarray(target_pos))))
            # 记录用于离线 prediction error 对齐
            prediction_records.append((
                step,
                np.asarray(pred_target_pos, dtype=np.float64),
                np.asarray(target_pos, dtype=np.float64),
            ))

        if save_trajectory:
            own_s = info.get("own_state", {})
            target_s = info.get("target_state", {})
            own_pos = own_s.get("position_m", own_s.get("position_neu", np.full(3, np.nan)))
            target_pos_arr = target_s.get("position_m", target_s.get("position_neu", np.full(3, np.nan)))
            target_vel = target_s.get("velocity_vector_mps", target_s.get("velocity_ned", np.full(3, np.nan)))
            pred_target = info.get("predicted_target_position", [np.nan, np.nan, np.nan])
            vp = info.get("virtual_point", {})
            vp_pos_arr = vp.get("position", np.full(3, np.nan))

            trajectory.append({
                "step": step,
                "time": step * env.env_config.get("high_level_dt", 0.2),
                "backend": env._backend,
                "method": method_name,
                "predictor_type": info.get("predictor_type", ""),
                "prediction_enabled": int(info.get("prediction_enabled", False)),
                "prediction_valid": int(info.get("prediction_valid", False)),
                "prediction_fallback_reason": info.get("prediction_fallback_reason", ""),
                "prediction_horizon_s": env.config.get("trajectory_prediction", {}).get("prediction", {}).get("lookahead_time_s", 1.0),
                "target_x": float(target_pos_arr[0]) if len(target_pos_arr) > 0 else np.nan,
                "target_y": float(target_pos_arr[1]) if len(target_pos_arr) > 1 else np.nan,
                "target_z": float(target_pos_arr[2]) if len(target_pos_arr) > 2 else np.nan,
                "target_vx": float(target_vel[0]) if len(target_vel) > 0 else np.nan,
                "target_vy": float(target_vel[1]) if len(target_vel) > 1 else np.nan,
                "target_vz": float(target_vel[2]) if len(target_vel) > 2 else np.nan,
                "predicted_target_x": float(pred_target[0]) if pred_target is not None else np.nan,
                "predicted_target_y": float(pred_target[1]) if pred_target is not None else np.nan,
                "predicted_target_z": float(pred_target[2]) if pred_target is not None else np.nan,
                "prediction_error_m": float(pred_error) if np.isfinite(pred_error) else np.nan,
                "virtual_x": float(vp_pos_arr[0]) if len(vp_pos_arr) > 0 else np.nan,
                "virtual_y": float(vp_pos_arr[1]) if len(vp_pos_arr) > 1 else np.nan,
                "virtual_z": float(vp_pos_arr[2]) if len(vp_pos_arr) > 2 else np.nan,
                "virtual_point_shift_m": virtual_point_shifts[-1] if virtual_point_shifts else np.nan,
                "ego_x": float(own_pos[0]) if len(own_pos) > 0 else np.nan,
                "ego_y": float(own_pos[1]) if len(own_pos) > 1 else np.nan,
                "ego_z": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
                "range_m": range_m,
                "ata_deg": ata_deg,
                "aspect_deg": aa_deg,
                "los_rate": rel_state.get("range_rate_mps", np.nan),
                "nz_cmd": info.get("nz_cmd", np.nan),
                "roll_rate_cmd": info.get("roll_rate_cmd", np.nan),
                "throttle_cmd": info.get("throttle_cmd", np.nan),
                "ego_score": info.get("ego_score", np.nan),
                "target_score": info.get("target_score", np.nan),
                "done": int(terminated or truncated),
                "termination_reason": info.get("reason", ""),
            })

        if terminated or truncated:
            reason = info.get("reason", "unknown")
            break

    # ---- 离线 prediction error 对齐 ----
    # step t 的 predicted_target_position 应与 step t + horizon_steps 的真实 target_position 对齐
    aligned_errors = []
    for i, (step_t, pred_pos, _) in enumerate(prediction_records):
        aligned_step = step_t + horizon_steps
        # 在 prediction_records 中查找 aligned_step 对应的 true_target_pos
        for j in range(i, len(prediction_records)):
            if prediction_records[j][0] == aligned_step:
                true_pos = prediction_records[j][2]
                err = float(np.linalg.norm(pred_pos - true_pos))
                aligned_errors.append(err)
                break
    # 如果有离线对齐误差，优先使用；否则保留 info 中的原始值（通常为 NaN）
    if aligned_errors:
        prediction_errors = aligned_errors

    return {
        "seed": seed,
        "scenario": scenario.get("name", "random") if isinstance(scenario, dict) else "random",
        "return": ep_reward,
        "length": ep_length,
        "min_range_m": min_range,
        "min_ata_deg": min_ata_deg,
        "final_range_m": final_range,
        "final_ata_deg": final_ata,
        "reason": reason,
        "is_success": reason == "success",
        "is_crash": reason == "crash",
        "is_timeout": reason == "timeout",
        "is_out_of_bounds": reason == "out_of_bounds",
        "score_win": ego_score_sum > target_score_sum,
        "prediction_enabled_rate": prediction_enabled_count / max(1, ep_length),
        "prediction_valid_rate": prediction_valid_count / max(1, ep_length),
        "prediction_fallback_rate": prediction_fallback_count / max(1, ep_length),
        "mean_prediction_error_m": float(np.mean(prediction_errors)) if prediction_errors else np.nan,
        "mean_virtual_point_shift_m": float(np.mean(virtual_point_shifts)) if virtual_point_shifts else np.nan,
        "mean_anchor_shift_m": float(np.mean(anchor_shifts)) if anchor_shifts else np.nan,
        "time_to_first_advantage_s": time_to_first_advantage if time_to_first_advantage is not None else np.nan,
        "advantage_hold_time_s": advantage_hold_steps * env.env_config.get("high_level_dt", 0.2),
    }, trajectory


def aggregate_metrics(episodes):
    """Aggregate metrics from a list of episode results."""
    if not episodes:
        return {}
    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    final_ranges = [e["final_range_m"] for e in episodes]
    final_atas = [e["final_ata_deg"] for e in episodes]
    min_ranges = [e["min_range_m"] for e in episodes]
    min_atas = [e["min_ata_deg"] for e in episodes]

    def safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else np.nan

    result = {
        "num_episodes": len(episodes),
        "mean_return": safe_mean(returns),
        "std_return": float(np.std(returns)) if returns else np.nan,
        "mean_length": safe_mean(lengths),
        "success_rate": sum(1 for e in episodes if e["is_success"]) / len(episodes),
        "crash_rate": sum(1 for e in episodes if e["is_crash"]) / len(episodes),
        "out_of_bounds_rate": sum(1 for e in episodes if e["is_out_of_bounds"]) / len(episodes),
        "timeout_rate": sum(1 for e in episodes if e["is_timeout"]) / len(episodes),
        "score_win_rate": sum(1 for e in episodes if e.get("score_win", False)) / len(episodes),
        "mean_final_range_m": safe_mean(final_ranges),
        "mean_final_ata_deg": safe_mean(final_atas),
        "mean_min_range_m": safe_mean(min_ranges),
        "mean_min_ata_deg": safe_mean(min_atas),
        "mean_prediction_enabled_rate": safe_mean([e["prediction_enabled_rate"] for e in episodes]),
        "mean_prediction_valid_rate": safe_mean([e["prediction_valid_rate"] for e in episodes]),
        "mean_prediction_fallback_rate": safe_mean([e["prediction_fallback_rate"] for e in episodes]),
        "mean_prediction_error_m": safe_mean([e["mean_prediction_error_m"] for e in episodes]),
        "mean_virtual_point_shift_m": safe_mean([e["mean_virtual_point_shift_m"] for e in episodes]),
        "mean_anchor_shift_m": safe_mean([e["mean_anchor_shift_m"] for e in episodes]),
        "mean_time_to_first_advantage_s": safe_mean([e["time_to_first_advantage_s"] for e in episodes]),
        "mean_advantage_hold_time_s": safe_mean([e["advantage_hold_time_s"] for e in episodes]),
    }
    # Unified field aliases for ablation / paper tables
    result["instant_success_rate"] = result["success_rate"]
    result["prediction_rmse_m"] = result["mean_prediction_error_m"]
    result["prediction_fallback_rate"] = result["mean_prediction_fallback_rate"]
    return result


def evaluate_method(env, agent, config, method_name, num_episodes=10, seeds=None, scenarios=None, save_trajectories=False, output_dir=None):
    """
    Evaluate a single method across multiple seeds and optional fixed scenarios.

    Returns:
        dict: Aggregated metrics with overall and per-scenario breakdown.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    all_episodes = []
    per_scenario_episodes = {}
    per_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        seed_episodes = []

        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)

            if scenarios:
                scenario_name = scenarios[ep % len(scenarios)]
                scenario = config.get("scenarios", {}).get(scenario_name)
            else:
                scenario = sample_scenario(config, rng)
                scenario_name = scenario.get("name", "random") if isinstance(scenario, dict) else "random"

            ep_result, trajectory = evaluate_single_episode(
                env, agent, config, scenario=scenario, seed=ep_seed,
                save_trajectory=save_trajectories, method_name=method_name,
            )
            all_episodes.append(ep_result)
            seed_episodes.append(ep_result)
            per_scenario_episodes.setdefault(scenario_name, []).append(ep_result)

            if save_trajectories and output_dir is not None and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories", method_name)
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

        per_seed_results[f"seed_{seed}"] = seed_episodes

    overall = aggregate_metrics(all_episodes)
    overall["method"] = method_name
    overall["scenario"] = "all"
    overall["seed"] = "all"
    overall["episodes"] = len(all_episodes)
    overall["per_scenario"] = {name: aggregate_metrics(eps) for name, eps in per_scenario_episodes.items()}
    overall["raw_episodes"] = all_episodes
    overall["per_seed"] = per_seed_results
    return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--method-checkpoint", type=str, action="append", default=[],
                        help="Per-method checkpoint override, e.g. no_prediction=path/to/best.pt. Can be repeated.")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Fixed scenario names to evaluate (e.g. favorable neutral disadvantage challenging)")
    parser.add_argument("--save-trajectories", action="store_true", help="Save per-episode trajectory CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    # Override backend
    config["backend"] = args.backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = args.backend
    config["env"]["use_jsbsim"] = (args.backend == "jsbsim")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "tables",
            config.get("experiment", {}).get("name", "prediction_comparison"),
            args.backend,
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Backend: {args.backend}")
    print(f"Output dir: {output_dir}")
    print(f"Episodes: {args.episodes} x {len(args.seeds)} seeds")
    if args.scenarios:
        print(f"Scenarios: {args.scenarios}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    # Parse per-method checkpoint overrides
    method_ckpt_overrides = {}
    for mk in args.method_checkpoint:
        if "=" in mk:
            k, v = mk.split("=", 1)
            method_ckpt_overrides[k.strip()] = v.strip()

    methods_cfg = config.get("methods", {})
    if not methods_cfg:
        print("ERROR: No methods defined in config.")
        sys.exit(1)

    all_method_metrics = []

    for method_name, method_override in methods_cfg.items():
        print(f"\n=== Evaluating method: {method_name} ===")
        method_config = merge_config(dict(config), method_override)

        env = CloseRangeTrackingEnv(method_config)
        print(f"  Environment backend: {env._backend}")

        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(method_config.get("policy", {}).get("action_dim", 3))

        device = "cpu"
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device=device)

        # Load checkpoint: CLI override > config override > global arg > none
        method_ckpt = method_ckpt_overrides.get(method_name, method_override.get("checkpoint", args.checkpoint))
        if method_ckpt is not None:
            ckpt_path = method_ckpt
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: Checkpoint not found: {ckpt_path}, using random policy")
            else:
                agent.load(ckpt_path)
                print(f"  Loaded checkpoint from {ckpt_path}")

        metrics = evaluate_method(
            env, agent, method_config, method_name,
            num_episodes=args.episodes,
            seeds=args.seeds,
            scenarios=args.scenarios,
            save_trajectories=args.save_trajectories,
            output_dir=output_dir,
        )
        all_method_metrics.append(metrics)

        env.close()

        print(
            f"  Mean return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f} | "
            f"Success: {metrics['success_rate']:.2%} | "
            f"Score Win: {metrics['score_win_rate']:.2%} | "
            f"Crash: {metrics['crash_rate']:.2%} | "
            f"OOB: {metrics['out_of_bounds_rate']:.2%}"
        )
        if metrics.get("per_scenario"):
            for sc_name, sc_metrics in metrics["per_scenario"].items():
                print(
                    f"    [{sc_name}] Success: {sc_metrics['success_rate']:.2%} | "
                    f"Score Win: {sc_metrics['score_win_rate']:.2%} | "
                    f"Mean Range: {sc_metrics['mean_final_range_m']:.1f} m"
                )

    # Save aggregated metrics
    json_path = os.path.join(output_dir, "prediction_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_method_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMetrics JSON saved to: {json_path}")

    # Save CSV (overall only; per-scenario in JSON)
    csv_path = os.path.join(output_dir, "prediction_metrics.csv")
    scalar_keys = [
        "method", "scenario", "seed", "episodes",
        "instant_success_rate", "score_win_rate", "mean_return",
        "mean_final_range_m", "mean_final_ata_deg",
        "prediction_rmse_m", "prediction_fallback_rate",
        "timeout_rate", "crash_rate", "out_of_bounds_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for m in all_method_metrics:
            writer.writerow({k: m.get(k, "") for k in scalar_keys})
    print(f"Metrics CSV saved to: {csv_path}")

    # Save per-scenario CSV
    for m in all_method_metrics:
        method_name = m["method"]
        per_scenario = m.get("per_scenario", {})
        if per_scenario:
            scenario_csv = os.path.join(output_dir, f"{method_name}_scenario_metrics.csv")
            with open(scenario_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["scenario"] + scalar_keys[1:])
                writer.writeheader()
                for sc_name, sc_metrics in per_scenario.items():
                    row = {
                        "scenario": sc_name,
                        "seed": "all",
                        "episodes": sc_metrics.get("num_episodes", ""),
                    }
                    for k in scalar_keys[3:]:
                        row[k] = sc_metrics.get(k, sc_metrics.get(k.replace("instant_", "").replace("prediction_", "mean_prediction_"), ""))
                    writer.writerow(row)
            print(f"Scenario CSV saved to: {scenario_csv}")


if __name__ == "__main__":
    main()
