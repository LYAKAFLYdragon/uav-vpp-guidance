"""
Prediction comparison evaluation runner.

Formal Stage 6F ablation comparison of five methods:
    no_prediction, cv_prediction, ca_prediction, lstm_frozen, gru_frozen.

Each method MUST declare a PPO policy checkpoint in the experiment config.
The comparison script loads those checkpoints and evaluates all methods on
identical scenarios and seeds.

IMPORTANT:
- Formal evaluation requires method checkpoints. Missing checkpoints raise.
- Use --allow-random-policy ONLY for smoke / debug.
- Stage 6F full ablation should NEVER use --allow-random-policy.

Supports:
- Fixed scenario evaluation (--scenarios favorable neutral ...)
- Loading trained checkpoints (per-method config)
- Per-method and per-scenario breakdown statistics
- Policy metadata provenance in JSON and CSV outputs

Usage:
    # Formal evaluation (requires all method checkpoints)
    python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --backend simple --episodes 50 --seeds 0 1 2 \
        --scenarios favorable neutral disadvantage challenging

    # Smoke / debug only: allow random policy fallback
    python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --backend simple --episodes 1 --seeds 0 \
        --allow-random-policy
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import warnings

import copy

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.trajectory_prediction.config_validator import validate_tp_config, validate_full_config
from uav_vpp_guidance.trajectory_prediction._telemetry import PredictorHealthAccumulator


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
    health_acc = PredictorHealthAccumulator()
    prediction_enabled_steps = 0
    virtual_point_shifts = []
    anchor_shifts = []
    time_to_first_advantage = None
    advantage_hold_steps = 0
    ego_score_sum = 0.0
    target_score_sum = 0.0

    # Per-step telemetry accumulators for command saturation, altitude, energy
    limits = config.get("limits", {})
    nz_min = float(limits.get("nz_min", -2.0))
    nz_max = float(limits.get("nz_max", 7.0))
    roll_rate_min = float(limits.get("roll_rate_min", -1.5))
    roll_rate_max = float(limits.get("roll_rate_max", 1.5))
    throttle_min = float(limits.get("throttle_min", 0.0))
    throttle_max = float(limits.get("throttle_max", 1.0))

    step_altitudes = []
    step_speeds = []
    step_nz_cmds = []
    step_roll_rate_cmds = []
    step_throttle_cmds = []
    step_raw_nz_cmds = []
    step_raw_roll_rate_cmds = []
    step_raw_throttle_cmds = []

    # Env-tracked prediction errors (delayed alignment via PredictionErrorTracker)
    env_prediction_errors = []

    # Offline aligned prediction error records
    # Each element: (step, predicted_target_pos, true_target_pos)
    prediction_records = []
    lookahead_time_s = env.config.get("trajectory_prediction", {}).get("prediction", {}).get("lookahead_time_s", 1.0)
    high_level_dt = env.env_config.get("high_level_dt", 0.2)
    horizon_steps = max(1, int(round(lookahead_time_s / high_level_dt)))

    for step in range(env.max_steps):
        obs_vec = obs["observation_vector"]
        action = agent.get_deterministic_action(obs_vec)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        if info.get("prediction_enabled", False):
            prediction_enabled_steps += 1
        health_acc.step(info)

        # Per-step telemetry extraction
        own_s = info.get("own_state", {})
        own_pos = own_s.get("position_m")
        if own_pos is None:
            own_pos = own_s.get("position_neu")
        if own_pos is not None and len(own_pos) > 2:
            step_altitudes.append(float(own_pos[2]))
        own_vel = own_s.get("velocity_vector_mps")
        if own_vel is None:
            own_vel = own_s.get("velocity_ned")
        if own_vel is not None:
            step_speeds.append(float(np.linalg.norm(np.asarray(own_vel))))

        nz_cmd = info.get("nz_cmd", np.nan)
        roll_rate_cmd = info.get("roll_rate_cmd", np.nan)
        throttle_cmd = info.get("throttle_cmd", np.nan)
        if np.isfinite(nz_cmd):
            step_nz_cmds.append(float(nz_cmd))
        if np.isfinite(roll_rate_cmd):
            step_roll_rate_cmds.append(float(roll_rate_cmd))
        if np.isfinite(throttle_cmd):
            step_throttle_cmds.append(float(throttle_cmd))

        raw_cmd = info.get("raw_command", {})
        raw_nz = raw_cmd.get("nz_cmd", np.nan)
        raw_roll = raw_cmd.get("roll_rate_cmd", np.nan)
        raw_throttle = raw_cmd.get("throttle_cmd", np.nan)
        if np.isfinite(raw_nz):
            step_raw_nz_cmds.append(float(raw_nz))
        if np.isfinite(raw_roll):
            step_raw_roll_rate_cmds.append(float(raw_roll))
        if np.isfinite(raw_throttle):
            step_raw_throttle_cmds.append(float(raw_throttle))

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
                time_to_first_advantage = step * high_level_dt
            advantage_hold_steps += 1

        pred_error = info.get("prediction_error_m", np.nan)
        if np.isfinite(pred_error):
            env_prediction_errors.append(pred_error)

        target_pos = info.get("target_state", {}).get("position_m")
        if target_pos is None:
            target_pos = info.get("target_state", {}).get("position_neu")
        vp_pos = info.get("virtual_point", {}).get("position")
        if target_pos is not None and vp_pos is not None:
            virtual_point_shifts.append(float(np.linalg.norm(np.asarray(vp_pos) - np.asarray(target_pos))))

        pred_target_pos = info.get("predicted_target_position")
        if target_pos is not None and pred_target_pos is not None:
            anchor_shifts.append(float(np.linalg.norm(np.asarray(pred_target_pos) - np.asarray(target_pos))))
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
                "time": step * high_level_dt,
                "backend": env._backend,
                "method": method_name,
                "predictor_type": info.get("predictor_type", ""),
                "prediction_enabled": int(info.get("prediction_enabled", False)),
                "prediction_valid": int(info.get("prediction_valid", False)),
                "prediction_fallback": int(info.get("prediction_fallback", False)),
                "prediction_fallback_reason": info.get("prediction_fallback_reason", ""),
                "prediction_fallback_phase": info.get("prediction_fallback_phase", ""),
                "prediction_horizon_s": lookahead_time_s,
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

    # ---- Offline aligned prediction error ----
    aligned_errors = []
    for i, (step_t, pred_pos, _) in enumerate(prediction_records):
        aligned_step = step_t + horizon_steps
        for j in range(i, len(prediction_records)):
            if prediction_records[j][0] == aligned_step:
                true_pos = prediction_records[j][2]
                err = float(np.linalg.norm(pred_pos - true_pos))
                aligned_errors.append(err)
                break

    rates = health_acc.rates(ep_length)

    # Compute command saturation / modification statistics
    def _sat_rate(filtered_vals, raw_vals, vmin, vmax, eps=1e-6):
        if not filtered_vals:
            return np.nan
        # Saturation: filtered value is at the limit boundary
        boundary_count = sum(1 for v in filtered_vals if v <= vmin + eps or v >= vmax - eps)
        # Modification: raw vs filtered differ (captures clip, energy comp, terminal protection, coordination)
        mod_count = 0
        if raw_vals and len(raw_vals) == len(filtered_vals):
            mod_count = sum(1 for rv, fv in zip(raw_vals, filtered_vals) if abs(rv - fv) > eps)
        return {
            "saturation_rate": boundary_count / len(filtered_vals),
            "modification_rate": mod_count / len(filtered_vals) if raw_vals else np.nan,
            "max": max(filtered_vals),
            "mean": float(np.mean(filtered_vals)),
        }

    nz_stats = _sat_rate(step_nz_cmds, step_raw_nz_cmds, nz_min, nz_max)
    roll_stats = _sat_rate(step_roll_rate_cmds, step_raw_roll_rate_cmds, roll_rate_min, roll_rate_max)
    throttle_stats = _sat_rate(step_throttle_cmds, step_raw_throttle_cmds, throttle_min, throttle_max)

    # Altitude / energy statistics
    altitude_stats = {}
    if step_altitudes:
        altitude_stats = {
            "min_altitude_m": min(step_altitudes),
            "max_altitude_m": max(step_altitudes),
            "final_altitude_m": step_altitudes[-1],
            "altitude_loss_rate": (step_altitudes[-1] - step_altitudes[0]) / max(1, ep_length) / high_level_dt,
        }
    else:
        altitude_stats = {
            "min_altitude_m": np.nan,
            "max_altitude_m": np.nan,
            "final_altitude_m": np.nan,
            "altitude_loss_rate": np.nan,
        }

    energy_proxy = np.nan
    if step_speeds and step_altitudes:
        g = 9.80665
        energy_proxy = step_speeds[-1] ** 2 / (2.0 * g) + step_altitudes[-1]

    return {
        "seed": seed,
        "scenario": scenario.get("name", "random") if isinstance(scenario, dict) else "random",
        "training_seed": None,
        "evaluation_seed": None,
        "episode_seed": seed,
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
        # Telemetry
        "prediction_enabled_rate": prediction_enabled_steps / max(1, ep_length),
        "prediction_valid_rate": rates["prediction_valid_rate"],
        "prediction_fallback_rate": rates["fallback_rate"],
        "warmup_fallback_rate": rates["warmup_fallback_rate"],
        "runtime_fallback_rate": rates["runtime_fallback_rate"],
        "post_warmup_fallback_rate": rates["post_warmup_fallback_rate"],
        "predictor_init_failed_count": health_acc.predictor_init_failed_steps,
        "unknown_fallback_phase_count": health_acc.unknown_fallback_phase_count,
        "missing_fallback_phase_count": health_acc.missing_fallback_phase_count,
        "configured_current_target_fallback_count": health_acc.configured_current_target_fallback_count,
        # Env-tracked prediction error (canonical, delayed alignment)
        "mean_env_prediction_error_m": float(np.mean(env_prediction_errors)) if env_prediction_errors else np.nan,
        "median_env_prediction_error_m": float(np.median(env_prediction_errors)) if env_prediction_errors else np.nan,
        "env_prediction_error_count": len(env_prediction_errors),
        # Offline aligned prediction error (separate metric)
        "mean_offline_aligned_error_m": float(np.mean(aligned_errors)) if aligned_errors else np.nan,
        "median_offline_aligned_error_m": float(np.median(aligned_errors)) if aligned_errors else np.nan,
        "offline_aligned_error_count": len(aligned_errors),
        # Legacy unified alias (prefer env-tracked if available, else offline)
        "mean_prediction_error_m": (
            float(np.mean(env_prediction_errors)) if env_prediction_errors
            else float(np.mean(aligned_errors)) if aligned_errors else np.nan
        ),
        "median_prediction_error_m": (
            float(np.median(env_prediction_errors)) if env_prediction_errors
            else float(np.median(aligned_errors)) if aligned_errors else np.nan
        ),
        "prediction_error_count": len(env_prediction_errors) if env_prediction_errors else len(aligned_errors),
        "mean_virtual_point_shift_m": float(np.mean(virtual_point_shifts)) if virtual_point_shifts else np.nan,
        "mean_anchor_shift_m": float(np.mean(anchor_shifts)) if anchor_shifts else np.nan,
        "time_to_first_advantage_s": time_to_first_advantage if time_to_first_advantage is not None else np.nan,
        "advantage_hold_time_s": advantage_hold_steps * high_level_dt,
        # Per-step telemetry aggregates (command saturation / altitude / energy)
        "nz_cmd_max": nz_stats["max"],
        "nz_cmd_mean": nz_stats["mean"],
        "nz_cmd_saturation_rate": nz_stats["saturation_rate"],
        "nz_cmd_modification_rate": nz_stats["modification_rate"],
        "roll_rate_cmd_max": roll_stats["max"],
        "roll_rate_cmd_mean": roll_stats["mean"],
        "roll_rate_cmd_saturation_rate": roll_stats["saturation_rate"],
        "roll_rate_cmd_modification_rate": roll_stats["modification_rate"],
        "throttle_cmd_max": throttle_stats["max"],
        "throttle_cmd_mean": throttle_stats["mean"],
        "throttle_cmd_saturation_rate": throttle_stats["saturation_rate"],
        "throttle_cmd_modification_rate": throttle_stats["modification_rate"],
        **altitude_stats,
        "energy_proxy": energy_proxy,
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
        "mean_warmup_fallback_rate": safe_mean([e["warmup_fallback_rate"] for e in episodes]),
        "mean_runtime_fallback_rate": safe_mean([e["runtime_fallback_rate"] for e in episodes]),
        "mean_post_warmup_fallback_rate": safe_mean([e["post_warmup_fallback_rate"] for e in episodes]),
        "predictor_init_failed_count": sum(e["predictor_init_failed_count"] for e in episodes),
        "unknown_fallback_phase_count": sum(e["unknown_fallback_phase_count"] for e in episodes),
        "missing_fallback_phase_count": sum(e["missing_fallback_phase_count"] for e in episodes),
        "configured_current_target_fallback_count": sum(e["configured_current_target_fallback_count"] for e in episodes),
        "mean_prediction_error_m": safe_mean([e["mean_prediction_error_m"] for e in episodes]),
        "median_prediction_error_m": safe_mean([e["median_prediction_error_m"] for e in episodes]),
        "mean_env_prediction_error_m": safe_mean([e["mean_env_prediction_error_m"] for e in episodes]),
        "median_env_prediction_error_m": safe_mean([e["median_env_prediction_error_m"] for e in episodes]),
        "mean_offline_aligned_error_m": safe_mean([e["mean_offline_aligned_error_m"] for e in episodes]),
        "median_offline_aligned_error_m": safe_mean([e["median_offline_aligned_error_m"] for e in episodes]),
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


def evaluate_method(env, agent, config, method_name, num_episodes=10, seeds=None, scenarios=None, save_trajectories=False, output_dir=None, training_seed=None):
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
            ep_result["training_seed"] = training_seed
            ep_result["evaluation_seed"] = seed
            ep_result["episode_seed"] = ep_seed
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
    # Scenario balance check
    if scenarios:
        scenario_counts = {name: len(eps) for name, eps in per_scenario_episodes.items()}
        # num_episodes is per eval seed; account for total across all seeds
        total_episodes = num_episodes * len(seeds)
        expected = total_episodes // len(scenarios)
        balance_ok = all(c == expected for c in scenario_counts.values())
        overall["scenario_episode_count"] = scenario_counts
        overall["episodes_per_scenario"] = expected
        overall["scenario_balance_ok"] = balance_ok
    else:
        overall["scenario_episode_count"] = {}
        overall["episodes_per_scenario"] = None
        overall["scenario_balance_ok"] = None
    return overall


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--method-checkpoint", type=str, action="append", default=[],
                        help="Per-method checkpoint override, e.g. no_prediction=path/to/best.pt. Can be repeated.")
    parser.add_argument("--backend", type=str, default="simple", choices=["simple", "jsbsim"],
                        help="Simulation backend")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes per seed (legacy; use --episodes-per-scenario for balanced evaluation)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Evaluation random seeds")
    parser.add_argument("--training-seed", type=int, default=None, help="Training seed of the policy being evaluated")
    parser.add_argument("--episodes-per-scenario", type=int, default=None, help="Episodes per fixed scenario (overrides --episodes for balanced design)")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Fixed scenario names to evaluate (e.g. favorable neutral disadvantage challenging)")
    parser.add_argument("--save-trajectories", action="store_true", help="Save per-episode trajectory CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--allow-random-policy", action="store_true",
                        help="Allow fallback to random policy when checkpoint is missing")
    parser.add_argument("--validation-mode", type=str, default="raise", choices=["raise", "warn"],
                        help="Config validation mode: raise (default) or warn")
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

    # Resolve episodes count
    num_episodes = args.episodes
    if args.episodes_per_scenario is not None:
        if args.scenarios:
            num_episodes = args.episodes_per_scenario * len(args.scenarios)
            print(f"Balanced evaluation: {args.episodes_per_scenario} episodes per scenario x {len(args.scenarios)} scenarios = {num_episodes} total")
        else:
            print("WARNING: --episodes-per-scenario requires --scenarios; falling back to --episodes")
            num_episodes = args.episodes or 10
    elif num_episodes is None:
        num_episodes = 10

    print(f"Backend: {args.backend}")
    print(f"Output dir: {output_dir}")
    print(f"Episodes: {num_episodes} x {len(args.seeds)} eval seeds")
    if args.training_seed is not None:
        print(f"Training seed: {args.training_seed}")
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
        method_config = merge_config(copy.deepcopy(config), copy.deepcopy(method_override))

        # Validate merged method config (cross-component + tp sub-config)
        try:
            validate_full_config(method_config, on_unknown=args.validation_mode)
        except ValueError as exc:
            print(f"  ERROR: Method config validation failed: {exc}")
            sys.exit(1)

        # Enforce checkpoint declaration for formal comparison
        method_ckpt = method_ckpt_overrides.get(
            method_name, method_override.get("checkpoint", args.checkpoint)
        )
        if method_ckpt is None and not args.allow_random_policy:
            print(
                f"  ERROR: Method '{method_name}' has no checkpoint declared. "
                f"Add 'checkpoint' to the method config or use --allow-random-policy."
            )
            sys.exit(1)

        # Resolve predictor checkpoint path for metadata
        predictor_ckpt = method_override.get("trajectory_prediction", {}).get("checkpoint_path")

        requested_policy_ckpt = method_ckpt
        loaded_policy_ckpt = None
        policy_type = "random_policy"
        if requested_policy_ckpt is not None:
            if os.path.exists(requested_policy_ckpt):
                loaded_policy_ckpt = requested_policy_ckpt
                policy_type = "trained_ppo"
            else:
                msg = f"Checkpoint not found for method '{method_name}': {requested_policy_ckpt}"
                if args.allow_random_policy:
                    print(f"  WARNING: {msg}, using random policy")
                    policy_type = "random_policy"
                else:
                    print(f"  ERROR: {msg}. Use --allow-random-policy to fall back to random policy.")
                    sys.exit(1)

        try:
            env = CloseRangeTrackingEnv(method_config)
        except RuntimeError as exc:
            msg = f"Environment creation failed for method '{method_name}': {exc}"
            if args.allow_random_policy:
                print(f"  WARNING: {msg}, skipping method")
                continue
            else:
                print(f"  ERROR: {msg}. Use --allow-random-policy to skip methods with initialization failures.")
                sys.exit(1)
        print(f"  Environment backend: {env._backend}")

        sample_obs = env.reset(seed=0)
        obs_dim = int(sample_obs["observation_vector"].shape[0])
        action_dim = int(method_config.get("policy", {}).get("action_dim", 3))

        device = "cpu"
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=method_config, device=device)

        if policy_type == "trained_ppo":
            agent.load(loaded_policy_ckpt)
            print(f"  Loaded checkpoint from {loaded_policy_ckpt}")

        metrics = evaluate_method(
            env, agent, method_config, method_name,
            num_episodes=num_episodes,
            seeds=args.seeds,
            scenarios=args.scenarios,
            save_trajectories=args.save_trajectories,
            output_dir=output_dir,
            training_seed=args.training_seed,
        )
        # Attach policy metadata and seed info
        # Guidance mode telemetry for auditability
        requested_guidance_mode = method_config.get("guidance", {}).get("mode", "unknown")
        # Prefer the new .mode attribute on guidance law instances (added in Stage 6G.1)
        effective_guidance_mode = getattr(env.guidance, "mode", None)
        if effective_guidance_mode is None:
            # Fallback to class-name mapping for backward compatibility
            effective_guidance_mode = type(env.guidance).__name__
            guidance_mode_map = {
                "LOSRateGuidance": "los_rate",
                "ProportionalNavigationGuidance": "proportional_navigation",
                "HybridGuidance": "hybrid",
            }
            effective_guidance_mode = guidance_mode_map.get(effective_guidance_mode, effective_guidance_mode)
        if requested_guidance_mode != effective_guidance_mode:
            raise RuntimeError(
                f"Guidance mode mismatch for {method_name}: "
                f"requested={requested_guidance_mode}, effective={effective_guidance_mode}"
            )

        metrics["policy_type"] = policy_type
        metrics["requested_policy_checkpoint_path"] = requested_policy_ckpt
        metrics["loaded_policy_checkpoint_path"] = loaded_policy_ckpt
        metrics["predictor_checkpoint_path"] = predictor_ckpt
        metrics["allow_random_policy"] = args.allow_random_policy
        metrics["validation_mode"] = args.validation_mode
        metrics["invalid_for_paper"] = args.allow_random_policy or (metrics.get("loaded_policy_checkpoint_path") is None)
        metrics["backend"] = args.backend
        metrics["config_path"] = args.config
        metrics["method_name"] = method_name
        metrics["training_seed"] = args.training_seed
        metrics["evaluation_seeds"] = args.seeds
        metrics["requested_guidance_mode"] = requested_guidance_mode
        metrics["effective_guidance_mode"] = effective_guidance_mode
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
        "method", "method_name", "scenario", "seed", "episodes",
        "training_seed", "evaluation_seeds",
        "policy_type", "requested_policy_checkpoint_path", "loaded_policy_checkpoint_path",
        "predictor_checkpoint_path", "allow_random_policy", "validation_mode", "backend", "config_path",
        "instant_success_rate", "score_win_rate", "mean_return",
        "mean_final_range_m", "mean_final_ata_deg",
        "prediction_rmse_m", "prediction_fallback_rate",
        "timeout_rate", "crash_rate", "out_of_bounds_rate",
        "mean_env_prediction_error_m", "median_env_prediction_error_m",
        "mean_offline_aligned_error_m", "median_offline_aligned_error_m",
        "unknown_fallback_phase_count", "missing_fallback_phase_count",
        "configured_current_target_fallback_count",
        "scenario_balance_ok", "episodes_per_scenario",
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
                    row = {"scenario": sc_name}
                    for k in scalar_keys[1:]:
                        if k == "scenario":
                            continue
                        if k == "seed":
                            row[k] = "all"
                        elif k == "episodes":
                            row[k] = sc_metrics.get("num_episodes", "")
                        else:
                            row[k] = sc_metrics.get(k, sc_metrics.get(k.replace("instant_", "").replace("prediction_", "mean_prediction_"), ""))
                    writer.writerow(row)
            print(f"Scenario CSV saved to: {scenario_csv}")


if __name__ == "__main__":
    main()
