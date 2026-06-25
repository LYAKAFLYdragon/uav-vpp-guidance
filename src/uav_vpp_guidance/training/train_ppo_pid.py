#!/usr/bin/env python3
"""
PPO+PID Hybrid Training.

PPO policy outputs 4D action: [Δx, Δy, Δz, aggressiveness]
- Δx, Δy, Δz: Virtual Pursuit Point offset (3D)
- aggressiveness: PID gain scaling factor in [-1, 1]
  → scale = 1.0 + agg * 0.5  →  [0.5, 1.5]
  → +1 = aggressive (higher PID gains, faster response)
  → -1 = conservative (lower PID gains, smoother response)

Architecture: PPO plans trajectories (VPP), PID handles low-level stability.
Benefits:
- Full maneuverability (VPP allows any geometry)
- High interpretability (single aggressiveness knob)
- Moderate training cost (action_dim=4, only 33% more than baseline)

Usage:
    # Smoke test (fast)
    python -m uav_vpp_guidance.training.train_ppo_pid \
        --config config/experiment/train_ppo_pid_jsbsim.yaml --smoke

    # Full training
    python -m uav_vpp_guidance.training.train_ppo_pid \
        --config config/experiment/train_ppo_pid_jsbsim.yaml
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from uav_vpp_guidance.common.provenance import record_config_override_if_changed

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


def run_evaluation(env, agent, config, num_episodes=10, seeds=None, save_trajectories=False, output_dir=None, domain_rand_scale=0.0):
    """
    Evaluate a trained PPO+PID policy.

    Returns:
        dict: Aggregated evaluation metrics including aggressiveness stats.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    if domain_rand_scale > 0.0 and hasattr(env, "set_domain_rand_scale"):
        env.set_domain_rand_scale(domain_rand_scale)

    all_episodes = []
    for seed in seeds:
        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenario = sample_scenario(config, rng)
            obs = env.reset(scenario=scenario, seed=ep_seed)

            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"
            trajectory = []
            ep_aggressiveness = []

            for step in range(env.max_steps):
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                min_range = min(min_range, range_m)
                final_range = range_m
                final_ata = ata_deg
                agg = info.get("aggressiveness")
                if agg is not None:
                    ep_aggressiveness.append(agg)

                if save_trajectories and output_dir is not None:
                    traj_point = {
                        "step": step,
                        "time": step * env.env_config.get("high_level_dt", 0.2),
                        "range_m": range_m,
                        "ata_deg": ata_deg,
                        "reward": reward,
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_z": float(action[2]),
                    }
                    if len(action) >= 4:
                        traj_point["action_agg"] = float(action[3])
                    trajectory.append(traj_point)

                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break

            ep_result = {
                "seed": seed,
                "episode": ep,
                "return": ep_reward,
                "length": ep_length,
                "min_range_m": min_range,
                "final_range_m": final_range,
                "final_ata_deg": final_ata,
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
                "mean_aggressiveness": float(np.mean(ep_aggressiveness)) if ep_aggressiveness else np.nan,
                "min_aggressiveness": float(np.min(ep_aggressiveness)) if ep_aggressiveness else np.nan,
                "max_aggressiveness": float(np.max(ep_aggressiveness)) if ep_aggressiveness else np.nan,
            }
            all_episodes.append(ep_result)

            if save_trajectories and output_dir is not None and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories", "eval")
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"eval_seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

    returns = [e["return"] for e in all_episodes]
    lengths = [e["length"] for e in all_episodes]
    success_count = sum(1 for e in all_episodes if e["is_success"])
    crash_count = sum(1 for e in all_episodes if e["is_crash"])
    oob_count = sum(1 for e in all_episodes if e["is_out_of_bounds"])
    timeout_count = sum(1 for e in all_episodes if e["is_timeout"])
    final_ranges = [e["final_range_m"] for e in all_episodes]
    final_atas = [e["final_ata_deg"] for e in all_episodes]
    all_agg = [e["mean_aggressiveness"] for e in all_episodes if not np.isnan(e["mean_aggressiveness"])]

    return {
        "num_episodes": len(all_episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": success_count / max(1, len(all_episodes)),
        "crash_rate": crash_count / max(1, len(all_episodes)),
        "out_of_bounds_rate": oob_count / max(1, len(all_episodes)),
        "timeout_rate": timeout_count / max(1, len(all_episodes)),
        "mean_final_range_m": float(np.mean(final_ranges)) if final_ranges else 0.0,
        "mean_final_ata_deg": float(np.mean(final_atas)) if final_atas else 0.0,
        "mean_aggressiveness": float(np.mean(all_agg)) if all_agg else np.nan,
        "std_aggressiveness": float(np.std(all_agg)) if all_agg else np.nan,
    }


def _compute_curriculum_scale(progress: float, schedule: list) -> float:
    """Compute domain-randomization scale from training progress."""
    scale = 0.0
    for thresh, s in schedule:
        if progress >= thresh:
            scale = s
    return scale


def _select_success_criterion_stage(progress: float, schedule: list) -> dict:
    """Select the current success-criterion stage from a curriculum schedule."""
    for stage in schedule:
        if progress <= stage.get("until_progress", 1.0):
            return stage
    return schedule[-1]


def _apply_success_criterion_stage(env, stage: dict):
    """Apply a success-criterion stage to the environment."""
    env.set_success_criteria(
        success_range_m=stage.get("success_range_m"),
        success_ata_deg=stage.get("success_ata_deg"),
        success_hold_time_s=stage.get("success_hold_time_s"),
        hysteresis_range_m=stage.get("hysteresis_range_m"),
        hysteresis_ata_deg=stage.get("hysteresis_ata_deg"),
    )


def _apply_final_success_criterion(env, schedule: list):
    """Apply the final (strictest) success-criterion stage for evaluation."""
    final_stage = schedule[-1]
    _apply_success_criterion_stage(env, final_stage)


def _maybe_relabel_episode_rewards(env, agent, episode_start_idx, can_relabel):
    """Replace raw sparse rewards with R2SP relabelled trajectory when enabled."""
    calc = getattr(env, "reward_calculator", None)
    if not can_relabel:
        return False
    if calc is None or not hasattr(calc, "finalize"):
        return False
    if not getattr(calc, "relabelling_enabled", False):
        return False

    relabelled = calc.finalize()
    end_idx = agent.buffer.ptr
    episode_len = end_idx - episode_start_idx
    if episode_len <= 0 or episode_len != len(relabelled):
        return False

    agent.buffer.rewards[episode_start_idx:end_idx] = relabelled.astype(np.float32)
    return True


def train_ppo(config, output_dir, smoke=False):
    """Main PPO+PID training loop."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    import yaml
    config_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    ppo_cfg = config.get("ppo", {})
    total_timesteps = int(ppo_cfg.get("total_timesteps", 200000))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
    eval_interval = int(config.get("evaluation", {}).get("eval_interval", 10000))
    save_interval = int(config.get("checkpoint", {}).get("save_interval", 10000))
    save_best = bool(config.get("checkpoint", {}).get("save_best", True))
    save_last = bool(config.get("checkpoint", {}).get("save_last", True))
    resume_step = 0  # updated by warm_start block below (must be BEFORE smoke block)

    dr_config = config.get("domain_randomization", {})
    dr_enabled = dr_config.get("enabled", False)
    dr_schedule = dr_config.get("curriculum_schedule", [
        (0.00, 0.05), (0.25, 0.10), (0.50, 0.15), (0.75, 0.20),
    ])
    dr_fixed_scale = dr_config.get("fixed_scale", None)
    if dr_fixed_scale is not None:
        dr_schedule = [(0.0, float(dr_fixed_scale))]

    sc_curriculum = config.get("success_criterion_curriculum", None)
    if sc_curriculum:
        print(f"Success-criterion curriculum enabled with {len(sc_curriculum)} stages")

    # Environment
    env = CloseRangeTrackingEnv(config)
    backend = env._backend
    print(f"Backend: {backend}")

    if dr_enabled:
        print(f"Domain randomization enabled with schedule: {dr_schedule}")

    # Get observation and action dimensions
    sample_obs = env.reset(seed=0)
    obs_vec = sample_obs["observation_vector"]
    obs_dim = int(obs_vec.shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 4))
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"PPO+PID mode: action[0:3] = VPP offset, action[3] = aggressiveness")

    # Agent
    device = ppo_cfg.get("device", "cpu")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    print(f"Network parameters: {agent.network.count_parameters()}")

    # Warm-start / resume from checkpoint
    resume_step = 0
    warm_start_cfg = config.get("warm_start", {})
    if warm_start_cfg.get("enabled", False):
        import torch as _torch
        checkpoint_path = warm_start_cfg.get("checkpoint") or warm_start_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("warm_start.enabled=true requires warm_start.checkpoint")
        ckpt = _torch.load(str(checkpoint_path), map_location=device)
        missing, unexpected = agent.network.load_state_dict(
            ckpt["network_state_dict"], strict=False,
        )
        if warm_start_cfg.get("load_optimizer", False) and "optimizer_state_dict" in ckpt:
            try:
                agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"Loaded optimizer state from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        resume_step = int(warm_start_cfg.get("step", 0))
        print(
            f"Warm-start loaded from {checkpoint_path}: "
            f"obs_dim={ckpt.get('obs_dim', '?')}, "
            f"action_dim={ckpt.get('action_dim', '?')}, "
            f"resume_step={resume_step}, "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )

    # Smoke mode timestep override (after warm-start: resume_step is correct)
    if smoke:
        total_timesteps = resume_step + 512 if resume_step > 0 else 512
        rollout_steps = 128
        eval_interval = 256
        save_interval = 256
        print("[SMOKE] Running smoke mode with reduced settings (resuming from step {})".format(resume_step))
        print(f"  total_timesteps={total_timesteps}, rollout_steps={rollout_steps}")

    # Training state
    global_step = resume_step
    episode_count = 0
    best_eval_return = -float("inf")
    last_sc_stage = None

    # CSV loggers (append mode when resuming)
    csv_mode = "a" if resume_step > 0 else "w"
    episode_log_path = os.path.join(log_dir, "episode_train_log.csv")
    update_log_path = os.path.join(log_dir, "update_train_log.csv")
    eval_log_path = os.path.join(log_dir, "eval_log.csv")

    episode_fieldnames = [
        "step", "episode", "episode_return", "episode_length",
        "success", "score_win", "crash", "out_of_bounds", "timeout",
        "mean_range", "final_range", "final_ata",
        "mean_aggressiveness", "min_aggressiveness", "max_aggressiveness",
    ]
    update_fieldnames = [
        "step", "update_num", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_fraction", "explained_variance", "learning_rate",
    ]
    eval_fieldnames = [
        "step", "num_episodes", "mean_return", "std_return",
        "success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        "mean_final_range_m", "mean_final_ata_deg",
        "mean_aggressiveness", "std_aggressiveness",
    ]

    with open(episode_log_path, csv_mode, newline="", encoding="utf-8") as f_ep:
        ep_writer = csv.DictWriter(f_ep, fieldnames=episode_fieldnames)
        if csv_mode == "w":
            ep_writer.writeheader()

        with open(update_log_path, csv_mode, newline="", encoding="utf-8") as f_up:
            up_writer = csv.DictWriter(f_up, fieldnames=update_fieldnames)
            if csv_mode == "w":
                up_writer.writeheader()

            with open(eval_log_path, csv_mode, newline="", encoding="utf-8") as f_eval:
                eval_writer = csv.DictWriter(f_eval, fieldnames=eval_fieldnames)
                if csv_mode == "w":
                    eval_writer.writeheader()

                # Main training loop
                rng = np.random.default_rng(config.get("experiment", {}).get("seed", 0))
                obs = env.reset(seed=rng.integers(0, 1000000))

                last_step_done = True
                episode_return = 0.0
                episode_length = 0
                episode_ranges = []
                episode_aggressiveness = []
                episode_success = False
                episode_crash = False
                episode_oob = False
                episode_timeout = False
                episode_score_win = False

                start_time = time.time()
                update_num = 0
                last_saved_step = 0

                while global_step < total_timesteps:
                    episode_start_idx = agent.buffer.ptr
                    can_relabel = last_step_done

                    # Collect rollout
                    for step in range(rollout_steps):
                        obs_vec = obs["observation_vector"]
                        action, log_prob, value = agent.select_action(obs_vec, deterministic=False, store=False)

                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        global_step += 1
                        episode_return += reward
                        episode_length += 1

                        # Update success-criterion curriculum
                        if sc_curriculum:
                            progress = min(1.0, global_step / max(1, total_timesteps))
                            sc_stage = _select_success_criterion_stage(progress, sc_curriculum)
                            if sc_stage != last_sc_stage:
                                _apply_success_criterion_stage(env, sc_stage)
                                print(
                                    f"[Curriculum] step={global_step} progress={progress:.2f} | "
                                    f"success_range={env.termination_checker.success_range_m:.0f}m "
                                    f"ata={env.termination_checker.success_ata_deg:.1f}deg "
                                    f"hold={env.termination_checker.success_hold_time_s:.1f}s"
                                )
                                last_sc_stage = sc_stage

                        # Store transition
                        agent.store_transition(obs_vec, action, log_prob, reward, done, value)

                        rel_state = obs.get("relative_state", {})
                        range_m = rel_state.get("range_m", 0.0)
                        episode_ranges.append(range_m)
                        agg = info.get("aggressiveness")
                        if agg is not None:
                            episode_aggressiveness.append(agg)

                        if terminated or truncated:
                            episode_count += 1
                            reason = info.get("reason", "unknown")
                            episode_success = reason == "success"
                            episode_crash = reason == "crash"
                            episode_oob = reason == "out_of_bounds"
                            episode_timeout = reason == "timeout"

                            ego_score = info.get("ego_score", 0.0)
                            target_score = info.get("target_score", 0.0)
                            episode_score_win = ego_score > target_score

                            final_range = range_m
                            final_ata = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                            mean_range = float(np.mean(episode_ranges)) if episode_ranges else 0.0

                            ep_row = {
                                "step": global_step,
                                "episode": episode_count,
                                "episode_return": episode_return,
                                "episode_length": episode_length,
                                "success": int(episode_success),
                                "score_win": int(episode_score_win),
                                "crash": int(episode_crash),
                                "out_of_bounds": int(episode_oob),
                                "timeout": int(episode_timeout),
                                "mean_range": mean_range,
                                "final_range": final_range,
                                "final_ata": final_ata,
                                "mean_aggressiveness": float(np.mean(episode_aggressiveness)) if episode_aggressiveness else np.nan,
                                "min_aggressiveness": float(np.min(episode_aggressiveness)) if episode_aggressiveness else np.nan,
                                "max_aggressiveness": float(np.max(episode_aggressiveness)) if episode_aggressiveness else np.nan,
                            }
                            ep_writer.writerow(ep_row)
                            f_ep.flush()

                            # Reset episode stats
                            episode_return = 0.0
                            episode_length = 0
                            episode_ranges = []
                            episode_aggressiveness = []

                            # Relabel sparse rewards if enabled
                            if _maybe_relabel_episode_rewards(env, agent, episode_start_idx, can_relabel):
                                pass

                            # Reset environment
                            scenario = sample_scenario(config, rng)
                            if dr_enabled:
                                progress = min(1.0, global_step / max(1, total_timesteps))
                                current_dr_scale = _compute_curriculum_scale(progress, dr_schedule)
                                env.set_domain_rand_scale(current_dr_scale)
                            obs = env.reset(scenario=scenario, seed=rng.integers(0, 1000000))

                            episode_start_idx = agent.buffer.ptr
                            can_relabel = True
                            last_step_done = True

                            if agent.buffer.full:
                                break

                        if global_step >= total_timesteps:
                            break

                    last_step_done = done

                    # PPO update when buffer is full or training ended
                    if agent.buffer.full or (global_step >= total_timesteps and len(agent.buffer) > 0):
                        next_obs_vec = obs["observation_vector"]
                        update_stats = agent.update(next_obs=next_obs_vec)
                        update_num += 1

                        up_row = {
                            "step": global_step,
                            "update_num": update_num,
                            "policy_loss": update_stats.get("policy_loss", ""),
                            "value_loss": update_stats.get("value_loss", ""),
                            "entropy": update_stats.get("entropy", ""),
                            "approx_kl": update_stats.get("approx_kl", ""),
                            "clip_fraction": update_stats.get("clip_fraction", ""),
                            "explained_variance": update_stats.get("explained_variance", ""),
                            "learning_rate": update_stats.get("learning_rate", ""),
                        }
                        up_writer.writerow(up_row)
                        f_up.flush()

                        print(
                            f"Step {global_step}/{total_timesteps} | "
                            f"Ep {episode_count} | "
                            f"Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                            f"Value Loss: {update_stats.get('value_loss', 0):.4f} | "
                            f"Entropy: {update_stats.get('entropy', 0):.4f} | "
                            f"Explained Var: {update_stats.get('explained_variance', 0):.4f}"
                        )

                    # Evaluation
                    if eval_interval > 0 and global_step % eval_interval == 0 and global_step > 0:
                        print(f"\n--- Evaluation at step {global_step} ---")
                        env.set_domain_rand_scale(0.0)
                        if sc_curriculum:
                            _apply_final_success_criterion(env, sc_curriculum)
                            last_sc_stage = None
                        eval_cfg = config.get("evaluation", {})
                        eval_metrics = run_evaluation(
                            env, agent, config,
                            num_episodes=eval_cfg.get("eval_episodes", 10),
                            seeds=eval_cfg.get("seeds", [0, 1, 2]),
                            save_trajectories=eval_cfg.get("save_trajectories", False),
                            output_dir=output_dir,
                            domain_rand_scale=eval_cfg.get("domain_rand_scale", 0.0),
                        )
                        eval_row = {
                            "step": global_step,
                            "num_episodes": eval_metrics["num_episodes"],
                            "mean_return": eval_metrics["mean_return"],
                            "std_return": eval_metrics["std_return"],
                            "success_rate": eval_metrics["success_rate"],
                            "crash_rate": eval_metrics["crash_rate"],
                            "out_of_bounds_rate": eval_metrics["out_of_bounds_rate"],
                            "timeout_rate": eval_metrics["timeout_rate"],
                            "mean_final_range_m": eval_metrics["mean_final_range_m"],
                            "mean_final_ata_deg": eval_metrics["mean_final_ata_deg"],
                            "mean_aggressiveness": eval_metrics.get("mean_aggressiveness", np.nan),
                            "std_aggressiveness": eval_metrics.get("std_aggressiveness", np.nan),
                        }
                        eval_writer.writerow(eval_row)
                        f_eval.flush()

                        print(
                            f"Eval Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f} | "
                            f"Success: {eval_metrics['success_rate']:.2%} | "
                            f"Crash: {eval_metrics['crash_rate']:.2%} | "
                            f"OOB: {eval_metrics['out_of_bounds_rate']:.2%} | "
                            f"Agg: {eval_metrics.get('mean_aggressiveness', np.nan):.3f}±{eval_metrics.get('std_aggressiveness', np.nan):.3f}"
                        )

                        # Save best checkpoint
                        if save_best and eval_metrics["mean_return"] > best_eval_return:
                            best_eval_return = eval_metrics["mean_return"]
                            best_path = os.path.join(checkpoint_dir, "best.pt")
                            agent.save(best_path)
                            print(f"  -> Saved best checkpoint (return={best_eval_return:.2f})")

                    # Periodic checkpoint save (guaranteed to fire even when
                    # save_interval is not an exact multiple of rollout_steps).
                    if save_interval > 0 and global_step - last_saved_step >= save_interval and global_step > 0:
                        step_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
                        agent.save(step_path)
                        last_saved_step = global_step

                # Save last checkpoint
                if save_last:
                    last_path = os.path.join(checkpoint_dir, "last.pt")
                    agent.save(last_path)
                    print(f"\nSaved last checkpoint to {last_path}")

                elapsed = time.time() - start_time
                print(f"\nTraining complete! Total steps: {global_step}, Episodes: {episode_count}, Time: {elapsed:.1f}s")

    env.close()

    # Smoke summary
    if smoke:
        smoke_summary = {
            "smoke": True,
            "total_timesteps": global_step,
            "episodes": episode_count,
            "elapsed_seconds": elapsed,
            "backend": backend,
            "checkpoint_dir": checkpoint_dir,
            "episode_train_log": episode_log_path,
            "update_train_log": update_log_path,
            "eval_log": eval_log_path,
        }
        smoke_path = os.path.join(log_dir, "smoke_summary.json")
        with open(smoke_path, "w", encoding="utf-8") as f:
            json.dump(smoke_summary, f, indent=2, ensure_ascii=False)
        print(f"Smoke summary saved to {smoke_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO+PID Hybrid Policy")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (minimal training)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override compute device")
    parser.add_argument("--backend", type=str, default=None, choices=["simple", "jsbsim"], help="Override simulation backend")
    parser.add_argument("--use-jsbsim", action="store_true", help="Force use_jsbsim=True")
    parser.add_argument("--domain-rand-scale", type=float, default=None, help="Override domain randomization scale")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt file")
    parser.add_argument("--resume-step", type=int, default=0, help="Global step at which checkpoint was saved")
    parser.add_argument("--resume-optimizer", action="store_true", help="Also load optimizer state when resuming")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    # Backend override
    backend = args.backend
    if args.use_jsbsim:
        backend = "jsbsim"
    if backend is not None:
        old_backend = config.get("backend")
        config["backend"] = backend
        record_config_override_if_changed(
            config,
            "backend",
            backend,
            old_value=old_backend,
            source="train_ppo_pid.py:--backend",
        )
        if "env" not in config:
            config["env"] = {}
        old_env_backend = config["env"].get("backend")
        old_use_jsbsim = config["env"].get("use_jsbsim")
        config["env"]["backend"] = backend
        config["env"]["use_jsbsim"] = (backend == "jsbsim")
        record_config_override_if_changed(
            config,
            "env.backend",
            backend,
            old_value=old_env_backend,
            source="train_ppo_pid.py:--backend",
        )
        record_config_override_if_changed(
            config,
            "env.use_jsbsim",
            backend == "jsbsim",
            old_value=old_use_jsbsim,
            source="train_ppo_pid.py:--backend",
        )
        print(f"Backend override: {backend}")

    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    if args.seed is not None:
        config.setdefault("experiment", {})
        old_seed = config["experiment"].get("seed")
        config["experiment"]["seed"] = int(seed)
        record_config_override_if_changed(
            config,
            "experiment.seed",
            int(seed),
            old_value=old_seed,
            source="train_ppo_pid.py:--seed",
        )
    set_seed(seed)

    exp_name = config.get("experiment", {}).get("name", "ppo_pid")
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "experiments",
            exp_name,
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"Seed: {seed}")

    # Domain randomization override
    if args.domain_rand_scale is not None:
        if "domain_randomization" not in config:
            config["domain_randomization"] = {}
        old_enabled = config["domain_randomization"].get("enabled")
        old_fixed_scale = config["domain_randomization"].get("fixed_scale")
        config["domain_randomization"]["enabled"] = (args.domain_rand_scale > 0.0)
        config["domain_randomization"]["fixed_scale"] = args.domain_rand_scale
        record_config_override_if_changed(
            config,
            "domain_randomization.enabled",
            args.domain_rand_scale > 0.0,
            old_value=old_enabled,
            source="train_ppo_pid.py:--domain-rand-scale",
        )
        record_config_override_if_changed(
            config,
            "domain_randomization.fixed_scale",
            args.domain_rand_scale,
            old_value=old_fixed_scale,
            source="train_ppo_pid.py:--domain-rand-scale",
        )
        print(f"Domain randomization scale override: {args.domain_rand_scale}")

    # Verify trajectory prediction is disabled
    tp_enabled = config.get("trajectory_prediction", {}).get("enabled", False)
    if tp_enabled:
        print("WARNING: trajectory_prediction.enabled is True! Forcing to False for this baseline.")
        config["trajectory_prediction"]["enabled"] = False
        record_config_override_if_changed(
            config,
            "trajectory_prediction.enabled",
            False,
            old_value=tp_enabled,
            source="train_ppo_pid.py:ppo_pid_baseline",
        )

    # Verify low-level controller is enhanced (PID-based, not baseline)
    ll_controller_cfg = config.get("low_level_controller", {})
    if isinstance(ll_controller_cfg, str):
        ll_controller = ll_controller_cfg
    else:
        ll_controller = ll_controller_cfg.get("type", "baseline")
    if ll_controller == "baseline":
        print("WARNING: low_level_controller is 'baseline'. Forcing to 'enhanced' for PPO+PID.")
        old_ll_controller = ll_controller_cfg
        if isinstance(ll_controller_cfg, dict):
            config["low_level_controller"] = {**ll_controller_cfg, "type": "enhanced"}
        else:
            config["low_level_controller"] = "enhanced"
        record_config_override_if_changed(
            config,
            "low_level_controller",
            config["low_level_controller"],
            old_value=old_ll_controller,
            source="train_ppo_pid.py:ppo_pid_controller_requirement",
        )
        ll_controller = "enhanced"
    print(f"Low-level controller: {ll_controller}")
    print(f"Action space: 4D [Δx, Δy, Δz, aggressiveness]")
    print(f"  aggressiveness ∈ [-1, 1] → PID gain scale ∈ [0.5, 1.5]")

    anchor_mode = config.get("virtual_point", {}).get("anchor_mode", "current_target")
    print(f"Anchor mode: {anchor_mode}")
    print(f"Trajectory prediction: {'enabled' if tp_enabled else 'disabled'}")

    # Device override
    if args.device is not None:
        if "ppo" not in config:
            config["ppo"] = {}
        old_device = config["ppo"].get("device")
        config["ppo"]["device"] = args.device
        record_config_override_if_changed(
            config,
            "ppo.device",
            args.device,
            old_value=old_device,
            source="train_ppo_pid.py:--device",
        )
        print(f"Device override: {args.device}")

    # Resume / warm-start
    if args.resume is not None:
        old_warm_start = config.get("warm_start")
        config["warm_start"] = {
            "enabled": True,
            "checkpoint": args.resume,
            "step": int(args.resume_step),
            "load_optimizer": bool(args.resume_optimizer),
            "strict_dims": True,
        }
        record_config_override_if_changed(
            config,
            "warm_start",
            config["warm_start"],
            old_value=old_warm_start,
            source="train_ppo_pid.py:--resume",
        )
        print(f"Resume enabled: checkpoint={args.resume}, step={args.resume_step}")

    train_ppo(config, output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
