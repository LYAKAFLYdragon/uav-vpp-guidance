#!/usr/bin/env python3
"""
Method 2A: Curriculum Learning for Crossing Scenario Breakthrough.

Trains PPO in stages:
  Stage 1 (0-25%): favorable + neutral only
  Stage 2 (25-50%): + disadvantage
  Stage 3 (50-75%): + challenging (crossing)
  Stage 4 (75-100%): all scenarios, with emphasis on crossing

Each stage transition is gated by performance: all current scenarios must
achieve SR >= 50% before advancing.
"""
import argparse
import csv
import json
import os
import time
import copy

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.cr_ppo_agent import CRPPOAgent
from uav_vpp_guidance.agents.intentional_ppo_agent import IntentionalPPOAgent


def load_experiment_config(config_path):
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def sample_scenario(config, rng):
    scenarios = config.get("scenarios", {})
    if not scenarios:
        return None
    name = rng.choice(list(scenarios.keys()))
    return scenarios[name]


def run_evaluation(env, agent, config, num_episodes=10, seeds=None, save_trajectories=False, output_dir=None):
    if seeds is None:
        seeds = [0, 1, 2]
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
                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break
            all_episodes.append({
                "seed": seed, "episode": ep, "return": ep_reward,
                "length": ep_length, "min_range_m": min_range,
                "final_range_m": final_range, "final_ata_deg": final_ata,
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_timeout": reason == "timeout",
                "is_out_of_bounds": reason == "out_of_bounds",
            })

    success_count = sum(1 for e in all_episodes if e["is_success"])
    crash_count = sum(1 for e in all_episodes if e["is_crash"])
    oob_count = sum(1 for e in all_episodes if e["is_out_of_bounds"])
    timeout_count = sum(1 for e in all_episodes if e["is_timeout"])
    returns = [e["return"] for e in all_episodes]

    return {
        "num_episodes": len(all_episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "success_rate": success_count / max(1, len(all_episodes)),
        "crash_rate": crash_count / max(1, len(all_episodes)),
        "out_of_bounds_rate": oob_count / max(1, len(all_episodes)),
        "timeout_rate": timeout_count / max(1, len(all_episodes)),
    }


def evaluate_scenarios(env, agent, scenarios, num_episodes=5, seed_base=1000):
    """Evaluate each scenario individually and return per-scenario SR."""
    results = {}
    for name, scenario in scenarios.items():
        successes = 0
        total = num_episodes
        for ep in range(num_episodes):
            ep_seed = seed_base + hash(name) % 10000 + ep
            obs = env.reset(scenario=scenario, seed=ep_seed)
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if info.get("reason") == "success":
                        successes += 1
                    break
        results[name] = successes / max(1, total)
    return results


# Curriculum stages: (progress_end, allowed_scenario_names)
DEFAULT_CURRICULUM = [
    (0.25, ["favorable", "neutral"]),
    (0.50, ["favorable", "neutral", "disadvantage"]),
    (0.75, ["favorable", "neutral", "disadvantage", "challenging"]),
    (1.00, ["favorable", "neutral", "disadvantage", "challenging"]),
]


def train_ppo_curriculum(config, output_dir, smoke=False, algorithm="ppo"):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    import yaml
    with open(os.path.join(output_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    ppo_cfg = config.get("ppo", {})
    total_timesteps = int(ppo_cfg.get("total_timesteps", 200000))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
    eval_interval = int(config.get("evaluation", {}).get("eval_interval", 10000))
    save_interval = int(config.get("checkpoint", {}).get("save_interval", 10000))
    save_best = bool(config.get("checkpoint", {}).get("save_best", True))
    save_last = bool(config.get("checkpoint", {}).get("save_last", True))

    if smoke:
        total_timesteps = 512
        rollout_steps = 128
        eval_interval = 256
        save_interval = 256
        print("[SMOKE] Running smoke mode")

    env = CloseRangeTrackingEnv(config)
    backend = env._backend
    print(f"Backend: {backend}")

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = ppo_cfg.get("device", "cpu")
    if algorithm == "cr_ppo":
        agent = CRPPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    elif algorithm == "intentional_ppo":
        agent = IntentionalPPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    else:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    print(f"Algorithm: {algorithm} | Network parameters: {agent.network.count_parameters()}")

    # Curriculum config
    curriculum = config.get("curriculum", {}).get("stages", DEFAULT_CURRICULUM)
    stage_gate_sr = config.get("curriculum", {}).get("stage_gate_sr", 0.50)
    all_scenarios = config.get("scenarios", {})

    global_step = 0
    episode_count = 0
    best_eval_return = -float("inf")
    rng = np.random.default_rng(config.get("experiment", {}).get("seed", 0))

    episode_log_path = os.path.join(log_dir, "episode_train_log.csv")
    update_log_path = os.path.join(log_dir, "update_train_log.csv")
    eval_log_path = os.path.join(log_dir, "eval_log.csv")
    curriculum_log_path = os.path.join(log_dir, "curriculum_log.csv")

    with open(episode_log_path, "w", newline="", encoding="utf-8") as f_ep, \
         open(update_log_path, "w", newline="", encoding="utf-8") as f_up, \
         open(eval_log_path, "w", newline="", encoding="utf-8") as f_eval, \
         open(curriculum_log_path, "w", newline="", encoding="utf-8") as f_cur:

        ep_writer = csv.DictWriter(f_ep, fieldnames=[
            "step", "episode", "episode_return", "episode_length",
            "success", "crash", "out_of_bounds", "timeout",
            "mean_range", "final_range", "final_ata",
        ])
        ep_writer.writeheader()

        up_writer = csv.DictWriter(f_up, fieldnames=[
            "step", "update_num", "policy_loss", "value_loss", "entropy",
            "approx_kl", "clip_fraction", "explained_variance", "learning_rate",
            # Algorithm-specific diagnostics (union across branches)
            "complexity",
            "scale_actor", "scale_critic", "ema_abs_adv",
        ])
        up_writer.writeheader()

        eval_writer = csv.DictWriter(f_eval, fieldnames=[
            "step", "num_episodes", "mean_return", "std_return",
            "success_rate", "crash_rate", "out_of_bounds_rate", "timeout_rate",
        ])
        eval_writer.writeheader()

        cur_writer = csv.DictWriter(f_cur, fieldnames=[
            "step", "stage", "allowed_scenarios", "scenario_sr",
        ])
        cur_writer.writeheader()

        obs = env.reset(seed=rng.integers(0, 1000000))
        episode_return = 0.0
        episode_length = 0
        episode_ranges = []
        start_time = time.time()
        update_num = 0
        current_stage = 0

        while global_step < total_timesteps:
            # Determine current curriculum stage
            progress = global_step / max(1, total_timesteps)
            current_stage = 0
            for i, stage in enumerate(curriculum):
                if isinstance(stage, dict):
                    thresh = stage.get("progress_end", (i + 1) / max(1, len(curriculum)))
                else:
                    thresh = stage[0]
                if progress <= thresh:
                    current_stage = i
                    break
            else:
                current_stage = len(curriculum) - 1

            stage_spec = curriculum[current_stage]
            if isinstance(stage_spec, dict):
                allowed_names = stage_spec.get("scenario_names", [])
            else:
                allowed_names = stage_spec[1]
            active_scenarios = {k: v for k, v in all_scenarios.items() if k in allowed_names}

            for step in range(rollout_steps):
                obs_dict = obs
                obs_vec = obs_dict["observation_vector"]
                action, log_prob, value = agent.select_action(obs_vec, deterministic=False, store=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                global_step += 1
                episode_return += reward
                episode_length += 1
                # Pass obs_dict so IntentionalPPOAgent can extract combat-aware phase features.
                agent.store_transition(obs_vec, action, log_prob, reward, done, value, info=obs_dict)

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                episode_ranges.append(range_m)

                if terminated or truncated:
                    episode_count += 1
                    reason = info.get("reason", "unknown")
                    final_ata = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                    ep_writer.writerow({
                        "step": global_step, "episode": episode_count,
                        "episode_return": episode_return, "episode_length": episode_length,
                        "success": int(reason == "success"),
                        "crash": int(reason == "crash"),
                        "out_of_bounds": int(reason == "out_of_bounds"),
                        "timeout": int(reason == "timeout"),
                        "mean_range": float(np.mean(episode_ranges)) if episode_ranges else 0.0,
                        "final_range": range_m, "final_ata": final_ata,
                    })
                    f_ep.flush()

                    episode_return = 0.0
                    episode_length = 0
                    episode_ranges = []

                    scenario = sample_scenario({"scenarios": active_scenarios}, rng)
                    obs = env.reset(scenario=scenario, seed=rng.integers(0, 1000000))

                    if agent.buffer.full:
                        break

                if global_step >= total_timesteps:
                    break

            if agent.buffer.full or (global_step >= total_timesteps and len(agent.buffer) > 0):
                next_obs_vec = obs["observation_vector"]
                update_stats = agent.update(next_obs=next_obs_vec)
                update_num += 1
                up_writer.writerow({
                    "step": global_step, "update_num": update_num,
                    "policy_loss": update_stats.get("policy_loss", ""),
                    "value_loss": update_stats.get("value_loss", ""),
                    "entropy": update_stats.get("entropy", ""),
                    "approx_kl": update_stats.get("approx_kl", ""),
                    "clip_fraction": update_stats.get("clip_fraction", ""),
                    "explained_variance": update_stats.get("explained_variance", ""),
                    "learning_rate": update_stats.get("learning_rate", ""),
                    "complexity": update_stats.get("complexity", ""),
                    "scale_actor": update_stats.get("scale_actor", ""),
                    "scale_critic": update_stats.get("scale_critic", ""),
                    "ema_abs_adv": update_stats.get("ema_abs_adv", ""),
                })
                f_up.flush()
                print(
                    f"Step {global_step}/{total_timesteps} | Stage {current_stage} | "
                    f"Ep {episode_count} | Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                    f"Value Loss: {update_stats.get('value_loss', 0):.4f} | "
                    f"Entropy: {update_stats.get('entropy', 0):.4f}"
                )

            # Evaluation
            if eval_interval > 0 and global_step % eval_interval == 0 and global_step > 0:
                print(f"\n--- Evaluation at step {global_step} ---")
                eval_cfg = config.get("evaluation", {})
                eval_metrics = run_evaluation(
                    env, agent, config,
                    num_episodes=eval_cfg.get("eval_episodes", 10),
                    seeds=eval_cfg.get("seeds", [0, 1, 2]),
                )
                eval_writer.writerow({
                    "step": global_step, "num_episodes": eval_metrics["num_episodes"],
                    "mean_return": eval_metrics["mean_return"],
                    "std_return": eval_metrics["std_return"],
                    "success_rate": eval_metrics["success_rate"],
                    "crash_rate": eval_metrics["crash_rate"],
                    "out_of_bounds_rate": eval_metrics["out_of_bounds_rate"],
                    "timeout_rate": eval_metrics["timeout_rate"],
                })
                f_eval.flush()
                print(
                    f"Eval Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f} | "
                    f"Success: {eval_metrics['success_rate']:.2%} | "
                    f"Crash: {eval_metrics['crash_rate']:.2%} | "
                    f"OOB: {eval_metrics['out_of_bounds_rate']:.2%}"
                )

                # Per-scenario evaluation for curriculum gate
                per_scenario = evaluate_scenarios(env, agent, all_scenarios, num_episodes=5)
                cur_writer.writerow({
                    "step": global_step, "stage": current_stage,
                    "allowed_scenarios": "|".join(allowed_names),
                    "scenario_sr": json.dumps(per_scenario),
                })
                f_cur.flush()
                print(f"Per-scenario SR: {per_scenario}")

                # Check if we should advance stage
                current_scenario_sr = [per_scenario.get(s, 0.0) for s in allowed_names]
                min_sr = min(current_scenario_sr) if current_scenario_sr else 0.0
                if min_sr >= stage_gate_sr and current_stage < len(curriculum) - 1:
                    print(f"*** Curriculum gate passed (min SR={min_sr:.2%}). Advancing to stage {current_stage+1} ***")
                elif min_sr < stage_gate_sr:
                    print(f"Curriculum gate NOT passed (min SR={min_sr:.2%}). Staying in stage {current_stage}")

                if save_best and eval_metrics["mean_return"] > best_eval_return:
                    best_eval_return = eval_metrics["mean_return"]
                    agent.save(os.path.join(checkpoint_dir, "best.pt"))
                    print(f"  -> Saved best checkpoint")

            if save_interval > 0 and global_step % save_interval == 0 and global_step > 0:
                agent.save(os.path.join(checkpoint_dir, f"step_{global_step}.pt"))

        if save_last:
            agent.save(os.path.join(checkpoint_dir, "last.pt"))
            print(f"\nSaved last checkpoint")

    elapsed = time.time() - start_time
    print(f"\nTraining complete! Steps: {global_step}, Episodes: {episode_count}, Time: {elapsed:.1f}s")
    env.close()
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train VPP PPO with Curriculum Learning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default=None, choices=["simple", "jsbsim"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "cr_ppo", "intentional_ppo"],
        help="RL algorithm to use for training",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.backend is not None:
        config["backend"] = args.backend
        config.setdefault("env", {})["backend"] = args.backend
        config.setdefault("env", {})["use_jsbsim"] = (args.backend == "jsbsim")
    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    set_seed(seed)
    exp_name = config.get("experiment", {}).get("name", "curriculum_ppo")
    output_dir = args.output_dir or os.path.join(
        config.get("experiment", {}).get("output_root", "outputs"), "experiments", exp_name
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.device is not None:
        config.setdefault("ppo", {})["device"] = args.device
    train_ppo_curriculum(config, output_dir, smoke=args.smoke, algorithm=args.algorithm)


if __name__ == "__main__":
    main()
