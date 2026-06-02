"""
JSBSim Sanity Evaluation.

Loads a checkpoint trained on the simple backend and evaluates it on the JSBSim
high-fidelity backend with a small number of episodes. Reports stability,
saturation, crash/stall/OOB rates, and mean control profiles.

Usage:
    python -m uav_vpp_guidance.evaluation.evaluate_jsbsim_sanity \
        --config config/experiment/train_vpp_ppo_cv.yaml \
        --checkpoint outputs/experiments/vpp_ppo_cv_prediction/checkpoints/best.pt \
        --episodes 5 \
        --seeds 0 \
        --save-trajectories \
        --output-dir outputs/tables/jsbsim_sanity/cv
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


def main():
    parser = argparse.ArgumentParser(description="JSBSim Sanity Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")
    parser.add_argument("--save-trajectories", action="store_true", help="Save trajectory CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    # Force JSBSim backend
    config["backend"] = "jsbsim"
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = "jsbsim"
    config["env"]["use_jsbsim"] = True

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("experiment", {}).get("output_root", "outputs"),
            "tables",
            "jsbsim_sanity",
            config.get("experiment", {}).get("name", "unknown"),
        )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backend: jsbsim")
    print(f"Output dir: {output_dir}")

    env = CloseRangeTrackingEnv(config)
    print(f"Environment backend: {env._backend}")

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))

    device = "cpu"
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    agent.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    all_episodes = []
    trajectories = []

    for seed in args.seeds:
        set_seed(seed)
        for ep in range(args.episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenario = sample_scenario(config, rng)
            obs = env.reset(scenario=scenario, seed=ep_seed)

            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            reason = "timeout"
            saturation_count = 0
            nz_cmds = []
            roll_rate_cmds = []
            throttle_cmds = []
            elevator_cmds = []
            aileron_cmds = []
            rudder_cmds = []
            trajectory = []

            for step in range(env.max_steps):
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += reward
                ep_length += 1
                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                min_range = min(min_range, range_m)
                final_range = range_m

                nz_cmds.append(info.get("nz_cmd", np.nan))
                roll_rate_cmds.append(info.get("roll_rate_cmd", np.nan))
                throttle_cmds.append(info.get("throttle_cmd", np.nan))
                elevator_cmds.append(info.get("elevator_cmd", np.nan))
                aileron_cmds.append(info.get("aileron_cmd", np.nan))
                rudder_cmds.append(info.get("rudder_cmd", np.nan))

                if info.get("saturation_flag"):
                    saturation_count += 1

                if args.save_trajectories:
                    own_s = info.get("own_state", {})
                    target_s = info.get("target_state", {})
                    own_pos = own_s.get("position_m", own_s.get("position_neu", np.full(3, np.nan)))
                    target_pos = target_s.get("position_m", target_s.get("position_neu", np.full(3, np.nan)))
                    trajectory.append({
                        "step": step,
                        "time": step * env.env_config.get("high_level_dt", 0.2),
                        "range_m": range_m,
                        "ata_deg": float(np.rad2deg(rel_state.get("ata_rad", 0.0))),
                        "nz_cmd": info.get("nz_cmd", np.nan),
                        "roll_rate_cmd": info.get("roll_rate_cmd", np.nan),
                        "throttle_cmd": info.get("throttle_cmd", np.nan),
                        "elevator_cmd": info.get("elevator_cmd", np.nan),
                        "aileron_cmd": info.get("aileron_cmd", np.nan),
                        "rudder_cmd": info.get("rudder_cmd", np.nan),
                        "saturation_flag": int(info.get("saturation_flag", False)),
                        "ego_x": float(own_pos[0]) if len(own_pos) > 0 else np.nan,
                        "ego_y": float(own_pos[1]) if len(own_pos) > 1 else np.nan,
                        "ego_z": float(own_pos[2]) if len(own_pos) > 2 else np.nan,
                        "target_x": float(target_pos[0]) if len(target_pos) > 0 else np.nan,
                        "target_y": float(target_pos[1]) if len(target_pos) > 1 else np.nan,
                        "target_z": float(target_pos[2]) if len(target_pos) > 2 else np.nan,
                    })

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
                "reason": reason,
                "is_success": reason == "success",
                "is_crash": reason == "crash",
                "is_stall": reason == "stall",
                "is_out_of_bounds": reason == "out_of_bounds",
                "is_timeout": reason == "timeout",
                "saturation_count": saturation_count,
                "saturation_rate": saturation_count / max(1, ep_length),
                "mean_nz_cmd": float(np.nanmean(nz_cmds)) if nz_cmds else np.nan,
                "mean_roll_rate_cmd": float(np.nanmean(roll_rate_cmds)) if roll_rate_cmds else np.nan,
                "mean_throttle_cmd": float(np.nanmean(throttle_cmds)) if throttle_cmds else np.nan,
                "mean_elevator_cmd": float(np.nanmean(elevator_cmds)) if elevator_cmds else np.nan,
                "mean_aileron_cmd": float(np.nanmean(aileron_cmds)) if aileron_cmds else np.nan,
                "mean_rudder_cmd": float(np.nanmean(rudder_cmds)) if rudder_cmds else np.nan,
            }
            all_episodes.append(ep_result)
            trajectories.append(trajectory)

            if args.save_trajectories and trajectory:
                traj_dir = os.path.join(output_dir, "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                traj_path = os.path.join(traj_dir, f"seed{seed}_ep{ep}.csv")
                with open(traj_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=trajectory[0].keys())
                    writer.writeheader()
                    writer.writerows(trajectory)

    env.close()

    # Aggregate
    success_rate = sum(1 for e in all_episodes if e["is_success"]) / len(all_episodes)
    crash_rate = sum(1 for e in all_episodes if e["is_crash"]) / len(all_episodes)
    stall_rate = sum(1 for e in all_episodes if e["is_stall"]) / len(all_episodes)
    oob_rate = sum(1 for e in all_episodes if e["is_out_of_bounds"]) / len(all_episodes)
    timeout_rate = sum(1 for e in all_episodes if e["is_timeout"]) / len(all_episodes)
    mean_saturation_rate = float(np.mean([e["saturation_rate"] for e in all_episodes]))

    summary = {
        "checkpoint": args.checkpoint,
        "backend": "jsbsim",
        "num_episodes": len(all_episodes),
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "stall_rate": stall_rate,
        "out_of_bounds_rate": oob_rate,
        "timeout_rate": timeout_rate,
        "mean_saturation_rate": mean_saturation_rate,
        "episodes": all_episodes,
    }

    json_path = os.path.join(output_dir, "jsbsim_sanity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSanity report saved to: {json_path}")

    csv_path = os.path.join(output_dir, "jsbsim_sanity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "episode", "return", "length", "reason",
            "saturation_rate", "mean_nz_cmd", "mean_roll_rate_cmd", "mean_throttle_cmd",
            "mean_elevator_cmd", "mean_aileron_cmd", "mean_rudder_cmd",
        ])
        writer.writeheader()
        for e in all_episodes:
            writer.writerow({k: e.get(k, "") for k in writer.fieldnames})
    print(f"Episode CSV saved to: {csv_path}")

    print("\n=== JSBSim Sanity Summary ===")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Success:  {success_rate:.2%}")
    print(f"  Crash:    {crash_rate:.2%}")
    print(f"  Stall:    {stall_rate:.2%}")
    print(f"  OOB:      {oob_rate:.2%}")
    print(f"  Timeout:  {timeout_rate:.2%}")
    print(f"  Saturation: {mean_saturation_rate:.2%}")

    # Stability check
    if crash_rate > 0.5:
        print("\nWARNING: Crash rate > 50%. Checkpoint may not be stable on JSBSim.")
    if mean_saturation_rate > 0.3:
        print("WARNING: Saturation rate > 30%. Control authority may be insufficient.")
    if stall_rate > 0.1:
        print("WARNING: Stall rate > 10%. Check low-level controller gains.")


if __name__ == "__main__":
    main()
