#!/usr/bin/env python3
"""
Generate an ACMI (TacView) file from a trained policy checkpoint.

Usage:
    python scripts/generate_acmi.py \
        --config config/experiment/train_no_prediction_vpp_ppo.yaml \
        --checkpoint outputs/experiments/baseline_10seed_s0/checkpoints/best.pt \
        --output outputs/acmi/baseline_s0.acmi

For end-to-end checkpoints:
    python scripts/generate_acmi.py \
        --config config/experiment/train_end_to_end_ppo.yaml \
        --checkpoint outputs/experiments/end_to_end_ppo_seed0/checkpoints/best.pt \
        --agent end_to_end \
        --output outputs/acmi/end_to_end_s0.acmi
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.end_to_end_ppo_agent import EndToEndPPOAgent


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


def neu_to_latlon(north_m, east_m, up_m, ref_lat_deg=0.0, ref_lon_deg=0.0):
    """Approximate NEU-to-lat/lon/alt conversion around reference origin."""
    R = 6371000.0  # Earth radius in meters
    lat_rad = math.radians(ref_lat_deg)
    lat_deg = ref_lat_deg + math.degrees(north_m / R)
    lon_deg = ref_lon_deg + math.degrees(east_m / (R * math.cos(lat_rad + 1e-12)))
    return lon_deg, lat_deg, up_m


def write_acmi(output_path, records, title="Flight"):
    """Write ACMI text file from list of records."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        f.write(f"0,Title={title}\n")
        f.write("0,ReferenceTime=2020-01-01T00:00:00Z\n")
        f.write("1000000,Type=Air+FixedWing,Name=Own\n")
        f.write("1000001,Type=Air+FixedWing,Name=Target\n")
        prev_time = None
        for rec in records:
            t = rec["time"]
            if prev_time is None or not math.isclose(t, prev_time, rel_tol=1e-9):
                f.write(f"#{t:.2f}\n")
                prev_time = t
            own = rec["own"]
            tgt = rec["target"]
            own_lon, own_lat, own_alt = neu_to_latlon(*own["pos"])
            tgt_lon, tgt_lat, tgt_alt = neu_to_latlon(*tgt["pos"])
            own_roll, own_pitch, own_yaw = own["roll"], own["pitch"], own["yaw"]
            tgt_roll, tgt_pitch, tgt_yaw = tgt["roll"], tgt["pitch"], tgt["yaw"]
            f.write(
                f"1000000,T={own_lon:.6f}|{own_lat:.6f}|{own_alt:.1f}|"
                f"{math.degrees(own_roll):.2f}|{math.degrees(own_pitch):.2f}|{math.degrees(own_yaw):.2f}|Own\n"
            )
            f.write(
                f"1000001,T={tgt_lon:.6f}|{tgt_lat:.6f}|{tgt_alt:.1f}|"
                f"{math.degrees(tgt_roll):.2f}|{math.degrees(tgt_pitch):.2f}|{math.degrees(tgt_yaw):.2f}|Target\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate ACMI flight trajectory from a checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--output", required=True, help="Output .acmi file path.")
    parser.add_argument("--agent", choices=["ppo", "end_to_end"], default="ppo", help="Agent class.")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed.")
    parser.add_argument("--scenario", default=None, help="Scenario name (default random).")
    parser.add_argument("--device", default="cpu", help="torch device.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenario = None
    if args.scenario:
        scenarios = config.get("scenarios", {})
        scenario = scenarios.get(args.scenario)
    if scenario is None:
        scenario = sample_scenario(config, rng)

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(scenario=scenario, seed=args.seed)

    # Helper to fetch current full states (env exposes them via private method)
    def get_states():
        return env._get_current_states()

    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))

    AgentClass = EndToEndPPOAgent if args.agent == "end_to_end" else PPOAgent
    agent = AgentClass(obs_dim, action_dim, config, device=args.device)
    agent.load(args.checkpoint)

    max_steps = args.max_steps if args.max_steps is not None else env.max_steps

    records = []
    done = False
    for step in range(max_steps):
        own_state, target_state = get_states()
        records.append({
            "time": getattr(env, "_sim_time_s", getattr(env, "time", env.current_step * env.env_config.get("high_level_dt", 0.2))),
            "own": {
                "pos": own_state.get("position_m", np.zeros(3)),
                "roll": own_state.get("roll_rad", 0.0),
                "pitch": own_state.get("pitch_rad", 0.0),
                "yaw": own_state.get("heading_rad", 0.0),
            },
            "target": {
                "pos": target_state.get("position_m", np.zeros(3)),
                "roll": target_state.get("roll_rad", 0.0),
                "pitch": target_state.get("pitch_rad", 0.0),
                "yaw": target_state.get("heading_rad", 0.0),
            },
        })

        action = agent.get_deterministic_action(obs["observation_vector"])
        if args.agent == "end_to_end":
            action = agent.clip_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            # Record final state
            own_state, target_state = get_states()
            records.append({
                "time": getattr(env, "_sim_time_s", getattr(env, "time", env.current_step * env.env_config.get("high_level_dt", 0.2))),
                "own": {
                    "pos": own_state.get("position_m", np.zeros(3)),
                    "roll": own_state.get("roll_rad", 0.0),
                    "pitch": own_state.get("pitch_rad", 0.0),
                    "yaw": own_state.get("heading_rad", 0.0),
                },
                "target": {
                    "pos": target_state.get("position_m", np.zeros(3)),
                    "roll": target_state.get("roll_rad", 0.0),
                    "pitch": target_state.get("pitch_rad", 0.0),
                    "yaw": target_state.get("heading_rad", 0.0),
                },
            })
            break

    reason = info.get("reason", "unknown")
    success = reason == "success"
    print(f"Episode finished: reason={reason}, success={success}, steps={len(records)}")

    write_acmi(args.output, records, title=f"{Path(args.checkpoint).stem}_seed{args.seed}_{reason}")
    print(f"ACMI written to {args.output}")


if __name__ == "__main__":
    main()
