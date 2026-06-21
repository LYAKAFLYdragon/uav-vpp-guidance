"""Diagnose VPP Stage 2 pursuer fine-tuning failure.

Compares VPP vs No-VPP Stage 2 pursuers against the same fixed target,
logging actions, virtual point behavior, range dynamics, and termination reasons.
"""
import os
import sys
import csv
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


def _policy_from_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def run_diagnosis(pursuer_condition: str, target_ckpt: str, num_eps: int = 10, out_csv: str = None):
    base_dir = f"outputs/adversarial_curriculum_pilot/{pursuer_condition}_new_target_s0"
    pursuer_ckpt = os.path.join(base_dir, "stage2", "pursuer", "checkpoints", "best.pt")

    cfg_path = ("config/experiment/close_range_curriculum_stage1.yaml" if pursuer_condition == "vpp"
                else "config/experiment/close_range_curriculum_stage1_no_vpp.yaml")
    config = load_yaml_config(cfg_path)
    p_cfg = load_yaml_config("config/adversarial/train_pursuer_v2.yaml")
    config = {**p_cfg, **config}
    config["pursuer_checkpoint"] = pursuer_ckpt
    config["target_checkpoint"] = target_ckpt

    env = AdversarialJSBSimEnv(config)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

    # Load pursuer with correct architecture
    pursuer_cfg = dict(config)
    p_policy = _policy_from_ckpt(pursuer_ckpt)
    if p_policy:
        pursuer_cfg["policy"] = p_policy
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=pursuer_cfg, device="cpu")
    pursuer.load(pursuer_ckpt)
    pursuer.network.eval()

    # Load target with correct architecture
    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
    target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
    t_policy = _policy_from_ckpt(target_ckpt)
    if t_policy:
        target_cfg["policy"] = t_policy
    target = AdversarialTargetAgent(config=target_cfg, device="cpu")
    target.load(target_ckpt)
    target.eval()

    # Optional virtual point generator for VPP diagnostics
    vp_generator = None
    if pursuer_condition == "vpp":
        vp_config = config.get("virtual_point", config.get("guidance", {}).get("virtual_point", {}))
        vp_generator = VirtualPointGenerator(vp_config)

    rows = []
    episodes = []
    for ep in range(num_eps):
        scenario = sampler.sample() if sampler else None
        p_obs, t_obs = env.reset(scenario=scenario, seed=ep)
        ep_data = {
            "actions": [],
            "ranges": [],
            "range_rates": [],
            "vp_offsets": [],
            "vp_distances": [],
            "rewards": [],
        }
        terminated = truncated = False
        step = 0
        while not (terminated or truncated) and step < env.max_steps:
            p_obs_vec = p_obs["observation_vector"]
            t_obs_vec = t_obs["observation_vector"]
            p_action = pursuer.get_deterministic_action(p_obs_vec)
            t_action = target.get_deterministic_action(t_obs_vec)

            # Compute virtual point if VPP
            vp_offset = None
            vp_distance = None
            if vp_generator is not None:
                p_state = p_obs.get("own_state", {})
                t_state = p_obs.get("target_state", {})
                rel = p_obs.get("relative_state", {})
                try:
                    vp = vp_generator.generate(t_state, rel, p_action)
                    if vp is not None and "position_m" in vp:
                        vp_pos = np.array(vp["position_m"])
                        tgt_pos = np.array(t_state.get("position_m", [0, 0, 0]))
                        vp_offset = float(np.linalg.norm(vp_pos - tgt_pos))
                        vp_distance = float(np.linalg.norm(vp_pos - np.array(p_state.get("position_m", [0, 0, 0]))))
                except Exception:
                    pass

            p_obs, t_obs, p_rew, t_rew, terminated, truncated, info = env.step(p_action, t_action)
            step += 1

            rel = p_obs.get("relative_state", {})
            range_m = float(rel.get("range_m", 0.0))
            range_rate = float(rel.get("range_rate_mps", 0.0))

            ep_data["actions"].append(p_action)
            ep_data["ranges"].append(range_m)
            ep_data["range_rates"].append(range_rate)
            ep_data["vp_offsets"].append(vp_offset if vp_offset is not None else np.nan)
            ep_data["vp_distances"].append(vp_distance if vp_distance is not None else np.nan)
            ep_data["rewards"].append(p_rew)

            rows.append({
                "ep": ep,
                "step": step,
                "condition": pursuer_condition,
                "range_m": range_m,
                "range_rate_mps": range_rate,
                "action_x": float(p_action[0]),
                "action_y": float(p_action[1]),
                "action_z": float(p_action[2]),
                "vp_offset_m": vp_offset if vp_offset is not None else np.nan,
                "vp_distance_m": vp_distance if vp_distance is not None else np.nan,
                "p_reward": p_rew,
            })

            if terminated or truncated:
                reason = info.get("termination", {}).get("reason", "unknown")
                episodes.append({
                    "length": step,
                    "reason": reason,
                    "final_range_m": range_m,
                    "min_range_m": min(ep_data["ranges"]) if ep_data["ranges"] else range_m,
                })
                break

    # Summary stats
    actions = np.array([r["action_x"] for r in rows] + [r["action_y"] for r in rows] + [r["action_z"] for r in rows])
    print(f"\n=== {pursuer_condition.upper()} Stage 2 pursuer diagnostics ===")
    print(f"Episodes: {len(episodes)}")
    reasons = {}
    for e in episodes:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    print(f"Reasons: {reasons}")
    print(f"Mean episode length: {np.mean([e['length'] for e in episodes]):.1f}")
    print(f"Mean min range: {np.mean([e['min_range_m'] for e in episodes]):.0f} m")
    print(f"Action distribution: mean={np.nanmean(actions):.3f}, std={np.nanstd(actions):.3f}, min={np.nanmin(actions):.3f}, max={np.nanmax(actions):.3f}")

    if vp_generator is not None:
        vp_offsets = [r["vp_offset_m"] for r in rows if not np.isnan(r["vp_offset_m"])]
        if vp_offsets:
            print(f"VP offset to target: mean={np.mean(vp_offsets):.0f} m, std={np.std(vp_offsets):.0f} m")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote step-level telemetry to {out_csv}")

    env.close()


if __name__ == "__main__":
    target_ckpt = "outputs/adversarial_curriculum_pilot/vpp_new_target_s0/stage2/target/checkpoints/best.pt"
    run_diagnosis("vpp", target_ckpt, num_eps=20, out_csv="outputs/adversarial/debug/vpp_stage2_diag.csv")
    run_diagnosis("no_vpp", target_ckpt, num_eps=20, out_csv="outputs/adversarial/debug/no_vpp_stage2_diag.csv")
