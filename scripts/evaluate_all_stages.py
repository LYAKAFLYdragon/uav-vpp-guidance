"""Evaluate saved Stage 2 and Stage 3 checkpoints after pilot runs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent


def _policy_from_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def evaluate(condition: str, stage: int, num_eps: int = 30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    base_dir = f"outputs/adversarial_curriculum_pilot/{condition}_full_gate25_s0"
    pursuer_dir = os.path.join(base_dir, f"stage{stage}", "pursuer")
    target_dir = os.path.join(base_dir, f"stage{stage}", "target")

    best_pt = os.path.join(pursuer_dir, "checkpoints", "best.pt")
    last_pt = os.path.join(pursuer_dir, "checkpoints", "last.pt")
    pursuer_ckpt = best_pt if os.path.exists(best_pt) else last_pt
    target_ckpt = os.path.join(target_dir, "checkpoints", "best.pt")

    if not os.path.exists(pursuer_ckpt):
        print(f"[{condition} stage{stage}] No pursuer checkpoint found, skipping")
        return None
    if not os.path.exists(target_ckpt):
        print(f"[{condition} stage{stage}] No target checkpoint found, skipping")
        return None

    cfg_path = ("config/experiment/close_range_curriculum_stage1.yaml" if condition == "vpp"
                else "config/experiment/close_range_curriculum_stage1_no_vpp.yaml")
    config = load_yaml_config(cfg_path)
    if condition == "vpp":
        p_cfg = load_yaml_config("config/adversarial/train_pursuer_v2_pilot_aggressive.yaml")
    else:
        p_cfg = load_yaml_config("config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml")
    config = {**p_cfg, **config}
    config["pursuer_checkpoint"] = pursuer_ckpt
    config["target_checkpoint"] = target_ckpt

    env = AdversarialJSBSimEnv(config)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

    # Load pursuer
    pursuer_cfg = dict(config)
    p_policy = _policy_from_ckpt(pursuer_ckpt)
    if p_policy:
        pursuer_cfg["policy"] = p_policy
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=pursuer_cfg, device="cpu")
    pursuer.load(pursuer_ckpt)
    pursuer.network.eval()

    # Load target
    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
    target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
    t_policy = _policy_from_ckpt(target_ckpt)
    if t_policy:
        target_cfg["policy"] = t_policy
    target = AdversarialTargetAgent(config=target_cfg, device="cpu")
    target.load(target_ckpt)
    target.eval()

    reasons = {}
    returns = []
    lengths = []
    min_ranges = []
    for seed in seeds:
        for ep in range(num_eps // max(1, len(seeds))):
            ep_seed = seed * 10000 + ep
            scenario = sampler.sample() if sampler else None
            p_obs, t_obs = env.reset(scenario=scenario, seed=ep_seed)
            ep_return = 0.0
            ep_len = 0
            min_range = float("inf")
            for _ in range(env.max_steps):
                p_action = pursuer.get_deterministic_action(p_obs["observation_vector"])
                t_action = target.get_deterministic_action(t_obs["observation_vector"])
                p_obs, t_obs, p_rew, _, terminated, truncated, info = env.step(p_action, t_action)
                ep_return += p_rew
                ep_len += 1
                rel = p_obs.get("relative_state", {})
                range_m = float(rel.get("range_m", 0.0))
                min_range = min(min_range, range_m)
                if terminated or truncated:
                    reason = info.get("termination", {}).get("reason", "unknown")
                    reasons[reason] = reasons.get(reason, 0) + 1
                    returns.append(ep_return)
                    lengths.append(ep_len)
                    min_ranges.append(min_range)
                    break
    env.close()
    total = sum(reasons.values())
    sr = reasons.get("success", 0) / max(1, total)
    print(f"\n[{condition.upper()} Stage {stage}] {pursuer_ckpt}")
    print(f"  Episodes: {total}  capture_rate={sr:.2%}")
    print(f"  Reasons: {reasons}")
    print(f"  Mean return: {np.mean(returns):.1f}  length: {np.mean(lengths):.1f}  min_range: {np.mean(min_ranges):.0f}m")
    return {"success_rate": sr, "reasons": reasons, "mean_return": np.mean(returns)}


if __name__ == "__main__":
    for cond in ["vpp", "no_vpp"]:
        for stage in [2, 3]:
            evaluate(cond, stage, num_eps=30)
