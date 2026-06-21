"""Manually evaluate existing Stage 2 checkpoints after orchestrator gate crash."""
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


def evaluate(condition: str, num_eps: int = 30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    base_dir = f"outputs/adversarial_curriculum_pilot/{condition}_new_target_s0"
    pursuer_ckpt = os.path.join(base_dir, "stage2", "pursuer", "checkpoints", "best.pt")
    target_ckpt = os.path.join(base_dir, "stage2", "target", "checkpoints", "best.pt")

    if condition == "vpp":
        cfg_path = "config/experiment/close_range_curriculum_stage1.yaml"
    else:
        cfg_path = "config/experiment/close_range_curriculum_stage1_no_vpp.yaml"

    config = load_yaml_config(cfg_path)
    # Use stage2 pursuer config for scenario sampler and eval settings
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

    episodes = []
    reasons = {}
    for seed in seeds:
        for ep in range(num_eps // max(1, len(seeds))):
            ep_seed = seed * 10000 + ep
            scenario = sampler.sample() if sampler else None
            p_obs, t_obs = env.reset(scenario=scenario, seed=ep_seed)
            for _ in range(env.max_steps):
                p_action = pursuer.get_deterministic_action(p_obs["observation_vector"])
                t_action = target.get_deterministic_action(t_obs["observation_vector"])
                p_obs, t_obs, _, _, terminated, truncated, info = env.step(p_action, t_action)
                if terminated or truncated:
                    reason = info.get("termination", {}).get("reason", "unknown")
                    reasons[reason] = reasons.get(reason, 0) + 1
                    episodes.append({"success": reason == "success"})
                    break
    env.close()
    sr = sum(1 for e in episodes if e["success"]) / max(1, len(episodes))
    print(f"\n[{condition.upper()}] Stage 2 pursuer vs target")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Success/capture rate: {sr:.2%}")
    print(f"  Reason breakdown: {reasons}")
    return sr, reasons


if __name__ == "__main__":
    vpp_sr, vpp_reasons = evaluate("vpp")
    no_vpp_sr, no_vpp_reasons = evaluate("no_vpp")
    print(f"\n=== Comparison ===")
    print(f"VPP    capture rate: {vpp_sr:.2%}")
    print(f"No-VPP capture rate: {no_vpp_sr:.2%}")
    print(f"Delta: {(vpp_sr - no_vpp_sr):.2%}")
