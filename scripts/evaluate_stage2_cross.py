"""Cross-evaluate Stage 2 pursuers against a common target to remove specialization bias."""
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


def evaluate_pair(pursuer_condition: str, target_condition: str, target_kind: str = "stage2", num_eps: int = 30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    base_dir = f"outputs/adversarial_curriculum_pilot/{pursuer_condition}_new_target_s0"
    pursuer_ckpt = os.path.join(base_dir, "stage2", "pursuer", "checkpoints", "best.pt")

    if target_kind == "stage2":
        target_base = f"outputs/adversarial_curriculum_pilot/{target_condition}_new_target_s0"
        target_ckpt = os.path.join(target_base, "stage2", "target", "checkpoints", "best.pt")
    elif target_kind == "ablation":
        target_ckpt = "outputs/adversarial_curriculum_pilot/target_stability_ablation_v2/checkpoints/best.pt"
    else:
        raise ValueError(target_kind)

    # Config for environment / scenario sampler
    cfg_path = ("config/experiment/close_range_curriculum_stage1.yaml" if pursuer_condition == "vpp"
                else "config/experiment/close_range_curriculum_stage1_no_vpp.yaml")
    config = load_yaml_config(cfg_path)
    p_cfg = load_yaml_config("config/adversarial/train_pursuer_v2.yaml")
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
                    break
    env.close()
    total = sum(reasons.values())
    sr = reasons.get("success", 0) / max(1, total)
    print(f"[{pursuer_condition:6s} pursuer vs {target_condition:6s} {target_kind:8s} target] "
          f"capture={sr:.2%} reasons={reasons}")
    return sr, reasons


if __name__ == "__main__":
    print("=== Same-condition Stage 2 evaluation ===")
    evaluate_pair("vpp", "vpp")
    evaluate_pair("no_vpp", "no_vpp")

    print("\n=== Cross-condition Stage 2 evaluation ===")
    evaluate_pair("vpp", "no_vpp")
    evaluate_pair("no_vpp", "vpp")

    print("\n=== Against independent ablation target ===")
    evaluate_pair("vpp", "ablation", target_kind="ablation")
    evaluate_pair("no_vpp", "ablation", target_kind="ablation")
