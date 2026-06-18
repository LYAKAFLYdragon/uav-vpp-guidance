"""Evaluate a trained target checkpoint with detailed reason breakdown."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def evaluate(config, target_ckpt, num_episodes=10, deterministic=True):
    env = AdversarialJSBSimEnv(config)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

    # Load pursuer
    pursuer_cfg = {
        "policy": {"hidden_sizes": [256, 256, 128], "activation": "relu"},
        "ppo": {"device": "cpu"},
    }
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=pursuer_cfg, device="cpu")
    pursuer_ckpt = config.get("pursuer_checkpoint")
    if pursuer_ckpt and os.path.exists(pursuer_ckpt):
        pursuer.load(pursuer_ckpt)
        print(f"Loaded pursuer from {pursuer_ckpt}")

    # Load target
    target = AdversarialTargetAgent(config, device="cpu")
    if target_ckpt and os.path.exists(target_ckpt):
        target.load(target_ckpt)
        print(f"Loaded target from {target_ckpt}")

    episodes = []
    for ep in range(num_episodes):
        scenario = sampler.sample() if sampler else None
        p_obs, t_obs = env.reset(scenario=scenario, seed=ep)
        ep_return = 0.0
        ep_length = 0
        min_range = float("inf")
        reason = "unknown"
        terminated = truncated = False
        while not (terminated or truncated) and ep_length < 2000:
            p_action, _, _ = pursuer.select_action(p_obs["observation_vector"], deterministic=True, store=False)
            t_action, _, _ = target.select_action(t_obs["observation_vector"], deterministic=deterministic, store=False)
            p_obs, t_obs, p_rew, t_rew, terminated, truncated, info = env.step(p_action, t_action)
            ep_return += t_rew
            ep_length += 1
            rel = t_obs.get("relative_state", {})
            range_m = float(rel.get("range_m", 0.0))
            min_range = min(min_range, range_m)
            if terminated or truncated:
                reason = info.get("termination", {}).get("reason", "unknown")
                break
        episodes.append({
            "return": ep_return,
            "length": ep_length,
            "reason": reason,
            "min_range_m": min_range,
        })
        print(f"Ep {ep}: return={ep_return:.1f} length={ep_length} reason={reason} min_range={min_range:.0f}")

    reasons = {}
    for e in episodes:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    print("\nReason breakdown:", reasons)
    print(f"Mean return: {np.mean([e['return'] for e in episodes]):.1f}")
    print(f"Mean length: {np.mean([e['length'] for e in episodes]):.1f}")
    print(f"Not captured (timeout+oob): {reasons.get('timeout',0)+reasons.get('out_of_bounds',0)}/{num_episodes}")


if __name__ == "__main__":
    cfg = load_yaml_config("config/adversarial/train_target_stability_ablation.yaml")
    evaluate(cfg, "outputs/adversarial_curriculum_pilot/target_stability_ablation/checkpoints/best.pt", num_episodes=10)
