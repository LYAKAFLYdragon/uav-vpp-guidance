"""Evaluate trained target per scenario to identify weaknesses."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def eval_scenario(env, pursuer, target, scenario_name, num_eps=5):
    scenario = ScenarioRegistry.get(scenario_name)
    if scenario is None:
        print(f"Scenario {scenario_name} not found")
        return
    reasons = {}
    returns = []
    lengths = []
    for ep in range(num_eps):
        p_obs, t_obs = env.reset(scenario=scenario, seed=ep)
        ep_ret = 0.0
        ep_len = 0
        reason = "unknown"
        terminated = truncated = False
        while not (terminated or truncated) and ep_len < 2000:
            p_action = pursuer.get_deterministic_action(p_obs["observation_vector"])
            t_action = target.get_deterministic_action(t_obs["observation_vector"])
            p_obs, t_obs, _, t_rew, terminated, truncated, info = env.step(p_action, t_action)
            ep_ret += t_rew
            ep_len += 1
            if terminated or truncated:
                reason = info.get("termination", {}).get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        returns.append(ep_ret)
        lengths.append(ep_len)
    print(f"{scenario_name:25s}: ret={np.mean(returns):7.1f} len={np.mean(lengths):5.0f} reasons={reasons}")


def main():
    cfg = load_yaml_config("config/adversarial/train_target_stability_ablation_v2.yaml")
    env = AdversarialJSBSimEnv(cfg)

    pursuer_cfg = {
        "policy": {"hidden_sizes": [256, 256, 128], "activation": "relu"},
        "ppo": {"device": "cpu"},
    }
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=pursuer_cfg, device="cpu")
    pursuer.load(cfg["pursuer_checkpoint"])

    target = AdversarialTargetAgent(cfg, device="cpu")
    target.load("outputs/adversarial_curriculum_pilot/target_stability_ablation_v2/checkpoints/best.pt")

    scenarios = [
        "smoke_head_on",
        "smoke_crossing_left",
        "smoke_crossing_right",
        "smoke_tail_chase",
        "smoke_offset_attack",
        "smoke_fleeing",
    ]
    for s in scenarios:
        eval_scenario(env, pursuer, target, s, num_eps=5)


if __name__ == "__main__":
    main()
