"""Quick diagnostic: fly trained pursuer vs zero-action target in smoke_fleeing."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def main():
    cfg = load_yaml_config("config/adversarial/train_target_stability_ablation.yaml")
    env = AdversarialJSBSimEnv(cfg)

    # Load trained pursuer checkpoint (used as adversary)
    pursuer_ckpt = cfg.get("pursuer_checkpoint")
    pursuer_cfg = {
        "policy": {"hidden_sizes": [256, 256, 128], "activation": "relu"},
        "ppo": {"device": "cpu"},
    }
    pursuer_agent = PPOAgent(
        obs_dim=16,
        action_dim=3,
        config=pursuer_cfg,
        device="cpu",
    )
    if pursuer_ckpt and os.path.exists(pursuer_ckpt):
        pursuer_agent.load(pursuer_ckpt)
        print(f"Loaded pursuer from {pursuer_ckpt}")
    else:
        print(f"Pursuer checkpoint not found: {pursuer_ckpt}")
        return

    scenario = ScenarioRegistry.get("smoke_fleeing")

    def run_baseline(target_action, label):
        p_obs, t_obs = env.reset(scenario=scenario, seed=0)
        terminated = truncated = False
        step = 0
        info = {}
        while not (terminated or truncated) and step < 2000:
            p_action, _, _ = pursuer_agent.select_action(p_obs["observation_vector"], deterministic=True, store=False)
            p_obs, t_obs, p_reward, t_reward, terminated, truncated, info = env.step(p_action, target_action)
            step += 1
            if step % 50 == 0:
                t_state = info.get("target_state", {})
                rel = info.get("relative_state", {})
                print(f"  [{label}] step={step} range={rel.get('range_m', -1):.0f} "
                      f"tgt_alt={t_state.get('altitude_m', -1):.0f} tgt_speed={t_state.get('velocity_mps', -1):.1f}")
        term = info.get("termination", {})
        print(f"[{label}] Episode length: {step} steps ({step * env._high_level_dt:.1f}s) "
              f"reason={term.get('reason')} is_crash={term.get('is_crash')} "
              f"is_oob={term.get('is_out_of_bounds')} is_success={term.get('is_success')}")

    # Test several baseline trim commands
    nz_mid = (env._nz_min + env._nz_max) / 2.0
    for nz, throttle in [(1.0, 0.7), (1.0, 0.8), (1.0, 0.9), (1.25, 0.8), (1.5, 0.8)]:
        a0 = (nz - nz_mid) / ((env._nz_max - env._nz_min) / 2.0)
        a2 = (throttle - 0.5) / 0.5
        action = np.array([a0, 0.0, a2], dtype=np.float32)
        run_baseline(action, f"nz={nz:.2f}_thr={throttle:.1f}")


if __name__ == "__main__":
    main()
