"""Final evaluation of VPP vs No-VPP adversarial curriculum pilots.

Evaluates Stage 2 and Stage 3 checkpoints with a large number of episodes
for stable success-rate estimates.
"""
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


def evaluate(condition: str, stage: int, run_dir: str, num_eps: int = 100, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    base_dir = f"outputs/adversarial_curriculum_pilot/{run_dir}"
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
    not_captured = total - reasons.get("success", 0)
    print(f"\n[{condition.upper()} {run_dir} Stage {stage}] checkpoint={os.path.basename(pursuer_ckpt)}")
    print(f"  Episodes: {total}")
    print(f"  Capture rate (SR): {sr:.2%}")
    print(f"  Target not captured: {not_captured/total:.2%}")
    print(f"  Reasons: {reasons}")
    print(f"  Mean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"  Mean length: {np.mean(lengths):.1f}")
    print(f"  Mean min range: {np.mean(min_ranges):.0f} m")
    return {"success_rate": sr, "reasons": reasons, "mean_return": np.mean(returns), "std_return": np.std(returns)}


def evaluate_stage1(condition: str, num_eps: int = 100, seeds=None):
    """Evaluate Stage 1 pursuer against rule-based maneuver target."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    run_dir = f"{condition}_full_gate10_s0" if condition == "no_vpp" else f"{condition}_full_gate25_s0"
    base_dir = f"outputs/adversarial_curriculum_pilot/{run_dir}"
    ckpt = os.path.join(base_dir, "stage1", "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        print(f"Stage 1 checkpoint not found for {condition}")
        return None

    cfg_path = ("config/experiment/close_range_curriculum_stage1.yaml" if condition == "vpp"
                else "config/experiment/close_range_curriculum_stage1_no_vpp.yaml")
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    config = load_yaml_config(cfg_path)
    env = CloseRangeTrackingEnv(config)
    p_policy = _policy_from_ckpt(ckpt)
    p_cfg = dict(config)
    if p_policy:
        p_cfg["policy"] = p_policy
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=p_cfg, device="cpu")
    pursuer.load(ckpt)
    pursuer.network.eval()

    successes = 0
    for seed in seeds:
        for ep in range(num_eps // max(1, len(seeds))):
            obs = env.reset(seed=seed * 10000 + ep)
            terminated = truncated = False
            for _ in range(env.max_steps):
                action = pursuer.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if info.get("termination_info", {}).get("is_success") or info.get("reason") == "success":
                        successes += 1
                    break
    env.close()
    sr = successes / num_eps
    print(f"\n[{condition.upper()} Stage 1] capture rate = {sr:.2%} ({successes}/{num_eps})")
    return {"success_rate": sr}


if __name__ == "__main__":
    print("=" * 70)
    print("FINAL EVALUATION: VPP vs No-VPP adversarial curriculum")
    print("=" * 70)

    # Stage 1 baselines
    evaluate_stage1("vpp")
    evaluate_stage1("no_vpp")

    # Stage 2 and 3
    results = {}
    for cond, run_dir in [("vpp", "vpp_full_gate25_s0"), ("no_vpp", "no_vpp_full_gate10_s0")]:
        for stage in [2, 3]:
            key = f"{cond}_stage{stage}"
            results[key] = evaluate(cond, stage, run_dir, num_eps=100)
