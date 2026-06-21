"""Evaluate Stage 3 checkpoints for the adversarial curriculum pilot monitor.

Compares:
  - outputs/adversarial_curriculum_pilot/no_vpp_full_gate10_s0/stage3
  - outputs/adversarial_curriculum_pilot/vpp_full_gate25_s0/stage3

Runs 30 episodes by default and returns per-condition success rates.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import torch
import numpy as np
from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.scenario_sampler import make_scenario_sampler
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.agents.adversarial_target_agent import AdversarialTargetAgent


CONDITIONS = {
    "no_vpp": {
        "base_dir": "outputs/adversarial_curriculum_pilot/no_vpp_full_gate10_s0",
        "stage1_cfg": "config/experiment/close_range_curriculum_stage1_no_vpp.yaml",
        "pursuer_cfg": "config/adversarial/train_pursuer_v2_no_vpp_pilot.yaml",
    },
    "vpp": {
        "base_dir": "outputs/adversarial_curriculum_pilot/vpp_full_gate25_s0",
        "stage1_cfg": "config/experiment/close_range_curriculum_stage1.yaml",
        "pursuer_cfg": "config/adversarial/train_pursuer_v2_pilot_aggressive.yaml",
    },
}


def _policy_from_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("config", {}).get("policy")
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def evaluate_condition(condition: str, num_eps: int = 30, seeds=None):
    if seeds is None:
        seeds = [0, 1, 2]
    meta = CONDITIONS[condition]
    base_dir = meta["base_dir"]
    stage_dir = os.path.join(base_dir, "stage3")
    pursuer_dir = os.path.join(stage_dir, "pursuer")
    target_dir = os.path.join(stage_dir, "target")

    best_pt = os.path.join(pursuer_dir, "checkpoints", "best.pt")
    last_pt = os.path.join(pursuer_dir, "checkpoints", "last.pt")
    pursuer_ckpt = best_pt if os.path.exists(best_pt) else last_pt

    target_best_pt = os.path.join(target_dir, "checkpoints", "best.pt")
    target_last_pt = os.path.join(target_dir, "checkpoints", "last.pt")
    target_ckpt = target_best_pt if os.path.exists(target_best_pt) else target_last_pt

    if not os.path.exists(pursuer_ckpt):
        print(f"[{condition.upper()} Stage 3] No pursuer checkpoint found, skipping")
        return None
    if not os.path.exists(target_ckpt):
        print(f"[{condition.upper()} Stage 3] No target checkpoint found, skipping")
        return None

    config = load_yaml_config(meta["stage1_cfg"])
    p_cfg = load_yaml_config(meta["pursuer_cfg"])
    config = {**p_cfg, **config}
    config["pursuer_checkpoint"] = pursuer_ckpt
    config["target_checkpoint"] = target_ckpt

    env = AdversarialJSBSimEnv(config)
    sampler = make_scenario_sampler(config.get("scenario_sampler", {}))

    pursuer_cfg = dict(config)
    p_policy = _policy_from_ckpt(pursuer_ckpt)
    if p_policy:
        pursuer_cfg["policy"] = p_policy
    pursuer = PPOAgent(obs_dim=16, action_dim=3, config=pursuer_cfg, device="cpu")
    pursuer.load(pursuer_ckpt)
    pursuer.network.eval()

    target_cfg = dict(config)
    target_cfg["ppo"] = config.get("target_ppo", config.get("ppo", {}))
    target_cfg["policy"] = config.get("target_policy", config.get("policy", {}))
    t_policy = _policy_from_ckpt(target_ckpt)
    if t_policy:
        target_cfg["policy"] = t_policy
    target = AdversarialTargetAgent(config=target_cfg, device="cpu")
    target.load(target_ckpt)
    target.eval()

    per_seed_sr = []
    reasons = {}
    returns = []
    lengths = []
    min_ranges = []
    episode_outcomes = []

    eps_per_seed = num_eps // max(1, len(seeds))
    for seed in seeds:
        seed_success = 0
        for ep in range(eps_per_seed):
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
                    captured = reason == "success"
                    reasons[reason] = reasons.get(reason, 0) + 1
                    returns.append(ep_return)
                    lengths.append(ep_len)
                    min_ranges.append(min_range)
                    episode_outcomes.append({
                        "seed": seed,
                        "episode": ep,
                        "captured": captured,
                        "reason": reason,
                        "return": float(ep_return),
                        "length": ep_len,
                        "min_range_m": float(min_range),
                    })
                    if captured:
                        seed_success += 1
                    break
        per_seed_sr.append(seed_success / eps_per_seed)

    env.close()
    total = sum(reasons.values())
    sr = reasons.get("success", 0) / max(1, total)
    variance = float(np.var(per_seed_sr, ddof=1)) if len(per_seed_sr) > 1 else 0.0
    std = float(np.std(per_seed_sr, ddof=1)) if len(per_seed_sr) > 1 else 0.0

    print(f"\n[{condition.upper()} Stage 3] {pursuer_ckpt}")
    print(f"  Episodes: {total}  capture_rate={sr:.2%}")
    print(f"  Per-seed SR: {per_seed_sr}")
    print(f"  Std (across seeds): {std:.3f}")
    print(f"  Reasons: {reasons}")
    print(f"  Mean return: {np.mean(returns):.1f}  length: {np.mean(lengths):.1f}  min_range: {np.mean(min_ranges):.0f}m")

    return {
        "condition": condition,
        "stage": 3,
        "pursuer_ckpt": pursuer_ckpt,
        "target_ckpt": target_ckpt,
        "num_episodes": total,
        "success_rate": sr,
        "per_seed_success_rate": per_seed_sr,
        "success_rate_std": std,
        "success_rate_variance": variance,
        "reasons": reasons,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_min_range_m": float(np.mean(min_ranges)),
        "episodes": episode_outcomes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 pilot checkpoints.")
    parser.add_argument("--num-eps", type=int, default=30, help="Episodes per condition (default: 30).")
    parser.add_argument("--output-name", type=str, default="stage3_eval_raw.json", help="Output JSON file name.")
    args = parser.parse_args()

    results = {}
    for cond in ["vpp", "no_vpp"]:
        results[cond] = evaluate_condition(cond, num_eps=args.num_eps)

    out_dir = "outputs/adversarial_pilot"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw evaluation results written to {out_path}")
