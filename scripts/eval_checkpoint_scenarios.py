"""Evaluate a trained checkpoint per scenario, optionally with a fixed VPP offset."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.virtual_point.fixed_offset_guidance import FixedOffsetGuidance


def load_cfg(path):
    base = load_yaml_config(path)
    includes = base.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = os.path.join(os.path.dirname(path), inc)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenarios", type=str, default="all", help="comma-separated or 'all'")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--fixed-offset", type=str, default=None, help="e.g. '500,0,0'")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["simple", "jsbsim"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    cfg["experiment"]["mode"] = "eval"
    if args.backend is not None:
        cfg["backend"] = args.backend
        cfg.setdefault("env", {})["backend"] = args.backend
        cfg.setdefault("env", {})["use_jsbsim"] = (args.backend == "jsbsim")

    env = CloseRangeTrackingEnv(cfg)
    env.set_domain_rand_scale(0.0)
    obs = env.reset(seed=0)
    agent = PPOAgent(obs["observation_vector"].shape[0], 3, env.config)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.network.eval()

    if args.fixed_offset is not None:
        offset = np.array([float(x) for x in args.fixed_offset.split(",")], dtype=np.float64)
        env.virtual_point_generator = FixedOffsetGuidance(offset)
        print(f"Using fixed offset: {offset.tolist()}")

    scenarios = cfg.get("scenarios", {})
    if args.scenarios != "all":
        names = [s.strip() for s in args.scenarios.split(",")]
        scenarios = {k: v for k, v in scenarios.items() if k in names}

    results = []
    for scen_name, scenario in scenarios.items():
        successes = crashes = timeouts = oobs = 0
        returns = []
        lengths = []
        for ep in range(args.episodes):
            seed = args.seed_base + ep
            obs = env.reset(seed=seed, scenario=scenario)
            done = False
            ep_return = 0.0
            length = 0
            while not done:
                action, _, _ = agent.select_action(
                    obs["observation_vector"], deterministic=True, store=False
                )
                obs, reward, term, trunc, info = env.step(action)
                ep_return += reward
                length += 1
                done = term or trunc
            reason = info.get("termination_info", {}).get("reason", "unknown")
            if reason == "success":
                successes += 1
            elif reason == "crash":
                crashes += 1
            elif reason == "out_of_bounds":
                oobs += 1
            else:
                timeouts += 1
            returns.append(ep_return)
            lengths.append(length)
        results.append(
            {
                "scenario": scen_name,
                "episodes": args.episodes,
                "success": successes,
                "success_rate": successes / args.episodes,
                "crash": crashes,
                "timeout": timeouts,
                "oob": oobs,
                "mean_return": float(np.mean(returns)),
                "mean_length": float(np.mean(lengths)),
            }
        )
        print(
            f"{scen_name:15s}: SR={successes}/{args.episodes} "
            f"(crash={crashes}, timeout={timeouts}, oob={oobs})"
        )
    env.close()

    out = args.output or "outputs/eval_checkpoint_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
