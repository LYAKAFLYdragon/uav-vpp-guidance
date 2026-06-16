"""Evaluate existing end-to-end PPO checkpoints zero-shot on JSBSim F-16."""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "src")
from uav_vpp_guidance.agents.end_to_end_ppo_agent import EndToEndPPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


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
    cfg = load_cfg("config/experiment/train_end_to_end_ppo.yaml")
    cfg["env"]["backend"] = "jsbsim"
    cfg["env"]["use_jsbsim"] = True
    cfg["experiment"]["mode"] = "eval"

    train_seeds = list(range(10))
    eval_eps = 10
    rows = []
    for seed in train_seeds:
        cfg["experiment"]["seed"] = seed
        env = CloseRangeTrackingEnv(cfg)
        obs = env.reset(seed=0)
        agent = EndToEndPPOAgent(obs["observation_vector"].shape[0], 3, cfg)
        ckpt_dir = f"outputs/experiments/end_to_end_ppo_seed{seed}/checkpoints"
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP seed {seed}: no checkpoint")
            env.close()
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        agent.network.load_state_dict(ckpt["network_state_dict"])
        agent.network.eval()

        for scen_name, scenario in cfg["scenarios"].items():
            successes = crashes = timeouts = oobs = 0
            returns = []
            lengths = []
            for ep in range(eval_eps):
                ep_seed = 2000 + seed * 1000 + ep
                obs = env.reset(seed=ep_seed, scenario=scenario)
                done = False
                ep_return = 0.0
                length = 0
                while not done:
                    action = agent.get_deterministic_action(obs["observation_vector"])
                    action = agent.clip_action(action)
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
            rows.append(
                {
                    "method": "End-to-End",
                    "train_seed": seed,
                    "scenario": scen_name,
                    "success": successes,
                    "crash": crashes,
                    "timeout": timeouts,
                    "oob": oobs,
                    "total": eval_eps,
                    "success_rate": successes / eval_eps,
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "mean_length": float(np.mean(lengths)),
                }
            )
        env.close()

    out_dir = "outputs/jsbsim_e2e_eval"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "raw.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n=== E2E zero-shot on JSBSim (per scenario over seeds) ===")
    summary = {}
    for r in rows:
        summary.setdefault(r["scenario"], []).append(r["success_rate"])
    agg = []
    for scen, rates in sorted(summary.items()):
        mean = float(np.mean(rates))
        std = float(np.std(rates))
        print(f"{scen:12s}: {mean:.3f} ± {std:.3f} (n={len(rates)})")
        agg.append({"scenario": scen, "mean_success_rate": mean, "std_success_rate": std, "n_seeds": len(rates)})
    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
