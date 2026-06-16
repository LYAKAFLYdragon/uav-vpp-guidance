"""Expanded JSBSim evaluation matrix: 5 seeds x 30 episodes per config/scenario."""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "src")
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
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


MATRIX = [
    {
        "method": "VPP",
        "target": "mild",
        "config": "config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml",
    },
    {
        "method": "No-VPP",
        "target": "mild",
        "config": "config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml",
    },
    {
        "method": "VPP",
        "target": "weave",
        "config": "config/experiment/train_no_prediction_vpp_ppo_jsbsim_weave.yaml",
    },
    {
        "method": "No-VPP",
        "target": "weave",
        "config": "config/experiment/train_no_vpp_ppo_jsbsim_weave.yaml",
    },
]

TRAIN_SEEDS = [0, 1, 2, 3, 4]
EVAL_EPISODES = 30
EVAL_SEED_BASE = 1000


def evaluate_entry(entry, train_seed, eval_eps=EVAL_EPISODES):
    cfg = load_cfg(entry["config"])
    cfg["experiment"]["mode"] = "eval"
    # Fix training seed for checkpoint selection
    cfg["experiment"]["seed"] = train_seed
    env = CloseRangeTrackingEnv(cfg)
    obs = env.reset(seed=0)
    agent = PPOAgent(obs["observation_vector"].shape[0], 3, env.config)
    ckpt_dir = os.path.join(
        "outputs", "experiments", cfg["experiment"]["name"], "checkpoints"
    )
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        # fallback to last.pt
        ckpt_path = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt_path):
        print(f"SKIP {entry['method']}/{entry['target']}/seed{train_seed}: no checkpoint")
        env.close()
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.network.eval()

    results = []
    for scen_name, scenario in cfg["scenarios"].items():
        successes = 0
        crashes = 0
        timeouts = 0
        oobs = 0
        returns = []
        lengths = []
        for ep in range(eval_eps):
            seed = EVAL_SEED_BASE + train_seed * 1000 + ep
            obs = env.reset(seed=seed, scenario=scenario)
            done = False
            ep_return = 0.0
            length = 0
            while not done:
                action = agent.select_action(
                    obs["observation_vector"], deterministic=True, store=False
                )[0]
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
                "method": entry["method"],
                "target": entry["target"],
                "train_seed": train_seed,
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
    return results


if __name__ == "__main__":
    out_dir = "outputs/jsbsim_matrix_eval"
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for entry in MATRIX:
        for train_seed in TRAIN_SEEDS:
            print(
                f"Evaluating {entry['method']} / {entry['target']} / train_seed={train_seed} ..."
            )
            rows = evaluate_entry(entry, train_seed)
            if rows is not None:
                all_rows.extend(rows)
                # Save incremental
                with open(os.path.join(out_dir, "raw.jsonl"), "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")

    # Aggregate
    summary = {}
    for r in all_rows:
        key = (r["method"], r["target"], r["scenario"])
        summary.setdefault(key, []).append(r["success_rate"])

    print("\n=== Aggregated success rate (mean ± std over train seeds) ===")
    agg_rows = []
    for (method, target, scenario), rates in sorted(summary.items()):
        mean = float(np.mean(rates))
        std = float(np.std(rates))
        print(f"{method:8s} {target:6s} {scenario:12s}: {mean:.3f} ± {std:.3f}")
        agg_rows.append(
            {
                "method": method,
                "target": target,
                "scenario": scenario,
                "mean_success_rate": mean,
                "std_success_rate": std,
                "n_seeds": len(rates),
            }
        )

    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg_rows, f, indent=2)
    print(f"\nSaved: {out_dir}/aggregate.json")
