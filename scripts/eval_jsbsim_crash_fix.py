"""Quick crash-fix evaluation across JSBSim maneuver/weave configs."""
import os
import sys
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


CONFIGS = [
    "config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml",
    "config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml",
    "config/experiment/train_no_prediction_vpp_ppo_jsbsim_weave.yaml",
    "config/experiment/train_no_vpp_ppo_jsbsim_weave.yaml",
]


def eval_config(cfg_path, n_eps=5):
    cfg = load_cfg(cfg_path)
    cfg["experiment"]["mode"] = "eval"
    env = CloseRangeTrackingEnv(cfg)
    obs = env.reset(seed=0)
    agent = PPOAgent(obs["observation_vector"].shape[0], 3, env.config)
    ckpt_path = os.path.join(
        "outputs", "experiments", cfg["experiment"]["name"], "checkpoints", "best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"SKIP {cfg_path}: checkpoint not found")
        env.close()
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.network.eval()

    results = {}
    for scen_name, scenario in cfg["scenarios"].items():
        stats = {"success": 0, "crash": 0, "oob": 0, "timeout": 0, "total": 0}
        for ep in range(n_eps):
            obs = env.reset(seed=ep + 1, scenario=scenario)
            done = False
            while not done:
                action = agent.select_action(
                    obs["observation_vector"], deterministic=True, store=False
                )[0]
                obs, rew, term, trunc, info = env.step(action)
                done = term or trunc
            reason = info.get("termination_info", {}).get("reason", "unknown")
            stats["total"] += 1
            if reason == "success":
                stats["success"] += 1
            elif reason == "crash":
                stats["crash"] += 1
            elif reason == "out_of_bounds":
                stats["oob"] += 1
            else:
                stats["timeout"] += 1
        results[scen_name] = stats
    env.close()
    return results


if __name__ == "__main__":
    n_eps = int(os.environ.get("N_EPS", "5"))
    for cfg_path in CONFIGS:
        print(f"\n=== {cfg_path} ===")
        results = eval_config(cfg_path, n_eps=n_eps)
        if results is None:
            continue
        for scen, stats in results.items():
            print(f"  {scen}: {stats}")
