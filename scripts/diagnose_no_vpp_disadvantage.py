"""Diagnose No-VPP mild-disadvantage timeouts (summary stats + sample trajectories)."""
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


CONFIGS = {
    "No-VPP": "config/experiment/train_no_vpp_ppo_jsbsim_maneuver.yaml",
    "VPP": "config/experiment/train_no_prediction_vpp_ppo_jsbsim_maneuver.yaml",
}
SCENARIO = "disadvantage"
TRAIN_SEEDS = [0, 1, 2, 3, 4]
N_EPS = 30
EVAL_SEED_BASE = 2000
FULL_TRAJ_SAMPLES = {"timeout": 3, "success": 3}


def run_method(method, config_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_finals = []
    full_traj_saved = {k: 0 for k in FULL_TRAJ_SAMPLES}

    for train_seed in TRAIN_SEEDS:
        cfg = load_cfg(config_path)
        cfg["experiment"]["mode"] = "eval"
        cfg["experiment"]["seed"] = train_seed

        env = CloseRangeTrackingEnv(cfg)
        env.set_domain_rand_scale(0.0)
        obs = env.reset(seed=0)
        agent = PPOAgent(obs["observation_vector"].shape[0], 3, env.config)

        ckpt_dir = os.path.join(
            "outputs", "experiments", cfg["experiment"]["name"], "checkpoints"
        )
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP {method} seed{train_seed}: no checkpoint")
            env.close()
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        agent.network.load_state_dict(ckpt["network_state_dict"])
        agent.network.eval()

        scenario = cfg["scenarios"][SCENARIO]
        for ep in range(N_EPS):
            eval_seed = EVAL_SEED_BASE + train_seed * 1000 + ep
            obs = env.reset(seed=eval_seed, scenario=scenario)
            traj = []
            done = False
            while not done:
                action, _, _ = agent.select_action(
                    obs["observation_vector"], deterministic=True, store=False
                )
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                own = info["own_state"]
                tgt = info["target_state"]
                traj.append(
                    {
                        "step": int(info["current_step"]),
                        "time_s": float(env._sim_time_s),
                        "own_n": float(own["position_m"][0]),
                        "own_e": float(own["position_m"][1]),
                        "own_alt": float(own["altitude_m"]),
                        "own_roll_deg": float(np.rad2deg(own["roll_rad"])),
                        "tgt_n": float(tgt["position_m"][0]),
                        "tgt_e": float(tgt["position_m"][1]),
                        "range_m": float(info["range_m"]),
                        "aspect_deg": float(info["aspect_deg"]),
                        "ata_deg": float(info["ata_deg"]),
                        "reward": float(reward),
                    }
                )
            reason = info.get("termination_info", {}).get("reason", "unknown")
            final = traj[-1]
            record = {
                "method": method,
                "train_seed": train_seed,
                "ep": ep,
                "eval_seed": eval_seed,
                "reason": reason,
                "length": len(traj),
                "final_range_m": final["range_m"],
                "final_ata_deg": final["ata_deg"],
                "final_aspect_deg": final["aspect_deg"],
                "final_alt_m": final["own_alt"],
                "min_range_m": min(t["range_m"] for t in traj),
                "max_roll_deg": max(abs(t["own_roll_deg"]) for t in traj),
                "mean_closing_speed": np.mean(
                    np.diff([t["range_m"] for t in traj]) / 0.2
                ) if len(traj) > 1 else 0.0,
            }
            all_finals.append(record)

            # Save a few full trajectories for plotting
            if reason in FULL_TRAJ_SAMPLES and full_traj_saved[reason] < FULL_TRAJ_SAMPLES[reason]:
                fname = os.path.join(
                    out_dir, f"{method}_seed{train_seed}_ep{ep}_{reason}_traj.json"
                )
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(traj, f)
                full_traj_saved[reason] += 1
        env.close()

    summary_path = os.path.join(out_dir, f"{method}_final_stats.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_finals, f, indent=2)
    print(f"[{method}] logged {len(all_finals)} episodes -> {summary_path}")
    return all_finals


def print_summary(finals):
    reasons = {}
    for r in finals:
        reasons.setdefault(r["reason"], 0)
        reasons[r["reason"]] += 1
    print("Reasons:", reasons)
    if finals:
        print(f"mean length: {np.mean([r['length'] for r in finals]):.1f}")
        print(f"mean final range: {np.mean([r['final_range_m'] for r in finals]):.1f} m")
        print(f"mean final ATA: {np.mean([abs(r['final_ata_deg']) for r in finals]):.1f} deg")
        print(f"mean min range: {np.mean([r['min_range_m'] for r in finals]):.1f} m")
        print(f"mean max |roll|: {np.mean([r['max_roll_deg'] for r in finals]):.1f} deg")
        print(f"mean closing speed: {np.mean([r['mean_closing_speed'] for r in finals]):.1f} m/s")


if __name__ == "__main__":
    out_dir = "outputs/jsbsim_diagnose"
    for method, path in CONFIGS.items():
        finals = run_method(method, path, out_dir)
        print(f"\n=== {method} mild-disadvantage summary ===")
        print_summary(finals)
