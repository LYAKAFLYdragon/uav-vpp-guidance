"""
Phase 7: JSBSim comparative evaluation for three guidance modes.
Runs short MC runs on JSBSim backend and reports terminal-phase stability metrics.

NOTE: This script uses an untrained PPO agent (random-initialized weights) for
smoke testing guidance laws. Results reflect guidance stability, NOT trained
policy performance.
"""

import argparse
import copy
import csv
import json
import os
import time

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent


def evaluate_mode(
    mode_name, config, seeds, episodes_per_seed, scenarios, require_backend=None
):
    config["guidance"]["mode"] = mode_name
    env = CloseRangeTrackingEnv(config)

    actual_backend = env._backend
    print(f"\n=== Evaluating {mode_name} (backend: {actual_backend}) ===")

    if require_backend and actual_backend != require_backend:
        print(
            f"  BACKEND MISMATCH: required {require_backend}, " f"got {actual_backend}"
        )
        env.close()
        return None

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))

    all_episodes = []
    nan_inf_issues = 0
    backend_violations = 0
    for seed in seeds:
        set_seed(seed)
        # Re-initialize agent per seed so weight init is deterministic
        agent = PPOAgent(
            obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu"
        )
        for ep in range(episodes_per_seed):
            ep_seed = seed * 10000 + ep
            scenario = scenarios[ep % len(scenarios)] if scenarios else None
            obs = env.reset(scenario=scenario, seed=ep_seed)
            ep_reward = 0.0
            ep_length = 0
            step_commands = []
            step_ranges = []
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1

                # Backend validation
                step_backend = info.get("backend")
                if require_backend and step_backend != require_backend:
                    backend_violations += 1

                # NaN/Inf checks on commands
                cmd = {
                    "nz_cmd": info.get("nz_cmd", np.nan),
                    "roll_rate_cmd": info.get("roll_rate_cmd", np.nan),
                    "throttle_cmd": info.get("throttle_cmd", np.nan),
                }
                for k, v in cmd.items():
                    if not np.isfinite(v):
                        nan_inf_issues += 1

                step_commands.append(cmd)
                rel_state = obs.get("relative_state", {})
                step_ranges.append(rel_state.get("range_m", np.nan))
                if terminated or truncated:
                    break

            # Terminal-phase metrics (last 20% steps, min 5)
            n = len(step_commands)
            t_start = max(0, n - max(5, int(np.ceil(0.2 * n))))
            t_cmds = step_commands[t_start:]
            t_ranges = step_ranges[t_start:]

            nz_vals = [c["nz_cmd"] for c in t_cmds if np.isfinite(c["nz_cmd"])]
            roll_vals = [
                c["roll_rate_cmd"] for c in t_cmds if np.isfinite(c["roll_rate_cmd"])
            ]
            throttle_vals = [
                c["throttle_cmd"] for c in t_cmds if np.isfinite(c["throttle_cmd"])
            ]

            limits = config.get("limits", {})
            nz_min, nz_max = limits.get("nz_min", -2.0), limits.get("nz_max", 7.0)
            rr_min, rr_max = limits.get("roll_rate_min", -1.5), limits.get(
                "roll_rate_max", 1.5
            )

            def safe_var(vals):
                clean = [v for v in vals if np.isfinite(v)]
                return float(np.var(clean, ddof=1)) if len(clean) > 1 else 0.0

            all_episodes.append(
                {
                    "seed": seed,
                    "episode": ep,
                    "scenario": (
                        scenario.get("name", "random")
                        if isinstance(scenario, dict)
                        else "random"
                    ),
                    "length": ep_length,
                    "return": ep_reward,
                    "reason": info.get("reason", "unknown"),
                    "backend": info.get("backend", "unknown"),
                    "terminal_nz_var": safe_var(nz_vals),
                    "terminal_roll_var": safe_var(roll_vals),
                    "terminal_throttle_var": safe_var(throttle_vals),
                    "terminal_nz_exceed": sum(
                        1 for v in nz_vals if v < nz_min or v > nz_max
                    )
                    / max(1, len(nz_vals)),
                    "terminal_roll_exceed": sum(
                        1 for v in roll_vals if v < rr_min or v > rr_max
                    )
                    / max(1, len(roll_vals)),
                    "terminal_mean_range_m": (
                        float(np.mean(t_ranges)) if t_ranges else np.nan
                    ),
                }
            )

    env.close()

    print(
        f"  Episodes: {len(all_episodes)}, "
        f"NaN/Inf issues: {nan_inf_issues}, "
        f"backend violations: {backend_violations}"
    )

    def safe_mean(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else np.nan

    def safe_std(vals):
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0

    returns = [e["return"] for e in all_episodes]
    success_rate = sum(1 for e in all_episodes if e["reason"] == "success") / max(
        1, len(all_episodes)
    )

    return {
        "mode": mode_name,
        "num_episodes": len(all_episodes),
        "mean_return": safe_mean(returns),
        "std_return": safe_std(returns),
        "success_rate": success_rate,
        "nan_inf_issues": nan_inf_issues,
        "backend_violations": backend_violations,
        "terminal_nz_var": safe_mean([e["terminal_nz_var"] for e in all_episodes]),
        "terminal_roll_var": safe_mean([e["terminal_roll_var"] for e in all_episodes]),
        "terminal_throttle_var": safe_mean(
            [e["terminal_throttle_var"] for e in all_episodes]
        ),
        "terminal_nz_exceed": safe_mean(
            [e["terminal_nz_exceed"] for e in all_episodes]
        ),
        "terminal_roll_exceed": safe_mean(
            [e["terminal_roll_exceed"] for e in all_episodes]
        ),
        "terminal_mean_range_m": safe_mean(
            [e["terminal_mean_range_m"] for e in all_episodes]
        ),
        "episodes": all_episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="JSBSim guidance mode comparison (untrained agent smoke test)"
    )
    parser.add_argument(
        "--config", default="config/experiment/no_prediction_vpp_jsbsim.yaml"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/jsbsim_guidance_comparison")
    parser.add_argument(
        "--require-backend",
        choices=["jsbsim", "simple"],
        default=None,
        help="Require every episode to use the specified backend",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    scenarios = list(config.get("scenarios", {}).values())
    if not scenarios:
        scenarios = [None]

    modes = ["los_rate", "proportional_navigation", "hybrid"]
    results = []
    total_issues = 0
    for mode in modes:
        start = time.time()
        mode_config = copy.deepcopy(config)
        res = evaluate_mode(
            mode,
            mode_config,
            args.seeds,
            args.episodes,
            scenarios,
            require_backend=args.require_backend,
        )
        elapsed = time.time() - start
        if res is None:
            total_issues += 1
            continue
        total_issues += res["nan_inf_issues"] + res["backend_violations"]
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Mean Return: {res['mean_return']:.1f} ± " f"{res['std_return']:.1f}")
        print(f"  Success Rate: {res['success_rate']:.1%}")
        print(f"  Terminal NZ Var: {res['terminal_nz_var']:.4f}")
        print(f"  Terminal Roll Var: {res['terminal_roll_var']:.4f}")
        print(f"  Terminal NZ Exceed: {res['terminal_nz_exceed']:.2%}")
        print(f"  Terminal Roll Exceed: {res['terminal_roll_exceed']:.2%}")
        print(f"  Terminal Mean Range: {res['terminal_mean_range_m']:.1f} m")
        results.append(res)

    # Save JSON
    json_path = os.path.join(args.output_dir, "results.json")
    output_payload = {
        "policy_type": "untrained_deterministic_ppo",
        "modes": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save CSV summary
    csv_path = os.path.join(args.output_dir, "summary.csv")
    keys = [
        "mode",
        "num_episodes",
        "mean_return",
        "std_return",
        "success_rate",
        "nan_inf_issues",
        "backend_violations",
        "terminal_nz_var",
        "terminal_roll_var",
        "terminal_throttle_var",
        "terminal_nz_exceed",
        "terminal_roll_exceed",
        "terminal_mean_range_m",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"CSV saved to: {csv_path}")

    if total_issues > 0:
        print(f"\nERROR: {total_issues} total issues detected")
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
