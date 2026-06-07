"""
No-Prediction VPP Baseline 评估入口。

功能：
  python -m uav_vpp_guidance.evaluation.evaluate_no_prediction \
      --config config/experiment/no_prediction_vpp.yaml

执行随机策略或规则策略的若干 episode，输出 metrics。
"""

import argparse
import json
import os
import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.baselines.rule_based_pursuit import RuleBasedPursuitPolicy


def evaluate(env: CloseRangeTrackingEnv, num_episodes: int = 10, policy=None, seed: int = 0):
    """
    评估策略在多个 episode 上的表现。

    Args:
        env (CloseRangeTrackingEnv): 环境实例。
        num_episodes (int): 评估 episode 数。
        policy: 策略对象（需有 get_action 方法）；若为 None 则使用随机策略。
        seed (int): 随机种子。

    Returns:
        dict: 评估指标。
    """
    rng = np.random.default_rng(seed)
    episodes = []

    for ep in range(num_episodes):
        obs = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_length = 0
        min_range = float("inf")
        min_ata_deg = float("inf")
        final_range = 0.0
        final_ata = 0.0
        reason = "timeout"

        for step in range(env.max_steps):
            if policy is not None:
                rel_state = obs.get("relative_state", {})
                own_state = obs.get("own_state", {})
                target_state = obs.get("target_state", {})
                action = policy.get_action(own_state, target_state, rel_state)
            else:
                action = rng.uniform(-1.0, 1.0, size=3).astype(np.float64)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            rel_state = obs.get("relative_state", {})
            range_m = rel_state.get("range_m", 0.0)
            ata_deg = np.rad2deg(rel_state.get("ata_rad", 0.0))
            min_range = min(min_range, range_m)
            min_ata_deg = min(min_ata_deg, ata_deg)
            final_range = range_m
            final_ata = rel_state.get("ata_rad", 0.0)

            if terminated or truncated:
                reason = info.get("reason", "unknown")
                break

        episodes.append({
            "episode": ep,
            "return": ep_reward,
            "length": ep_length,
            "min_range_m": min_range,
            "min_ata_deg": min_ata_deg,
            "final_range_m": final_range,
            "final_ata_deg": float(np.rad2deg(final_ata)),
            "reason": reason,
            "is_success": reason == "success",
            "is_crash": reason == "crash",
            "is_timeout": reason == "timeout",
            "is_out_of_bounds": reason == "out_of_bounds",
        })

    # 汇总指标
    returns = [ep["return"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]
    min_ranges = [ep["min_range_m"] for ep in episodes]
    min_atas = [ep["min_ata_deg"] for ep in episodes]
    final_ranges = [ep["final_range_m"] for ep in episodes]
    final_atas = [ep["final_ata_deg"] for ep in episodes]

    metrics = {
        "num_episodes": num_episodes,
        "success_rate": sum(1 for ep in episodes if ep["is_success"]) / num_episodes,
        "crash_rate": sum(1 for ep in episodes if ep["is_crash"]) / num_episodes,
        "timeout_rate": sum(1 for ep in episodes if ep["is_timeout"]) / num_episodes,
        "out_of_bounds_rate": sum(1 for ep in episodes if ep["is_out_of_bounds"]) / num_episodes,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_min_range_m": float(np.mean(min_ranges)),
        "avg_final_range_m": float(np.mean(final_ranges)),
        "avg_final_ata_deg": float(np.mean(final_atas)),
        "avg_min_ata_deg": float(np.mean(min_atas)),
        "episodes": episodes,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate No-Prediction VPP Baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes (overrides config)")
    parser.add_argument("--rule-mode", type=str, default=None, choices=["pure_pursuit", "lag_pursuit", "lead_pursuit"],
                        help="Use rule-based policy instead of random")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    base_config = load_yaml_config(args.config)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(args.config), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    config = merge_config(merged, base_config)

    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    set_seed(seed)

    exp_name = config.get("experiment", {}).get("name", "no_prediction_vpp")

    # Episode count: CLI override > config.evaluation.episodes > default 10
    num_episodes = args.episodes
    if num_episodes is None:
        num_episodes = config.get("evaluation", {}).get("episodes", 10)

    env = CloseRangeTrackingEnv(config)

    policy = None
    if args.rule_mode is not None:
        policy = RuleBasedPursuitPolicy(mode=args.rule_mode)
        print(f"Using rule-based policy: {args.rule_mode}")
    else:
        print("Using random policy")

    print(f"Evaluating {num_episodes} episodes...")
    metrics = evaluate(env, num_episodes=num_episodes, policy=policy, seed=seed)
    env.close()

    print("\n=== Evaluation Metrics ===")
    for k in ["success_rate", "crash_rate", "timeout_rate", "out_of_bounds_rate",
              "avg_return", "avg_episode_length", "avg_min_range_m",
              "avg_final_range_m", "avg_final_ata_deg", "avg_min_ata_deg"]:
        print(f"  {k}: {metrics[k]:.4f}")

    # 保存结果到 outputs/tables/no_prediction_vpp/metrics.json
    tables_dir = os.path.join("outputs", "tables", exp_name)
    os.makedirs(tables_dir, exist_ok=True)
    metrics_path = os.path.join(tables_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
