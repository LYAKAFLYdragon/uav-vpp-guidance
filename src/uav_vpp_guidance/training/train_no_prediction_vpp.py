"""
No-Prediction VPP Baseline 训练入口。

本阶段主要实现 smoke rollout 验证：
  python -m uav_vpp_guidance.training.train_no_prediction_vpp \
      --config config/experiment/no_prediction_vpp.yaml --smoke

完整 PPO 训练逻辑保留 TODO，后续在此文件接入。
"""

import argparse
import json
import os
import numpy as np
from datetime import datetime

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.utils.logger import create_experiment_dir, save_config_snapshot
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv


def smoke_rollout(env: CloseRangeTrackingEnv, num_steps: int = 100, seed: int = 0):
    """
    执行 smoke rollout：随机动作，验证闭环不崩溃。

    Args:
        env (CloseRangeTrackingEnv): 环境实例。
        num_steps (int): rollout 步数。
        seed (int): 随机种子。

    Returns:
        dict: Rollout summary statistics.
    """
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)

    rewards = []
    final_range = 0.0
    final_ata = 0.0
    done = False
    reason = None
    min_range = float("inf")

    for step in range(num_steps):
        # 随机动作（3 维偏移，归一化 [-1, 1]）
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float64)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        rel_state = obs.get("relative_state", {})
        range_m = rel_state.get("range_m", 0.0)
        min_range = min(min_range, range_m)
        final_range = range_m
        final_ata = rel_state.get("ata_rad", 0.0)

        if terminated or truncated:
            done = True
            reason = info.get("reason", "unknown")
            break

    summary = {
        "num_steps": len(rewards),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "final_range_m": final_range,
        "final_ata_deg": float(np.rad2deg(final_ata)),
        "done": done,
        "reason": reason,
        "min_range_m": min_range,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train No-Prediction VPP Baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run smoke rollout instead of training")
    parser.add_argument("--smoke-steps", type=int, default=100, help="Smoke rollout steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    # 加载配置
    base_config = load_yaml_config(args.config)
    # 合并 includes（简单处理：逐个加载并 merge）
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

    print(f"Experiment: {exp_name}")
    print(f"Backend: {'JSBSim' if config.get('env', {}).get('use_jsbsim', True) else 'SimplePointMass'}")

    if args.smoke:
        print(f"\nRunning smoke rollout ({args.smoke_steps} steps)...")
        env = CloseRangeTrackingEnv(config)
        summary = smoke_rollout(env, num_steps=args.smoke_steps, seed=seed)
        env.close()

        print("\n=== Smoke Rollout Summary ===")
        print(f"  total_reward: {summary['total_reward']:.4f}")
        print(f"  final_range: {summary['final_range_m']:.2f} m")
        print(f"  final_ata: {summary['final_ata_deg']:.2f} deg")
        print(f"  done: {summary['done']}")
        print(f"  reason: {summary['reason']}")

        # 保存到 outputs/tables/no_prediction_vpp/smoke_summary.json
        tables_dir = os.path.join("outputs", "tables", exp_name)
        os.makedirs(tables_dir, exist_ok=True)
        summary_path = os.path.join(tables_dir, "smoke_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_path}")
        return

    # TODO: 接入完整 PPO 训练逻辑
    print("\nFull training is not yet implemented. Use --smoke for rollout validation.")
    print("TODO: Integrate PPOAgent training loop here.")


if __name__ == "__main__":
    main()
