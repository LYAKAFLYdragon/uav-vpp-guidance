#!/usr/bin/env python3
"""
快速运行 Expert VPP Policy 作为对照基准。
用于验证：0% Success 是 Simple 后端固有问题，还是 No-Pred 方法本身弱。
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.expert_system import ExpertVPPPolicy
import yaml


def load_config():
    with open(project_root / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml") as f:
        config = yaml.safe_load(f)
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    config["trajectory_prediction"]["enabled"] = False
    return config


def run_episode(env, policy, scenario, seed):
    obs = env.reset(scenario=scenario, seed=seed)
    policy.reset_history()
    total_reward = 0.0
    length = 0
    min_range = float("inf")
    
    while True:
        action = policy.get_action(
            obs["own_state"], obs["target_state"], obs["relative_state"]
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        length += 1
        min_range = min(min_range, obs["relative_state"]["range_m"])
        
        if terminated or truncated:
            reason = info.get("reason", "unknown")
            is_success = info.get("is_success", False)
            final_range = obs["relative_state"]["range_m"]
            return {
                "return": total_reward,
                "length": length,
                "min_range_m": min_range,
                "final_range_m": final_range,
                "reason": reason,
                "is_success": is_success,
            }


def main():
    config = load_config()
    env = CloseRangeTrackingEnv(config)
    policy = ExpertVPPPolicy(config.get("expert_vpp", {}))
    
    scenario = config["scenarios"]["favorable"]
    
    print("=" * 60)
    print("Expert VPP Policy Baseline (favorable scenario)")
    print("=" * 60)
    
    results = []
    for ep in range(3):
        result = run_episode(env, policy, scenario, seed=ep)
        results.append(result)
        status = "SUCCESS" if result["is_success"] else result["reason"].upper()
        print(f"  Ep {ep}: {status:15s} | return={result['return']:+.1f} | "
              f"len={result['length']:3d} | min_range={result['min_range_m']:.0f}m | "
              f"final_range={result['final_range_m']:.0f}m")
    
    success_rate = sum(1 for r in results if r["is_success"]) / len(results)
    mean_return = np.mean([r["return"] for r in results])
    mean_min_range = np.mean([r["min_range_m"] for r in results])
    
    print("-" * 60)
    print(f"Summary: Success rate = {success_rate*100:.0f}% | Mean return = {mean_return:.1f} | Mean min_range = {mean_min_range:.0f}m")
    print("=" * 60)
    
    env.close()
    return success_rate > 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
