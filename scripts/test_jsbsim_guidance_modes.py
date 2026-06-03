"""
Quick JSBSim smoke test for all three guidance modes.
Runs 2 episodes per mode with random policy and checks for NaN/inf/crashes.
"""

import numpy as np
import sys

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config


def run_mode(mode_name, guidance_mode, n_eps=2, seed=0):
    config = load_yaml_config("config/experiment/no_prediction_vpp_jsbsim.yaml")
    config["guidance"]["mode"] = guidance_mode

    env = CloseRangeTrackingEnv(config)
    print(f"\n=== Mode: {mode_name} (backend: {env._backend}) ===")

    issues = []
    for ep in range(n_eps):
        obs = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(env.max_steps):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Check commands for NaN/Inf
            cmd = info.get("guidance_command", {})
            for k, v in cmd.items():
                if not np.isfinite(v):
                    issues.append(f"Ep{ep} Step{step}: {k}={v}")

            # Check own state for NaN/Inf
            own = info.get("own_state", {})
            pos = own.get("position_m")
            if pos is None:
                pos = own.get("position_neu")
            if pos is not None and not np.all(np.isfinite(pos)):
                issues.append(f"Ep{ep} Step{step}: own_pos non-finite")

            if terminated or truncated:
                break

        reason = info.get("reason", "unknown")
        print(f"  Ep{ep}: reward={ep_reward:.1f}, steps={step+1}, reason={reason}")

    env.close()
    if issues:
        print(f"  ISSUES ({len(issues)}):")
        for i in issues[:5]:
            print(f"    - {i}")
    else:
        print("  OK: no NaN/Inf issues")
    return len(issues)


def main():
    modes = [
        ("LOS Rate", "los_rate"),
        ("Proportional Navigation", "proportional_navigation"),
        ("Hybrid", "hybrid"),
    ]
    total_issues = 0
    for name, mode in modes:
        try:
            total_issues += run_mode(name, mode)
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            total_issues += 1

    print("\n=== Summary ===")
    print(f"Total issues: {total_issues}")
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
