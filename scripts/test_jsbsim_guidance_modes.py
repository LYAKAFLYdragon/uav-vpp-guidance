"""
Quick JSBSim smoke test for all three guidance modes.
Runs 2 episodes per mode with random policy and checks for NaN/inf/crashes.
"""

import argparse
import numpy as np
import sys

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config


def run_mode(mode_name, guidance_mode, n_eps=2, seed=0, require_backend=None):
    config = load_yaml_config("config/experiment/no_prediction_vpp_jsbsim.yaml")
    config["guidance"]["mode"] = guidance_mode

    env = CloseRangeTrackingEnv(config)
    print(f"\n=== Mode: {mode_name} (backend: {env._backend}) ===")

    if require_backend and env._backend != require_backend:
        print(f"  BACKEND MISMATCH: required {require_backend}, got {env._backend}")
        return 1

    issues = []
    backend_violations = 0
    for ep in range(n_eps):
        obs = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(env.max_steps):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Check backend consistency
            actual_backend = info.get("backend")
            if require_backend and actual_backend != require_backend:
                backend_violations += 1
                issues.append(
                    f"Ep{ep} Step{step}: backend={actual_backend}, "
                    f"required={require_backend}"
                )

            # Check commands for NaN/Inf
            cmd = info.get("guidance_command", {})
            for k, v in cmd.items():
                if not np.isfinite(v):
                    issues.append(f"Ep{ep} Step{step}: {k}={v}")

            # Check own state for NaN/Inf
            own = info.get("own_state", {})
            pos_keys = ("position_m", "position_neu", "position_ned")
            pos = None
            for pk in pos_keys:
                pos = own.get(pk)
                if pos is not None:
                    break
            if pos is not None and not np.all(np.isfinite(pos)):
                issues.append(f"Ep{ep} Step{step}: own_pos non-finite")

            if terminated or truncated:
                break

        reason = info.get("reason", "unknown")
        print(
            f"  Ep{ep}: reward={ep_reward:.1f}, steps={step+1}, "
            f"reason={reason}, backend={actual_backend}"
        )

    env.close()
    if issues:
        print(f"  ISSUES ({len(issues)}) :")
        for i in issues[:10]:
            print(f"    - {i}")
    else:
        print("  OK: no NaN/Inf/backend issues")
    return len(issues)


def main():
    parser = argparse.ArgumentParser(description="JSBSim smoke test for guidance modes")
    parser.add_argument(
        "--require-backend",
        choices=["jsbsim", "simple"],
        default=None,
        help="Require every episode to use the specified backend",
    )
    args = parser.parse_args()

    modes = [
        ("LOS Rate", "los_rate"),
        ("Proportional Navigation", "proportional_navigation"),
        ("Hybrid", "hybrid"),
    ]
    total_issues = 0
    for name, mode in modes:
        try:
            total_issues += run_mode(name, mode, require_backend=args.require_backend)
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            total_issues += 1

    print("\n=== Summary ===")
    print(f"Total issues: {total_issues}")
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
