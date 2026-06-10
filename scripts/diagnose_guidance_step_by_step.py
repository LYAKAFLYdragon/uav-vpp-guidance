#!/usr/bin/env python3
"""
Step-by-step diagnosis of the guidance command chain.
Prints every action, virtual point, and guidance command to find the root cause.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.expert_system import ExpertVPPPolicy
import yaml


def load_config(backend="simple"):
    with open(project_root / "config" / "experiment" / "train_no_prediction_vpp_ppo.yaml") as f:
        config = yaml.safe_load(f)
    config["backend"] = backend
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = (backend == "jsbsim")
    config["trajectory_prediction"]["enabled"] = False
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="simple", choices=["simple", "jsbsim"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    config = load_config(backend=args.backend)
    env = CloseRangeTrackingEnv(config)
    policy = ExpertVPPPolicy(config.get("expert_vpp", {}))

    scenario = config["scenarios"]["favorable"]
    obs = env.reset(scenario=scenario, seed=0)
    policy.reset_history()

    print("=" * 80)
    print(f"Step-by-step diagnosis ({args.backend} backend)")
    print("=" * 80)
    print(f"\nInitial state:")
    print(f"  Own:     pos={obs['own_state']['position_m']}, "
          f"vel={obs['own_state']['velocity_vector_mps']}, "
          f"speed={obs['own_state']['speed_mps']:.1f}")
    print(f"  Target:  pos={obs['target_state']['position_m']}, "
          f"vel={obs['target_state']['velocity_vector_mps']}, "
          f"speed={obs['target_state']['speed_mps']:.1f}")
    print(f"  Relative: range={obs['relative_state']['range_m']:.0f}m, "
          f"ata={np.rad2deg(obs['relative_state']['ata_rad']):.1f}deg")

    print(f"\n{'Step':>4} | {'Action':>24} | {'VP offset':>30} | "
          f"{'nz':>6} | {'roll':>6} | {'throttle':>8} | "
          f"{'Range':>8} | {'Alt':>8} | {'Speed':>8}")
    print("-" * 120)

    for step in range(args.steps):
        action = policy.get_action(
            obs["own_state"], obs["target_state"], obs["relative_state"]
        )
        obs, reward, terminated, truncated, info = env.step(action)

        vp = info.get("virtual_point", {}).get("virtual_point", [0, 0, 0])
        cmd = info.get("guidance_command", {})
        own = obs["own_state"]

        print(f"{step:>4} | {str(np.round(action, 2)):>24} | {str(np.round(vp, 0)):>30} | "
              f"{cmd.get('nz_cmd', 0):>6.2f} | {cmd.get('roll_rate_cmd', 0):>6.2f} | "
              f"{cmd.get('throttle_cmd', 0):>8.3f} | "
              f"{obs['relative_state']['range_m']:>8.0f} | "
              f"{own['position_m'][2]:>8.0f} | {own['speed_mps']:>8.1f}")

        if terminated or truncated:
            print(f"\nTerminated at step {step}: {info.get('reason', 'unknown')}")
            break

    env.close()


if __name__ == "__main__":
    main()
