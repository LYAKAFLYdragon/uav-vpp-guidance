#!/usr/bin/env python3
"""Demo: run CloseRangeTrackingEnv with a maneuver-library-driven bandit.

Usage:
    source activate jsbenv
    python scripts/demo_bandit_env.py --config config/env_bandit.yaml --steps 100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def _resolve_config(config_path: str) -> dict:
    """Load config and inline includes the same way training scripts do."""
    root_dir = Path(config_path).parent
    base = load_yaml_config(config_path)
    includes = base.pop("includes", []) or []
    merged = {}
    for inc in includes:
        inc_path = root_dir / inc
        if inc_path.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_path)))
    return merge_config(merged, base)


def main():
    parser = argparse.ArgumentParser(description="Bandit maneuver baseline demo")
    parser.add_argument("--config", type=str, default="config/env_bandit.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Override bandit difficulty (easy/medium/hard)")
    args = parser.parse_args()

    config = _resolve_config(args.config)
    if args.difficulty is not None:
        config.setdefault("env", {})["bandit"] = config.get("env", {}).get("bandit", {})
        config["env"]["bandit"]["difficulty"] = args.difficulty

    env = CloseRangeTrackingEnv(config)
    print(f"Backend: {env._backend}")
    print(f"Bandit enabled: {env._bandit_enabled}")
    print(f"Missile enabled: {env._missile_tracker is not None}")

    obs = env.reset()
    print(f"Initial range: {obs['relative_state']['range_m']:.1f} m")

    for step in range(args.steps):
        action = np.zeros(3, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)

        line = (
            f"step={step:3d}  "
            f"range={info.get('range_m', np.nan):.0f}m  "
            f"ata={info.get('ata_deg', np.nan):.1f}\u00b0  "
            f"bandit_maneuver={info.get('bandit_maneuver', 'N/A'):12s}  "
            f"reward={reward:7.2f}"
        )
        missiles = info.get("missiles", [])
        if missiles:
            line += f"  missiles={len(missiles)}"
        if info.get("termination_info", {}).get("reason"):
            line += f"  reason={info['termination_info']['reason']}"
        print(line)

        if terminated or truncated:
            print(f"Episode ended at step {step}: {info.get('termination_info', {})}")
            break

    env.close()


if __name__ == "__main__":
    main()
