"""Task 5 — AirCombatMVPEnv ↔ PPOAgent compatibility check.

Constructs AirCombatMVPEnv (simple backend, pilot-like config), resets, reads
obs_dim from obs["observation_vector"].shape[0], constructs PPOAgent, and runs a
full select_action → env.step → store_transition → agent.update() cycle to
confirm there are no shape/dtype errors.

Run (Python 3.11 / PowerShell)::

    $env:PYTHONPATH="src"; & "C:/Users/admin/.conda/envs/py3.11/python.exe" \
        scripts/check_mvp_ppo_compat.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent  # noqa: E402
from uav_vpp_guidance.envs.air_combat_mvp_env import AirCombatMVPEnv  # noqa: E402
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv  # noqa: E402

# Reuse the proven config / scenario helpers from the smoke training harness.
from train_mvp_air_combat_smoke import (  # noqa: E402
    DEFAULT_PILOT_CONFIG,
    build_config,
    build_episode_scenario,
)


def main() -> int:
    config = build_config(DEFAULT_PILOT_CONFIG, reward="sparse", vpp=True)
    # PPO hyperparams: ensure a small rollout so update() exercises real data.
    config.setdefault("ppo", {})
    config["ppo"]["rollout_steps"] = 64
    config["ppo"]["minibatch_size"] = 32
    config["ppo"]["device"] = "cpu"
    config.setdefault("policy", {})
    config["policy"]["action_dim"] = 3

    env = AirCombatMVPEnv(config)

    # --- parent_dim: construct a bare parent env to measure parent obs dim ---
    parent_env = CloseRangeTrackingEnv(config)
    parent_obs = parent_env.reset(seed=0)
    parent_dim = int(parent_obs["observation_vector"].shape[0])
    parent_env.close()

    scenario_cfg = config.get("scenario", {})
    rng = np.random.default_rng(0)
    scenario = build_episode_scenario(scenario_cfg, rng)
    obs = env.reset(scenario=scenario, seed=0)

    obs_vec = obs["observation_vector"]
    obs_dim = int(obs_vec.shape[0])
    action_dim = int(config["policy"]["action_dim"])

    print("=" * 70)
    print("Task 5 — AirCombatMVPEnv ↔ PPOAgent compatibility check")
    print("=" * 70)
    print(f"  parent observation dim         = {parent_dim}")
    print(f"  air-combat feature dim         = {AirCombatMVPEnv._AIR_COMBAT_FEATURE_DIM}")
    print(f"  AirCombatMVPEnv obs_dim        = {obs_dim}")
    print(f"  expected (parent + 6)          = {parent_dim + 6}")
    print(f"  obs_dim == parent + 6 ?        = {obs_dim == parent_dim + 6}")
    print(f"  obs_vec dtype                  = {obs_vec.dtype}")
    print(f"  action_dim                     = {action_dim}")

    assert obs_dim == parent_dim + 6, (
        f"obs_dim {obs_dim} != parent_dim+6 {parent_dim + 6}"
    )

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")
    print(f"  PPOAgent network parameters    = {agent.network.count_parameters()}")

    # Run a short rollout: select_action (store=True) → step → store_transition.
    n_steps = 0
    for _ in range(64):
        obs_vec = obs["observation_vector"]
        action, log_prob, value = agent.select_action(obs_vec, deterministic=False, store=True)
        assert action.shape == (action_dim,), f"action shape {action.shape}"
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        agent.store_transition(obs_vec, action, log_prob, reward, done, value, info=info)
        n_steps += 1
        obs = next_obs
        if done:
            scenario = build_episode_scenario(scenario_cfg, rng)
            obs = env.reset(scenario=scenario, seed=int(rng.integers(0, 1_000_000)))

    # One PPO update on the collected rollout.
    next_obs_vec = obs["observation_vector"]
    stats = agent.update(next_obs=next_obs_vec)

    print("-" * 70)
    print(f"  rollout steps collected        = {n_steps}")
    print(f"  update() stats keys            = {sorted(stats.keys())}")
    print(f"  policy_loss                    = {stats.get('policy_loss')}")
    print(f"  value_loss                     = {stats.get('value_loss')}")
    print(f"  entropy                        = {stats.get('entropy')}")
    finite = all(
        np.isfinite(v) for v in stats.values() if isinstance(v, (int, float))
    )
    print(f"  all update stats finite ?      = {finite}")
    assert stats, "update() returned empty stats"
    assert finite, "update() produced non-finite stats"

    env.close()
    print("=" * 70)
    print("RESULT: PASS — full select→step→store→update cycle runs with no "
          "shape/dtype errors.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
