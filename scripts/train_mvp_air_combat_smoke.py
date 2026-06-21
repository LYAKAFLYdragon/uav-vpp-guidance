"""Task 4 / 6 / 7 — Minimal PPO training harness for AirCombatMVPEnv (smoke).

A compact, standalone PPO training loop for the air-combat MVP scenario, modeled
on the rollout/update loop in
``src/uav_vpp_guidance/training/train_no_prediction_vpp_ppo.py`` but self-contained.

Goals (SMOKE-level integration validation):
    - Task 4: verify PPO can run stably for ~10K timesteps on the MVP scenario
      (simple backend), randomizing target geometry per episode, logging per-episode
      return / length / hit, and printing a short training summary.
    - Task 6: --reward {dense,sparse} toggles air_combat.reward.use_sparse to compare
      dense (base RewardCalculator) vs sparse (event) reward end-to-end.
    - Task 7: --vpp / --no-vpp toggles virtual_point.mode (normal vs zero_offset) to
      compare VPP guidance vs No-VPP baseline end-to-end.

Run (Python 3.11 / PowerShell)::

    $env:PYTHONPATH="src"; & "C:/Users/admin/.conda/envs/py3.11/python.exe" \
        scripts/train_mvp_air_combat_smoke.py --timesteps 10000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent  # noqa: E402
from uav_vpp_guidance.envs.air_combat_mvp_env import AirCombatMVPEnv  # noqa: E402
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config  # noqa: E402

MAX_STEPS = 400

DEFAULT_PILOT_CONFIG = os.path.join(
    _REPO_ROOT, "config", "experiment", "air_combat_mvp_pilot.yaml"
)
DEFAULT_PPO_CONFIG = os.path.join(_REPO_ROOT, "config", "ppo.yaml")


# ---------------------------------------------------------------------------
# Config construction (reuses the proven minimal simple-backend base from
# scripts/validate_mvp_air_combat.py and tests/test_mvp_air_combat.py).
# ---------------------------------------------------------------------------
def _base_working_config() -> dict:
    """Minimal, verified simple-backend base config for AirCombatMVPEnv."""
    return {
        "experiment": {
            "name": "train_mvp_air_combat_smoke",
            "seed": 0,
            "output_root": "outputs",
        },
        "env": {
            "use_jsbsim": False,
            "backend": "simple",
            "decision_freq": 5,
            "sim_freq": 60,
            "max_high_level_steps": MAX_STEPS,
            "success_range_m": 900.0,
            "success_ata_deg": 25.0,
            "success_hold_time_s": 0.2,
            "hysteresis_range_m": 950.0,
            "hysteresis_ata_deg": 30.0,
            "min_altitude_m": 500.0,
            "max_altitude_m": 15000.0,
            "max_range_m": 8000.0,
            "target_mode": "constant_velocity",
            "high_level_dt": 0.2,
        },
        "virtual_point": {
            "enabled": True,
            "mode": "normal",  # toggled by --no-vpp -> "zero_offset"
            "anchor_mode": "current_target",
            "action_dim": 3,
            "d_long_range": [-1500.0, 1500.0],
            "d_lat_range": [-800.0, 800.0],
            "d_vert_range": [-500.0, 500.0],
            "smoothing_alpha": 0.3,
        },
        "trajectory_prediction": {"enabled": False},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {
            "w_range": 0.5,
            "w_angle": 0.8,
            "w_energy": 0.2,
            "w_safety": 2.0,
            "w_saturation": 1.0,
            "w_smooth": 0.1,
            "terminal_success": 200.0,
            "terminal_failure": -200.0,
            "terminal_crash": -300.0,
            "min_altitude_m": 500.0,
        },
        "guidance": {
            "mode": "los_rate",
            "use_gain_adapter": False,
            "gains": {
                "k_los": 1.0,
                "k_pos": 0.5,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "alpha_filter": 0.3,
            },
        },
        "air_combat": {
            "missile_config": {
                "ego": {"enabled": True, "nav_constant": 5.0, "kill_radius_m": 120.0},
                "target": {"enabled": False},
            },
            "radar_config": {
                "max_range_m": 10000.0,
                "azimuth_fov_deg": 60.0,
                "elevation_fov_deg": 30.0,
                "lock_time_steps": 3,
            },
            "reward": {"use_sparse": True},  # toggled by --reward
        },
    }


def _load_pilot_overrides(pilot_path: str) -> dict:
    """Extract only env / scenario / air_combat from pilot yaml (no includes)."""
    if not os.path.exists(pilot_path):
        print(f"[warn] pilot config not found: {pilot_path}; using built-in base only.")
        return {}
    raw = load_yaml_config(pilot_path)
    raw.pop("includes", None)
    overrides: dict = {}
    for key in ("env", "scenario", "air_combat"):
        if key in raw and raw[key] is not None:
            overrides[key] = raw[key]
    overrides.setdefault("env", {})
    overrides["env"]["use_jsbsim"] = False
    overrides["env"]["backend"] = "simple"
    overrides["env"]["max_high_level_steps"] = MAX_STEPS
    return overrides


def _load_ppo_hyperparams(ppo_path: str) -> dict:
    """Load PPO hyperparams (ppo + policy) from config/ppo.yaml, force cpu."""
    out: dict = {}
    if os.path.exists(ppo_path):
        raw = load_yaml_config(ppo_path)
        if raw.get("ppo"):
            out["ppo"] = dict(raw["ppo"])
        if raw.get("policy"):
            out["policy"] = dict(raw["policy"])
    out.setdefault("ppo", {})
    out["ppo"]["device"] = "cpu"  # smoke runs on CPU
    return out


def build_config(pilot_path: str, reward: str = "sparse", vpp: bool = True,
                 launch_action: bool = False) -> dict:
    """Build full AirCombatMVPEnv config: base + pilot + ppo, with toggles.

    Args:
        pilot_path: pilot yaml path.
        reward: "sparse" or "dense" (toggles air_combat.reward.use_sparse).
        vpp: True -> virtual_point.mode="normal"; False -> "zero_offset" (No-VPP).
        launch_action: True -> 4-dim action with policy launch decision
            (air_combat.action.use_launch_action); False -> 3-dim rule-launch.

    Returns:
        dict: full config for AirCombatMVPEnv + PPOAgent.
    """
    config = _base_working_config()
    config = merge_config(config, _load_pilot_overrides(pilot_path))
    config = merge_config(config, _load_ppo_hyperparams(DEFAULT_PPO_CONFIG))

    # Ensure policy.action_dim present.
    config.setdefault("policy", {})
    config["policy"]["action_dim"] = config["policy"].get("action_dim", 3)

    # Task 6: reward toggle.
    config.setdefault("air_combat", {}).setdefault("reward", {})
    config["air_combat"]["reward"]["use_sparse"] = (reward == "sparse")

    # Task 7: VPP toggle.
    config.setdefault("virtual_point", {})
    config["virtual_point"]["enabled"] = True
    config["virtual_point"]["mode"] = "normal" if vpp else "zero_offset"

    # Defect 2: launch-decision action toggle.
    config["air_combat"].setdefault("action", {})
    config["air_combat"]["action"]["use_launch_action"] = bool(launch_action)

    return config


# ---------------------------------------------------------------------------
# Per-episode target geometry sampler (polar -> NEU), reused from validate script.
# ---------------------------------------------------------------------------
def _sample_target_neu_position(scenario_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    tgt = scenario_cfg.get("target_init", {})
    range_rng = tgt.get("range_m", [3000.0, 5000.0])
    az_rng = tgt.get("azimuth_deg", [-30.0, 30.0])
    el_rng = tgt.get("elevation_deg", [-10.0, 10.0])

    range_m = float(rng.uniform(range_rng[0], range_rng[1]))
    az = math.radians(float(rng.uniform(az_rng[0], az_rng[1])))
    el = math.radians(float(rng.uniform(el_rng[0], el_rng[1])))

    ego_up = 5000.0
    north = range_m * math.cos(el) * math.cos(az)
    east = range_m * math.cos(el) * math.sin(az)
    up = ego_up + range_m * math.sin(el)
    return np.array([north, east, up], dtype=float)


def build_episode_scenario(scenario_cfg: dict, rng: np.random.Generator) -> dict:
    ego_init_cfg = scenario_cfg.get("ego_init", {})
    ego_vel = float(ego_init_cfg.get("velocity_mps", 250.0))
    ego_heading = float(ego_init_cfg.get("heading_deg", 0.0))

    tgt_cfg = scenario_cfg.get("target_init", {})
    tgt_vel = float(tgt_cfg.get("velocity_mps", 250.0))
    tgt_heading = float(tgt_cfg.get("heading_deg", 180.0))

    target_pos = _sample_target_neu_position(scenario_cfg, rng)
    return {
        "name": "air_combat_mvp_smoke",
        "own_init": {
            "position_m": np.array([0.0, 0.0, 5000.0], dtype=float),
            "velocity_mps": ego_vel,
            "heading_deg": ego_heading,
        },
        "target_init": {
            "position_m": target_pos,
            "velocity_mps": tgt_vel,
            "heading_deg": tgt_heading,
        },
    }


def _episode_is_hit(info: dict) -> bool:
    """Whether the (final) info indicates a successful missile hit."""
    term = info.get("termination_info", {}) or {}
    if term.get("is_success"):
        return True
    return "missile_hit" in (info.get("events") or [])


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train(config: dict, total_timesteps: int, seed: int, label: str,
          output_dir: str) -> dict:
    """Run a compact PPO training loop on AirCombatMVPEnv.

    Returns:
        dict: summary metrics (timesteps, episodes, mean returns, hit rate, etc.).
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    scenario_cfg = config.get("scenario", {})
    rollout_steps = int(config.get("ppo", {}).get("rollout_steps", 1024))

    env = AirCombatMVPEnv(config)
    rng = np.random.default_rng(seed)

    obs = env.reset(scenario=build_episode_scenario(scenario_cfg, rng), seed=seed)
    obs_dim = int(obs["observation_vector"].shape[0])
    # action_dim from the env (4 when launch-decision action is enabled, else 3),
    # keeping policy.action_dim in sync so the PPO network output matches.
    action_dim = int(env.action_dim)
    config.setdefault("policy", {})["action_dim"] = action_dim

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device="cpu")

    print("=" * 72)
    print(f"[{label}] PPO smoke training on AirCombatMVPEnv (simple backend)")
    print(f"  reward={'sparse' if config['air_combat']['reward']['use_sparse'] else 'dense'}"
          f"  vpp_mode={config['virtual_point']['mode']}"
          f"  launch_action={config.get('air_combat', {}).get('action', {}).get('use_launch_action', False)}")
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  rollout_steps={rollout_steps}"
          f"  total_timesteps={total_timesteps}  seed={seed}")
    print("=" * 72)

    global_step = 0
    episode_count = 0
    update_num = 0
    episode_return = 0.0
    episode_length = 0

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_hits: list[int] = []

    episode_log: list[dict] = []
    last_info: dict = {}

    start = time.time()
    anomalies: list[str] = []

    while global_step < total_timesteps:
        for _ in range(rollout_steps):
            obs_vec = obs["observation_vector"]
            try:
                action, log_prob, value = agent.select_action(
                    obs_vec, deterministic=False, store=False
                )
            except ValueError as e:  # NaN/inf guard tripped
                anomalies.append(f"select_action non-finite @step {global_step}: {e}")
                raise

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if not np.isfinite(reward):
                anomalies.append(f"non-finite reward @step {global_step}")
                reward = 0.0
            agent.store_transition(obs_vec, action, log_prob, reward, done, value, info=info)

            global_step += 1
            episode_return += reward
            episode_length += 1
            last_info = info
            obs = next_obs

            if done:
                episode_count += 1
                hit = int(_episode_is_hit(info))
                ep_returns.append(episode_return)
                ep_lengths.append(episode_length)
                ep_hits.append(hit)
                reason = (info.get("termination_info", {}) or {}).get("reason")
                episode_log.append({
                    "episode": episode_count,
                    "global_step": global_step,
                    "return": episode_return,
                    "length": episode_length,
                    "hit": hit,
                    "reason": reason,
                })
                episode_return = 0.0
                episode_length = 0
                obs = env.reset(
                    scenario=build_episode_scenario(scenario_cfg, rng),
                    seed=int(rng.integers(0, 1_000_000)),
                )
                if agent.buffer.full:
                    break
            if global_step >= total_timesteps:
                break

        if agent.buffer.full or (global_step >= total_timesteps and len(agent.buffer) > 0):
            next_obs_vec = obs["observation_vector"]
            stats = agent.update(next_obs=next_obs_vec)
            update_num += 1
            for k, v in stats.items():
                if isinstance(v, (int, float)) and not np.isfinite(v):
                    anomalies.append(f"non-finite update stat {k}={v} @step {global_step}")
            recent_ret = float(np.mean(ep_returns[-10:])) if ep_returns else float("nan")
            recent_hit = float(np.mean(ep_hits[-10:])) if ep_hits else float("nan")
            print(
                f"[{label}] step {global_step}/{total_timesteps} | upd {update_num} | "
                f"eps {episode_count} | mean_ret(10)={recent_ret:.2f} | "
                f"hit(10)={recent_hit:.2f} | "
                f"ploss={stats.get('policy_loss', float('nan')):.3f} "
                f"vloss={stats.get('value_loss', float('nan')):.3f} "
                f"ent={stats.get('entropy', float('nan')):.3f}"
            )

    elapsed = time.time() - start
    env.close()

    # Summary metrics.
    n_last = min(20, len(ep_returns))
    mean_ret_last = float(np.mean(ep_returns[-n_last:])) if ep_returns else float("nan")
    mean_ret_first = float(np.mean(ep_returns[:n_last])) if ep_returns else float("nan")
    hit_rate = float(np.mean(ep_hits)) if ep_hits else float("nan")
    trends_up = (
        len(ep_returns) >= 2 * n_last and mean_ret_last > mean_ret_first
    )

    summary = {
        "label": label,
        "reward": "sparse" if config["air_combat"]["reward"]["use_sparse"] else "dense",
        "vpp_mode": config["virtual_point"]["mode"],
        "launch_action": bool(
            config.get("air_combat", {}).get("action", {}).get("use_launch_action", False)
        ),
        "completed_timesteps": global_step,
        "target_timesteps": total_timesteps,
        "episodes": episode_count,
        "updates": update_num,
        "mean_return_last": mean_ret_last,
        "mean_return_first": mean_ret_first,
        "n_window": n_last,
        "hit_rate": hit_rate,
        "n_hits": int(sum(ep_hits)),
        "mean_episode_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
        "return_trends_up": bool(trends_up),
        "elapsed_seconds": elapsed,
        "anomalies": anomalies,
    }

    # Persist logs under outputs/.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"summary_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, f"episodes_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(episode_log, f, indent=2, ensure_ascii=False)

    print("-" * 72)
    print(f"[{label}] DONE: {global_step} steps, {episode_count} episodes, "
          f"{update_num} updates, {elapsed:.1f}s")
    print(f"[{label}] mean_return(first {n_last})={mean_ret_first:.2f}  "
          f"mean_return(last {n_last})={mean_ret_last:.2f}  trends_up={trends_up}")
    print(f"[{label}] hit_rate={hit_rate:.3f} ({int(sum(ep_hits))}/{episode_count})  "
          f"mean_ep_len={summary['mean_episode_length']:.1f}")
    if anomalies:
        print(f"[{label}] ANOMALIES: {anomalies}")
    else:
        print(f"[{label}] no anomalies (no NaN/inf guards tripped).")
    print("-" * 72)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal PPO smoke training for AirCombatMVPEnv (Tasks 4/6/7)."
    )
    parser.add_argument("--config", default=DEFAULT_PILOT_CONFIG, help="pilot yaml path")
    parser.add_argument("--timesteps", type=int, default=10000, help="total timesteps")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--reward", choices=["dense", "sparse"], default="sparse",
                        help="reward mode (Task 6 toggle)")
    parser.add_argument("--vpp", dest="vpp", action="store_true", default=True,
                        help="use VPP guidance (default)")
    parser.add_argument("--no-vpp", dest="vpp", action="store_false",
                        help="use No-VPP baseline (virtual_point.mode=zero_offset)")
    parser.add_argument("--launch-action", dest="launch_action", action="store_true",
                        default=False,
                        help="enable 4-dim action with policy launch decision "
                             "(air_combat.action.use_launch_action; defect 2)")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="override PPO rollout_steps")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "mvp_air_combat_smoke"),
                        help="output dir for logs (under outputs/)")
    parser.add_argument("--label", default=None, help="run label for log filenames")
    args = parser.parse_args()

    config = build_config(
        args.config, reward=args.reward, vpp=args.vpp,
        launch_action=args.launch_action,
    )
    if args.rollout_steps is not None:
        config["ppo"]["rollout_steps"] = args.rollout_steps

    la_tag = "_launch" if args.launch_action else ""
    label = args.label or (
        f"{args.reward}_{'vpp' if args.vpp else 'novpp'}{la_tag}_seed{args.seed}"
    )
    summary = train(config, args.timesteps, args.seed, label, args.output_dir)

    print("\n=== Task summary (JSON) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
