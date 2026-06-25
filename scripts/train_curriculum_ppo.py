#!/usr/bin/env python3
"""
Method 2A: Curriculum Learning for Crossing Scenario Breakthrough.

Trains PPO in stages:
  Stage 1 (0-25%): favorable + neutral only
  Stage 2 (25-50%): + disadvantage
  Stage 3 (50-75%): + challenging (crossing)
  Stage 4 (75-100%): all scenarios, with emphasis on crossing

Scenario progression is progress-based up to a maximum stage, but that maximum
stage is unlocked by performance: all currently allowed scenarios must achieve
SR >= 50% before the next stage becomes available.  This prevents the agent
from being exposed to harder scenarios before it masters the current set.
"""
import argparse
import csv
import json
import os
import time
import copy
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from uav_vpp_guidance.utils.config import load_yaml_config, merge_config
from uav_vpp_guidance.utils.seed import set_seed
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.envs.opponent_policy import OpponentPolicy
from uav_vpp_guidance.envs.expert_opponent import ExpertOpponent
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.common.provenance import record_config_override_if_changed
from uav_vpp_guidance.metrics.combat_evaluator import compute_combat_metrics


def load_experiment_config(config_path):
    base_config = load_yaml_config(config_path)
    includes = base_config.pop("includes", [])
    merged = {}
    for inc_path in includes:
        inc_full = os.path.join(os.path.dirname(config_path), inc_path)
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_yaml_config(inc_full))
    return merge_config(merged, base_config)


def sample_scenario(config, rng):
    scenarios = config.get("scenarios", {})
    if not scenarios:
        return None
    name = rng.choice(list(scenarios.keys()))
    return scenarios[name]


def _env_limit(env, config, key, default):
    env_cfg = getattr(env, "env_config", None)
    if isinstance(env_cfg, dict) and key in env_cfg:
        return float(env_cfg.get(key, default))
    cfg_env = config.get("env", {}) if isinstance(config, dict) else {}
    if isinstance(cfg_env, dict) and key in cfg_env:
        return float(cfg_env.get(key, default))
    return float(config.get(key, default)) if isinstance(config, dict) else float(default)


def _safe_float(value, default=np.nan):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out


def _raw_failure_diagnostics(info, reason, env, config, final_range_m=np.nan):
    term_info = info.get("termination_info") if isinstance(info, dict) else {}
    term_info = term_info if isinstance(term_info, dict) else {}
    raw_reason = term_info.get("raw_termination_reason") or term_info.get("reason") or reason
    raw_is_crash = bool(term_info.get("raw_is_crash", term_info.get("is_crash", raw_reason == "crash")))
    raw_is_oob = bool(
        term_info.get("raw_is_out_of_bounds", term_info.get("is_out_of_bounds", raw_reason == "out_of_bounds"))
    )
    raw_is_timeout = bool(term_info.get("raw_is_timeout", term_info.get("is_timeout", raw_reason == "timeout")))
    own_state = info.get("own_state", {}) if isinstance(info, dict) else {}
    altitude_m = _safe_float(own_state.get("altitude_m"), np.nan)
    min_alt = _env_limit(env, config, "min_altitude_m", 500.0)
    max_alt = _env_limit(env, config, "max_altitude_m", 15000.0)
    max_range = _env_limit(env, config, "max_range_m", 8000.0)
    high_altitude_failure = bool(raw_is_crash and np.isfinite(altitude_m) and altitude_m > max_alt)
    low_altitude_failure = bool(raw_is_crash and np.isfinite(altitude_m) and altitude_m < min_alt)
    range_oob_failure = bool(raw_is_oob or (np.isfinite(final_range_m) and final_range_m > max_range))
    return {
        "raw_termination_reason": raw_reason,
        "raw_is_crash": raw_is_crash,
        "raw_is_out_of_bounds": raw_is_oob,
        "raw_is_timeout": raw_is_timeout,
        "final_altitude_m": altitude_m,
        "high_altitude_failure": high_altitude_failure,
        "low_altitude_failure": low_altitude_failure,
        "range_oob_failure": range_oob_failure,
    }


def run_evaluation(env, agent, config, num_episodes=10, seeds=None, save_trajectories=False, output_dir=None):
    if seeds is None:
        seeds = [0, 1, 2]
    all_episodes = []
    for seed in seeds:
        for ep in range(num_episodes):
            ep_seed = seed * 10000 + ep
            rng = np.random.default_rng(ep_seed)
            scenario = sample_scenario(config, rng)
            obs = env.reset(scenario=scenario, seed=ep_seed)
            ep_reward = 0.0
            ep_length = 0
            min_range = float("inf")
            final_range = 0.0
            final_ata = 0.0
            reason = "timeout"
            info = {}
            for step in range(env.max_steps):
                obs_vec = obs["observation_vector"]
                action = agent.get_deterministic_action(obs_vec)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_length += 1
                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                ata_deg = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                min_range = min(min_range, range_m)
                final_range = range_m
                final_ata = ata_deg
                if terminated or truncated:
                    reason = info.get("reason", "unknown")
                    break
            combat_enabled = bool(getattr(getattr(env, "combat_hp", None), "enabled", False))
            combat_success = bool(info.get("combat_success", False))
            combat_reason = str(info.get("combat_reason", ""))
            is_success = combat_success if combat_enabled else (reason == "success")
            if combat_enabled:
                is_crash = combat_reason in {
                    "ego_crash_or_out_of_bounds",
                    "target_crash_or_out_of_bounds",
                    "crash",
                }
                is_timeout = combat_reason in {
                    "timeout_hp_advantage",
                    "timeout_hp_disadvantage",
                    "timeout_draw",
                    "timeout",
                }
                is_out_of_bounds = combat_reason in {
                    "ego_crash_or_out_of_bounds",
                    "out_of_bounds",
                }
            else:
                is_crash = reason == "crash"
                is_timeout = reason == "timeout"
                is_out_of_bounds = reason == "out_of_bounds"
            diagnostics = _raw_failure_diagnostics(
                info,
                reason,
                env,
                config,
                final_range_m=final_range,
            )
            all_episodes.append({
                "seed": seed, "episode": ep, "return": ep_reward,
                "length": ep_length, "min_range_m": min_range,
                "final_range_m": final_range, "final_ata_deg": final_ata,
                "reason": reason,
                "is_success": is_success,
                "is_crash": is_crash,
                "is_timeout": is_timeout,
                "is_out_of_bounds": is_out_of_bounds,
                "termination_reason": reason,
                "combat_outcome": info.get("combat_outcome"),
                "combat_success": combat_success,
                "combat_reason": info.get("combat_reason"),
                "ego_hp": info.get("ego_hp"),
                "target_hp": info.get("target_hp"),
                "combat_time_to_kill": info.get("combat_time_to_kill"),
                **diagnostics,
            })

    success_count = sum(1 for e in all_episodes if e["is_success"])
    crash_count = sum(1 for e in all_episodes if e["is_crash"])
    oob_count = sum(1 for e in all_episodes if e["is_out_of_bounds"])
    timeout_count = sum(1 for e in all_episodes if e["is_timeout"])
    raw_crash_count = sum(1 for e in all_episodes if e["raw_is_crash"])
    raw_oob_count = sum(1 for e in all_episodes if e["raw_is_out_of_bounds"])
    high_alt_count = sum(1 for e in all_episodes if e["high_altitude_failure"])
    low_alt_count = sum(1 for e in all_episodes if e["low_altitude_failure"])
    range_oob_count = sum(1 for e in all_episodes if e["range_oob_failure"])
    returns = [e["return"] for e in all_episodes]
    combat_metrics = compute_combat_metrics(all_episodes)

    return {
        "num_episodes": len(all_episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "success_rate": success_count / max(1, len(all_episodes)),
        "win_rate": combat_metrics["win_rate"],
        "survival_rate": combat_metrics["survival_rate"],
        "hp_advantage": combat_metrics["hp_advantage"],
        "mean_time_to_kill": combat_metrics["mean_time_to_kill"],
        "crash_rate": crash_count / max(1, len(all_episodes)),
        "out_of_bounds_rate": oob_count / max(1, len(all_episodes)),
        "raw_crash_rate": raw_crash_count / max(1, len(all_episodes)),
        "raw_out_of_bounds_rate": raw_oob_count / max(1, len(all_episodes)),
        "high_altitude_failure_rate": high_alt_count / max(1, len(all_episodes)),
        "low_altitude_failure_rate": low_alt_count / max(1, len(all_episodes)),
        "range_oob_failure_rate": range_oob_count / max(1, len(all_episodes)),
        "timeout_rate": timeout_count / max(1, len(all_episodes)),
    }


def evaluate_scenarios(env, agent, scenarios, num_episodes=5, seed_base=1000):
    """Evaluate each scenario individually and return per-scenario SR."""
    results = {}
    combat_enabled = bool(getattr(getattr(env, "combat_hp", None), "enabled", False))
    for name, scenario in scenarios.items():
        successes = 0
        total = num_episodes
        for ep in range(num_episodes):
            ep_seed = seed_base + hash(name) % 10000 + ep
            obs = env.reset(scenario=scenario, seed=ep_seed)
            for step in range(env.max_steps):
                action = agent.get_deterministic_action(obs["observation_vector"])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    if combat_enabled:
                        is_success = bool(info.get("combat_success", False))
                    else:
                        is_success = info.get("reason") == "success"
                    if is_success:
                        successes += 1
                    break
        results[name] = successes / max(1, total)
    return results


# Curriculum stages: (progress_end, allowed_scenario_names)
DEFAULT_CURRICULUM = [
    (0.25, ["favorable", "neutral"]),
    (0.50, ["favorable", "neutral", "disadvantage"]),
    (0.75, ["favorable", "neutral", "disadvantage", "challenging"]),
    (1.00, ["favorable", "neutral", "disadvantage", "challenging"]),
]


def load_warm_start_checkpoint(
    agent,
    checkpoint_path,
    strict_dims=True,
    load_optimizer=False,
):
    """Load policy weights from an existing PPO checkpoint for warm-starting."""
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location=agent.device)
    ckpt_obs_dim = int(checkpoint.get("obs_dim", agent.obs_dim))
    ckpt_action_dim = int(checkpoint.get("action_dim", agent.action_dim))
    if strict_dims and (ckpt_obs_dim != agent.obs_dim or ckpt_action_dim != agent.action_dim):
        raise ValueError(
            "Warm-start checkpoint dim mismatch: "
            f"checkpoint obs/action=({ckpt_obs_dim}, {ckpt_action_dim}) "
            f"agent obs/action=({agent.obs_dim}, {agent.action_dim})"
        )
    missing, unexpected = agent.network.load_state_dict(
        checkpoint["network_state_dict"],
        strict=False,
    )
    optimizer_loaded = False
    if load_optimizer and "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True
    return {
        "loaded": True,
        "checkpoint_path": str(path),
        "checkpoint_obs_dim": ckpt_obs_dim,
        "checkpoint_action_dim": ckpt_action_dim,
        "agent_obs_dim": int(agent.obs_dim),
        "agent_action_dim": int(agent.action_dim),
        "optimizer_loaded": optimizer_loaded,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


class SafetyPreCurriculumGate:
    """Block adversarial opponents until basic survival behavior is demonstrated."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.disable_adversary = bool(cfg.get("disable_adversary", True))
        self.min_steps = int(cfg.get("min_steps", 0))
        self.survival_threshold = float(cfg.get("survival_threshold", 0.5))
        self.max_crash_rate = float(cfg.get("max_crash_rate", 0.5))
        self.max_out_of_bounds_rate = float(cfg.get("max_out_of_bounds_rate", 0.5))
        self.passed = not self.enabled
        self.passed_step = None
        self.last_metrics = {}

    def allows_adversary(self, global_step=0):
        if not self.enabled or not self.disable_adversary:
            return True
        return bool(self.passed)

    def update_from_eval(self, metrics, global_step):
        self.last_metrics = dict(metrics or {})
        if not self.enabled or self.passed:
            return self.passed
        if int(global_step) < self.min_steps:
            return False
        survival = _finite_metric(metrics.get("survival_rate"))
        crash = _finite_metric(metrics.get("raw_crash_rate", metrics.get("crash_rate")))
        oob = _finite_metric(metrics.get("raw_out_of_bounds_rate", metrics.get("out_of_bounds_rate")))
        if (
            np.isfinite(survival)
            and np.isfinite(crash)
            and np.isfinite(oob)
            and survival >= self.survival_threshold
            and crash <= self.max_crash_rate
            and oob <= self.max_out_of_bounds_rate
        ):
            self.passed = True
            self.passed_step = int(global_step)
        return self.passed

    def status(self, global_step=0):
        if not self.enabled:
            return "disabled"
        if self.passed:
            return "passed"
        if int(global_step) < self.min_steps:
            return "min_steps"
        return "waiting_survival"

    def as_dict(self, global_step=0):
        return {
            "enabled": self.enabled,
            "disable_adversary": self.disable_adversary,
            "min_steps": self.min_steps,
            "survival_threshold": self.survival_threshold,
            "max_crash_rate": self.max_crash_rate,
            "max_out_of_bounds_rate": self.max_out_of_bounds_rate,
            "passed": self.passed,
            "passed_step": self.passed_step,
            "status": self.status(global_step),
            "last_metrics": dict(self.last_metrics),
        }


def _finite_metric(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def compute_rollout_safety_metrics(episodes):
    rows = list(episodes or [])
    count = len(rows)
    raw_crash = sum(int(_finite_metric(row.get("raw_crash", 0)) > 0.5) for row in rows)
    raw_oob = sum(int(_finite_metric(row.get("raw_out_of_bounds", 0)) > 0.5) for row in rows)
    high_alt = sum(int(_finite_metric(row.get("high_altitude_failure", 0)) > 0.5) for row in rows)
    low_alt = sum(int(_finite_metric(row.get("low_altitude_failure", 0)) > 0.5) for row in rows)
    range_oob = sum(int(_finite_metric(row.get("range_oob_failure", 0)) > 0.5) for row in rows)
    denom = max(1, count)
    return {
        "train_episode_count": count,
        "train_raw_crash_rate": raw_crash / denom,
        "train_raw_out_of_bounds_rate": raw_oob / denom,
        "train_high_altitude_failure_rate": high_alt / denom,
        "train_low_altitude_failure_rate": low_alt / denom,
        "train_range_oob_failure_rate": range_oob / denom,
    }


def rollout_safety_gate_passed(metrics, cfg=None):
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return True, "disabled"

    min_episodes = int(cfg.get("min_episodes", 0))
    episode_count = int(metrics.get("train_episode_count", 0))
    if episode_count < min_episodes:
        return False, "insufficient_episodes"

    checks = [
        ("raw_crash_rate", "max_raw_crash_rate"),
        ("raw_out_of_bounds_rate", "max_raw_out_of_bounds_rate"),
        ("high_altitude_failure_rate", "max_high_altitude_failure_rate"),
        ("low_altitude_failure_rate", "max_low_altitude_failure_rate"),
        ("range_oob_failure_rate", "max_range_oob_failure_rate"),
    ]
    for metric_name, threshold_name in checks:
        threshold = cfg.get(threshold_name)
        if threshold is None:
            continue
        value = _finite_metric(
            metrics.get(f"train_{metric_name}", metrics.get(metric_name))
        )
        if np.isfinite(value) and value > float(threshold):
            return False, metric_name
    return True, "passed"


@dataclass
class OpponentPoolEntry:
    path: str
    step: int
    elo: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0


class SelfPlayCheckpointOpponent(OpponentPolicy):
    """PPO checkpoint opponent that consumes role-reversed observations."""

    def __init__(self, checkpoint_path, device="cpu", action_mode=None):
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.checkpoint_path = checkpoint_path
        self.obs_dim = int(checkpoint.get("obs_dim", 16))
        self.action_dim = int(checkpoint.get("action_dim", 3))
        cfg = checkpoint.get("config", {})
        self.action_mode = action_mode or self._infer_action_mode(cfg)
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=cfg,
            device=device,
        )
        self.agent.network.load_state_dict(checkpoint["network_state_dict"], strict=False)
        self.agent.network.eval()

    def act(self, opponent_obs):
        obs_vec = opponent_obs["observation_vector"]
        return self.agent.get_deterministic_action(obs_vec)

    def get_diagnostics(self):
        return {
            "opponent_type": "self_play_checkpoint",
            "opponent_checkpoint": self.checkpoint_path,
            "opponent_action_mode": self.action_mode,
        }

    @staticmethod
    def _infer_action_mode(config):
        if config.get("end_to_end", {}).get("enabled", False):
            return "direct_command"
        if not config.get("virtual_point", {}).get("enabled", True):
            return "direct_command"
        return "vpp"


class LightweightEloOpponentPool:
    """Minimal opponent pool for curriculum scheduling."""

    def __init__(self, cfg, pool_dir):
        self.cfg = cfg or {}
        self.pool_dir = pool_dir
        os.makedirs(self.pool_dir, exist_ok=True)
        self.entries = []
        self.threshold = float(self.cfg.get("switch_threshold", 0.7))
        self.weak_max = float(self.cfg.get("weak_elo_max", 1050.0))
        self.medium_max = float(self.cfg.get("medium_elo_max", 1150.0))

    def add_checkpoint(self, checkpoint_path, step, elo=None):
        entry = OpponentPoolEntry(
            path=str(checkpoint_path),
            step=int(step),
            elo=float(elo if elo is not None else self.cfg.get("initial_elo", 1000.0)),
        )
        self.entries.append(entry)
        self.save_manifest()
        return entry

    def bucket_for(self, entry):
        if entry.elo < self.weak_max:
            return "weak"
        if entry.elo < self.medium_max:
            return "medium"
        return "strong"

    def entries_for_bucket(self, bucket):
        selected = [e for e in self.entries if self.bucket_for(e) == bucket]
        if selected:
            return selected
        if bucket == "medium":
            return [e for e in self.entries if self.bucket_for(e) in {"weak", "medium"}]
        if bucket == "strong":
            return list(self.entries)
        return selected

    def sample(self, bucket, rng):
        candidates = self.entries_for_bucket(bucket)
        if not candidates:
            return None
        idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]

    @staticmethod
    def next_bucket(bucket):
        if bucket == "weak":
            return "medium"
        if bucket == "medium":
            return "strong"
        return "strong"

    def update_from_eval(self, entry, ego_win_rate, num_episodes=1):
        if entry is None or not np.isfinite(ego_win_rate):
            return
        expected = 1.0 / (1.0 + 10.0 ** ((entry.elo - 1000.0) / 400.0))
        k_factor = float(self.cfg.get("elo_k_factor", 32.0))
        entry.elo += k_factor * (expected - float(ego_win_rate))
        episodes = max(1, int(num_episodes))
        ego_wins = int(round(float(ego_win_rate) * episodes))
        ego_wins = max(0, min(episodes, ego_wins))
        opponent_wins = episodes - ego_wins
        if ego_win_rate > 0.5:
            entry.losses += ego_wins
            entry.wins += opponent_wins
        elif ego_win_rate < 0.5:
            entry.wins += opponent_wins
            entry.losses += ego_wins
        else:
            entry.draws += episodes
        self.save_manifest()

    def save_manifest(self):
        path = os.path.join(self.pool_dir, "opponent_pool.json")
        entries = []
        for entry in self.entries:
            item = asdict(entry)
            item["bucket"] = self.bucket_for(entry)
            entries.append(item)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"entries": entries},
                f,
                indent=2,
                ensure_ascii=False,
            )


def _set_training_opponent(
    env,
    pool,
    bucket,
    rng,
    device,
    cfg,
    adversary_allowed=True,
    blocked_reason=None,
):
    if not cfg.get("enabled", False):
        return None
    if not adversary_allowed:
        env.set_opponent_policy(
            None,
            {
                "stage": "safety_precurriculum",
                "bucket": bucket,
                "type": "none",
                "blocked_reason": blocked_reason or "safety_gate",
            },
        )
        return None
    final_opponent = cfg.get("final_opponent")
    if bucket == "strong" and final_opponent == "expert":
        policy = ExpertOpponent(cfg.get("expert", {}))
        env.set_opponent_policy(policy, {"stage": "curriculum", "bucket": bucket, "type": "expert"})
        return None
    entry = pool.sample(bucket, rng)
    if entry is None:
        env.set_opponent_policy(None, {"stage": "curriculum", "bucket": bucket, "type": "none"})
        return None
    policy = SelfPlayCheckpointOpponent(
        entry.path,
        device=device,
        action_mode=cfg.get("self_play_action_mode"),
    )
    env.set_opponent_policy(
        policy,
        {
            "stage": "curriculum",
            "bucket": bucket,
            "type": "self_play_checkpoint",
            "checkpoint": entry.path,
            "elo": entry.elo,
        },
    )
    return entry


def train_ppo_curriculum(config, output_dir, smoke=False):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    adversarial_cfg = config.get("adversarial_curriculum", config.get("opponent_curriculum", {}))
    if adversarial_cfg.get("enabled", False):
        attack_cfg = config.setdefault("attack_zone", {})
        old_attack_enabled = attack_cfg.get("enabled")
        attack_cfg["enabled"] = True
        record_config_override_if_changed(
            config,
            "attack_zone.enabled",
            True,
            old_value=old_attack_enabled,
            source="train_curriculum_ppo.py:adversarial_curriculum",
        )

    import yaml
    with open(os.path.join(output_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    ppo_cfg = config.get("ppo", {})
    total_timesteps = int(ppo_cfg.get("total_timesteps", 200000))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 2048))
    eval_interval = int(config.get("evaluation", {}).get("eval_interval", 10000))
    save_interval = int(config.get("checkpoint", {}).get("save_interval", 10000))
    save_best = bool(config.get("checkpoint", {}).get("save_best", True))
    save_last = bool(config.get("checkpoint", {}).get("save_last", True))

    if smoke:
        total_timesteps = 512
        rollout_steps = 128
        eval_interval = 256
        save_interval = 256
        print("[SMOKE] Running smoke mode")

    env = CloseRangeTrackingEnv(config)
    backend = env._backend
    print(f"Backend: {backend}")

    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = ppo_cfg.get("device", "cpu")
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    print(f"Network parameters: {agent.network.count_parameters()}")

    training_metadata = {
        "warm_start": {
            "enabled": bool(config.get("warm_start", {}).get("enabled", False)),
            "loaded": False,
        },
        "safety_precurriculum": {},
        "rollout_gate": {},
    }
    warm_start_cfg = config.get("warm_start", {})
    if warm_start_cfg.get("enabled", False):
        checkpoint_path = warm_start_cfg.get("checkpoint") or warm_start_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("warm_start.enabled=true requires warm_start.checkpoint")
        warm_start_meta = load_warm_start_checkpoint(
            agent,
            checkpoint_path=checkpoint_path,
            strict_dims=bool(warm_start_cfg.get("strict_dims", True)),
            load_optimizer=bool(warm_start_cfg.get("load_optimizer", False)),
        )
        training_metadata["warm_start"] = {
            "enabled": True,
            **warm_start_meta,
        }
        print(f"Warm-start loaded from {warm_start_meta['checkpoint_path']}")

    # Curriculum config
    curriculum = config.get("curriculum", {}).get("stages", DEFAULT_CURRICULUM)
    stage_gate_sr = config.get("curriculum", {}).get("stage_gate_sr", 0.50)
    all_scenarios = config.get("scenarios", {})
    opponent_pool = LightweightEloOpponentPool(
        adversarial_cfg,
        os.path.join(output_dir, "opponent_pool"),
    )
    opponent_bucket = str(adversarial_cfg.get("initial_bucket", "weak"))
    active_opponent_entry = None
    safety_gate = SafetyPreCurriculumGate(config.get("safety_precurriculum", {}))
    rollout_gate_cfg = config.get("curriculum", {}).get("rollout_gate", {})
    rollout_window = max(1, int(rollout_gate_cfg.get("window_episodes", 16)))
    recent_rollout_episodes = deque(maxlen=rollout_window)
    env.set_command_post_processor_safety_mode(not safety_gate.passed)

    global_step = 0
    episode_count = 0
    best_eval_return = -float("inf")
    rng = np.random.default_rng(config.get("experiment", {}).get("seed", 0))

    episode_log_path = os.path.join(log_dir, "episode_train_log.csv")
    update_log_path = os.path.join(log_dir, "update_train_log.csv")
    eval_log_path = os.path.join(log_dir, "eval_log.csv")
    curriculum_log_path = os.path.join(log_dir, "curriculum_log.csv")

    with open(episode_log_path, "w", newline="", encoding="utf-8") as f_ep, \
         open(update_log_path, "w", newline="", encoding="utf-8") as f_up, \
         open(eval_log_path, "w", newline="", encoding="utf-8") as f_eval, \
         open(curriculum_log_path, "w", newline="", encoding="utf-8") as f_cur:

        ep_writer = csv.DictWriter(f_ep, fieldnames=[
            "step", "episode", "episode_return", "episode_length",
            "success", "crash", "out_of_bounds", "timeout",
            "mean_range", "final_range", "max_range", "final_ata",
            "termination_reason", "raw_termination_reason", "combat_reason",
            "combat_outcome", "raw_crash", "raw_out_of_bounds",
            "raw_timeout", "high_altitude_failure", "low_altitude_failure",
            "range_oob_failure", "final_altitude_m", "min_altitude_m",
            "max_altitude_m", "final_pitch_deg", "mean_nz_cmd",
            "max_abs_nz_cmd",
        ])
        ep_writer.writeheader()

        up_writer = csv.DictWriter(f_up, fieldnames=[
            "step", "update_num", "policy_loss", "value_loss", "entropy",
            "approx_kl", "clip_fraction", "explained_variance", "learning_rate",
        ])
        up_writer.writeheader()

        eval_writer = csv.DictWriter(f_eval, fieldnames=[
            "step", "num_episodes", "mean_return", "std_return",
            "success_rate", "win_rate", "survival_rate", "hp_advantage",
            "mean_time_to_kill", "crash_rate", "out_of_bounds_rate",
            "raw_crash_rate", "raw_out_of_bounds_rate",
            "high_altitude_failure_rate", "low_altitude_failure_rate",
            "range_oob_failure_rate", "timeout_rate",
        ])
        eval_writer.writeheader()

        cur_writer = csv.DictWriter(f_cur, fieldnames=[
            "step", "stage", "allowed_scenarios", "scenario_sr",
            "opponent_bucket", "opponent_pool_size", "opponent_eval_win_rate",
            "safety_gate_status", "safety_gate_passed", "adversary_allowed",
            "safety_gate_passed_step",
            "train_rollout_episode_count", "train_raw_crash_rate",
            "train_raw_out_of_bounds_rate", "train_high_altitude_failure_rate",
            "train_low_altitude_failure_rate", "train_range_oob_failure_rate",
            "rollout_gate_passed", "rollout_gate_reason",
        ])
        cur_writer.writeheader()

        adversary_allowed = safety_gate.allows_adversary(global_step)
        active_opponent_entry = _set_training_opponent(
            env,
            opponent_pool,
            opponent_bucket,
            rng,
            device,
            adversarial_cfg,
            adversary_allowed=adversary_allowed,
            blocked_reason=safety_gate.status(global_step),
        )
        obs = env.reset(seed=rng.integers(0, 1000000))
        episode_return = 0.0
        episode_length = 0
        episode_ranges = []
        episode_altitudes = []
        episode_nz_cmds = []
        episode_pitch_degs = []
        start_time = time.time()
        update_num = 0
        current_stage = 0
        unlocked_stage = 0
        combat_enabled = bool(getattr(getattr(env, "combat_hp", None), "enabled", False))

        while global_step < total_timesteps:
            # Determine current curriculum stage.
            # Progress suggests a stage, but scenarios beyond unlocked_stage are
            # withheld until the performance gate is cleared.
            progress = global_step / max(1, total_timesteps)
            progress_stage = 0
            for i, (thresh, _) in enumerate(curriculum):
                if progress <= thresh:
                    progress_stage = i
                    break
            else:
                progress_stage = len(curriculum) - 1

            current_stage = min(progress_stage, unlocked_stage)
            allowed_names = curriculum[current_stage][1]
            active_scenarios = {k: v for k, v in all_scenarios.items() if k in allowed_names}

            for step in range(rollout_steps):
                obs_vec = obs["observation_vector"]
                action, log_prob, value = agent.select_action(obs_vec, deterministic=False, store=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                global_step += 1
                episode_return += reward
                episode_length += 1
                agent.store_transition(obs_vec, action, log_prob, reward, done, value)

                rel_state = obs.get("relative_state", {})
                range_m = rel_state.get("range_m", 0.0)
                episode_ranges.append(range_m)
                own_state = info.get("own_state", {})
                altitude_m = _safe_float(own_state.get("altitude_m"), np.nan)
                if np.isfinite(altitude_m):
                    episode_altitudes.append(altitude_m)
                guidance_command = info.get("guidance_command", {})
                nz_cmd = _safe_float(info.get("nz_cmd", guidance_command.get("nz_cmd")), np.nan)
                if np.isfinite(nz_cmd):
                    episode_nz_cmds.append(nz_cmd)
                pitch_rad = _safe_float(own_state.get("pitch_rad"), np.nan)
                if np.isfinite(pitch_rad):
                    episode_pitch_degs.append(float(np.rad2deg(pitch_rad)))

                if terminated or truncated:
                    episode_count += 1
                    reason = info.get("reason", "unknown")
                    final_ata = float(np.rad2deg(rel_state.get("ata_rad", 0.0)))
                    combat_reason = str(info.get("combat_reason", ""))
                    if combat_enabled:
                        ep_success = bool(info.get("combat_success", False))
                        ep_crash = combat_reason in {
                            "ego_crash_or_out_of_bounds",
                            "target_crash_or_out_of_bounds",
                            "crash",
                        }
                        ep_oob = combat_reason in {"ego_crash_or_out_of_bounds", "out_of_bounds"}
                        ep_timeout = combat_reason in {
                            "timeout_hp_advantage",
                            "timeout_hp_disadvantage",
                            "timeout_draw",
                            "timeout",
                        }
                    else:
                        ep_success = reason == "success"
                        ep_crash = reason == "crash"
                        ep_oob = reason == "out_of_bounds"
                        ep_timeout = reason == "timeout"
                    diagnostics = _raw_failure_diagnostics(
                        info,
                        reason,
                        env,
                        config,
                        final_range_m=range_m,
                    )
                    recent_rollout_episodes.append({
                        "raw_crash": int(diagnostics["raw_is_crash"]),
                        "raw_out_of_bounds": int(diagnostics["raw_is_out_of_bounds"]),
                        "high_altitude_failure": int(diagnostics["high_altitude_failure"]),
                        "low_altitude_failure": int(diagnostics["low_altitude_failure"]),
                        "range_oob_failure": int(diagnostics["range_oob_failure"]),
                    })
                    ep_writer.writerow({
                        "step": global_step, "episode": episode_count,
                        "episode_return": episode_return, "episode_length": episode_length,
                        "success": int(ep_success),
                        "crash": int(ep_crash),
                        "out_of_bounds": int(ep_oob),
                        "timeout": int(ep_timeout),
                        "mean_range": float(np.mean(episode_ranges)) if episode_ranges else 0.0,
                        "final_range": range_m,
                        "max_range": float(np.max(episode_ranges)) if episode_ranges else range_m,
                        "final_ata": final_ata,
                        "termination_reason": reason,
                        "raw_termination_reason": diagnostics["raw_termination_reason"],
                        "combat_reason": info.get("combat_reason", ""),
                        "combat_outcome": info.get("combat_outcome", ""),
                        "raw_crash": int(diagnostics["raw_is_crash"]),
                        "raw_out_of_bounds": int(diagnostics["raw_is_out_of_bounds"]),
                        "raw_timeout": int(diagnostics["raw_is_timeout"]),
                        "high_altitude_failure": int(diagnostics["high_altitude_failure"]),
                        "low_altitude_failure": int(diagnostics["low_altitude_failure"]),
                        "range_oob_failure": int(diagnostics["range_oob_failure"]),
                        "final_altitude_m": diagnostics["final_altitude_m"],
                        "min_altitude_m": float(np.min(episode_altitudes)) if episode_altitudes else "",
                        "max_altitude_m": float(np.max(episode_altitudes)) if episode_altitudes else "",
                        "final_pitch_deg": episode_pitch_degs[-1] if episode_pitch_degs else "",
                        "mean_nz_cmd": float(np.mean(episode_nz_cmds)) if episode_nz_cmds else "",
                        "max_abs_nz_cmd": float(np.max(np.abs(episode_nz_cmds))) if episode_nz_cmds else "",
                    })
                    f_ep.flush()

                    episode_return = 0.0
                    episode_length = 0
                    episode_ranges = []
                    episode_altitudes = []
                    episode_nz_cmds = []
                    episode_pitch_degs = []

                    adversary_allowed = safety_gate.allows_adversary(global_step)
                    active_opponent_entry = _set_training_opponent(
                        env,
                        opponent_pool,
                        opponent_bucket,
                        rng,
                        device,
                        adversarial_cfg,
                        adversary_allowed=adversary_allowed,
                        blocked_reason=safety_gate.status(global_step),
                    )
                    scenario = sample_scenario({"scenarios": active_scenarios}, rng)
                    obs = env.reset(scenario=scenario, seed=rng.integers(0, 1000000))

                    if agent.buffer.full:
                        break

                if global_step >= total_timesteps:
                    break

            if agent.buffer.full or (global_step >= total_timesteps and len(agent.buffer) > 0):
                next_obs_vec = obs["observation_vector"]
                update_stats = agent.update(next_obs=next_obs_vec)
                update_num += 1
                up_writer.writerow({
                    "step": global_step, "update_num": update_num,
                    "policy_loss": update_stats.get("policy_loss", ""),
                    "value_loss": update_stats.get("value_loss", ""),
                    "entropy": update_stats.get("entropy", ""),
                    "approx_kl": update_stats.get("approx_kl", ""),
                    "clip_fraction": update_stats.get("clip_fraction", ""),
                    "explained_variance": update_stats.get("explained_variance", ""),
                    "learning_rate": update_stats.get("learning_rate", ""),
                })
                f_up.flush()
                print(
                    f"Step {global_step}/{total_timesteps} | Stage {current_stage} | "
                    f"Ep {episode_count} | Policy Loss: {update_stats.get('policy_loss', 0):.4f} | "
                    f"Value Loss: {update_stats.get('value_loss', 0):.4f} | "
                    f"Entropy: {update_stats.get('entropy', 0):.4f}"
                )

            # Evaluation
            if eval_interval > 0 and global_step % eval_interval == 0 and global_step > 0:
                print(f"\n--- Evaluation at step {global_step} ---")
                eval_cfg = config.get("evaluation", {})
                adversary_allowed_for_eval = safety_gate.allows_adversary(global_step)
                safety_status_before_eval = safety_gate.status(global_step)
                eval_metrics = run_evaluation(
                    env, agent, config,
                    num_episodes=eval_cfg.get("eval_episodes", 10),
                    seeds=eval_cfg.get("seeds", [0, 1, 2]),
                )
                rollout_metrics = compute_rollout_safety_metrics(recent_rollout_episodes)
                rollout_gate_ok, rollout_gate_reason = rollout_safety_gate_passed(
                    rollout_metrics,
                    rollout_gate_cfg,
                )
                safety_gate.update_from_eval(eval_metrics, global_step)
                env.set_command_post_processor_safety_mode(not safety_gate.passed)
                adversary_allowed_after_eval = safety_gate.allows_adversary(global_step)
                eval_writer.writerow({
                    "step": global_step, "num_episodes": eval_metrics["num_episodes"],
                    "mean_return": eval_metrics["mean_return"],
                    "std_return": eval_metrics["std_return"],
                    "success_rate": eval_metrics["success_rate"],
                    "win_rate": eval_metrics["win_rate"],
                    "survival_rate": eval_metrics["survival_rate"],
                    "hp_advantage": eval_metrics["hp_advantage"],
                    "mean_time_to_kill": eval_metrics["mean_time_to_kill"],
                    "crash_rate": eval_metrics["crash_rate"],
                    "out_of_bounds_rate": eval_metrics["out_of_bounds_rate"],
                    "raw_crash_rate": eval_metrics["raw_crash_rate"],
                    "raw_out_of_bounds_rate": eval_metrics["raw_out_of_bounds_rate"],
                    "high_altitude_failure_rate": eval_metrics["high_altitude_failure_rate"],
                    "low_altitude_failure_rate": eval_metrics["low_altitude_failure_rate"],
                    "range_oob_failure_rate": eval_metrics["range_oob_failure_rate"],
                    "timeout_rate": eval_metrics["timeout_rate"],
                })
                f_eval.flush()
                print(
                    f"Eval Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f} | "
                    f"Success: {eval_metrics['success_rate']:.2%} | "
                    f"Win: {eval_metrics['win_rate']:.2%} | "
                    f"Crash: {eval_metrics['crash_rate']:.2%} | "
                    f"OOB: {eval_metrics['out_of_bounds_rate']:.2%}"
                )

                # Per-scenario evaluation for curriculum gate
                per_scenario = evaluate_scenarios(env, agent, all_scenarios, num_episodes=5)
                cur_writer.writerow({
                    "step": global_step, "stage": current_stage,
                    "allowed_scenarios": "|".join(allowed_names),
                    "scenario_sr": json.dumps(per_scenario),
                    "opponent_bucket": opponent_bucket,
                    "opponent_pool_size": len(opponent_pool.entries),
                    "opponent_eval_win_rate": eval_metrics["win_rate"],
                    "safety_gate_status": safety_gate.status(global_step),
                    "safety_gate_passed": int(safety_gate.passed),
                    "adversary_allowed": int(adversary_allowed_after_eval),
                    "safety_gate_passed_step": safety_gate.passed_step
                    if safety_gate.passed_step is not None else "",
                    "train_rollout_episode_count": rollout_metrics["train_episode_count"],
                    "train_raw_crash_rate": rollout_metrics["train_raw_crash_rate"],
                    "train_raw_out_of_bounds_rate": rollout_metrics["train_raw_out_of_bounds_rate"],
                    "train_high_altitude_failure_rate": rollout_metrics["train_high_altitude_failure_rate"],
                    "train_low_altitude_failure_rate": rollout_metrics["train_low_altitude_failure_rate"],
                    "train_range_oob_failure_rate": rollout_metrics["train_range_oob_failure_rate"],
                    "rollout_gate_passed": int(rollout_gate_ok),
                    "rollout_gate_reason": rollout_gate_reason,
                })
                f_cur.flush()
                print(f"Per-scenario SR: {per_scenario}")
                print(
                    "Recent rollout safety: "
                    f"episodes={rollout_metrics['train_episode_count']} | "
                    f"raw_crash={rollout_metrics['train_raw_crash_rate']:.2%} | "
                    f"high_alt={rollout_metrics['train_high_altitude_failure_rate']:.2%} | "
                    f"gate={rollout_gate_reason}"
                )

                if (
                    adversarial_cfg.get("enabled", False)
                    and adversary_allowed_for_eval
                    and active_opponent_entry is not None
                ):
                    opponent_metric = eval_metrics["win_rate"]
                    if not np.isfinite(opponent_metric):
                        opponent_metric = eval_metrics["success_rate"]
                    opponent_pool.update_from_eval(
                        active_opponent_entry,
                        opponent_metric,
                        num_episodes=eval_metrics.get("num_episodes", 1),
                    )
                    if (
                        opponent_metric >= opponent_pool.threshold
                        and opponent_bucket != "strong"
                    ):
                        if rollout_gate_ok:
                            opponent_bucket = opponent_pool.next_bucket(opponent_bucket)
                            print(
                                f"*** Opponent curriculum advanced to {opponent_bucket} "
                                f"(win_rate={opponent_metric:.2%}) ***"
                            )
                        else:
                            print(
                                "Opponent curriculum gate blocked by rollout safety "
                                f"({rollout_gate_reason})"
                            )
                elif adversarial_cfg.get("enabled", False):
                    skip_reason = (
                        safety_status_before_eval
                        if not adversary_allowed_for_eval
                        else "no_active_opponent"
                    )
                    print(
                        "Opponent curriculum: skipping opponent ELO update "
                        f"({skip_reason})"
                    )

                if (
                    adversarial_cfg.get("enabled", False)
                    and adversary_allowed_after_eval != adversary_allowed_for_eval
                ):
                    active_opponent_entry = _set_training_opponent(
                        env,
                        opponent_pool,
                        opponent_bucket,
                        rng,
                        device,
                        adversarial_cfg,
                        adversary_allowed=adversary_allowed_after_eval,
                        blocked_reason=safety_gate.status(global_step),
                    )

                # Check if we should advance stage
                current_scenario_sr = [per_scenario.get(s, 0.0) for s in allowed_names]
                min_sr = min(current_scenario_sr) if current_scenario_sr else 0.0
                if min_sr >= stage_gate_sr and unlocked_stage < len(curriculum) - 1:
                    if rollout_gate_ok:
                        unlocked_stage += 1
                        current_stage = min(progress_stage, unlocked_stage)
                        print(
                            f"*** Curriculum gate passed (min SR={min_sr:.2%}). "
                            f"Unlocked stage {unlocked_stage} (current {current_stage}) ***"
                        )
                    else:
                        print(
                            "Curriculum gate blocked by rollout safety "
                            f"({rollout_gate_reason}); staying in stage {current_stage}"
                        )
                elif min_sr < stage_gate_sr:
                    print(f"Curriculum gate NOT passed (min SR={min_sr:.2%}). Staying in stage {current_stage}")

                if save_best and eval_metrics["mean_return"] > best_eval_return:
                    best_eval_return = eval_metrics["mean_return"]
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    agent.save(best_path)
                    if adversarial_cfg.get("enabled", False):
                        opponent_pool.add_checkpoint(best_path, global_step)
                    print(f"  -> Saved best checkpoint")

            if save_interval > 0 and global_step % save_interval == 0 and global_step > 0:
                step_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
                agent.save(step_path)
                if adversarial_cfg.get("enabled", False):
                    opponent_pool.add_checkpoint(step_path, global_step)

        if save_last:
            last_path = os.path.join(checkpoint_dir, "last.pt")
            agent.save(last_path)
            if adversarial_cfg.get("enabled", False):
                opponent_pool.add_checkpoint(last_path, global_step)
            print(f"\nSaved last checkpoint")

    elapsed = time.time() - start_time
    training_metadata["safety_precurriculum"] = safety_gate.as_dict(global_step)
    training_metadata["rollout_gate"] = {
        "config": dict(rollout_gate_cfg or {}),
        "last_metrics": compute_rollout_safety_metrics(recent_rollout_episodes),
        "last_result": rollout_safety_gate_passed(
            compute_rollout_safety_metrics(recent_rollout_episodes),
            rollout_gate_cfg,
        ),
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f_meta:
        json.dump(_json_safe(training_metadata), f_meta, indent=2, ensure_ascii=False)
    print(f"\nTraining complete! Steps: {global_step}, Episodes: {episode_count}, Time: {elapsed:.1f}s")
    env.close()
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train VPP PPO with Curriculum Learning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--backend", type=str, default=None, choices=["simple", "jsbsim"])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.backend is not None:
        old_backend = config.get("backend")
        config["backend"] = args.backend
        record_config_override_if_changed(
            config,
            "backend",
            args.backend,
            old_value=old_backend,
            source="train_curriculum_ppo.py:--backend",
        )
        env_cfg = config.setdefault("env", {})
        old_env_backend = env_cfg.get("backend")
        old_use_jsbsim = env_cfg.get("use_jsbsim")
        env_cfg["backend"] = args.backend
        env_cfg["use_jsbsim"] = (args.backend == "jsbsim")
        record_config_override_if_changed(
            config,
            "env.backend",
            args.backend,
            old_value=old_env_backend,
            source="train_curriculum_ppo.py:--backend",
        )
        record_config_override_if_changed(
            config,
            "env.use_jsbsim",
            args.backend == "jsbsim",
            old_value=old_use_jsbsim,
            source="train_curriculum_ppo.py:--backend",
        )
    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 0)
    if args.seed is not None:
        config.setdefault("experiment", {})
        old_seed = config["experiment"].get("seed")
        config["experiment"]["seed"] = int(seed)
        record_config_override_if_changed(
            config,
            "experiment.seed",
            int(seed),
            old_value=old_seed,
            source="train_curriculum_ppo.py:--seed",
        )
    set_seed(seed)
    exp_name = config.get("experiment", {}).get("name", "curriculum_ppo")
    output_dir = args.output_dir or os.path.join(
        config.get("experiment", {}).get("output_root", "outputs"), "experiments", exp_name
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.device is not None:
        ppo_cfg = config.setdefault("ppo", {})
        old_device = ppo_cfg.get("device")
        ppo_cfg["device"] = args.device
        record_config_override_if_changed(
            config,
            "ppo.device",
            args.device,
            old_value=old_device,
            source="train_curriculum_ppo.py:--device",
        )
    train_ppo_curriculum(config, output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
