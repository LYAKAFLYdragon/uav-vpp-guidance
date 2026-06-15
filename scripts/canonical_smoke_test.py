#!/usr/bin/env python3
"""End-to-end smoke test for frozen canonical configurations.

Validates that the canonical guidance, reward, gain-space, and VPP configs can
initialize and run the close-range tracking pipeline without exceptions.

Usage:
    python scripts/canonical_smoke_test.py
"""

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_vpp_guidance.envs.scenario_registry import initialize_canonical_scenarios
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.gain_optimizer.gain_space import GainSpace
from uav_vpp_guidance.utils.config import merge_config


CANONICAL_DIR = PROJECT_ROOT / "config" / "canonical"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_canonical_config(extra: dict = None) -> dict:
    """Merge canonical configs into a single dict usable by CloseRangeTrackingEnv."""
    config = {}
    for name in ("guidance", "reward", "virtual_point", "gain_space"):
        config = merge_config(config, load_yaml(CANONICAL_DIR / f"{name}.yaml"))
    # env and termination defaults
    config = merge_config(config, load_yaml(PROJECT_ROOT / "config" / "env.yaml"))
    config = merge_config(config, load_yaml(PROJECT_ROOT / "config" / "success_criteria" / "medium.yaml"))
    # Default PPO / agent placeholders so downstream code can read obs dims safely
    config.setdefault("ppo", {})
    config["ppo"].setdefault("hidden_dims", [256, 256])
    config["ppo"].setdefault("activation", "relu")
    config["ppo"].setdefault("lr", 3e-4)
    config["ppo"].setdefault("gamma", 0.99)
    config["ppo"].setdefault("gae_lambda", 0.95)
    config["ppo"].setdefault("clip_epsilon", 0.2)
    config["ppo"].setdefault("value_coef", 0.5)
    config["ppo"].setdefault("entropy_coef", 0.01)
    config["ppo"].setdefault("max_grad_norm", 0.5)
    # Use simple backend for fast, dependency-light smoke testing
    config["backend"] = "simple"
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    # Disable mode switch for pure canonical smoke
    config.setdefault("guidance", {})
    config["guidance"].setdefault("mode_switch", {})
    config["guidance"]["mode_switch"]["enabled"] = False
    if extra:
        config = merge_config(config, extra)
    return config


def smoke_init_env(config: dict) -> dict:
    """Initialize CloseRangeTrackingEnv and GainSpace."""
    report = {"ok": True, "errors": []}
    env = None
    try:
        initialize_canonical_scenarios()
        env = CloseRangeTrackingEnv(config)
        report["backend"] = env._backend
        report["guidance_mode"] = env.guidance.__class__.__name__
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"CloseRangeTrackingEnv init failed: {exc}\n{traceback.format_exc()}")
        return report

    try:
        gain_space = GainSpace(config["gain_space"]["bounds"])
        report["gain_space"] = {
            "names": gain_space.names,
            "low": gain_space.low.tolist(),
            "high": gain_space.high.tolist(),
        }
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"GainSpace init failed: {exc}\n{traceback.format_exc()}")
        return report

    return report, env, gain_space


def smoke_run_episode(env: CloseRangeTrackingEnv, max_steps: int = 128) -> dict:
    """Run a short episode with random actions and report outcome."""
    report = {"ok": True, "errors": [], "steps": 0, "done": False, "reason": None}
    try:
        obs = env.reset(seed=42)
        action_dim = env.virtual_point_generator.action_dim if env.virtual_point_generator else 3
        for step in range(max_steps):
            action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            report["steps"] = step + 1
            done = terminated or truncated or info.get("done", False)
            if done:
                report["done"] = True
                report["reason"] = info.get("termination", {}).get("reason", "unknown")
                break
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"Episode rollout failed at step {report['steps']}: {exc}\n{traceback.format_exc()}")
    return report


def smoke_gain_only_cem(config: dict, checkpoint: Path = None, n_iter: int = 1, candidates: int = 4) -> dict:
    """Verify the gain-only CEM initialization / dry-run path."""
    report = {"ok": True, "errors": []}
    try:
        from uav_vpp_guidance.agents.ppo_agent import PPOAgent
        from uav_vpp_guidance.envs.scenario_registry import ScenarioRegistry
        from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
        from uav_vpp_guidance.gain_optimizer.cem import CEMEMAGainOptimizer
        from uav_vpp_guidance.guidance.gain_config import GuidanceGains

        gain_space = GainSpace(config["gain_space"]["bounds"])
        cem_config = {
            "candidates": candidates,
            "elite_ratio": 0.25,
            "noise_floor": 0.05,
            "convergence_tol": 0.001,
            "beta_ema": 0.7,
        }
        cem = CEMEMAGainOptimizer(gain_space, cem_config)

        env = CloseRangeTrackingEnv(config)
        obs = env.reset(seed=0)
        obs_dim = int(obs["observation_vector"].shape[0])
        agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")
        if checkpoint and checkpoint.exists():
            agent.load(str(checkpoint))
            report["checkpoint_used"] = str(checkpoint)
        else:
            report["checkpoint_used"] = None
            report["note"] = "No checkpoint provided; using random policy weights for smoke test."

        scenarios = ScenarioRegistry.get_regression_suite()[:2]

        def evaluator(gains_dict: dict) -> float:
            env.current_gains = GuidanceGains(**gains_dict)
            successes = 0
            total = 0
            for scen in scenarios:
                for seed in (0, 1):
                    result, _ = evaluate_single_episode(
                        env=env,
                        agent=agent,
                        config=env.config,
                        scenario=scen,
                        seed=seed,
                        save_trajectory=False,
                        method_name="gain_only_cem_smoke",
                    )
                    if result.get("is_success", False):
                        successes += 1
                    total += 1
            return successes / total if total > 0 else 0.0

        best_gains, history = cem.optimize(evaluator, n_iter=n_iter)
        report["best_gains"] = best_gains
        report["history_length"] = len(history)
        if history:
            report["final_best_score"] = float(history[-1].get("best_score", float("nan")))
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"Gain-only CEM smoke failed: {exc}\n{traceback.format_exc()}")
    return report


def find_any_checkpoint() -> Path:
    """Return any available .pt checkpoint for dry-run validation."""
    candidates = list(PROJECT_ROOT.glob("outputs/**/checkpoints/*.pt"))
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Canonical config end-to-end smoke test")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional PPO checkpoint for CEM smoke")
    parser.add_argument("--max-episode-steps", type=int, default=128)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else find_any_checkpoint()

    full_report = {
        "title": "Canonical Configuration Smoke Test",
        "config_files": [
            "config/canonical/guidance.yaml",
            "config/canonical/reward.yaml",
            "config/canonical/virtual_point.yaml",
            "config/canonical/gain_space.yaml",
            "config/env.yaml",
            "config/success_criteria/medium.yaml",
        ],
        "steps": {},
    }

    print("[SmokeTest] Building canonical config...")
    config = build_canonical_config()
    full_report["steps"]["build_config"] = {"ok": True}

    print("[SmokeTest] Initializing CloseRangeTrackingEnv and GainSpace...")
    init_report, env, gain_space = smoke_init_env(config)
    full_report["steps"]["init_env_and_gain_space"] = {
        k: v for k, v in init_report.items() if k != "errors"
    }
    if not init_report["ok"]:
        full_report["overall"] = "FAILED"
        full_report["steps"]["init_env_and_gain_space"]["errors"] = init_report["errors"]
        _write_report(full_report, args.output)
        sys.exit(1)

    print("[SmokeTest] Running one episode...")
    episode_report = smoke_run_episode(env, max_steps=args.max_episode_steps)
    full_report["steps"]["run_episode"] = {k: v for k, v in episode_report.items() if k != "errors"}
    if not episode_report["ok"]:
        full_report["overall"] = "FAILED"
        full_report["steps"]["run_episode"]["errors"] = episode_report["errors"]
        _write_report(full_report, args.output)
        sys.exit(1)

    print("[SmokeTest] Verifying gain-only CEM flow...")
    cem_report = smoke_gain_only_cem(config, checkpoint=checkpoint)
    full_report["steps"]["gain_only_cem"] = {k: v for k, v in cem_report.items() if k != "errors"}
    if not cem_report["ok"]:
        full_report["overall"] = "FAILED"
        full_report["steps"]["gain_only_cem"]["errors"] = cem_report["errors"]
        _write_report(full_report, args.output)
        sys.exit(1)

    full_report["overall"] = "PASSED"
    _write_report(full_report, args.output)
    print("[SmokeTest] PASSED")


def _write_report(report: dict, output_path: str = None):
    text = json.dumps(report, indent=2, default=str)
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"[SmokeTest] Report written to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
