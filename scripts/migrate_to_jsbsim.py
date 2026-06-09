#!/usr/bin/env python3
"""
JSBSim Backend Policy Migration Validation Script.

Loads a policy trained on the simple backend and evaluates it on both
simple and JSBSim backends under identical scenarios and seeds.

Produces:
    outputs/jsbsim_migration/report.md       - Human-readable migration report
    outputs/jsbsim_migration/comparison.csv  - Quantitative comparison table
    outputs/jsbsim_migration/results.json    - Full raw results

Usage (full validation):
    python scripts/migrate_to_jsbsim.py \
        --checkpoint outputs/experiments/no_prediction/checkpoints/best.pt \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --scenarios favorable neutral disadvantage challenging \
        --seeds 0 1 2 3 4 \
        --episodes-per-scenario 10

Usage (smoke test, 1 episode):
    python scripts/migrate_to_jsbsim.py \
        --checkpoint outputs/experiments/no_prediction/checkpoints/best.pt \
        --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
        --smoke

Requirements:
    - JSBSim Python bindings and data files must be installed.
    - Set JSBSIM_ROOT environment variable or configure legacy_project_root
      in config/env.yaml.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, "src")

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_method,
    load_experiment_config,
)
from uav_vpp_guidance.utils.config import merge_config
import copy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS_TO_COMPARE = [
    ("Success Rate", "instant_success_rate", ".2%"),
    ("Score Win Rate", "score_win_rate", ".2%"),
    ("Mean Return", "mean_return", ".1f"),
    ("Std Return", "std_return", ".1f"),
    ("Mean Final Range (m)", "mean_final_range_m", ".1f"),
    ("Mean Final ATA (deg)", "mean_final_ata_deg", ".1f"),
    ("Crash Rate", "crash_rate", ".2%"),
    ("OOB Rate", "out_of_bounds_rate", ".2%"),
    ("Timeout Rate", "timeout_rate", ".2%"),
    ("Mean Episode Length", "mean_length", ".1f"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate policy migration from simple to JSBSim backend"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PPO checkpoint trained on simple backend (.pt)"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/experiment/evaluate_vpp_prediction_comparison.yaml",
        help="Base experiment config YAML"
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+",
        default=["favorable", "neutral", "disadvantage", "challenging"],
        help="Scenarios to evaluate"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Evaluation random seeds"
    )
    parser.add_argument(
        "--episodes-per-scenario", type=int, default=10,
        help="Episodes per scenario (total = episodes * scenarios * seeds)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/jsbsim_migration",
        help="Output directory"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 seed, 1 episode per scenario, max 2 scenarios"
    )
    parser.add_argument(
        "--jsbsim-root", type=str, default=None,
        help="JSBSim root path (overrides config/env.yaml)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device"
    )
    parser.add_argument(
        "--skip-jsbsim-if-missing", action="store_true",
        help="Skip JSBSim evaluation if JSBSim is unavailable (simple-only report)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_backend_config(base_config: dict, backend: str, jsbsim_root: Optional[str]) -> dict:
    """Create a config copy with the specified backend."""
    config = copy.deepcopy(base_config)
    config["backend"] = backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = (backend == "jsbsim")
    if jsbsim_root is not None:
        config["env"]["legacy_project_root"] = jsbsim_root
    # For JSBSim, enforce strict_backend so we don't silently fall back to simple
    if backend == "jsbsim":
        config["env"]["strict_backend"] = True
    return config


# ---------------------------------------------------------------------------
# Observation adapter (dimension mismatch handling)
# ---------------------------------------------------------------------------


class ObservationAdapter:
    """
    Adapts observations when simple and JSBSim backends produce different
    observation dimensions.

    Strategy:
      - If dims match: passthrough (identity).
      - If JSBSim has fewer dims than simple: pad with zeros (policy was trained
        on more features; missing features are neutral).
      - If JSBSim has more dims than simple: truncate (extra features are ignored;
        policy was not trained on them).
    """

    def __init__(self, simple_dim: int, jsbsim_dim: int):
        self.simple_dim = simple_dim
        self.jsbsim_dim = jsbsim_dim
        self.mismatch = simple_dim != jsbsim_dim
        self.mode = "passthrough"
        if self.mismatch:
            if jsbsim_dim < simple_dim:
                self.mode = "pad"
            else:
                self.mode = "truncate"

    def adapt(self, obs_vec: np.ndarray, backend: str) -> np.ndarray:
        """Adapt observation vector for the policy network."""
        if not self.mismatch:
            return obs_vec
        vec = np.asarray(obs_vec, dtype=np.float32).flatten()
        if backend == "jsbsim":
            if self.mode == "pad":
                # Pad with zeros to match simple_dim
                if vec.shape[0] < self.simple_dim:
                    padded = np.zeros(self.simple_dim, dtype=np.float32)
                    padded[:vec.shape[0]] = vec
                    return padded
                return vec
            elif self.mode == "truncate":
                return vec[:self.simple_dim]
        return vec

    def summary(self) -> str:
        if not self.mismatch:
            return "No adaptation needed (dimensions match)."
        return f"Mode: {self.mode} (simple={self.simple_dim}, jsbsim={self.jsbsim_dim})"


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class AdaptedAgent:
    """Wrapper around PPOAgent that applies observation adaptation."""

    def __init__(self, agent: PPOAgent, adapter: ObservationAdapter, backend: str):
        self.agent = agent
        self.adapter = adapter
        self.backend = backend

    def get_deterministic_action(self, obs_vec: np.ndarray) -> np.ndarray:
        adapted = self.adapter.adapt(obs_vec, self.backend)
        return self.agent.get_deterministic_action(adapted)

    def get_value(self, obs_vec: np.ndarray):
        adapted = self.adapter.adapt(obs_vec, self.backend)
        return self.agent.get_value(adapted)


def run_backend_evaluation(
    config, agent: PPOAgent, adapter: ObservationAdapter, backend_name: str, args
) -> dict:
    """Run evaluation on a single backend."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating on backend: {backend_name}")
    print(f"{'=' * 60}")

    env = CloseRangeTrackingEnv(config)
    print(f"Environment backend: {env._backend}")

    adapted_agent = AdaptedAgent(agent, adapter, backend_name)

    num_episodes = args.episodes_per_scenario * len(args.scenarios)
    backend_output = os.path.join(args.output_dir, backend_name)

    metrics = evaluate_method(
        env=env,
        agent=adapted_agent,
        config=config,
        method_name=backend_name,
        num_episodes=num_episodes,
        seeds=args.seeds,
        scenarios=args.scenarios,
        save_trajectories=False,
        output_dir=backend_output,
        training_seed=None,
    )
    env.close()
    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(val, fmt: str) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    try:
        return f"{val:{fmt}}"
    except Exception:
        return str(val)


def generate_report(results: dict, output_dir: str, args) -> None:
    """Generate markdown migration validation report."""
    simple = results.get("simple", {})
    jsbsim = results.get("jsbsim", {})
    jsbsim_error = results.get("jsbsim_error")

    report_path = Path(output_dir) / "report.md"

    lines = [
        "# JSBSim Backend Policy Migration Validation Report",
        "",
        f"**Checkpoint**: `{args.checkpoint}`",
        f"**Config**: `{args.config}`",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Evaluation Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Scenarios | {', '.join(args.scenarios)} |",
        f"| Seeds | {', '.join(map(str, args.seeds))} |",
        f"| Episodes per scenario | {args.episodes_per_scenario} |",
        f"| Total episodes per backend | {args.episodes_per_scenario * len(args.scenarios) * len(args.seeds)} |",
        f"| Device | {args.device} |",
        "",
        "## Observation Compatibility",
        "",
        "| Backend | Observation Dim |",
        "|---------|-----------------|",
        f"| Simple | {results.get('simple_obs_dim', 'N/A')} |",
        f"| JSBSim | {results.get('jsbsim_obs_dim', 'N/A')} |",
        "",
    ]

    adapter_summary = results.get("obs_adapter_summary", "Unknown")
    if results.get("obs_dim_mismatch"):
        lines.extend([
            f"**WARNING**: Observation dimensions differ between backends.",
            f"Adapter: {adapter_summary}",
            "",
        ])
    else:
        lines.extend([
            "**OK**: Observation dimensions are compatible.",
            f"Adapter: {adapter_summary}",
            "",
        ])

    if jsbsim_error:
        lines.extend([
            "## JSBSim Evaluation Status",
            "",
            f"**FAILED**: {jsbsim_error}",
            "",
            "Only simple backend results are available.",
            "",
        ])

    lines.extend([
        "## Performance Comparison",
        "",
        "| Metric | Simple | JSBSim | Delta |",
        "|--------|--------|--------|-------|",
    ])

    for label, key, fmt in METRICS_TO_COMPARE:
        s_val = simple.get(key)
        j_val = jsbsim.get(key) if not jsbsim_error else None
        s_str = _fmt(s_val, fmt)
        j_str = _fmt(j_val, fmt)

        if s_val is not None and j_val is not None and np.isfinite(s_val) and np.isfinite(j_val):
            delta = j_val - s_val
            if key.endswith("_rate"):
                d_str = f"{delta:+.2%}"
            else:
                d_str = f"{delta:+.1f}"
        else:
            d_str = "N/A"

        lines.append(f"| {label} | {s_str} | {j_str} | {d_str} |")

    lines.extend([
        "",
        "## Per-Scenario Breakdown",
        "",
    ])

    simple_per_scenario = simple.get("per_scenario", {})
    jsbsim_per_scenario = jsbsim.get("per_scenario", {}) if not jsbsim_error else {}

    for scenario in args.scenarios:
        lines.extend([
            f"### {scenario}",
            "",
            "| Metric | Simple | JSBSim | Delta |",
            "|--------|--------|--------|-------|",
        ])
        s_sc = simple_per_scenario.get(scenario, {})
        j_sc = jsbsim_per_scenario.get(scenario, {})
        for label, key, fmt in METRICS_TO_COMPARE:
            s_val = s_sc.get(key)
            j_val = j_sc.get(key)
            s_str = _fmt(s_val, fmt)
            j_str = _fmt(j_val, fmt)
            if s_val is not None and j_val is not None and np.isfinite(s_val) and np.isfinite(j_val):
                delta = j_val - s_val
                if key.endswith("_rate"):
                    d_str = f"{delta:+.2%}"
                else:
                    d_str = f"{delta:+.1f}"
            else:
                d_str = "N/A"
            lines.append(f"| {label} | {s_str} | {j_str} | {d_str} |")
        lines.append("")

    lines.extend([
        "## Conclusion",
        "",
    ])

    if jsbsim_error:
        lines.extend([
            "- **JSBSim evaluation could not be performed** due to environment initialization failure.",
            "- Check JSBSIM_ROOT or config/env.yaml `legacy_project_root`.",
        ])
    elif results.get("obs_dim_mismatch"):
        lines.append(
            "- **Observation dimension mismatch detected**. "
            "The policy was trained with a different observation shape than JSBSim provides. "
            "An automatic adapter was applied, but performance may be degraded."
        )
        sr_simple = simple.get("instant_success_rate")
        sr_jsbsim = jsbsim.get("instant_success_rate")
        if sr_simple is not None and sr_jsbsim is not None and np.isfinite(sr_simple) and np.isfinite(sr_jsbsim):
            sr_drop = sr_simple - sr_jsbsim
            lines.append(f"- Success rate drop: {sr_drop:.1%} (simple {sr_simple:.1%} → jsbsim {sr_jsbsim:.1%}).")
    else:
        sr_simple = simple.get("instant_success_rate")
        sr_jsbsim = jsbsim.get("instant_success_rate")
        if sr_simple is not None and sr_jsbsim is not None and np.isfinite(sr_simple) and np.isfinite(sr_jsbsim):
            sr_drop = sr_simple - sr_jsbsim
            if sr_drop > 0.2:
                lines.append(
                    f"- **Significant performance degradation**: Success rate drops by {sr_drop:.1%} on JSBSim. "
                    "The policy likely overfits to simplified dynamics."
                )
            elif sr_drop > 0.05:
                lines.append(
                    f"- **Moderate performance degradation**: Success rate drops by {sr_drop:.1%} on JSBSim. "
                    "Some dynamics mismatch exists; consider fine-tuning or curriculum transfer."
                )
            elif sr_drop < -0.05:
                lines.append(
                    f"- **Unexpected improvement**: Success rate increases by {abs(sr_drop):.1%} on JSBSim. "
                    "This may indicate the simplified model is overly pessimistic."
                )
            else:
                lines.append(
                    f"- **Good migration**: Success rate difference is only {abs(sr_drop):.1%}. "
                    "Policy transfers well to JSBSim."
                )
        else:
            lines.append("- **Insufficient data** to draw a conclusion.")

    lines.extend([
        "",
        f"---",
        f"Generated by `scripts/migrate_to_jsbsim.py`",
        f"Total validation time: {results.get('total_time_s', 0):.1f}s",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[REPORT] {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Smoke mode overrides
    if args.smoke:
        args.seeds = [0]
        args.episodes_per_scenario = 1
        if len(args.scenarios) > 2:
            args.scenarios = args.scenarios[:2]
        print("[SMOKE] Smoke mode enabled: 1 seed, 1 episode per scenario, max 2 scenarios")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    print(f"Loading config: {args.config}")
    base_config = load_experiment_config(args.config)

    # Load checkpoint metadata
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    ckpt_obs_dim = checkpoint.get("obs_dim")
    ckpt_action_dim = checkpoint.get("action_dim")
    ckpt_config = checkpoint.get("config", {})
    print(f"  Checkpoint metadata: obs_dim={ckpt_obs_dim}, action_dim={ckpt_action_dim}")

    # Build backend configs
    simple_config = build_backend_config(base_config, "simple", args.jsbsim_root)
    jsbsim_config = build_backend_config(base_config, "jsbsim", args.jsbsim_root)

    # ------------------------------------------------------------------
    # Observation compatibility check
    # ------------------------------------------------------------------
    print("\nChecking observation compatibility...")
    simple_env = CloseRangeTrackingEnv(simple_config)
    simple_obs = simple_env.reset(seed=0)
    simple_dim = int(simple_obs["observation_vector"].shape[0])
    simple_env.close()
    print(f"  Simple backend obs_dim: {simple_dim}")

    jsbsim_env = None
    jsbsim_dim = None
    jsbsim_error = None
    try:
        jsbsim_env = CloseRangeTrackingEnv(jsbsim_config)
        jsbsim_obs = jsbsim_env.reset(seed=0)
        jsbsim_dim = int(jsbsim_obs["observation_vector"].shape[0])
        jsbsim_env.close()
        print(f"  JSBSim backend obs_dim: {jsbsim_dim}")
    except Exception as exc:
        jsbsim_error = str(exc)
        print(f"  JSBSim backend init FAILED: {exc}")
        if jsbsim_env is not None:
            try:
                jsbsim_env.close()
            except Exception:
                pass
        if not args.skip_jsbsim_if_missing:
            print("\nTo skip JSBSim and generate a simple-only report, use --skip-jsbsim-if-missing")
            sys.exit(1)

    policy_obs_dim = ckpt_obs_dim if ckpt_obs_dim is not None else simple_dim
    policy_action_dim = ckpt_action_dim if ckpt_action_dim is not None else 3

    if jsbsim_dim is not None:
        obs_dim_mismatch = (simple_dim != jsbsim_dim) or (policy_obs_dim != jsbsim_dim)
    else:
        obs_dim_mismatch = False

    adapter = ObservationAdapter(policy_obs_dim, jsbsim_dim if jsbsim_dim is not None else policy_obs_dim)
    print(f"  Observation adapter: {adapter.summary()}")

    # ------------------------------------------------------------------
    # Create agent
    # ------------------------------------------------------------------
    agent_config = merge_config(copy.deepcopy(base_config), copy.deepcopy(ckpt_config))
    agent = PPOAgent(
        obs_dim=policy_obs_dim,
        action_dim=policy_action_dim,
        config=agent_config,
        device=args.device,
    )
    agent.load(args.checkpoint)
    print(f"Agent loaded: obs_dim={policy_obs_dim}, action_dim={policy_action_dim}")

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    results = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "simple_obs_dim": simple_dim,
        "jsbsim_obs_dim": jsbsim_dim,
        "policy_obs_dim": policy_obs_dim,
        "policy_action_dim": policy_action_dim,
        "obs_dim_mismatch": obs_dim_mismatch,
        "obs_adapter_summary": adapter.summary(),
        "jsbsim_error": jsbsim_error,
    }

    start_time = time.time()

    # Simple backend (always run)
    simple_start = time.time()
    try:
        metrics = run_backend_evaluation(simple_config, agent, adapter, "simple", args)
        results["simple"] = metrics
        print(f"[SIMPLE] Completed in {time.time() - simple_start:.1f}s")
    except Exception as exc:
        print(f"[ERROR] Simple backend evaluation failed: {exc}")
        traceback.print_exc()
        results["simple"] = {"error": str(exc)}

    # JSBSim backend (if available)
    if jsbsim_error is None:
        jsbsim_start = time.time()
        try:
            metrics = run_backend_evaluation(jsbsim_config, agent, adapter, "jsbsim", args)
            results["jsbsim"] = metrics
            print(f"[JSBSIM] Completed in {time.time() - jsbsim_start:.1f}s")
        except Exception as exc:
            print(f"[ERROR] JSBSim backend evaluation failed: {exc}")
            traceback.print_exc()
            results["jsbsim"] = {"error": str(exc)}
            results["jsbsim_error"] = str(exc)
    else:
        results["jsbsim"] = {"error": jsbsim_error}
        results["jsbsim_error"] = jsbsim_error

    total_time = time.time() - start_time
    results["total_time_s"] = total_time
    results["args"] = vars(args)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[JSON] Results saved: {json_path}")

    # ------------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------------
    comparison_rows = []
    simple_metrics = results.get("simple", {})
    jsbsim_metrics = results.get("jsbsim", {})
    jsbsim_has_error = results.get("jsbsim_error") is not None or "error" in jsbsim_metrics

    def _add_row(metric_key: str, label: str, s_dict: dict, j_dict: dict):
        s_val = s_dict.get(metric_key)
        j_val = j_dict.get(metric_key) if not jsbsim_has_error else None
        if s_val is not None and np.isfinite(s_val):
            s_out = float(s_val)
        else:
            s_out = None
        if j_val is not None and np.isfinite(j_val):
            j_out = float(j_val)
        else:
            j_out = None
        delta = None
        rel_delta = None
        if s_out is not None and j_out is not None:
            delta = j_out - s_out
            if s_out != 0:
                rel_delta = delta / abs(s_out) * 100.0
        comparison_rows.append({
            "metric": label,
            "simple": s_out,
            "jsbsim": j_out,
            "delta": delta,
            "relative_delta_pct": rel_delta,
        })

    for label, key, _fmt in METRICS_TO_COMPARE:
        _add_row(key, label, simple_metrics, jsbsim_metrics)

    # Per-scenario comparison
    simple_sc = simple_metrics.get("per_scenario", {})
    jsbsim_sc = jsbsim_metrics.get("per_scenario", {}) if not jsbsim_has_error else {}
    for scenario in args.scenarios:
        for label, key, _fmt in METRICS_TO_COMPARE:
            _add_row(key, f"{scenario}:{label}", simple_sc.get(scenario, {}), jsbsim_sc.get(scenario, {}))

    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "simple", "jsbsim", "delta", "relative_delta_pct"]
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"[CSV] Comparison saved: {csv_path}")

    # ------------------------------------------------------------------
    # Generate report
    # ------------------------------------------------------------------
    generate_report(results, str(output_dir), args)

    print(f"\n{'=' * 60}")
    print("Migration validation complete.")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
