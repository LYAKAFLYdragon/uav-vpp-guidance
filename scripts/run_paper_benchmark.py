#!/usr/bin/env python3
"""Stage 9A: Frozen paper benchmark artifact contract.

Evaluates all methods and generates:
- summary.md: Full text report with statistical comparison and reproducibility metadata
- results.csv: Raw data
- figures/*.png: Paper figures
- tables/*.md: Markdown tables
- run_manifest.json: Audit manifest with exact CLI, git commit, config hash,
  per-method checkpoint/gains provenance, and paper_safe flag.
"""

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.evaluation.statistical_comparison import paired_t_test, cohens_d
from uav_vpp_guidance.guidance.gain_config import GuidanceGains


METHODS = {
    "no_prediction": {
        "name": "No-Prediction",
        "checkpoint": "outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt",
        "config_method": "no_prediction",
    },
    "cv_prediction": {
        "name": "CV Prediction",
        "checkpoint": "outputs/experiments/stage6b_cv_s1/checkpoints/best.pt",
        "config_method": "cv_prediction",
    },
    "ca_prediction": {
        "name": "CA Prediction",
        "checkpoint": "outputs/experiments/stage6b_ca_s1/checkpoints/best.pt",
        "config_method": "ca_prediction",
    },
    "gain_only": {
        "checkpoint": "outputs/audit_no_pred_final/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/gain_only_cem/cem_results.json",
        "note": "Same policy as no_prediction but with CEM-optimized gains",
    },
    "bilevel": {
        "checkpoint": "outputs/bilevel_training/checkpoints/best.pt",
        "config_method": "no_prediction",
        "gains_path": "outputs/bilevel_training/bilevel_results.json",
        "note": "Bilevel co-optimization: PPO policy + CEM gains",
    },
    "lstm_frozen": {
        "checkpoint": "outputs/experiments/vpp_ppo_lstm_frozen/checkpoints/best.pt",
        "config_method": "lstm_frozen",
    },
    "gru_frozen": {
        "checkpoint": "outputs/experiments/vpp_ppo_gru_frozen/checkpoints/best.pt",
        "config_method": "gru_frozen",
    },
}


# ---------------------------------------------------------------------------
# Config / provenance helpers
# ---------------------------------------------------------------------------
def _config_hash(config: dict) -> str:
    """Compute a simple hash of the resolved config for reproducibility tracking."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _get_git_info() -> dict:
    """Capture git commit, dirty status, and branch."""
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
        info["dirty"] = (
            len(subprocess.check_output(["git", "status", "--short"], text=True).strip())
            > 0
        )
        info["branch"] = (
            subprocess.check_output(["git", "branch", "--show-current"], text=True)
            .strip()
        )
    except Exception:
        pass
    return info


def load_config(config_path: str, method_name: str) -> dict:
    full_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get(method_name, {})
    base_config = copy.deepcopy(full_config)
    for k, v in method_override.items():
        if isinstance(v, dict) and k in base_config and isinstance(base_config[k], dict):
            base_config[k].update(copy.deepcopy(v))
        else:
            base_config[k] = copy.deepcopy(v)
    return base_config


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
def _resolve_checkpoint(
    method_name: str,
    methods_default_cfg: dict,
    config_method_override: dict,
    checkpoint_map: dict,
) -> tuple:
    """Resolve final checkpoint path with explicit precedence.

    Precedence (high to low):
      1. CLI --checkpoint-map
      2. Config method override
      3. METHODS default

    Returns:
        (final_path, source) where source is one of
        "cli_checkpoint_map", "config_method", "methods_default".
    """
    if checkpoint_map and method_name in checkpoint_map:
        return checkpoint_map[method_name], "cli_checkpoint_map"
    if config_method_override.get("checkpoint"):
        return config_method_override["checkpoint"], "config_method"
    return methods_default_cfg["checkpoint"], "methods_default"


# ---------------------------------------------------------------------------
# Gains schema validation
# ---------------------------------------------------------------------------
_GAIN_FIELD_NAMES = {f.name for f in fields(GuidanceGains)}


def _load_method_gains(method_cfg: dict, allow_random_smoke: bool, method_name: str = "gain_only") -> dict:
    """Load and validate CEM-optimized gains for a method.

    Returns:
        {
            "loaded_gains": dict,
            "ignored_gain_fields": list,
            "gains_schema_valid": bool,
            "gains_exists": bool,
        }

    Raises:
        FileNotFoundError: gains file missing (unless allow_random_smoke).
        ValueError: JSON parse error, missing/empty best_gains, or no supported fields.
    """
    default_invalid = {
        "loaded_gains": {},
        "ignored_gain_fields": [],
        "gains_schema_valid": False,
        "gains_exists": False,
    }

    gains_path = method_cfg.get("gains_path")
    if not gains_path:
        if allow_random_smoke:
            print(f"WARNING: gains_path not configured for {method_name}")
            return default_invalid
        raise FileNotFoundError(
            f"gains_path not configured for {method_name}. "
            "Use --allow-random-smoke to proceed without gains."
        )

    gains_file = Path(gains_path)
    if not gains_file.exists():
        if allow_random_smoke:
            print(f"WARNING: Gains file not found for {method_name}: {gains_path}")
            return default_invalid
        raise FileNotFoundError(
            f"Gains file not found for {method_name}: {gains_path}. "
            "Use --allow-random-smoke to proceed without gains."
        )

    try:
        data = json.loads(gains_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if allow_random_smoke:
            print(f"WARNING: Invalid JSON in gains file {gains_path}: {exc}")
            return default_invalid
        raise ValueError(f"Invalid JSON in gains file {gains_path}: {exc}")

    best = data.get("best_gains")
    if not isinstance(best, dict) or not best:
        if allow_random_smoke:
            print(f"WARNING: No valid best_gains in {gains_path}")
            return default_invalid
        raise ValueError(
            f"Gains file {gains_path} must contain a non-empty 'best_gains' object."
        )

    loaded_gains = {}
    ignored_gain_fields = []
    for k, v in best.items():
        if k in _GAIN_FIELD_NAMES:
            loaded_gains[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
        else:
            ignored_gain_fields.append(k)

    if not loaded_gains:
        if allow_random_smoke:
            print(
                f"WARNING: best_gains in {gains_path} contains no supported gain fields. "
                f"Supported: {_GAIN_FIELD_NAMES}"
            )
            return {
                "loaded_gains": {},
                "ignored_gain_fields": ignored_gain_fields,
                "gains_schema_valid": False,
                "gains_exists": True,
            }
        raise ValueError(
            f"best_gains in {gains_path} contains no supported gain fields. "
            f"Supported: {_GAIN_FIELD_NAMES}"
        )

    return {
        "loaded_gains": loaded_gains,
        "ignored_gain_fields": ignored_gain_fields,
        "gains_schema_valid": True,
        "gains_exists": True,
    }


# ---------------------------------------------------------------------------
# Method evaluation
# ---------------------------------------------------------------------------
def evaluate_method(
    method_name: str,
    method_cfg: dict,
    scenarios: list,
    seeds: tuple,
    backend: str,
    config_path: str,
    full_config: dict,
    checkpoint_map: dict = None,
    allow_random_smoke: bool = False,
) -> dict:
    """Evaluate a single method across all scenarios and seeds."""
    config = load_config(config_path, method_cfg["config_method"])
    config["backend"] = backend
    if "env" not in config:
        config["env"] = {}
    config["env"]["backend"] = backend
    config["env"]["use_jsbsim"] = backend == "jsbsim"

    # Disable mode-switch for clean comparison
    if "guidance" not in config:
        config["guidance"] = {}
    if "mode_switch" not in config["guidance"]:
        config["guidance"]["mode_switch"] = {}
    config["guidance"]["mode_switch"]["enabled"] = False

    # gain_only: load and apply CEM-optimized gains
    gains_info = {
        "loaded_gains": None,
        "ignored_gain_fields": None,
        "gains_schema_valid": None,
        "gains_exists": False,
    }
    if method_name in ("gain_only", "bilevel"):
        gains_info = _load_method_gains(method_cfg, allow_random_smoke, method_name=method_name)
        loaded_gains = gains_info["loaded_gains"]
        if loaded_gains:
            if "gains" not in config["guidance"]:
                config["guidance"]["gains"] = {}
            config["guidance"]["gains"].update(copy.deepcopy(loaded_gains))

    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])

    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=config, device="cpu")

    config_method_override = full_config.get("methods", {}).get(method_name, {})
    ckpt_path, ckpt_source = _resolve_checkpoint(
        method_name, method_cfg, config_method_override, checkpoint_map or {}
    )
    ckpt_exists = Path(ckpt_path).exists()
    if ckpt_exists:
        agent.load(ckpt_path)
    else:
        if not allow_random_smoke:
            raise FileNotFoundError(
                f"Checkpoint not found for {method_name}: {ckpt_path}. "
                "Use --allow-random-smoke to proceed with random policy."
            )
        print(f"WARNING: Checkpoint not found for {method_name}: {ckpt_path}")
        print("Using random policy (results marked as invalid_for_paper)")

    episodes = []
    for scen in scenarios:
        for seed in seeds:
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scen,
                seed=seed,
                save_trajectory=False,
                method_name=method_name,
            )
            result["method"] = method_name
            result["scenario"] = scen.get("name", "unknown")
            episodes.append(result)

    env.close()

    # Paper-safe eligibility
    invalid_reasons = []
    if not ckpt_exists:
        invalid_reasons.append("missing_checkpoint")
    if method_name in ("gain_only", "bilevel"):
        if not gains_info["gains_exists"]:
            invalid_reasons.append("missing_gains_file")
        if not gains_info["gains_schema_valid"]:
            invalid_reasons.append("invalid_gains_schema")
        if gains_info["gains_exists"] and not gains_info["loaded_gains"]:
            invalid_reasons.append("no_loaded_gains")

    invalid_for_paper = bool(invalid_reasons)

    metadata = {
        "method": method_name,
        "config_path": config_path,
        "resolved_config_hash": _config_hash(config),
        "method_override_name": method_cfg["config_method"],
        "backend": backend,
        "scenarios": [s.get("name", "unknown") for s in scenarios],
        "seeds": list(seeds),
        "prediction_mode": method_cfg["config_method"],
        "guidance_mode": config.get("guidance", {}).get("mode", "unknown"),
        "gain_source": "cem" if method_name in ("gain_only", "bilevel") else "default",
        # Checkpoint provenance (final/frozen contract)
        "checkpoint_path_final": ckpt_path,
        "checkpoint_source": ckpt_source,
        "checkpoint_exists": ckpt_exists,
        # Backward-compatible alias
        "policy_checkpoint": ckpt_path,
        # Gains provenance
        "gains_path": method_cfg.get("gains_path"),
        "gains_exists": gains_info["gains_exists"],
        "gains_schema_valid": gains_info["gains_schema_valid"],
        "loaded_gains": gains_info["loaded_gains"],
        "ignored_gain_fields": gains_info["ignored_gain_fields"],
        "is_random_smoke": not ckpt_exists,
        "invalid_for_paper": invalid_for_paper,
        "invalid_for_paper_reasons": invalid_reasons,
        "note": method_cfg.get("note", ""),
    }

    return {
        "method": method_name,
        "episodes": episodes,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize(obj):
    """Serialize numpy types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return bool(obj)
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------
def generate_figures(results: list, output_dir: Path):
    """Generate paper figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = []
    means = []
    stds = []
    for r in results:
        method = r["method"]
        returns = [ep.get("return", 0) for ep in r["episodes"]]
        methods.append(method)
        means.append(np.mean(returns) if returns else 0)
        stds.append(np.std(returns, ddof=1) if len(returns) > 1 else 0)

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Return")
    ax.set_title("Method Comparison (Mean Return ± Std)")
    ax.grid(axis="y", alpha=0.3)

    fig_path = output_dir / "figures" / "figure1_method_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return [fig_path]


def generate_tables(results: list, output_dir: Path):
    """Generate Markdown and CSV tables."""
    rows = []
    baseline = None
    for r in results:
        method = r["method"]
        episodes = r["episodes"]
        returns = [ep.get("return", 0) for ep in episodes]
        successes = sum(1 for ep in episodes if ep.get("is_success", False))
        total = len(episodes)

        sr = successes / total if total > 0 else 0
        mean_ret = np.mean(returns) if returns else 0
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0

        row = {
            "Method": method,
            "Success Rate": f"{sr:.2%}",
            "Mean Return": f"{mean_ret:.2f} ± {std_ret:.2f}",
            "N Episodes": total,
        }

        if baseline is not None and method != baseline["method"]:
            method_by_seed = {
                (ep.get("scenario"), ep.get("seed")): ep.get("return", 0)
                for ep in episodes
            }
            baseline_by_seed = {
                (ep.get("scenario"), ep.get("seed")): ep.get("return", 0)
                for ep in baseline["episodes"]
            }
            common_keys = sorted(
                set(method_by_seed.keys()) & set(baseline_by_seed.keys())
            )
            if common_keys:
                a_vals = [baseline_by_seed[k] for k in common_keys]
                b_vals = [method_by_seed[k] for k in common_keys]
                ttest = paired_t_test(a_vals, b_vals)
                d = cohens_d(a_vals, b_vals)
                sig = "*" if ttest["significant_at_05"] else ""
                row["vs Baseline p"] = f"{ttest['p_value']:.4f}{sig}"
                row["Cohen's d"] = f"{d['d']:.3f} ({d['magnitude']})"

        rows.append(row)
        if baseline is None:
            baseline = r

    df = pd.DataFrame(rows)
    md_path = output_dir / "tables" / "comparison_table.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join([" --- " for _ in cols]) + "|"
    lines = [header, separator]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return md_path, csv_path


def generate_raw_csv(results: list, output_dir: Path):
    """Export one row per episode as raw data CSV with full telemetry."""
    # Define the full set of fields we want to preserve for analysis.
    # This must stay in sync with evaluate_single_episode return dict.
    _fields = [
        "method",
        "scenario",
        "seed",
        "return",
        "length",
        "is_success",
        "is_crash",
        "is_timeout",
        "is_out_of_bounds",
        "reason",
        "min_range_m",
        "min_ata_deg",
        "final_range_m",
        "final_ata_deg",
        "score_win",
        "mode_switch_effective",
        "effective_guidance_mode",
        "prediction_enabled_rate",
        "prediction_valid_rate",
        "prediction_fallback_rate",
        "warmup_fallback_rate",
        "runtime_fallback_rate",
        "post_warmup_fallback_rate",
        "predictor_init_failed_count",
        "unknown_fallback_phase_count",
        "missing_fallback_phase_count",
        "configured_current_target_fallback_count",
        "mean_env_prediction_error_m",
        "median_env_prediction_error_m",
        "env_prediction_error_count",
        "mean_offline_aligned_error_m",
        "median_offline_aligned_error_m",
        "offline_aligned_error_count",
        "mean_prediction_error_m",
        "median_prediction_error_m",
        "prediction_error_count",
        "mean_virtual_point_shift_m",
        "mean_anchor_shift_m",
        "time_to_first_advantage_s",
        "advantage_hold_time_s",
        "nz_cmd_max",
        "nz_cmd_mean",
        "nz_cmd_saturation_rate",
        "nz_cmd_modification_rate",
        "roll_rate_cmd_max",
        "roll_rate_cmd_mean",
        "roll_rate_cmd_saturation_rate",
        "roll_rate_cmd_modification_rate",
        "throttle_cmd_max",
        "throttle_cmd_mean",
        "throttle_cmd_saturation_rate",
        "throttle_cmd_modification_rate",
        "min_altitude_m",
        "max_altitude_m",
        "final_altitude_m",
        "altitude_loss_rate",
        "energy_proxy",
    ]
    rows = []
    for r in results:
        method = r["method"]
        for ep in r["episodes"]:
            row = {"method": method}
            for key in _fields[1:]:
                val = ep.get(key)
                if val is None:
                    val = float("nan")
                row[key] = val
            rows.append(row)
    if rows:
        import csv
        raw_path = output_dir / "raw_episodes.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {raw_path}")
        return raw_path
    return None


# ---------------------------------------------------------------------------
# Summary and manifest
# ---------------------------------------------------------------------------
def generate_summary(
    results: list,
    output_dir: Path,
    backend: str,
    args: argparse.Namespace,
    manifest_path: Path,
):
    """Generate summary.md report with full reproducibility metadata."""
    git_info = _get_git_info()
    lines = [
        "# UAV VPP Guidance — Paper Benchmark Report",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Git Commit**: `{git_info['commit']}` (dirty={git_info['dirty']}, branch={git_info['branch']})",
        f"**Backend**: {backend}",
        f"**Config**: {args.config}",
        f"**Methods**: {', '.join(r['method'] for r in results)}",
        f"**Scenarios**: {args.scenarios}",
        f"**Seeds**: {args.seeds}",
        f"**Allow Random Smoke**: {args.allow_random_smoke}",
        f"**Run Manifest**: `{manifest_path}`",
        "",
        "## Benchmark Type",
        "",
    ]

    any_invalid = any(r["metadata"]["invalid_for_paper"] for r in results)
    if any_invalid:
        lines.append("- ⚠️ **SMOKE BENCHMARK**: At least one method uses random policy or missing/invalid gains.")
        lines.append("- **NOT PAPER-SAFE**: Do not use these results for paper claims.")
    else:
        lines.append("- ✅ **PAPER-SAFE BENCHMARK**: All checkpoints and gains loaded successfully.")
    lines.append("")

    lines.append("## Results Summary")
    lines.append("")

    for r in results:
        method = r["method"]
        meta = r["metadata"]
        episodes = r["episodes"]
        successes = sum(1 for ep in episodes if ep.get("is_success", False))
        total = len(episodes)
        sr = successes / total if total > 0 else 0
        lines.append(f"### {method}")
        lines.append(f"- Success Rate: {sr:.2%} ({successes}/{total})")
        lines.append(f"- Config Path: {meta['config_path']}")
        lines.append(f"- Resolved Config Hash: {meta['resolved_config_hash']}")
        lines.append(f"- Method Override: {meta['method_override_name']}")
        lines.append(f"- Prediction Mode: {meta['prediction_mode']}")
        lines.append(f"- Guidance Mode: {meta['guidance_mode']}")
        lines.append(f"- Gain Source: {meta['gain_source']}")
        lines.append(f"- Checkpoint Path (Final): {meta['checkpoint_path_final']}")
        lines.append(f"- Checkpoint Source: {meta['checkpoint_source']}")
        lines.append(f"- Checkpoint Exists: {meta['checkpoint_exists']}")
        if meta.get("gains_path"):
            lines.append(f"- Gains Path: {meta['gains_path']}")
            lines.append(f"- Gains Exists: {meta['gains_exists']}")
            lines.append(f"- Gains Schema Valid: {meta['gains_schema_valid']}")
            if meta.get("loaded_gains"):
                lines.append(f"- Loaded Gains: {meta['loaded_gains']}")
            if meta.get("ignored_gain_fields"):
                lines.append(f"- Ignored Gain Fields: {meta['ignored_gain_fields']}")
        if meta["invalid_for_paper"]:
            lines.append(
                f"- ⚠️ **INVALID FOR PAPER**: {', '.join(meta['invalid_for_paper_reasons'])}"
            )
        if meta.get("note"):
            lines.append(f"- Note: {meta['note']}")
        lines.append("")

    lines.extend(
        [
            "## Statistical Comparison",
            "See `tables/comparison_table.md` for paired t-test and Cohen's d.",
            "",
            "## Figures",
            "See `figures/` directory.",
            "",
            "## Reproducibility",
            "Exact command used to produce this run:",
            "",
            "```bash",
            shlex.join(sys.argv),
            "```",
            "",
        ]
    )

    summary_path = output_dir / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {summary_path}")

    return summary_path


def generate_manifest(
    results: list,
    output_dir: Path,
    args: argparse.Namespace,
    start_time: str,
    end_time: str,
) -> Path:
    """Generate run_manifest.json with full provenance and paper-safe audit."""
    git_info = _get_git_info()

    method_entries = {}
    invalid_reasons = []
    for r in results:
        m = r["metadata"]
        method_entries[m["method"]] = {
            "checkpoint_path_final": m["checkpoint_path_final"],
            "checkpoint_source": m["checkpoint_source"],
            "checkpoint_exists": m["checkpoint_exists"],
            "gains_path": m["gains_path"],
            "gains_exists": m["gains_exists"],
            "gains_schema_valid": m["gains_schema_valid"],
            "loaded_gains": m["loaded_gains"],
            "ignored_gain_fields": m["ignored_gain_fields"],
            "resolved_config_hash": m["resolved_config_hash"],
            "invalid_for_paper": m["invalid_for_paper"],
            "invalid_for_paper_reasons": m["invalid_for_paper_reasons"],
        }
        if m["invalid_for_paper"]:
            invalid_reasons.append(
                {
                    "method": m["method"],
                    "reasons": m["invalid_for_paper_reasons"],
                }
            )

    manifest = {
        "start_time": start_time,
        "end_time": end_time,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "git_branch": git_info["branch"],
        "command_line": sys.argv,
        "config_path": args.config,
        "backend": args.backend,
        "methods": [r["method"] for r in results],
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "output_dir": str(output_dir),
        "allow_random_smoke": args.allow_random_smoke,
        "paper_safe": not any(r["metadata"]["invalid_for_paper"] for r in results),
        "invalid_for_paper_reasons": invalid_reasons,
        "method_provenance": method_entries,
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(serialize(manifest), f, indent=2, ensure_ascii=False)
    print(f"Saved: {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment/stage6f5_feasible_geometry.yaml",
    )
    parser.add_argument(
        "--backend", type=str, default="simple", choices=["simple", "jsbsim"]
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="regression",
        choices=["regression", "candidate", "all"],
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/paper_benchmark"
    )
    parser.add_argument(
        "--allow-random-smoke",
        action="store_true",
        help="Allow evaluation with missing checkpoints or gains. Results will be marked invalid_for_paper.",
    )
    parser.add_argument(
        "--allow-missing-methods",
        action="store_true",
        help="Skip unknown methods instead of raising an error.",
    )
    parser.add_argument(
        "--checkpoint-map",
        type=str,
        action="append",
        default=[],
        help="Per-method checkpoint override, e.g. no_prediction=path/to/best.pt. Can be repeated.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full config once for checkpoint resolution and provenance
    full_config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )

    # Parse per-method checkpoint map
    checkpoint_map = {}
    for mapping in args.checkpoint_map:
        if "=" not in mapping:
            raise ValueError(
                f"Invalid --checkpoint-map value '{mapping}'. Expected format: method=path"
            )
        method_key, ckpt_path = mapping.split("=", 1)
        checkpoint_map[method_key.strip()] = ckpt_path.strip()

    initialize_canonical_scenarios()

    if args.scenarios == "regression":
        scenarios = ScenarioRegistry.get_regression_suite()
    elif args.scenarios == "candidate":
        scenarios = ScenarioRegistry.get_candidate_suite()
    else:
        scenarios = (
            ScenarioRegistry.get_regression_suite()
            + ScenarioRegistry.get_candidate_suite()
        )

    methods_to_run = args.methods or list(METHODS.keys())

    # Validate that gain_only and no_prediction are semantically different
    if "gain_only" in methods_to_run and "no_prediction" in methods_to_run:
        gain_cfg = METHODS["gain_only"]
        no_pred_cfg = METHODS["no_prediction"]
        if gain_cfg.get("gains_path") == no_pred_cfg.get("gains_path") and not gain_cfg.get("note"):
            raise ValueError(
                "gain_only and no_prediction must have distinct configuration. "
                "gain_only requires a gains_path or note to differentiate from no_prediction."
            )

    start_time = datetime.now(timezone.utc).isoformat()
    results = []
    for method_name in methods_to_run:
        if method_name not in METHODS:
            msg = f"Unknown method '{method_name}'. Available: {list(METHODS.keys())}"
            if args.allow_missing_methods:
                print(f"WARNING: {msg}, skipping")
                continue
            raise ValueError(msg)
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        result = evaluate_method(
            method_name,
            METHODS[method_name],
            scenarios,
            tuple(args.seeds),
            args.backend,
            config_path=args.config,
            full_config=full_config,
            checkpoint_map=checkpoint_map,
            allow_random_smoke=args.allow_random_smoke,
        )
        results.append(result)
        sr = sum(1 for ep in result["episodes"] if ep.get("is_success", False)) / max(
            1, len(result["episodes"])
        )
        print(f"Success Rate: {sr:.2%}")

    end_time = datetime.now(timezone.utc).isoformat()

    # Generate outputs
    print(f"\n{'='*50}")
    print("Generating figures, tables, summary, and manifest...")
    generate_figures(results, output_dir)
    generate_tables(results, output_dir)
    generate_raw_csv(results, output_dir)
    manifest_path = generate_manifest(results, output_dir, args, start_time, end_time)
    generate_summary(results, output_dir, args.backend, args, manifest_path)

    print(f"\n{'='*50}")
    print("Benchmark Complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
