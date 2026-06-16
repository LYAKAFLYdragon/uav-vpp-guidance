#!/usr/bin/env python3
"""
P0 Fair Comparison: VPP vs No-VPP vs E2E across 4 scenarios on JSBSim backend.

Architecture:
  VPP (5 seeds)     — hierarchical: PPO → VPP offset → LOS-rate guidance → controls
  No-VPP (5 seeds)  — hierarchical with zero-offset: PPO → [0,0,0] → LOS-rate guidance → controls
  E2E (2 seeds × 4 step counts) — flat: PPO → direct [nz, roll_rate, throttle]

All methods evaluated on the same 4 canonical scenarios:
  favorable, neutral, disadvantage, challenging

Statistical outputs (per-scenario + aggregated):
  - Success rate with bootstrap 95% CI
  - Crash rate, timeout rate, OOB rate
  - Paired t-test (VPP vs No-VPP) across 5 training-seed pairs
  - Cohen's d effect size
  - LaTeX-formatted summary table

Usage:
    # Full evaluation (JSBSim, 20 episodes per scenario per seed):
    python scripts/run_fair_comparison.py --backend jsbsim --episodes-per-scenario 20

    # Quick smoke test (simple, 2 episodes per scenario per seed):
    python scripts/run_fair_comparison.py --backend simple --episodes-per-scenario 2

    # E2E only (skip VPP and No-VPP):
    python scripts/run_fair_comparison.py --methods e2e --steps 200000 500000
"""

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import (
    evaluate_single_episode,
)
from uav_vpp_guidance.evaluation.statistical_comparison import (
    bootstrap_success_rate_ci,
    bootstrap_ci,
    paired_t_test,
    cohens_d,
    mcnemar_exact_pvalue,
)
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENARIO_NAMES = ["favorable", "neutral", "disadvantage", "challenging"]

# Default checkpoint directories (post-training, pre-evaluation)
CHECKPOINT_ROOTS = {
    "vpp": _PROJECT_ROOT / "outputs/validate_vpp_5seed",
    "no_vpp": _PROJECT_ROOT / "outputs/experiments/no_vpp_validate_5seed",
    "e2e": _PROJECT_ROOT / "outputs/validate_e2e_extended",
}

# Training seeds available for each method
TRAINING_SEEDS = {
    "vpp": [0, 1, 2, 3, 4],
    "no_vpp": [0, 1, 2, 3, 4],
    "e2e": [0, 1],
}

# E2E step counts trained
E2E_STEP_COUNTS = [200000, 500000, 1000000, 2000000]

# Config paths for each method (used to construct env during evaluation)
CONFIG_PATHS = {
    "vpp": _PROJECT_ROOT / "config/experiment/train_no_prediction_vpp_ppo.yaml",
    "no_vpp": _PROJECT_ROOT / "config/experiment/train_no_vpp_ppo.yaml",
    "e2e": _PROJECT_ROOT / "config/experiment/train_end_to_end_ppo.yaml",
}

# Default output
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs/fair_comparison"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _resolve_config(config_path: Path) -> dict:
    """Load a YAML config and resolve its includes."""
    cfg = load_yaml_config(str(config_path))
    includes = cfg.pop("includes", [])
    merged: dict = {}
    for inc_path in includes:
        inc_full = Path(config_path).parent / inc_path
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(str(inc_full)))
    return merge_config(merged, cfg)


def _make_env_config(config: dict, backend: str) -> dict:
    """Prepare a config dict for evaluation: set backend, disable mode-switch."""
    cfg = copy.deepcopy(config)
    cfg["backend"] = backend
    if "env" not in cfg:
        cfg["env"] = {}
    cfg["env"]["backend"] = backend
    cfg["env"]["use_jsbsim"] = (backend == "jsbsim")
    # Disable mode-switch for clean comparison
    if "guidance" not in cfg:
        cfg["guidance"] = {}
    if "mode_switch" not in cfg["guidance"]:
        cfg["guidance"]["mode_switch"] = {}
    cfg["guidance"]["mode_switch"]["enabled"] = False
    # Disable eval-time domain randomization for deterministic comparison
    if "evaluation" not in cfg:
        cfg["evaluation"] = {}
    cfg["evaluation"]["domain_rand_scale"] = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
def _get_checkpoint(method: str, seed: int, step_count: Optional[int] = None) -> Path:
    """Resolve checkpoint path for a given method, training seed, and step count."""
    if method == "e2e":
        if step_count is None:
            step_count = E2E_STEP_COUNTS[0]
        ckpt = (CHECKPOINT_ROOTS["e2e"]
                / f"e2e_steps_{step_count}"
                / f"seed_{seed}"
                / "checkpoints" / "best.pt")
    elif method == "vpp":
        ckpt = (CHECKPOINT_ROOTS["vpp"]
                / f"seed_{seed}"
                / "checkpoints" / "best.pt")
    elif method == "no_vpp":
        ckpt = (CHECKPOINT_ROOTS["no_vpp"]
                / f"seed_{seed}"
                / "checkpoints" / "best.pt")
    else:
        raise ValueError(f"Unknown method: {method}")
    return ckpt


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_method(
    method: str,
    config: dict,
    checkpoint_path: Path,
    training_seed: int,
    eval_seeds: List[int],
    scenarios: List[dict],
    backend: str,
    step_count: Optional[int] = None,
) -> List[dict]:
    """Evaluate a single checkpoint across all scenarios and eval seeds."""
    env_cfg = _make_env_config(config, backend)
    env = CloseRangeTrackingEnv(env_cfg)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = env_cfg.get("policy", {}).get("action_dim", 3)

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=env_cfg, device="cpu")
    if checkpoint_path.exists():
        agent.load(str(checkpoint_path))
    else:
        msg = f"Checkpoint not found: {checkpoint_path}"
        print(f"  ERROR: {msg}")
        env.close()
        raise FileNotFoundError(msg)

    episodes = []
    for scen_idx, scenario in enumerate(scenarios):
        scen_name = scenario.get("name", f"unknown_{scen_idx}")
        for es_idx, eval_seed in enumerate(eval_seeds):
            # Derive a deterministic episode seed from scenario × eval_seed index
            episode_seed = scen_idx * 100000 + es_idx * 1000 + training_seed
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=env_cfg,
                scenario=scenario,
                seed=episode_seed,
                save_trajectory=False,
                method_name=method,
            )
            result["method"] = method
            result["training_seed"] = training_seed
            result["eval_seed"] = eval_seed
            result["scenario"] = scen_name
            if step_count is not None:
                result["step_count"] = step_count
            episodes.append(result)

    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------
def _safe_mean(values: list) -> float:
    clean = [v for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else np.nan


def aggregate_per_scenario(episodes: List[dict]) -> Dict[str, dict]:
    """Aggregate metrics per scenario. Returns {scenario_name: metrics_dict}."""
    by_scenario: Dict[str, List[dict]] = {}
    for ep in episodes:
        by_scenario.setdefault(ep["scenario"], []).append(ep)

    result = {}
    for scen_name, eps in by_scenario.items():
        n = len(eps)
        outcomes = [1 if e["is_success"] else 0 for e in eps]
        sr, sr_lo, sr_hi = bootstrap_success_rate_ci(outcomes, n_bootstrap=2000)
        returns = [e["return"] for e in eps]
        mean_ret, ret_lo, ret_hi = bootstrap_ci(returns, n_bootstrap=2000)
        result[scen_name] = {
            "n_episodes": n,
            "success_rate": sr,
            "success_rate_ci_lo": sr_lo,
            "success_rate_ci_hi": sr_hi,
            "crash_rate": _safe_mean([1.0 if e["is_crash"] else 0.0 for e in eps]),
            "timeout_rate": _safe_mean([1.0 if e["is_timeout"] else 0.0 for e in eps]),
            "oob_rate": _safe_mean([1.0 if e["is_out_of_bounds"] else 0.0 for e in eps]),
            "mean_return": mean_ret,
            "mean_return_ci_lo": ret_lo,
            "mean_return_ci_hi": ret_hi,
            "mean_final_range_m": _safe_mean([e["final_range_m"] for e in eps]),
            "mean_final_ata_deg": _safe_mean([e["final_ata_deg"] for e in eps]),
            "mean_min_range_m": _safe_mean([e["min_range_m"] for e in eps]),
            "mean_min_ata_deg": _safe_mean([e["min_ata_deg"] for e in eps]),
            "mean_virtual_point_shift_m": _safe_mean(
                [e.get("mean_virtual_point_shift_m", np.nan) for e in eps]
            ),
        }
    return result


def aggregate_per_training_seed(episodes: List[dict]) -> Dict[int, dict]:
    """Aggregate metrics per training seed (across all scenarios)."""
    by_seed: Dict[int, List[dict]] = {}
    for ep in episodes:
        by_seed.setdefault(ep["training_seed"], []).append(ep)

    result = {}
    for seed, eps in by_seed.items():
        outcomes = [1 if e["is_success"] else 0 for e in eps]
        sr, sr_lo, sr_hi = bootstrap_success_rate_ci(outcomes, n_bootstrap=2000)
        result[seed] = {
            "n_episodes": len(eps),
            "success_rate": sr,
            "success_rate_ci_lo": sr_lo,
            "success_rate_ci_hi": sr_hi,
            "crash_rate": _safe_mean([1.0 if e["is_crash"] else 0.0 for e in eps]),
            "timeout_rate": _safe_mean([1.0 if e["is_timeout"] else 0.0 for e in eps]),
            "mean_return": _safe_mean([e["return"] for e in eps]),
        }
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def paired_comparison(
    vpp_episodes: List[dict],
    other_episodes: List[dict],
    scenario_name: str,
    metric: str = "success",
) -> dict:
    """Paired comparison of VPP vs another method on one scenario.

    Pairs are across training seeds: for each training seed, the mean metric
    across eval episodes within this scenario forms a single data point.
    """
    def _per_seed_metric(episodes, scen, metric_key):
        by_seed = {}
        for ep in episodes:
            if ep["scenario"] != scen:
                continue
            ts = ep["training_seed"]
            by_seed.setdefault(ts, []).append(ep)
        if metric_key == "success":
            return {ts: _safe_mean([1.0 if e["is_success"] else 0.0 for e in eps])
                    for ts, eps in by_seed.items()}
        elif metric_key == "return":
            return {ts: _safe_mean([e["return"] for e in eps])
                    for ts, eps in by_seed.items()}
        else:
            return {ts: _safe_mean([e.get(metric_key, np.nan) for e in eps])
                    for ts, eps in by_seed.items()}

    vpp_vals = _per_seed_metric(vpp_episodes, scenario_name, metric)
    other_vals = _per_seed_metric(other_episodes, scenario_name, metric)

    common_seeds = sorted(set(vpp_vals.keys()) & set(other_vals.keys()))
    if not common_seeds:
        return {"error": "No common training seeds", "n_pairs": 0}

    a_vals = [vpp_vals[s] for s in common_seeds]
    b_vals = [other_vals[s] for s in common_seeds]

    tt = paired_t_test(a_vals, b_vals)
    cd = cohens_d(a_vals, b_vals)

    return {
        "scenario": scenario_name,
        "metric": metric,
        "n_pairs": len(common_seeds),
        "vpp_mean": float(np.mean(a_vals)),
        "vpp_std": float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else 0.0,
        "other_mean": float(np.mean(b_vals)),
        "other_std": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0,
        "mean_diff": tt["mean_diff"],
        "t_statistic": tt["t_statistic"],
        "p_value": tt["p_value"],
        "significant_at_05": tt["significant_at_05"],
        "significant_at_01": tt["significant_at_01"],
        "cohens_d": cd["d"],
        "d_magnitude": cd["magnitude"],
    }


def mcnemar_comparison(
    vpp_episodes: List[dict],
    other_episodes: List[dict],
    scenario_name: str,
) -> dict:
    """McNemar test on paired binary outcomes (success/failure) across training seeds.

    Within each training seed, count successes and failures.  Pair across seeds:
      b = #seeds where VPP fails and other succeeds
      c = #seeds where VPP succeeds and other fails
    """
    def _success_by_seed(episodes, scen):
        by_seed = {}
        for ep in episodes:
            if ep["scenario"] != scen:
                continue
            ts = ep["training_seed"]
            by_seed.setdefault(ts, {"n": 0, "s": 0})
            by_seed[ts]["n"] += 1
            if ep["is_success"]:
                by_seed[ts]["s"] += 1
        return {ts: v["s"] > v["n"] / 2 for ts, v in by_seed.items()}

    vpp_win = _success_by_seed(vpp_episodes, scenario_name)
    other_win = _success_by_seed(other_episodes, scenario_name)

    common = sorted(set(vpp_win.keys()) & set(other_win.keys()))
    b = sum(1 for s in common if not vpp_win[s] and other_win[s])
    c_val = sum(1 for s in common if vpp_win[s] and not other_win[s])

    p = mcnemar_exact_pvalue(b, c_val)
    return {
        "scenario": scenario_name,
        "n_seeds": len(common),
        "vpp_success_seeds": sum(1 for s in common if vpp_win[s]),
        "other_success_seeds": sum(1 for s in common if other_win[s]),
        "b_vpp_fail_other_success": b,
        "c_vpp_success_other_fail": c_val,
        "p_value": p,
        "significant_at_05": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def generate_latex_table(
    results: Dict[str, dict],
    output_path: Path,
) -> None:
    """Generate a LaTeX-formatted comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Fair Comparison: VPP vs No-VPP vs E2E on JSBSim Backend}",
        r"\label{tab:fair_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Scenario & SR (\%) & Crash (\%) & Timeout (\%) & "
        r"Mean Range (m) & Mean ATA ($^\circ$) \\",
        r"\midrule",
    ]

    method_order = ["vpp", "no_vpp", "e2e"]
    method_labels = {"vpp": "VPP", "no_vpp": "No-VPP", "e2e": "E2E"}

    for method in method_order:
        if method not in results:
            continue
        r = results[method]
        label = method_labels.get(method, method)
        per_scenario = r.get("per_scenario", {})

        for si, scen in enumerate(SCENARIO_NAMES):
            sm = per_scenario.get(scen, {})
            if not sm:
                continue
            sr_str = f"{sm['success_rate']*100:.1f} [{sm['success_rate_ci_lo']*100:.1f}, {sm['success_rate_ci_hi']*100:.1f}]"
            crash_str = f"{sm['crash_rate']*100:.1f}"
            timeout_str = f"{sm['timeout_rate']*100:.1f}"
            range_str = f"{sm['mean_final_range_m']:.0f}"
            ata_str = f"{sm['mean_final_ata_deg']:.1f}"
            prefix = label if si == 0 else ""
            lines.append(
                f"{prefix} & {scen} & {sr_str} & {crash_str} & {timeout_str} & "
                f"{range_str} & {ata_str} \\\\"
            )
        if method != method_order[-1]:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table: {output_path}")


def generate_statistical_table(
    comparisons: List[dict],
    output_path: Path,
) -> None:
    """Generate a LaTeX table of statistical comparisons."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Comparison: Paired t-test and Cohen's d (5 training seeds)}",
        r"\label{tab:statistical}",
        r"\small",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Comparison & Scenario & \Delta SR (pp) & t & p & sig. & d & Magnitude \\",
        r"\midrule",
    ]

    for comp in comparisons:
        delta_pp = comp.get("mean_diff", 0) * 100  # convert to percentage points
        t_val = comp.get("t_statistic", np.nan)
        p_val = comp.get("p_value", np.nan)
        sig = ""
        if comp.get("significant_at_01"):
            sig = "**"
        elif comp.get("significant_at_05"):
            sig = "*"
        d_val = comp.get("cohens_d", np.nan)
        mag = comp.get("d_magnitude", "")

        t_str = f"{t_val:.2f}" if np.isfinite(t_val) else "—"
        p_str = f"{p_val:.4f}" if np.isfinite(p_val) else "—"
        d_str = f"{d_val:.3f}" if np.isfinite(d_val) else "—"

        lines.append(
            f"{comp['comparison']} & {comp['scenario']} & {delta_pp:+.1f} & "
            f"{t_str} & {p_str} & {sig} & {d_str} & {mag} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Statistical table: {output_path}")


def generate_summary_json(
    results: Dict[str, dict],
    comparisons: List[dict],
    output_path: Path,
    meta: dict,
) -> None:
    """Write comprehensive summary JSON."""
    # Convert numpy types
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload = {
        "meta": meta,
        "results": results,
        "statistical_comparisons": comparisons,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2, ensure_ascii=False)
    print(f"  Summary JSON: {output_path}")


def generate_raw_csv(all_episodes: List[dict], output_path: Path) -> None:
    """Write per-episode raw data CSV."""
    # Collect all possible keys
    fieldnames = [
        "method", "training_seed", "step_count", "scenario", "eval_seed",
        "seed", "return", "length",
        "is_success", "is_crash", "is_timeout", "is_out_of_bounds",
        "reason", "final_range_m", "final_ata_deg",
        "min_range_m", "min_ata_deg",
        "mean_virtual_point_shift_m",
        "min_altitude_m", "max_altitude_m", "final_altitude_m",
        "nz_cmd_max", "nz_cmd_mean",
        "roll_rate_cmd_max", "roll_rate_cmd_mean",
        "throttle_cmd_max", "throttle_cmd_mean",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ep in all_episodes:
            writer.writerow(ep)
    print(f"  Raw CSV: {output_path}")


def generate_figures(results: Dict[str, dict], output_dir: Path) -> None:
    """Generate grouped bar charts per scenario."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping figures")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods = ["vpp", "no_vpp", "e2e"]
    method_labels = {"vpp": "VPP", "no_vpp": "No-VPP", "e2e": "E2E"}
    method_colors = {"vpp": "#2E86AB", "no_vpp": "#A23B72", "e2e": "#F18F01"}

    for metric in ["success_rate", "crash_rate", "timeout_rate"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(SCENARIO_NAMES))
        width = 0.25
        multiplier = 0

        for method in methods:
            if method not in results:
                continue
            r = results[method]
            per_scenario = r.get("per_scenario", {})
            values = []
            errors_lo = []
            errors_hi = []
            for scen in SCENARIO_NAMES:
                sm = per_scenario.get(scen, {})
                v = sm.get(metric, 0) * 100
                values.append(v)
                if metric == "success_rate":
                    errors_lo.append(max(0, v - sm.get("success_rate_ci_lo", v) * 100))
                    errors_hi.append(max(0, sm.get("success_rate_ci_hi", v) * 100 - v))
                else:
                    errors_lo.append(0)
                    errors_hi.append(0)

            offset = width * multiplier
            ax.bar(
                x + offset, values, width,
                label=method_labels.get(method, method),
                color=method_colors.get(method, "gray"),
                alpha=0.85,
            )
            if any(e > 0 for e in errors_lo) or any(e > 0 for e in errors_hi):
                ax.errorbar(
                    x + offset, values,
                    yerr=[errors_lo, errors_hi],
                    fmt="none", ecolor="black", capsize=4, linewidth=0.8,
                )
            multiplier += 1

        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
        ax.set_title(f"Fair Comparison: {metric.replace('_', ' ').title()} by Scenario")
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.capitalize() for s in SCENARIO_NAMES])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig_path = fig_dir / f"comparison_{metric}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Figure: {fig_path}")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------
def _discover_checkpoints(
    method: str,
    seeds: List[int],
    step_counts: Optional[List[int]] = None,
) -> List[Tuple[int, Path, Optional[int]]]:
    """Discover available checkpoints. Returns [(training_seed, path, step_count)]."""
    found = []
    if method == "e2e":
        for sc in (step_counts or E2E_STEP_COUNTS):
            for seed in seeds:
                ckpt = _get_checkpoint(method, seed, sc)
                if ckpt.exists():
                    found.append((seed, ckpt, sc))
                else:
                    print(f"  WARNING: Missing checkpoint: {ckpt}")
    else:
        for seed in seeds:
            ckpt = _get_checkpoint(method, seed)
            if ckpt.exists():
                found.append((seed, ckpt, None))
            else:
                print(f"  WARNING: Missing checkpoint: {ckpt}")
    return found


def _resolve_scenarios(config: dict) -> List[dict]:
    """Resolve the 4 canonical scenarios from a config dict."""
    scenarios_cfg = config.get("scenarios", {})
    resolved = []
    for name in SCENARIO_NAMES:
        sc = scenarios_cfg.get(name)
        if sc is not None:
            sc_copy = copy.deepcopy(sc)
            sc_copy.setdefault("name", name)
            resolved.append(sc_copy)
    if not resolved:
        raise ValueError(
            f"Config missing required scenarios: {SCENARIO_NAMES}. "
            f"Available: {list(scenarios_cfg.keys())}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P0 Fair Comparison: VPP vs No-VPP vs E2E"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["vpp", "no_vpp", "e2e"],
        choices=["vpp", "no_vpp", "e2e"],
        help="Methods to evaluate (default: all three)"
    )
    parser.add_argument(
        "--e2e-steps", type=int, nargs="+",
        default=E2E_STEP_COUNTS,
        help=f"E2E step counts to evaluate (default: {E2E_STEP_COUNTS})"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=None,
        help="Training seeds per method (default: uses all discovered; VPP/No-VPP=5, E2E=2)"
    )
    parser.add_argument(
        "--episodes-per-scenario", type=int, default=20,
        help="Independent eval episodes per scenario per training seed (default: 20)"
    )
    parser.add_argument(
        "--backend", type=str, default="jsbsim",
        choices=["simple", "jsbsim"],
        help="Simulation backend (default: jsbsim)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip evaluation if output files already exist"
    )
    parser.add_argument(
        "--vpp-config", type=str, default=str(CONFIG_PATHS["vpp"]),
        help="VPP config YAML path"
    )
    parser.add_argument(
        "--no-vpp-config", type=str, default=str(CONFIG_PATHS["no_vpp"]),
        help="No-VPP config YAML path"
    )
    parser.add_argument(
        "--e2e-config", type=str, default=str(CONFIG_PATHS["e2e"]),
        help="E2E config YAML path"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"Output already exists: {summary_path}. Use --no-skip-existing to rerun.")
        return 0

    start_time = datetime.now(timezone.utc).isoformat()

    # Resolve seeds
    seeds_map = {}
    for method in args.methods:
        if args.seeds:
            seeds_map[method] = args.seeds
        else:
            seeds_map[method] = TRAINING_SEEDS[method]

    # Load configs
    configs = {}
    scenario_names = None
    for method in args.methods:
        cp = Path(getattr(args, f"{method.replace('-', '_')}_config"))
        cfg = _resolve_config(cp)
        configs[method] = cfg
        # Verify scenarios are consistent
        sc_names = [s.get("name", "") for s in cfg.get("scenarios", {}).values()]
        if scenario_names is None:
            scenario_names = sc_names
        elif sc_names != scenario_names:
            print(f"WARNING: Scenario mismatch: {method} has {sc_names}, expected {scenario_names}")

    scenarios = _resolve_scenarios(configs[args.methods[0]])

    # Eval seeds: n_eps independent seeds, each reused across all scenarios
    n_eps = args.episodes_per_scenario
    eval_seeds = list(range(n_eps))

    # -----------------------------------------------------------------------
    # Evaluate each method
    # -----------------------------------------------------------------------
    all_episodes: List[dict] = []
    per_method_episodes: Dict[str, List[dict]] = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'='*60}")

        config = configs[method]
        step_counts = args.e2e_steps if method == "e2e" else None
        checkpoints = _discover_checkpoints(method, seeds_map[method], step_counts)

        if not checkpoints:
            print(f"  No checkpoints found for {method}, skipping.")
            continue

        method_episodes: List[dict] = []
        for training_seed, ckpt_path, step_count in checkpoints:
            label = f"seed={training_seed}"
            if step_count is not None:
                label += f", steps={step_count}"
            print(f"  [{label}] Loading {ckpt_path} ...")
            t0 = time.perf_counter()
            episodes = evaluate_method(
                method=method,
                config=config,
                checkpoint_path=ckpt_path,
                training_seed=training_seed,
                eval_seeds=eval_seeds,
                scenarios=scenarios,
                backend=args.backend,
                step_count=step_count,
            )
            elapsed = time.perf_counter() - t0
            successes = sum(1 for e in episodes if e["is_success"])
            crashes = sum(1 for e in episodes if e["is_crash"])
            timeouts = sum(1 for e in episodes if e["is_timeout"])
            total = len(episodes)
            print(f"    {total} episodes in {elapsed:.1f}s | "
                  f"SR={successes/total:.1%} | Crash={crashes/total:.1%} | TO={timeouts/total:.1%}")
            method_episodes.extend(episodes)

        all_episodes.extend(method_episodes)
        per_method_episodes[method] = method_episodes

    # -----------------------------------------------------------------------
    # Aggregate results per method
    # -----------------------------------------------------------------------
    results = {}
    for method in args.methods:
        episodes = per_method_episodes.get(method, [])
        if not episodes:
            continue
        per_scenario = aggregate_per_scenario(episodes)
        per_seed = aggregate_per_training_seed(episodes)
        results[method] = {
            "per_scenario": per_scenario,
            "per_training_seed": per_seed,
            "total_episodes": len(episodes),
        }

    # -----------------------------------------------------------------------
    # Statistical comparisons (VPP as baseline)
    # -----------------------------------------------------------------------
    comparisons: List[dict] = []
    vpp_eps = per_method_episodes.get("vpp", [])
    for other in ["no_vpp", "e2e"]:
        other_eps = per_method_episodes.get(other, [])
        if not vpp_eps or not other_eps:
            continue
        for scen in SCENARIO_NAMES:
            comp = paired_comparison(vpp_eps, other_eps, scen, metric="success")
            comp["comparison"] = f"VPP vs {other.upper()}"
            comparisons.append(comp)
            # Also compare returns
            comp_ret = paired_comparison(vpp_eps, other_eps, scen, metric="return")
            comp_ret["comparison"] = f"VPP vs {other.upper()}"
            comparisons.append(comp_ret)

    # McNemar tests
    for other in ["no_vpp", "e2e"]:
        other_eps = per_method_episodes.get(other, [])
        if not vpp_eps or not other_eps:
            continue
        for scen in SCENARIO_NAMES:
            mc = mcnemar_comparison(vpp_eps, other_eps, scen)
            mc["comparison"] = f"VPP vs {other.upper()} (McNemar)"
            comparisons.append(mc)

    # -----------------------------------------------------------------------
    # Generate outputs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Generating outputs...")
    print(f"{'='*60}")

    end_time = datetime.now(timezone.utc).isoformat()
    meta = {
        "start_time": start_time,
        "end_time": end_time,
        "backend": args.backend,
        "episodes_per_scenario": args.episodes_per_scenario,
        "methods": args.methods,
        "seeds_used": seeds_map,
        "e2e_step_counts": args.e2e_steps,
    }

    generate_raw_csv(all_episodes, output_dir / "raw_episodes.csv")
    generate_summary_json(results, comparisons, summary_path, meta)
    generate_latex_table(results, output_dir / "table.tex")
    generate_statistical_table(comparisons, output_dir / "statistical_tests.tex")
    generate_figures(results, output_dir)

    # -----------------------------------------------------------------------
    # Print quick summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("QUICK SUMMARY")
    print(f"{'='*60}")
    for method in args.methods:
        r = results.get(method, {})
        if not r:
            continue
        per_scenario = r.get("per_scenario", {})
        print(f"\n--- {method.upper()} ---")
        for scen in SCENARIO_NAMES:
            sm = per_scenario.get(scen, {})
            if not sm:
                continue
            sr = sm['success_rate']
            sr_lo = sm['success_rate_ci_lo']
            sr_hi = sm['success_rate_ci_hi']
            print(f"  {scen:15s}: SR={sr:.1%} [{sr_lo:.1%}, {sr_hi:.1%}] "
                  f"Crash={sm['crash_rate']:.1%} TO={sm['timeout_rate']:.1%} "
                  f"Range={sm['mean_final_range_m']:.0f}m ATA={sm['mean_final_ata_deg']:.1f}°")

    print(f"\n--- Statistical Highlights ---")
    for comp in comparisons:
        if comp.get("significant_at_05") and comp.get("metric") == "success":
            delta_pp = comp.get("mean_diff", 0) * 100
            direction = "higher" if delta_pp > 0 else "lower"
            print(f"  ** {comp['comparison']} / {comp['scenario']}: "
                  f"ΔSR={delta_pp:+.1f}pp ({direction}), "
                  f"p={comp['p_value']:.4f}, d={comp['cohens_d']:.3f}")

    print(f"\nAll outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
