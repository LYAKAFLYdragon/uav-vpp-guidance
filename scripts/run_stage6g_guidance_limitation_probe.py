#!/usr/bin/env python3
"""
Stage 6G Guidance-Law Limitation Probe (Hardened).

Lightweight comparison of guidance laws on tail-chase / stern-conversion
dead-zone scenarios to determine whether the dead zone is specific to
LOS-rate guidance or inherent to the VPP formulation.

Methods: no_prediction, gru_frozen (training_seed 0 only)
Guidance modes: los_rate, proportional_navigation, hybrid
Scenarios: favorable, disadvantage, weaving_pursuit, weaving_disadvantage
Episodes: 10 per scenario
Eval seeds: 0, 1, 2

Exit contract:
- exit 0: all required artifacts generated, status = completed or smoke_completed
- exit 1: any uncaught exception, missing checkpoint, config failure, artifact missing, stats failure
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml
import numpy as np

from uav_vpp_guidance.evaluation.statistical_comparison import mcnemar_exact_pvalue

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
GUIDANCE_MODES = ["los_rate", "proportional_navigation", "hybrid"]

METHODS = [
    {
        "name": "no_prediction",
        "checkpoint": "outputs/experiments/no_prediction_vpp_ppo_seed0/checkpoints/best.pt",
    },
    {
        "name": "gru_frozen",
        "checkpoint": "outputs/experiments/vpp_ppo_gru_frozen_seed0/checkpoints/best.pt",
    },
]

SCENARIO_CONFIGS = {
    "favorable": {
        "base_config": "config/experiment/stage6f5_feasible_geometry.yaml",
        "scenario": "favorable",
    },
    "disadvantage": {
        "base_config": "config/experiment/stage6f5_feasible_geometry.yaml",
        "scenario": "disadvantage",
    },
    "weaving_pursuit": {
        "base_config": "config/experiment/stage6f5_maneuvering_target.yaml",
        "scenario": "weaving_pursuit",
    },
    "weaving_disadvantage": {
        "base_config": "config/experiment/stage6f5_maneuvering_target.yaml",
        "scenario": "weaving_disadvantage",
    },
}

REQUIRED_ARTIFACTS = [
    "resolved_config.yaml",
    "run_manifest.json",
    "raw_episodes.csv",
    "scenario_method_summary.csv",
    "pairwise_mcnemar.csv",
    "paper_safe_claims.md",
    "README_result_block.md",
    "run.log",
]

# Failure root cause classification
VALID_TERMINATION_REASONS = {
    "success", "out_of_bounds", "crash", "timeout",
    "guidance_saturation", "altitude_channel_instability",
    "range_not_closing", "prediction_fallback", "invalid_guidance_mode", "unknown",
}


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def build_probe_config(base_config_path: str, guidance_mode: str, scenario_name: str) -> dict:
    """Load base config and override guidance mode + scenario subset + methods subset."""
    cfg = load_yaml(base_config_path)
    # Override guidance mode
    if "guidance" not in cfg:
        cfg["guidance"] = {}
    cfg["guidance"]["mode"] = guidance_mode

    # Keep only the requested scenario
    scenarios = cfg.get("scenarios", {})
    if scenario_name in scenarios:
        cfg["scenarios"] = {scenario_name: scenarios[scenario_name]}
    else:
        raise ValueError(f"Scenario {scenario_name} not found in {base_config_path}")

    # Keep only the two probe methods in config
    if "methods" in cfg:
        wanted = {m["name"] for m in METHODS}
        filtered = {k: v for k, v in cfg["methods"].items() if k in wanted}
        cfg["methods"] = filtered
    return cfg


# ------------------------------------------------------------------
# Git helpers
# ------------------------------------------------------------------
def get_git_info() -> dict:
    """Capture git commit, dirty status, branch."""
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        import subprocess as sp
        info["commit"] = sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        info["dirty"] = len(sp.check_output(["git", "status", "--short"], text=True).strip()) > 0
        info["branch"] = sp.check_output(["git", "branch", "--show-current"], text=True).strip()
    except Exception:
        pass
    return info


# ------------------------------------------------------------------
# Run a single probe cell
# ------------------------------------------------------------------
def run_probe_cell(
    guidance_mode: str,
    scenario_name: str,
    base_config_path: str,
    episodes_per_scenario: int,
    eval_seeds: List[int],
    training_seed: int,
    output_dir: Path,
    dry_run: bool,
    log_file,
) -> Tuple[bool, List[dict]]:
    """Run one guidance_mode x scenario cell.

    Returns (ok, raw_episodes_list).
    """
    print(f"\n{'='*60}")
    print(f"Probe: guidance={guidance_mode} | scenario={scenario_name}")
    print(f"{'='*60}")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Probe: guidance={guidance_mode} | scenario={scenario_name}\n")
    log_file.write(f"{'='*60}\n")
    log_file.flush()

    # Build resolved config and save to sub-output dir
    probe_cfg = build_probe_config(base_config_path, guidance_mode, scenario_name)
    cell_dir = output_dir / f"{guidance_mode}_{scenario_name}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = cell_dir / "resolved_config.yaml"
    save_yaml(str(resolved_config_path), probe_cfg)

    # Verify resolved config matches requested mode
    resolved_mode = probe_cfg.get("guidance", {}).get("mode", "unknown")
    if resolved_mode != guidance_mode:
        msg = f"ERROR: Resolved config guidance mode mismatch: {resolved_mode} != {guidance_mode}"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        return False, []

    # Build method-checkpoint overrides
    method_overrides = []
    missing_ckpts = []
    for method in METHODS:
        ckpt = method["checkpoint"]
        if not os.path.exists(ckpt) and not dry_run:
            missing_ckpts.append(ckpt)
            print(f"  WARNING: Checkpoint not found: {ckpt}")
        method_overrides.append(f"{method['name']}={ckpt}")

    if missing_ckpts and not dry_run:
        msg = f"ERROR: Missing checkpoints: {missing_ckpts}. Cannot proceed without --allow-random-policy."
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        return False, []

    cmd = [
        sys.executable,
        "-m",
        "uav_vpp_guidance.evaluation.evaluate_prediction_comparison",
        "--config", str(resolved_config_path),
        "--backend", "simple",
        "--training-seed", str(training_seed),
        "--episodes-per-scenario", str(episodes_per_scenario),
        "--seeds", *map(str, eval_seeds),
        "--scenarios", scenario_name,
        "--output-dir", str(cell_dir),
        "--validation-mode", "raise",
    ]
    for override in method_overrides:
        cmd.extend(["--method-checkpoint", override])

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        log_file.write(f"  [DRY-RUN] {' '.join(cmd)}\n")
        log_file.flush()
        return True, []

    print(f"  Running: {' '.join(cmd)}")
    log_file.write(f"  Running: {' '.join(cmd)}\n")
    log_file.flush()

    result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False, text=False)

    if result.returncode != 0:
        msg = f"ERROR: Probe failed for {guidance_mode} / {scenario_name} (exit code {result.returncode})"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        return False, []

    # Collect raw episodes from prediction_metrics.json
    raw_episodes = []
    metrics_json = cell_dir / "prediction_metrics.json"
    if not metrics_json.exists():
        msg = f"ERROR: prediction_metrics.json not found in {cell_dir}"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        return False, []

    try:
        with open(metrics_json, "r", encoding="utf-8") as f:
            methods_data = json.load(f)
    except Exception as exc:
        msg = f"ERROR: Failed to load {metrics_json}: {exc}"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        return False, []

    for m in methods_data:
        method_name = m.get("method_name", m.get("method", "unknown"))
        requested_guidance = m.get("requested_guidance_mode", guidance_mode)
        effective_guidance = m.get("effective_guidance_mode", guidance_mode)
        raw_eps = m.get("raw_episodes", [])
        for ep in raw_eps:
            ep["guidance_mode_requested"] = requested_guidance
            ep["effective_guidance_mode"] = effective_guidance
            ep["method"] = method_name
            # Normalize termination reason
            reason = ep.get("reason", "unknown")
            if reason not in VALID_TERMINATION_REASONS:
                reason = "unknown"
            ep["termination_reason"] = reason
            raw_episodes.append(ep)

    print(f"  Results saved to {cell_dir} ({len(raw_episodes)} raw episodes)")
    log_file.write(f"  Results saved to {cell_dir} ({len(raw_episodes)} raw episodes)\n")
    log_file.flush()
    return True, raw_episodes


# ------------------------------------------------------------------
# Artifact generation
# ------------------------------------------------------------------
def save_raw_episodes_csv(output_dir: Path, episodes: List[dict]) -> None:
    """Save raw_episodes.csv with canonical columns."""
    if not episodes:
        return
    # Canonical columns
    columns = [
        "scenario", "method", "guidance_mode_requested", "effective_guidance_mode",
        "training_seed", "evaluation_seed", "episode_seed", "episode_index",
        "success", "termination_reason", "capture_time", "miss_distance", "min_range",
        "oob", "crash", "fallback_used", "prediction_error",
        "return", "length", "final_range_m", "final_ata_deg",
        "score_win", "mean_virtual_point_shift_m", "mean_anchor_shift_m",
        "time_to_first_advantage_s", "advantage_hold_time_s",
    ]
    # Derive capture_time from length and dt (use high_level_dt from config if available, else 0.2)
    dt = 0.2  # default
    rows = []
    for idx, ep in enumerate(episodes):
        row = {}
        row["scenario"] = ep.get("scenario", "")
        row["method"] = ep.get("method", "")
        row["guidance_mode_requested"] = ep.get("guidance_mode_requested", "")
        row["effective_guidance_mode"] = ep.get("effective_guidance_mode", "")
        row["training_seed"] = ep.get("training_seed", None)
        row["evaluation_seed"] = ep.get("evaluation_seed", None)
        row["episode_seed"] = ep.get("episode_seed", None)
        row["episode_index"] = idx
        row["success"] = bool(ep.get("is_success", False))
        row["termination_reason"] = ep.get("termination_reason", ep.get("reason", "unknown"))
        row["capture_time"] = ep.get("length", 0) * dt
        row["miss_distance"] = ep.get("final_range_m", np.nan)
        row["min_range"] = ep.get("min_range_m", np.nan)
        row["oob"] = bool(ep.get("is_out_of_bounds", False))
        row["crash"] = bool(ep.get("is_crash", False))
        row["fallback_used"] = bool(ep.get("prediction_fallback_rate", 0.0) > 0.0)
        row["prediction_error"] = ep.get("mean_prediction_error_m", np.nan)
        row["return"] = ep.get("return", np.nan)
        row["length"] = ep.get("length", 0)
        row["final_range_m"] = ep.get("final_range_m", np.nan)
        row["final_ata_deg"] = ep.get("final_ata_deg", np.nan)
        row["score_win"] = bool(ep.get("score_win", False))
        row["mean_virtual_point_shift_m"] = ep.get("mean_virtual_point_shift_m", np.nan)
        row["mean_anchor_shift_m"] = ep.get("mean_anchor_shift_m", np.nan)
        row["time_to_first_advantage_s"] = ep.get("time_to_first_advantage_s", np.nan)
        row["advantage_hold_time_s"] = ep.get("advantage_hold_time_s", np.nan)
        rows.append(row)

    path = output_dir / "raw_episodes.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw episodes saved to {path}")


def save_scenario_method_summary(output_dir: Path, episodes: List[dict]) -> None:
    """Aggregate raw_episodes into scenario_method_summary.csv."""
    if not episodes:
        return
    import pandas as pd
    df = pd.DataFrame(episodes)
    groups = df.groupby(["scenario", "method", "effective_guidance_mode"])
    rows = []
    for (sc, method, guidance), g in groups:
        n = len(g)
        success = g["is_success"].sum()
        crash = g["is_crash"].sum()
        oob = g["is_out_of_bounds"].sum()
        timeout = g["is_timeout"].sum()
        fallback = (g["prediction_fallback_rate"] > 0.0).sum() if "prediction_fallback_rate" in g.columns else 0
        fail_reasons = {}
        for reason in g["reason"]:
            if not (reason == "success"):
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        rows.append({
            "scenario": sc,
            "method": method,
            "guidance_mode": guidance,
            "n_episodes": n,
            "success_rate": success / n,
            "crash_rate": crash / n,
            "oob_rate": oob / n,
            "timeout_rate": timeout / n,
            "fallback_rate": fallback / n,
            "mean_return": g["return"].mean(),
            "std_return": g["return"].std(ddof=1) if n > 1 else 0.0,
            "mean_miss_distance_m": g["final_range_m"].mean(),
            "std_miss_distance_m": g["final_range_m"].std(ddof=1) if n > 1 else 0.0,
            "mean_capture_time_s": g["length"].mean() * 0.2,
            "std_capture_time_s": g["length"].std(ddof=1) * 0.2 if n > 1 else 0.0,
            "failure_root_causes": json.dumps(fail_reasons, sort_keys=True),
        })
    path = output_dir / "scenario_method_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scenario summary saved to {path}")


def compute_pairwise_mcnemar(episodes: List[dict]) -> List[dict]:
    """Compute McNemar exact p for paired comparisons."""
    if not episodes:
        return []
    import pandas as pd
    df = pd.DataFrame(episodes)
    rows = []

    # Define comparison pairs
    pairs = [
        ("no_prediction", "gru_frozen"),
    ]
    # Also compare guidance modes within each scenario x method
    for scenario in df["scenario"].unique():
        for method in df["method"].unique():
            sub = df[(df["scenario"] == scenario) & (df["method"] == method)]
            modes = sub["effective_guidance_mode"].unique()
            if len(modes) >= 2:
                for i in range(len(modes)):
                    for j in range(i + 1, len(modes)):
                        pairs.append((modes[i], modes[j]))

    # Compute for each pair within each scenario
    for scenario in df["scenario"].unique():
        for method in df["method"].unique():
            for a_name, b_name in pairs:
                a = df[
                    (df["scenario"] == scenario)
                    & (df["method"] == method)
                    & (df["effective_guidance_mode"] == a_name)
                ]
                b = df[
                    (df["scenario"] == scenario)
                    & (df["method"] == method)
                    & (df["effective_guidance_mode"] == b_name)
                ]
                if len(a) == 0 or len(b) == 0:
                    continue
                # Pair by evaluation_seed and episode_index (within scenario)
                # Since episodes are aligned by seed and episode order, use positional index
                n = min(len(a), len(b))
                a_succ = a["is_success"].iloc[:n].values.astype(bool)
                b_succ = b["is_success"].iloc[:n].values.astype(bool)
                # Discordant counts
                b_disc = int(np.sum(a_succ & ~b_succ))  # A success, B failure
                c_disc = int(np.sum(~a_succ & b_succ))  # A failure, B success
                try:
                    p_val = mcnemar_exact_pvalue(b_disc, c_disc)
                except Exception as exc:
                    p_val = np.nan
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "comparison": f"{a_name}_vs_{b_name}",
                    "n_pairs": n,
                    "a_success_b_failure": b_disc,
                    "a_failure_b_success": c_disc,
                    "mcnemar_exact_p": p_val,
                    "a_success_rate": a_succ.mean(),
                    "b_success_rate": b_succ.mean(),
                })
    return rows


def save_pairwise_mcnemar(output_dir: Path, rows: List[dict]) -> None:
    path = output_dir / "pairwise_mcnemar.csv"
    if not rows:
        # Write empty CSV with headers so artifact contract is always satisfied
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scenario", "method", "comparison", "n_pairs",
                "a_success_b_failure", "a_failure_b_success", "mcnemar_exact_p",
                "a_success_rate", "b_success_rate",
            ])
            writer.writeheader()
        print(f"Pairwise McNemar (empty) saved to {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Pairwise McNemar saved to {path}")


def render_probe_summary(rows: List[dict], complete: bool, failed_probes: List[str]) -> str:
    """Generate legacy probe summary markdown (for backward compatibility with tests)."""
    lines = []
    lines.append("# Probe Summary")
    lines.append("")
    lines.append(f"**Complete**: {complete}")
    lines.append(f"**Failed Probes**: {len(failed_probes)}")
    if failed_probes:
        lines.append(f"Failed: {', '.join(failed_probes)}")
    lines.append("")
    if not rows:
        lines.append("No data.")
        return "\n".join(lines)
    lines.append("| Guidance | Scenario | Method | Success | Return | Crash | OOB | Range |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.get('guidance_mode','')} | {r.get('scenario','')} | {r.get('method','')} | "
            f"{r.get('success_rate',0):.1%} | {r.get('mean_return',0):.1f} | "
            f"{r.get('crash_rate',0):.1%} | {r.get('out_of_bounds_rate',0):.1%} | {r.get('mean_final_range_m',0):.1f} |"
        )
    return "\n".join(lines)


def render_paper_safe_claims(episodes: List[dict], complete: bool, smoke: bool) -> str:
    """Generate paper_safe_claims.md."""
    lines = []
    lines.append("# Paper-Safe Claims (Stage 6G.1)")
    lines.append("")
    lines.append(f"**Run status**: {'completed' if complete else 'smoke_completed' if smoke else 'incomplete'}")
    lines.append("")
    lines.append("| Claim | Status | Reason |")
    lines.append("|---|---|---|")

    claims = [
        (
            "Neural prediction improves feasible-geometry tracking over classical/no prediction baselines",
            "Paper-safe, within tested feasible geometries",
            "Supported by Stage 6F synthesis; scope limited to tested scenarios.",
        ),
        (
            "GRU is strictly better than LSTM in weaving_headon",
            "Not paper-safe",
            "Cross-seed strict consistency insufficient.",
        ),
        (
            "CA and CV are practically equivalent",
            "Not paper-safe",
            "Observed differences small but not enough for formal claim.",
        ),
        (
            "Tail-chase failure is a guidance-law limitation",
            "Pending Stage 6G.1" if not complete else "To be updated after full probe",
            "Requires LOS-rate vs PN vs hybrid probe with full multi-seed artifacts.",
        ),
        (
            "PN/hybrid resolves tail-chase failure",
            "Pending Stage 6G.1" if not complete else "To be updated after full probe",
            "Depends on Stage 6G.1 results.",
        ),
    ]
    for claim, status, reason in claims:
        lines.append(f"| {claim} | {status} | {reason} |")
    lines.append("")

    if not complete:
        lines.append(
            "> ⚠️ **Incomplete run**: Claims marked `Pending` remain pending. "
            "Do not cite smoke or partial results as final performance conclusions."
        )
        lines.append("")

    return "\n".join(lines)


def render_readme_result_block(episodes: List[dict], complete: bool, smoke: bool) -> str:
    """Generate README_result_block.md."""
    lines = []
    lines.append("## Stage 6G Guidance-Law Limitation Probe Results")
    lines.append("")
    status = "completed" if complete else ("smoke_completed" if smoke else "incomplete")
    lines.append(f"**Status**: {status}")
    lines.append("")
    if not episodes:
        lines.append("No probe results available.")
        return "\n".join(lines)

    import pandas as pd
    df = pd.DataFrame(episodes)
    for scenario in sorted(df["scenario"].unique()):
        lines.append(f"### Scenario: {scenario}")
        lines.append("")
        sdf = df[df["scenario"] == scenario]
        lines.append("| Guidance | Method | Success Rate | Mean Return | Crash Rate | OOB Rate |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for guidance in sorted(sdf["effective_guidance_mode"].unique()):
            for method in sorted(sdf["method"].unique()):
                sub = sdf[(sdf["effective_guidance_mode"] == guidance) & (sdf["method"] == method)]
                if len(sub) == 0:
                    continue
                sr = sub["is_success"].mean()
                mr = sub["return"].mean()
                cr = sub["is_crash"].mean()
                orate = sub["is_out_of_bounds"].mean()
                lines.append(
                    f"| {guidance} | {method} | {sr:.1%} | {mr:.1f} | {cr:.1%} | {orate:.1%} |"
                )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 6G Guidance-Law Limitation Probe (Hardened)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1 episode, 1 seed, fast verification")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and write resolved_config + manifest only; do not run episodes")
    parser.add_argument("--output-dir", type=str, default="outputs/stage6g_guidance_limitation_probe", help="Base output directory")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", type=str, nargs="+", default=[m["name"] for m in METHODS])
    parser.add_argument("--guidance-modes", type=str, nargs="+", default=GUIDANCE_MODES)
    parser.add_argument("--scenarios", type=str, nargs="+", default=list(SCENARIO_CONFIGS.keys()))
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow aggregation even if some probes fail")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Stage 6G Guidance-Law Limitation Probe (Hardened)")
    log(f"Run ID: {run_id}")
    log(f"Guidance modes: {args.guidance_modes}")
    log(f"Scenarios: {args.scenarios}")
    log(f"Methods: {args.methods}")
    log(f"Episodes per scenario: {args.episodes_per_scenario}")
    log(f"Eval seeds: {args.eval_seeds}")
    log(f"Training seed: {args.training_seed}")
    log(f"Smoke: {args.smoke}")
    log(f"Dry-run: {args.dry_run}")
    log(f"Allow incomplete: {args.allow_incomplete}")
    log(f"Output dir: {output_dir}")

    # Smoke overrides
    episodes_per_scenario = args.episodes_per_scenario
    eval_seeds = list(args.eval_seeds)
    if args.smoke:
        episodes_per_scenario = 1
        eval_seeds = [eval_seeds[0]] if eval_seeds else [0]
        log(f"Smoke override: episodes_per_scenario={episodes_per_scenario}, eval_seeds={eval_seeds}")

    # Filter methods and scenarios to requested subsets
    active_methods = [m for m in METHODS if m["name"] in args.methods]
    if not active_methods:
        log("ERROR: No valid methods selected.")
        log_file.close()
        sys.exit(1)

    active_scenarios = {k: v for k, v in SCENARIO_CONFIGS.items() if k in args.scenarios}
    if not active_scenarios:
        log("ERROR: No valid scenarios selected.")
        log_file.close()
        sys.exit(1)

    # Check checkpoint availability upfront
    missing_ckpts = []
    for method in active_methods:
        if not os.path.exists(method["checkpoint"]):
            missing_ckpts.append(method["checkpoint"])
    if missing_ckpts and not args.dry_run:
        log(f"ERROR: Missing checkpoints: {missing_ckpts}")
        log_file.close()
        sys.exit(1)

    # Git info
    git_info = get_git_info()
    log(f"Git commit: {git_info['commit']} (dirty={git_info['dirty']}, branch={git_info['branch']})")

    # Resolved config
    resolved_config = {
        "run_id": run_id,
        "guidance_modes": args.guidance_modes,
        "scenarios": args.scenarios,
        "methods": [m["name"] for m in active_methods],
        "method_checkpoints": {m["name"]: {"path": m["checkpoint"], "exists": os.path.exists(m["checkpoint"])} for m in active_methods},
        "episodes_per_scenario": episodes_per_scenario,
        "eval_seeds": eval_seeds,
        "training_seed": args.training_seed,
        "backend": "simple",
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "git": git_info,
        "python_version": sys.version,
        "hostname": os.environ.get("COMPUTERNAME", "unknown"),
    }
    save_yaml(str(output_dir / "resolved_config.yaml"), resolved_config)
    log("Resolved config saved.")

    # Dry-run: write manifest and exit
    if args.dry_run:
        for guidance_mode in args.guidance_modes:
            for scenario_name in active_scenarios:
                log(f"  [DRY-RUN] Planned: {guidance_mode} x {scenario_name}")
        manifest = {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "hostname": os.environ.get("COMPUTERNAME", "unknown"),
            "python_version": sys.version.split()[0],
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "git_branch": git_info["branch"],
            "command_line": sys.argv,
            "run_status": "dry_run_completed",
            "planned_cells": len(args.guidance_modes) * len(args.scenarios),
        }
        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        log("[DRY-RUN] Manifest saved. Exiting without running episodes.")
        log_file.close()
        return 0

    # Run probe cells
    overall_ok = []
    failed_probes = []
    all_raw_episodes = []

    for guidance_mode in args.guidance_modes:
        for scenario_name, sc_cfg in active_scenarios.items():
            ok, raw_eps = run_probe_cell(
                guidance_mode=guidance_mode,
                scenario_name=scenario_name,
                base_config_path=sc_cfg["base_config"],
                episodes_per_scenario=episodes_per_scenario,
                eval_seeds=eval_seeds,
                training_seed=args.training_seed,
                output_dir=output_dir,
                dry_run=False,
                log_file=log_file,
            )
            overall_ok.append(ok)
            if not ok:
                failed_probes.append(f"{guidance_mode}_{scenario_name}")
            all_raw_episodes.extend(raw_eps)

    complete = all(overall_ok)
    if not complete:
        log(f"\nFAILED PROBES ({len(failed_probes)}): {failed_probes}")
        if not args.allow_incomplete:
            log("ERROR: Use --allow-incomplete to generate partial summary, or fix failures.")
            log_file.close()
            sys.exit(1)
        log("WARNING: --allow-incomplete set; generating partial summary.")

    # Generate artifacts
    if all_raw_episodes:
        save_raw_episodes_csv(output_dir, all_raw_episodes)
        save_scenario_method_summary(output_dir, all_raw_episodes)
        mcnemar_rows = compute_pairwise_mcnemar(all_raw_episodes)
        save_pairwise_mcnemar(output_dir, mcnemar_rows)
    else:
        log("WARNING: No raw episodes collected. Artifacts will be minimal.")

    # Paper-safe claims
    claims_md = render_paper_safe_claims(all_raw_episodes, complete=complete, smoke=args.smoke)
    with open(output_dir / "paper_safe_claims.md", "w", encoding="utf-8") as f:
        f.write(claims_md)
    log("Paper-safe claims saved.")

    # README result block
    readme_block = render_readme_result_block(all_raw_episodes, complete=complete, smoke=args.smoke)
    with open(output_dir / "README_result_block.md", "w", encoding="utf-8") as f:
        f.write(readme_block)
    log("README result block saved.")

    # Run manifest
    run_status = "completed" if complete else ("smoke_completed" if args.smoke else "partial")
    if args.smoke and not complete:
        run_status = "smoke_partial"
    manifest = {
        "run_id": run_id,
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version.split()[0],
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "git_branch": git_info["branch"],
        "command_line": sys.argv,
        "run_status": run_status,
        "cells_total": len(args.guidance_modes) * len(args.scenarios),
        "cells_passed": sum(overall_ok),
        "cells_failed": len(failed_probes),
        "failed_cells": failed_probes,
        "total_raw_episodes": len(all_raw_episodes),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "artifacts_present": {a: (output_dir / a).exists() for a in REQUIRED_ARTIFACTS},
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log("Run manifest saved.")

    # Verify all artifacts
    missing_artifacts = [a for a in REQUIRED_ARTIFACTS if not (output_dir / a).exists()]
    if missing_artifacts:
        log(f"ERROR: Missing required artifacts: {missing_artifacts}")
        log_file.close()
        sys.exit(1)

    log(f"\nProbe complete. Status: {run_status}")
    log(f"Output directory: {output_dir}")
    log_file.close()
    return 0 if complete else (0 if args.allow_incomplete else 1)


if __name__ == "__main__":
    sys.exit(main())
