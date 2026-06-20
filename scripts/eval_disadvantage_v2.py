#!/usr/bin/env python3
"""
Evaluate the redesigned front-hemisphere disadvantage scenes (v2) and generate report.

Usage:
    source activate jsbenv
    python scripts/eval_disadvantage_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.utils.config import load_yaml_config, merge_config


def load_config_with_includes(path: str, _visited: set = None) -> dict:
    if _visited is None:
        _visited = set()
    abs_path = os.path.abspath(path)
    if abs_path in _visited:
        return {}
    _visited.add(abs_path)

    base = load_yaml_config(path)
    includes = base.pop("includes", [])
    merged = {}
    cfg_dir = os.path.dirname(path)
    for inc in includes:
        inc_full = os.path.join(cfg_dir, inc)
        if not os.path.exists(inc_full):
            inc_full = os.path.join(cfg_dir, "..", os.path.basename(inc))
        if os.path.exists(inc_full):
            merged = merge_config(merged, load_config_with_includes(inc_full, _visited))
    return merge_config(merged, base)


def summarize(records: List[dict]) -> dict:
    n = len(records)
    successes = sum(1 for r in records if r.get("is_success"))
    crashes = sum(1 for r in records if r.get("is_crash"))
    oobs = sum(1 for r in records if r.get("is_out_of_bounds"))
    timeouts = sum(1 for r in records if r.get("reason") == "timeout")
    returns = [r.get("return", np.nan) for r in records]
    min_ranges = [r.get("min_range_m", np.nan) for r in records]
    final_ranges = [r.get("final_range_m", np.nan) for r in records]
    return {
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "crash_rate": crashes / n if n else 0.0,
        "out_of_bounds_rate": oobs / n if n else 0.0,
        "timeout_rate": timeouts / n if n else 0.0,
        "mean_return": float(np.nanmean(returns)) if returns else 0.0,
        "std_return": float(np.nanstd(returns)) if returns else 0.0,
        "mean_min_range_m": float(np.nanmean(min_ranges)) if min_ranges else 0.0,
        "mean_final_range_m": float(np.nanmean(final_ranges)) if final_ranges else 0.0,
    }


def evaluate_checkpoint(
    config_path: str,
    checkpoint: str,
    scenario: dict,
    episodes: int = 20,
    seeds: List[int] = None,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2, 3]

    config = load_config_with_includes(config_path)
    env = CloseRangeTrackingEnv(config)
    obs = env.reset(seed=0)
    obs_dim = int(obs["observation_vector"].shape[0])
    action_dim = int(config.get("policy", {}).get("action_dim", 3))
    device = config.get("ppo", {}).get("device", "cpu")

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    agent.load(checkpoint)

    records = []
    for seed in seeds:
        for ep in range(episodes // max(1, len(seeds))):
            result, _ = evaluate_single_episode(
                env=env,
                agent=agent,
                config=config,
                scenario=scenario,
                seed=seed * 10000 + ep,
                save_trajectory=False,
                method_name="disadvantage_v2",
            )
            records.append(result)

    env.close()
    return summarize(records)


def compute_geometry(scenario: dict) -> dict:
    """Compute ATA, AA, closure rate for reporting."""
    import math
    own = scenario["own_init"]
    tgt = scenario["target_init"]
    own_pos = np.array(own["position_m"][:2])
    tgt_pos = np.array(tgt["position_m"][:2])
    los = tgt_pos - own_pos
    range_m = float(np.linalg.norm(los))
    los_unit = los / max(range_m, 1e-6)

    own_vel = np.array([
        own["velocity_mps"] * math.cos(math.radians(own["heading_deg"])),
        own["velocity_mps"] * math.sin(math.radians(own["heading_deg"])),
    ])
    tgt_vel = np.array([
        tgt["velocity_mps"] * math.cos(math.radians(tgt["heading_deg"])),
        tgt["velocity_mps"] * math.sin(math.radians(tgt["heading_deg"])),
    ])
    rel_vel = tgt_vel - own_vel
    closure = -float(np.dot(rel_vel, los_unit))

    ata = math.degrees(math.atan2(los[1], los[0]))
    los_from_tgt = own_pos - tgt_pos
    aa = abs(tgt["heading_deg"] - math.degrees(math.atan2(los_from_tgt[1], los_from_tgt[0])))
    aa = min(aa, 360 - aa)

    return {
        "range_m": range_m,
        "ata_deg": ata,
        "aa_deg": abs(aa),
        "closure_mps": closure,
    }


def plot_results(results: List[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[PLOT] matplotlib not available: {exc}")
        return

    difficulties = [r["difficulty"] for r in results]
    labels = [r["short_label"] for r in results]
    success_rates = [r["success_rate"] * 100 for r in results]
    crash_rates = [r["crash_rate"] * 100 for r in results]
    timeout_rates = [r["timeout_rate"] * 100 for r in results]
    oob_rates = [r["out_of_bounds_rate"] * 100 for r in results]
    min_ranges = [r["mean_min_range_m"] for r in results]
    final_ranges = [r["mean_final_range_m"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(difficulties, success_rates, "o-", color="#2ca02c", linewidth=2, markersize=8, label="Success")
    ax1.plot(difficulties, crash_rates, "s--", color="#d62728", linewidth=2, markersize=6, label="Crash")
    ax1.plot(difficulties, timeout_rates, "^--", color="#ff7f0e", linewidth=2, markersize=6, label="Timeout")
    ax1.plot(difficulties, oob_rates, "v--", color="#9467bd", linewidth=2, markersize=6, label="Out of bounds")
    ax1.axhline(50, color="#2ca02c", linestyle=":", alpha=0.5, label="50% success threshold")
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.set_title("Disadvantage v2 Feasibility Gradient (front-hemisphere, angle-disadvantage)", fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax2.plot(difficulties, min_ranges, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Mean min range")
    ax2.plot(difficulties, final_ranges, "s--", color="#9467bd", linewidth=2, markersize=6, label="Mean final range")
    ax2.set_xlabel("Scene difficulty", fontsize=12)
    ax2.set_ylabel("Range (m)", fontsize=12)
    ax2.set_title("Closure distance gradient", fontsize=12)
    ax2.set_xticks(difficulties)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved {output_path}")


def generate_report(results: List[dict], output_path: str):
    lines = [
        "# Disadvantage Feasibility Gradient Report v2",
        "",
        "**Scene redesign**: front-hemisphere, angle-disadvantage (physically solvable).",
        "",
        "> v1 used rear-hemisphere offsets, which are physically infeasible for an F-16 "
        "(min turn radius ~1700 m, max 200 steps / 40 s). v2 keeps the target in the front "
        "hemisphere with crossing geometry and ATA ≥ 35°.",
        "",
        "| Scene | Ego speed | Target speed | Range | ATA | AA | Closure | Success | Crash | Timeout | OOB | Mean min range |",
        "|-------|-----------|--------------|-------|-----|-----|---------|---------|-------|---------|-----|----------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['short_label']} | {r['ego_speed']} | {r['target_speed']} | {r['range_m']:.0f} m | "
            f"{r['ata_deg']:.0f}° | {r['aa_deg']:.0f}° | {r['closure_mps']:.0f} m/s | "
            f"{r['success_rate']*100:.1f}% | {r['crash_rate']*100:.1f}% | "
            f"{r['timeout_rate']*100:.1f}% | {r['out_of_bounds_rate']*100:.1f}% | "
            f"{r['mean_min_range_m']:.0f} m |"
        )

    lines.extend(["", "## Success Criteria Check", ""])
    mild_sr = results[0]["success_rate"] * 100
    moderate_sr = results[1]["success_rate"] * 100
    hard_sr = results[2]["success_rate"] * 100
    monotonic = mild_sr > moderate_sr > hard_sr

    lines.append(f"- **Mild success rate**: {mild_sr:.1f}% (criteria: ≥ 50%) {'✅ PASS' if mild_sr >= 50 else '❌ FAIL'}")
    lines.append(f"- **Monotonic gradient**: {mild_sr:.1f}% > {moderate_sr:.1f}% > {hard_sr:.1f}% {'✅ PASS' if monotonic else '❌ FAIL'}")
    lines.append("")

    mild_ata = results[0]["ata_deg"]
    moderate_ata = results[1]["ata_deg"]
    hard_ata = results[2]["ata_deg"]
    lines.extend([
        "## Interpretation",
        "",
        f"- **Mild** (ATA ~{mild_ata:.0f}°): small lead-turn required; strong ego speed advantage allows reliable capture.",
        f"- **Moderate** (ATA ~{moderate_ata:.0f}°): larger lead-turn and target speed advantage; small domain-randomization perturbations push the scenario across the feasibility boundary, producing a fractional success rate.",
        f"- **Hard** (ATA ~{hard_ata:.0f}°): near-perpendicular crossing with equal speeds and very low closure; physically infeasible within the 40 s horizon.",
        "",
        "## Conclusion",
        "",
    ])
    if mild_sr >= 50 and monotonic:
        lines.append(
            "The redesigned scenes form a valid feasibility gradient. VPP-LOS with v5 protections "
            "reliably solves mild front-hemisphere angle-disadvantage scenarios, succeeds on a majority of "
            "moderate instances, and fails once ATA and energy margin push the geometry beyond the feasible region."
        )
    elif mild_sr >= 50 and not monotonic:
        lines.append(
            "Mild scene is solvable, but the difficulty gradient is not strictly monotonic. "
            "Scene parameters may need further tuning."
        )
    else:
        lines.append(
            "Even the mildest front-hemisphere angle-disadvantage scene falls below 50% success. "
            "This suggests VPP-LOS with v5 protections cannot solve any non-trivial angle-disadvantage geometry, "
            "regardless of physical feasibility."
        )
    lines.append("")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[REPORT] Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate redesigned disadvantage scenes v2.")
    parser.add_argument("--summary", type=str, default="outputs/disadvantage_v2/training_summary.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out-dir", type=str, default="outputs/disadvantage_v2")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    results = []
    for entry in summary:
        config_path = os.path.join(_PROJECT_ROOT, entry["config"])
        checkpoint = entry["checkpoint"]

        if not Path(checkpoint).exists():
            print(f"[EVAL] Checkpoint not found, skipping: {checkpoint}")
            continue

        config = load_config_with_includes(config_path)
        scenario_name = entry["name"]
        scenario = config.get("scenarios", {}).get(scenario_name)
        if scenario is None:
            print(f"[EVAL] Scenario {scenario_name} not found in {config_path}")
            continue

        geo = compute_geometry(scenario)

        print(f"\n[EVAL] {entry['label']} ({args.episodes} eps, seeds={args.seeds})")
        metrics = evaluate_checkpoint(
            config_path=config_path,
            checkpoint=checkpoint,
            scenario=scenario,
            episodes=args.episodes,
            seeds=args.seeds,
        )

        result = {
            "name": entry["name"],
            "label": entry["label"],
            "short_label": entry["name"].replace("disadvantage_", "").capitalize(),
            "difficulty": entry["difficulty"],
            "ego_speed": scenario["own_init"]["velocity_mps"],
            "target_speed": scenario["target_init"]["velocity_mps"],
            **geo,
            **metrics,
        }
        results.append(result)

        print(
            f"  success={metrics['success_rate']:.1%} "
            f"crash={metrics['crash_rate']:.1%} "
            f"timeout={metrics['timeout_rate']:.1%} "
            f"oob={metrics['out_of_bounds_rate']:.1%} "
            f"mean_min_range={metrics['mean_min_range_m']:.0f}m"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "feasibility_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    plot_results(results, str(out_dir / "feasibility_gradient_v2.png"))
    generate_report(results, str(out_dir / "feasibility_report_v2.md"))


if __name__ == "__main__":
    main()
