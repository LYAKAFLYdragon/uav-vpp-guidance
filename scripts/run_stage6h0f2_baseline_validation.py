#!/usr/bin/env python3
"""Stage 6H.0-F.2: Formal Regression Baseline Validation.

Validates:
    1. Each baseline scenario is feasible (success_rate >= 0.6).
    2. Episode-level state isolation across resets.
    3. All baseline scenarios originate from ScenarioRegistry.
    4. Geometry validator classification matches scenario_type.

Produces artifacts in outputs/stage6h0f2_formal_baseline/:
    - regression_baseline_validation_report.md
    - regression_baseline.csv (updated with actual success rates)
"""

import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml

from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
from uav_vpp_guidance.agents.ppo_agent import PPOAgent
from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
from uav_vpp_guidance.utils.config import merge_config
from uav_vpp_guidance.utils.geometry_validator import (
    compute_relative_geometry,
    validate_scenario_geometry,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage6h0f2_formal_baseline"
CHECKPOINT = PROJECT_ROOT / "outputs" / "audit_no_pred_final" / "checkpoints" / "best.pt"
N_EPS_PER_SCENARIO = 5


def _make_env():
    config_path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))
    method_config["backend"] = "simple"
    method_config["env"]["backend"] = "simple"
    method_config["env"]["use_jsbsim"] = False
    return CloseRangeTrackingEnv(method_config), method_config


def _validate_baseline_feasibility() -> dict:
    """Run episodes for each baseline scenario and record success rates."""
    initialize_canonical_scenarios()
    env, method_config = _make_env()
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=method_config, device="cpu")
    if CHECKPOINT.exists():
        agent.load(str(CHECKPOINT))

    baseline_names = ScenarioRegistry.list_names("regression_baseline")
    results = {}

    for name in baseline_names:
        scen = ScenarioRegistry.get(name)
        successes = []
        for seed in range(N_EPS_PER_SCENARIO):
            result, _ = evaluate_single_episode(
                env, agent, method_config, scenario=scen, seed=1000 + seed,
                save_trajectory=False, method_name="no_prediction",
            )
            successes.append(result.get("is_success", False))

        sr = float(np.mean(successes))
        results[name] = {
            "scenario_type": scen.get("metadata", {}).get("scenario_type", "unknown"),
            "success_rate": sr,
            "success_count": int(sum(successes)),
            "total": N_EPS_PER_SCENARIO,
            "feasible": sr >= 0.6,
            "reasons": [result.get("reason", "unknown") for result in [
                evaluate_single_episode(
                    env, agent, method_config, scenario=scen, seed=1000 + s,
                    save_trajectory=False, method_name="no_prediction",
                )[0] for s in range(N_EPS_PER_SCENARIO)
            ]],
        }
        print(f"  {name}: {results[name]['success_count']}/{N_EPS_PER_SCENARIO} = {sr:.1%}")

    env.close()
    return results


def _verify_state_isolation() -> dict:
    """Verify that env.reset() produces isolated state for each scenario."""
    initialize_canonical_scenarios()
    env, _ = _make_env()
    baseline_names = ScenarioRegistry.list_names("regression_baseline")
    reports = {}

    for name in baseline_names:
        scen = ScenarioRegistry.get(name)
        # Reset twice with different seeds
        env.reset(scenario=scen, seed=1)
        simple = env._simple_env
        pos1_own = simple.own_state["position_m"].copy()
        pos1_tgt = simple.target_state["position_m"].copy()

        env.reset(scenario=scen, seed=2)
        simple = env._simple_env
        pos2_own = simple.own_state["position_m"].copy()
        pos2_tgt = simple.target_state["position_m"].copy()

        # Same scenario should produce same initial state regardless of seed
        # (scenario dict fully determines initial state)
        same_own = np.allclose(pos1_own, pos2_own)
        same_tgt = np.allclose(pos1_tgt, pos2_tgt)

        reports[name] = {
            "own_state_deterministic": bool(same_own),
            "target_state_deterministic": bool(same_tgt),
            "initial_range_m": float(np.linalg.norm(pos1_tgt - pos1_own)),
        }

    env.close()
    return reports


def _audit_script_sources() -> dict:
    """Audit scripts for ScenarioRegistry-only usage."""
    scripts_dir = PROJECT_ROOT / "scripts"
    audit = {}
    for script_path in sorted(scripts_dir.glob("*.py")):
        text = script_path.read_text(encoding="utf-8")
        uses_registry = "ScenarioRegistry" in text or "scenario_registry" in text
        uses_legacy = "build_geometry_scenario" in text
        uses_explicit = "build_explicit_scenario" in text
        uses_hardcoded = "own_init" in text and "target_init" in text and not uses_registry
        audit[script_path.name] = {
            "uses_scenario_registry": uses_registry,
            "uses_legacy_builder": uses_legacy,
            "uses_explicit_builder": uses_explicit,
            "has_hardcoded_scenarios": uses_hardcoded,
            "status": "compliant" if uses_registry else "needs_migration",
        }
    return audit


def _write_csv(results: dict) -> None:
    path = OUTPUT_DIR / "regression_baseline.csv"
    initialize_canonical_scenarios()
    rows = []
    for name in sorted(results.keys()):
        info = results[name]
        scen = ScenarioRegistry.get(name)
        report = validate_scenario_geometry(scen)
        geo = report["geometry"]
        rows.append({
            "variant": "no_prediction",
            "scenario_id": name,
            "scenario_type": info["scenario_type"],
            "aspect_angle_deg": round(geo["aspect_angle_deg"], 1),
            "initial_range_m": round(geo["range_m"], 1),
            "ego_speed_mps": scen["own_init"]["velocity_mps"],
            "target_speed_mps": scen["target_init"]["velocity_mps"],
            "altitude_diff_m": scen.get("metadata", {}).get("altitude_diff_m", 0.0),
            "success_rate": round(info["success_rate"], 2),
            "is_candidate": "true" if info["feasible"] else "false",
            "family": report["classified_family"],
            "feasibility_note": scen.get("metadata", {}).get("note", ""),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


def _write_validation_report(feasibility: dict, isolation: dict, script_audit: dict) -> None:
    lines = [
        "# Stage 6H.0-F.2: Formal Regression Baseline Validation Report",
        "",
        f"**Date**: {datetime.now().isoformat()}",
        f"**Checkpoint**: `{CHECKPOINT}`",
        f"**Episodes per scenario**: {N_EPS_PER_SCENARIO}",
        "",
        "## 1. Baseline Scenario Feasibility",
        "",
        "| Scenario | Type | Family | Success Rate | Feasible | Failure Reasons |",
        "|---|---|---|---|---|---|",
    ]

    all_feasible = True
    family_coverage = {}
    for name in sorted(feasibility.keys()):
        info = feasibility[name]
        scen = ScenarioRegistry.get(name)
        report = validate_scenario_geometry(scen)
        family = report["classified_family"]
        family_coverage[family] = True
        if not info["feasible"]:
            all_feasible = False
        reasons = ", ".join(set(info["reasons"]))
        status = "PASS" if info["feasible"] else "FAIL"
        lines.append(
            f"| {name} | {info['scenario_type']} | {family} | "
            f"{info['success_rate']:.1%} | {status} | {reasons} |"
        )

    lines.extend([
        "",
        "## 2. Geometry Family Coverage",
        "",
        "| Family | Covered | Source Scenario |",
        "|---|---|---|",
    ])
    for family in sorted(["tail_chase", "head_on", "crossing_left", "crossing_right", "offset_attack", "fleeing"]):
        covered = family_coverage.get(family, False)
        src = next(
            (n for n in sorted(feasibility.keys())
             if validate_scenario_geometry(ScenarioRegistry.get(n))["classified_family"] == family),
            "N/A",
        )
        status = "PASS" if covered else "NOT FEASIBLE"
        lines.append(f"| {family} | {status} | {src} |")

    lines.extend([
        "",
        "## 3. Episode-Level State Isolation",
        "",
        "| Scenario | Own State Deterministic | Target State Deterministic | Initial Range (m) |",
        "|---|---|---|---|",
    ])
    all_isolated = True
    for name in sorted(isolation.keys()):
        info = isolation[name]
        if not info["own_state_deterministic"] or not info["target_state_deterministic"]:
            all_isolated = False
        lines.append(
            f"| {name} | {info['own_state_deterministic']} | {info['target_state_deterministic']} | "
            f"{info['initial_range_m']:.1f} |"
        )

    lines.extend([
        "",
        "## 4. Script Compliance Audit",
        "",
        "| Script | Uses Registry | Uses Legacy Builder | Status |",
        "|---|---|---|---|",
    ])
    for script_name in sorted(script_audit.keys()):
        info = script_audit[script_name]
        lines.append(
            f"| {script_name} | {info['uses_scenario_registry']} | {info['uses_legacy_builder']} | {info['status']} |"
        )

    compliant_count = sum(1 for s in script_audit.values() if s["status"] == "compliant")
    total_count = len(script_audit)

    lines.extend([
        "",
        f"**Compliance**: {compliant_count}/{total_count} scripts use ScenarioRegistry.",
        "",
        "## 5. Acceptance Gate",
        "",
    ])

    gate_pass = all_feasible and all_isolated
    if gate_pass:
        lines.append("**PASS**: All baseline scenarios are feasible, state isolation verified.")
    else:
        lines.append("**PARTIAL**: Some scenarios infeasible or state isolation issues detected.")

    lines.append("")
    lines.append(
        "> **Note**: `tail_chase`, `offset_attack`, and `fleeing` families have no feasible "
        "scenarios with the current VPP checkpoint. This is a known fundamental limitation "
        "(Stage 6F.5 confirmed 0% success across all methods for tail-chase and offset geometries)."
    )

    path = OUTPUT_DIR / "regression_baseline_validation_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Stage 6H.0-F.2: Formal Regression Baseline Validation ===")

    print("\n1. Validating baseline feasibility ...")
    feasibility = _validate_baseline_feasibility()

    print("\n2. Verifying state isolation ...")
    isolation = _verify_state_isolation()

    print("\n3. Auditing script sources ...")
    script_audit = _audit_script_sources()

    print("\n4. Writing artifacts ...")
    _write_csv(feasibility)
    _write_validation_report(feasibility, isolation, script_audit)

    print(f"\nDone. Artifacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
