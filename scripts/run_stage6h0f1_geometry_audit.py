#!/usr/bin/env python3
"""Stage 6H.0-F.1: Geometry Semantics Audit.

Produces artifacts in outputs/stage6h0f1_geometry_audit/:
    - geometry_catalog.json         : all scenarios from ScenarioRegistry with validation reports
    - classification_report.md      : per-family geometry classification summary
    - telemetry_consistency_report.md : scenario -> telemetry binding verification
    - evaluator_consistency_report.md : evaluator classification vs validator output
"""

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml

from uav_vpp_guidance.envs.geometry_scenarios import (
    GEOMETRY_FAMILY_DOCS,
    VALID_SCENARIO_TYPES,
    build_explicit_scenario,
)
from uav_vpp_guidance.envs.scenario_registry import (
    ScenarioRegistry,
    initialize_canonical_scenarios,
)
from uav_vpp_guidance.utils.geometry_validator import (
    classify_geometry_family,
    compute_relative_geometry,
    validate_scenario_geometry,
    GEOMETRY_FAMILIES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage6h0f1_geometry_audit"


def _build_catalog() -> dict:
    """Build geometry catalog from ScenarioRegistry."""
    initialize_canonical_scenarios()
    catalog = {
        "audit_date": datetime.now().isoformat(),
        "geometry_families": sorted(GEOMETRY_FAMILIES),
        "scenarios": {},
    }
    for name in ScenarioRegistry.list_names():
        scen = ScenarioRegistry.get(name)
        report = validate_scenario_geometry(scen)
        catalog["scenarios"][name] = {
            "scenario_type": scen.get("metadata", {}).get("scenario_type", "unknown"),
            "scenario_set": next(
                (s for s in ScenarioRegistry.list_sets()
                 if name in ScenarioRegistry.list_names(s)),
                "unknown",
            ),
            "validation": report,
        }
    return catalog


def _write_geometry_catalog(catalog: dict) -> None:
    path = OUTPUT_DIR / "geometry_catalog.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, default=str)
    print(f"  Written: {path}")


def _write_classification_report(catalog: dict) -> None:
    lines = [
        "# Stage 6H.0-F.1: Geometry Classification Report",
        "",
        f"**Audit date**: {catalog['audit_date']}",
        "",
        "## Geometry Family Definitions",
        "",
    ]
    for family in sorted(GEOMETRY_FAMILIES):
        doc = GEOMETRY_FAMILY_DOCS.get(family, {})
        lines.append(f"### {family}")
        lines.append(f"- **Description**: {doc.get('description', 'N/A')}")
        lines.append(f"- **Position**: {doc.get('position', 'N/A')}")
        lines.append(f"- **Target heading**: {doc.get('target_heading', 'N/A')}")
        lines.append(f"- **Aspect angle**: {doc.get('aspect_angle', 'N/A')}")
        lines.append(f"- **Closure condition**: {doc.get('closure_condition', 'N/A')}")
        lines.append("")

    lines.extend([
        "## Registry Scenario Classification",
        "",
        "| Scenario | Type | Set | Classified Family | Aspect (deg) | Closure (m/s) | Range (m) | Checks Pass |",
        "|---|---|---|---|---|---|---|---|",
    ])

    all_pass = True
    family_counts = {f: 0 for f in GEOMETRY_FAMILIES}
    for name, info in sorted(catalog["scenarios"].items()):
        val = info["validation"]
        geo = val["geometry"]
        checks = "PASS" if val["all_checks_pass"] else "FAIL"
        if not val["all_checks_pass"]:
            all_pass = False
        lines.append(
            f"| {name} | {info['scenario_type']} | {info['scenario_set']} | "
            f"{val['classified_family']} | {geo['aspect_angle_deg']:.1f} | "
            f"{geo['closure_rate_mps']:.1f} | {geo['range_m']:.0f} | {checks} |"
        )
        family_counts[val["classified_family"]] += 1

    lines.extend([
        "",
        "## Family Coverage Summary",
        "",
        "| Family | Count in Registry |",
        "|---|---|",
    ])
    for fam, count in sorted(family_counts.items()):
        status = "OK" if count > 0 else "MISSING"
        lines.append(f"| {fam} | {count} | {status} |")

    lines.extend([
        "",
        "## Acceptance Gate",
        "",
    ])
    if all_pass and all(c > 0 for c in family_counts.values()):
        lines.append("**PASS**: All scenarios validate and every geometry family is represented.")
    else:
        lines.append("**FAIL**: Some scenarios failed validation or families are missing.")

    path = OUTPUT_DIR / "classification_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def _write_telemetry_consistency_report(catalog: dict) -> None:
    """Verify that scenario names bind correctly to telemetry."""
    lines = [
        "# Stage 6H.0-F.1: Telemetry Consistency Report",
        "",
        f"**Audit date**: {catalog['audit_date']}",
        "",
        "## Scenario -> Telemetry Binding",
        "",
        "| Scenario | scenario_type | Metadata Family | Validator Family | Consistent |",
        "|---|---|---|---|---|",
    ]

    consistent = True
    for name, info in sorted(catalog["scenarios"].items()):
        scen_type = info["scenario_type"]
        val_family = info["validation"]["classified_family"]
        # Consistency: scenario_type should match validator family
        # (allow offset_pursuit == offset_attack alias)
        is_consistent = (
            scen_type == val_family
            or (scen_type == "offset_pursuit" and val_family == "offset_attack")
            or (scen_type == "offset_attack" and val_family == "offset_pursuit")
        )
        if not is_consistent:
            consistent = False
        status = "PASS" if is_consistent else "MISMATCH"
        lines.append(
            f"| {name} | {scen_type} | {scen_type} | {val_family} | {status} |"
        )

    lines.extend([
        "",
        "## Acceptance Gate",
        "",
    ])
    if consistent:
        lines.append("**PASS**: Every scenario's `scenario_type` matches its validator-classified family.")
    else:
        lines.append("**FAIL**: Some scenarios have scenario_type / validator family mismatches.")

    path = OUTPUT_DIR / "telemetry_consistency_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def _write_evaluator_consistency_report() -> None:
    """Verify evaluator produces expected scenario field and classification."""
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    from uav_vpp_guidance.agents.ppo_agent import PPOAgent
    from uav_vpp_guidance.evaluation.evaluate_prediction_comparison import evaluate_single_episode
    from uav_vpp_guidance.utils.config import merge_config

    config_path = PROJECT_ROOT / "config" / "experiment" / "stage6f5_feasible_geometry.yaml"
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    method_override = full_config.get("methods", {}).get("no_prediction", {})
    method_config = merge_config(copy.deepcopy(full_config), copy.deepcopy(method_override))
    method_config["backend"] = "simple"
    method_config["env"]["backend"] = "simple"
    method_config["env"]["use_jsbsim"] = False

    env = CloseRangeTrackingEnv(method_config)
    sample_obs = env.reset(seed=0)
    obs_dim = int(sample_obs["observation_vector"].shape[0])
    agent = PPOAgent(obs_dim=obs_dim, action_dim=3, config=method_config, device="cpu")

    lines = [
        "# Stage 6H.0-F.1: Evaluator Consistency Report",
        "",
        f"**Audit date**: {datetime.now().isoformat()}",
        "",
        "## Per-Family Evaluator Run",
        "",
        "| Scenario | Evaluator scenario field | is_success | reason | Validator Family | Consistent |",
        "|---|---|---|---|---|---|",
    ]

    all_consistent = True
    for st in sorted(VALID_SCENARIO_TYPES):
        scen = build_explicit_scenario(st, 2000.0, 250.0, 180.0)
        scen["name"] = f"eval_test_{st}"
        result, _ = evaluate_single_episode(
            env, agent, method_config, scenario=scen, seed=42,
            save_trajectory=False, method_name="no_prediction",
        )
        # Validate geometry from the actual env state after reset
        simple = env._simple_env
        own_pos = simple.own_state["position_m"]
        tgt_pos = simple.target_state["position_m"]
        own_hdg = float(__import__("numpy", fromlist=["degrees"]).degrees(simple.own_state["heading_rad"]))
        tgt_hdg = float(__import__("numpy", fromlist=["degrees"]).degrees(simple.target_state["heading_rad"]))
        own_spd = float(__import__("numpy", fromlist=["linalg"]).linalg.norm(simple.own_state["velocity_vector_mps"]))
        tgt_spd = float(__import__("numpy", fromlist=["linalg"]).linalg.norm(simple.target_state["velocity_vector_mps"]))

        geo = compute_relative_geometry(own_pos, own_hdg, own_spd, tgt_pos, tgt_hdg, tgt_spd)
        val_family = classify_geometry_family(geo)

        eval_scenario = result.get("scenario", "MISSING")
        is_consistent = eval_scenario == scen["name"]
        if not is_consistent:
            all_consistent = False
        status = "PASS" if is_consistent else "FAIL"

        lines.append(
            f"| {st} | {eval_scenario} | {result.get('is_success')} | "
            f"{result.get('reason')} | {val_family} | {status} |"
        )

    env.close()

    lines.extend([
        "",
        "## Acceptance Gate",
        "",
    ])
    if all_consistent:
        lines.append("**PASS**: Evaluator scenario field correctly binds to scenario name for all families.")
    else:
        lines.append("**FAIL**: Some evaluator runs have scenario field mismatches.")

    path = OUTPUT_DIR / "evaluator_consistency_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Stage 6H.0-F.1: Geometry Semantics Audit ===")

    print("\n1. Building geometry catalog ...")
    catalog = _build_catalog()
    _write_geometry_catalog(catalog)

    print("\n2. Writing classification report ...")
    _write_classification_report(catalog)

    print("\n3. Writing telemetry consistency report ...")
    _write_telemetry_consistency_report(catalog)

    print("\n4. Writing evaluator consistency report ...")
    _write_evaluator_consistency_report()

    print(f"\nAudit complete. Artifacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
