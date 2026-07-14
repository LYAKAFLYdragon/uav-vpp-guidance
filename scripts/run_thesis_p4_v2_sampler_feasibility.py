#!/usr/bin/env python
"""Validate the P4-v2 sampler-feasibility design without running JSBSim.

This command is deliberately a design validator.  ``--execute`` is present
only to fail closed, making it impossible to mistake this entry point for an
authorized physical probe or a training launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_vpp_guidance.evaluation.p4_v2_sampler_feasibility import (  # noqa: E402
    PreflightContractError,
    build_sampler_feasibility_plan,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "thesis_five_state_shared_skills_geometry_v2_sampler_feasibility.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise PreflightContractError(f"YAML root must be a mapping: {path}")
    return loaded


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def validate_design(config_path: Path) -> dict[str, Any]:
    """Load the design inputs and return a JSON-serializable dry-run plan."""

    config = _load_yaml(config_path)
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise PreflightContractError("config.sources must be a mapping")
    evidence_manifest = _resolve(
        REPO_ROOT, str(sources["p4_evidence_bundle"])
    ) / "evidence_manifest.json"
    if not evidence_manifest.is_file():
        raise PreflightContractError(
            f"frozen P4 evidence manifest is missing: {evidence_manifest}"
        )
    registry = _load_yaml(_resolve(REPO_ROOT, str(sources["skill_registry"])))
    train_manifest = _load_yaml(
        _resolve(REPO_ROOT, str(sources["train_candidate_manifest"]))
    )
    dev_manifest = _load_yaml(_resolve(REPO_ROOT, str(sources["dev_candidate_manifest"])))
    plan = build_sampler_feasibility_plan(config, registry, train_manifest, dev_manifest)
    return {
        "mode": "design_validation_only",
        "execution_permitted": False,
        "training_permitted": False,
        "p4_evidence_manifest": evidence_manifest.relative_to(REPO_ROOT).as_posix(),
        "declared_cell_count": len(plan.declared_cells),
        "declared_cells": [cell.key for cell in plan.declared_cells],
        "train_source_id": plan.train_source_id,
        "dev_source_id": plan.dev_source_id,
        "train_signature_namespace": plan.train_signature_namespace,
        "dev_signature_namespace": plan.dev_signature_namespace,
        "coverage_support_gate": {
            "min_episodes_per_cell_per_opponent": plan.min_episodes_per_cell_per_opponent,
            "min_policy_steps_per_cell_per_opponent": plan.min_policy_steps_per_cell_per_opponent,
            "min_distinct_scenario_signatures": plan.min_distinct_scenario_signatures,
        },
        "next_action": "explicit_authorization_required_before_any_jsbsim_probe",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate only (default behavior).")
    parser.add_argument("--execute", action="store_true", help="Always refused by this design-only runner.")
    args = parser.parse_args(argv)
    if args.execute:
        print("REFUSED: P4-v2 JSBSim probe is not authorized by this runner.", file=sys.stderr)
        return 2
    try:
        plan = validate_design(args.config)
    except (KeyError, OSError, yaml.YAMLError, PreflightContractError) as exc:
        print(f"INVALID P4-v2 sampler-feasibility design: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
