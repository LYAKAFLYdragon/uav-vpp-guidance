"""Validate the design-only v2 physical-continuation handoff contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_vpp_guidance.evaluation.phase_reachability_handoff_contract import (  # noqa: E402
    PhaseReachabilityContractError,
    build_phase_reachability_plan,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment" / "thesis_phase_reachability_v2_run_in_handoff_design.yaml"


def validate_design(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plan = build_phase_reachability_plan(config)
    frozen = config["phase_reachability_v2"]["frozen_v1_negative_evidence"]
    gate_path = Path(str(frozen["gate_path"]))
    if not gate_path.is_file():
        raise PhaseReachabilityContractError(f"missing frozen v1 gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("verdict") != frozen["required_verdict"]:
        raise PhaseReachabilityContractError("frozen v1 verdict mismatch")
    return {
        "mode": "design_validation_only",
        "execution_permitted": False,
        "training_permitted": False,
        "source_id": plan.source_id,
        "run_in_controller": plan.run_in_controller,
        "handoff_mode": plan.handoff_mode,
        "opponents": plan.opponents,
        "phase_gate": {"post_merge_steps": plan.post_merge_steps, "re_entry_steps": plan.re_entry_steps, "qualifying_fraction": plan.qualifying_fraction},
        "next_action": "explicit_execution_authorization_required_after_continuity_sidecar_implementation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Always refused: this entry point validates design only.")
    args = parser.parse_args()
    if args.execute:
        print("REFUSED: phase-reachability v2 execution is not authorized.", file=sys.stderr)
        return 2
    try:
        print(json.dumps(validate_design(args.config), indent=2))
    except (OSError, ValueError, yaml.YAMLError, PhaseReachabilityContractError) as error:
        print(f"INVALID phase-reachability v2 design: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
