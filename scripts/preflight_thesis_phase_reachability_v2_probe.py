"""Fail-closed preflight for the separately authorised v2 observation probe."""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_phase_reachability_v2_run_in_handoff_r1.yaml"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    prepared = json.loads(json.dumps(manifest))
    prepared.get("integrity", {}).pop("payload_sha256", None)
    encoded = json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def validate_probe(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plan = build_phase_reachability_plan(config)
    if not plan.execution_permitted:
        raise PhaseReachabilityContractError("v2 probe execution is not authorised")
    protocol = config["phase_reachability_v2"]
    frozen = protocol["frozen_v1_negative_evidence"]
    gate_path = Path(str(frozen["gate_path"]))
    if not gate_path.is_file():
        raise PhaseReachabilityContractError(f"missing frozen v1 gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("verdict") != frozen["required_verdict"]:
        raise PhaseReachabilityContractError("frozen v1 negative-evidence verdict mismatch")
    scenario_config = config.get("scenario_manifest", {})
    manifest_path = (path.parent / str(scenario_config["path"])).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if manifest.get("source_id") != plan.source_id or len(manifest.get("scenarios", [])) != 12:
        raise PhaseReachabilityContractError("v2 scenario manifest identity mismatch")
    if manifest.get("independence", {}).get("v1_manifest_reused") is not False:
        raise PhaseReachabilityContractError("v2 scenario manifest must be independent from v1")
    if _manifest_payload_sha256(manifest) != scenario_config.get("payload_sha256"):
        raise PhaseReachabilityContractError("v2 scenario manifest payload SHA mismatch")
    for scenario in manifest["scenarios"]:
        metadata = scenario.get("metadata", {})
        if metadata.get("taxonomy_geometry_state") != "disadvantage":
            raise PhaseReachabilityContractError("v2 scenario taxonomy drift")
        if float(metadata.get("initial_range_m", 0.0)) <= 1000.0:
            raise PhaseReachabilityContractError("v2 must physically run into, not reset inside, merge range")
    return {
        "mode": "authorised_evaluation_only",
        "execution_permitted": True,
        "training_permitted": False,
        "action_replacement_permitted": False,
        "source_id": plan.source_id,
        "opponents": plan.opponents,
        "scenario_count": len(manifest["scenarios"]),
        "next_action": "run_three_strict_jsbsim_formal_small_probes_then_analyze_once",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_probe(args.config), indent=2))
    except (OSError, ValueError, yaml.YAMLError, PhaseReachabilityContractError) as error:
        print(f"INVALID v2 phase-reachability probe: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
