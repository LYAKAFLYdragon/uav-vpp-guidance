#!/usr/bin/env python3
"""Independent, non-executing preflight for the Taxonomy60 formal universe."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_jsbsim_hrl_comparison as comparison  # noqa: E402
from preflight_formal_reproducibility import (  # noqa: E402
    _collect_frozen_assets,
    _directory_manifest_sha256,
)
from verify_thesis_dualskill_taxonomy60_p2_smoke import (  # noqa: E402
    verify_smoke_root,
)


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
FORMAL_CONFIG = REPO_ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_dualskill_taxonomy60_formal.yaml"
MANIFEST_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION_CONTRACT_PATH = (
    REPO_ROOT
    / "config"
    / "experiment"
    / "decision_contracts"
    / "thesis_dualskill_taxonomy60_no_go_v3.yaml"
)
P2_SMOKE_ROOT = Path(
    "E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE-R1"
)
FORMAL_OUTPUT_ROOT = Path(
    "E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
)
MIN_FREE_BYTES = 50 * 1024**3
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
WARMUP_CONTRACT = {
    "warmup_fallback_high_level_steps": 9,
    "first_required_valid_high_level_step": 10,
    "fallback_after_warmup_permitted": False,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static_contract(
    *,
    manifest: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    decision_contract: Mapping[str, Any],
) -> List[str]:
    """Validate facts that must hold before any simulator process is created."""

    errors: List[str] = []
    if manifest.get("source_id") != SOURCE_ID or manifest.get("manifest_version") != 4:
        errors.append("formal manifest source/version mismatch")
    if manifest.get("disjointness", {}).get("intersection_count") != 0:
        errors.append("formal manifest has historical scenario intersections")
    source_files = manifest.get("disjointness", {}).get("source_files", [])
    if len(source_files) != 49 or any(
        source.get("provenance", {}).get("kind") != "git_blob" for source in source_files
    ):
        errors.append("historical disjointness proof is not portable Git evidence")
    scenarios = manifest.get("scenarios", [])
    if len(scenarios) != 60:
        errors.append("formal manifest does not have 60 scenarios")
    else:
        initial_counts = Counter(
            scenario.get("metadata", {}).get("initial_class") for scenario in scenarios
        )
        height_counts = Counter(
            scenario.get("metadata", {}).get("height_condition") for scenario in scenarios
        )
        mirror_counts = Counter(
            scenario.get("metadata", {}).get("mirror_sign") for scenario in scenarios
        )
        if initial_counts != Counter(
            {"advantage": 12, "head_on": 12, "disadvantage": 12, "neutral": 12, "crossing_entry": 12}
        ):
            errors.append("formal initial-class balance mismatch")
        if height_counts != Counter(
            {"own_below": 20, "co_altitude": 20, "own_above": 20}
        ):
            errors.append("formal height balance mismatch")
        if mirror_counts != Counter({"negative": 30, "positive": 30}):
            errors.append("formal mirror balance mismatch")
    if tuple(manifest.get("protocol", {}).get("methods", [])) != METHODS:
        errors.append("formal manifest method contract mismatch")
    if tuple(manifest.get("protocol", {}).get("opponent_stages", [])) != OPPONENTS:
        errors.append("formal manifest opponent contract mismatch")
    if {
        key: manifest.get("prediction_receipt_contract", {}).get(key)
        for key in WARMUP_CONTRACT
    } != WARMUP_CONTRACT:
        errors.append("formal manifest prediction warm-up contract mismatch")

    protocol = formal_config.get("thesis_dualskill_taxonomy60_formal", {})
    if protocol.get("source_id") != SOURCE_ID or protocol.get("expected_episodes") != 900:
        errors.append("formal config Source/episode contract mismatch")
    if tuple(protocol.get("methods", [])) != METHODS:
        errors.append("formal config methods mismatch")
    if tuple(protocol.get("opponents", [])) != OPPONENTS:
        errors.append("formal config opponents mismatch")
    if {
        key: protocol.get("prediction_receipt_contract", {}).get(key)
        for key in WARMUP_CONTRACT
    } != WARMUP_CONTRACT:
        errors.append("formal config prediction warm-up contract mismatch")
    if formal_config.get("env", {}).get("strict_backend") is not True:
        errors.append("formal config does not require strict backend")

    decision = manifest.get("decision_contract", {})
    if decision.get("id") != decision_contract.get("contract_id"):
        errors.append("manifest decision contract ID mismatch")
    if decision.get("sha256") != _sha256(DECISION_CONTRACT_PATH):
        errors.append("manifest decision contract hash mismatch")
    if {
        key: decision_contract.get("formal_validity", {})
        .get("prediction_receipt_contract", {})
        .get(key)
        for key in WARMUP_CONTRACT
    } != WARMUP_CONTRACT:
        errors.append("decision contract prediction warm-up boundary mismatch")
    precedence = decision_contract.get("classification", {}).get("precedence", [])
    route_mapping = decision_contract.get("classification_routes", {})
    allowed_routes = decision_contract.get("allowed_routes", {})
    missing_route_mappings = [label for label in precedence if label not in route_mapping]
    if missing_route_mappings:
        errors.append("decision contract has unmapped classifications")
    invalid_route_mappings = [
        label
        for label, route in route_mapping.items()
        if route.get("selected_study_route") not in allowed_routes
    ]
    if invalid_route_mappings:
        errors.append("decision contract maps classifications to invalid routes")
    artifact_contract = decision_contract.get("route_decision_artifact", {})
    if artifact_contract.get("producer") != "P4_formal_verifier":
        errors.append("route decision artifact producer must be P4 formal verifier")
    required_route_fields = {
        "route_candidates_in_precedence_order",
        "selected_study_route",
        "route_transition_reason",
        "post_closure_eligibility",
        "recovery_state",
    }
    if not required_route_fields.issubset(set(artifact_contract.get("required_fields", []))):
        errors.append("route decision artifact lacks transition fields")
    return errors


def _collect_runtime_checks(formal_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = formal_config["thesis_dualskill_taxonomy60_formal"]
    jsbsim_root = Path(str(runtime["jsbsim_legacy_root"]))
    jsbsim_data = jsbsim_root / "envs" / "JSBSim" / "data"
    asset_config = copy.deepcopy(formal_config)
    source_worktree = Path(
        str(asset_config.get("frozen_assets", {}).get("source_worktree", ""))
    )
    source_worktree_git_verifiable = bool(source_worktree and (source_worktree / ".git").exists())
    if not source_worktree_git_verifiable:
        # The portable asset bundle is content-addressed, not a Git checkout.
        asset_config["frozen_assets"].pop("source_worktree", None)
        asset_config["frozen_assets"].pop("source_git_sha", None)
    frozen_assets = _collect_frozen_assets(repo_root=REPO_ROOT, config=asset_config)
    asset_checks = []
    for asset in frozen_assets["artifacts"]:
        asset_checks.append(
            {
                "id": asset["id"],
                "exists": asset["exists"],
                "hash_matches": asset["expected_sha256"] == asset["actual_sha256"],
                "obs_dim_matches": asset["expected_obs_dim"] in (None, asset["checkpoint_obs_dim"]),
                "action_dim_matches": asset["expected_action_dim"] in (None, asset["checkpoint_action_dim"]),
            }
        )
    return {
        "assets": asset_checks,
        "source_worktree_git_verifiable": source_worktree_git_verifiable,
        "source_worktree_git_matches": (
            frozen_assets["expected_source_git_sha"] == frozen_assets["actual_source_git_sha"]
        ),
        "jsbsim_data_dir": str(jsbsim_data),
        "jsbsim_data_exists": jsbsim_data.is_dir(),
        "jsbsim_data_expected_sha256": runtime["jsbsim_data_manifest_sha256"],
        "jsbsim_data_actual_sha256": (
            _directory_manifest_sha256(jsbsim_data) if jsbsim_data.is_dir() else None
        ),
    }


def run_preflight(
    *,
    formal_config_path: Path = FORMAL_CONFIG,
    formal_output_root: Path = FORMAL_OUTPUT_ROOT,
    p2_smoke_root: Path = P2_SMOKE_ROOT,
) -> Dict[str, Any]:
    manifest = _load_yaml(MANIFEST_PATH)
    decision_contract = _load_yaml(DECISION_CONTRACT_PATH)
    formal_config = comparison._load_config_with_includes(formal_config_path)
    materialized = comparison._materialize_scenario_manifest(formal_config, formal_config_path)
    errors = validate_static_contract(
        manifest=manifest, formal_config=materialized, decision_contract=decision_contract
    )

    expected_counts = {"head_on": 48, "crossing_feasible": 12}
    if materialized.get("scenario_manifest_resolution", {}).get("task_counts") != expected_counts:
        errors.append("formal materialized task counts mismatch")
    config_shapes = []
    for opponent in OPPONENTS:
        for method in METHODS:
            for task in expected_counts:
                try:
                    config = comparison.build_eval_config(
                        materialized, method, task, backend="jsbsim", opponent_stage=opponent
                    )
                    config_shapes.append(
                        {
                            "opponent": opponent,
                            "method": method,
                            "task": task,
                            "strict_backend": config.get("env", {}).get("strict_backend"),
                            "opponent_stage": config.get("opponent_stage"),
                        }
                    )
                except Exception as error:  # pragma: no cover - reported as receipt data
                    errors.append("config build {}:{}:{}: {}".format(opponent, method, task, error))
    if len(config_shapes) != len(OPPONENTS) * len(METHODS) * len(expected_counts):
        errors.append("formal config matrix did not build all 30 cells")
    if any(not row["strict_backend"] or row["opponent_stage"] != row["opponent"] for row in config_shapes):
        errors.append("formal config matrix has backend/opponent drift")

    runtime = _collect_runtime_checks(materialized)
    if runtime["source_worktree_git_verifiable"] and not runtime["source_worktree_git_matches"]:
        errors.append("frozen asset source worktree SHA mismatch")
    if not runtime["jsbsim_data_exists"] or (
        runtime["jsbsim_data_expected_sha256"] != runtime["jsbsim_data_actual_sha256"]
    ):
        errors.append("frozen JSBSim data directory hash mismatch")
    if any(
        not asset["exists"]
        or not asset["hash_matches"]
        or not asset["obs_dim_matches"]
        or not asset["action_dim_matches"]
        for asset in runtime["assets"]
    ):
        errors.append("frozen checkpoint asset preflight mismatch")

    try:
        smoke = verify_smoke_root(p2_smoke_root)
    except Exception as error:  # pragma: no cover - emitted in receipt
        smoke = {"verdict": "fail", "error": str(error)}
        errors.append("P2 smoke receipt is invalid")
    else:
        if smoke.get("verdict") != "pass" or smoke.get("records") != 15:
            errors.append("P2 smoke receipt does not pass the 15-record gate")
        if smoke.get("prediction_fallback_steps") != 135:
            errors.append("P2 smoke prediction warm-up receipt drifted")

    drive = formal_output_root.drive or str(formal_output_root.anchor)
    free_bytes = shutil.disk_usage(drive).free
    if formal_output_root.exists():
        errors.append("formal output root already exists")
    if free_bytes < MIN_FREE_BYTES:
        errors.append("formal output drive is below the 50 GiB reserve")

    return {
        "source_id": SOURCE_ID,
        "formal_config": str(formal_config_path),
        "formal_output_root": str(formal_output_root),
        "formal_output_root_exists": formal_output_root.exists(),
        "free_bytes": free_bytes,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "static_errors": errors,
        "config_matrix_cells": config_shapes,
        "runtime": runtime,
        "p2_smoke": smoke,
        "verdict": "pass" if not errors else "fail",
        "formal_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-config", type=Path, default=FORMAL_CONFIG)
    parser.add_argument("--formal-output-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    parser.add_argument("--p2-smoke-root", type=Path, default=P2_SMOKE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "thesis_dualskill_taxonomy60_p3_independent_preflight.json",
    )
    args = parser.parse_args()
    result = run_preflight(
        formal_config_path=args.formal_config.resolve(),
        formal_output_root=args.formal_output_root,
        p2_smoke_root=args.p2_smoke_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
