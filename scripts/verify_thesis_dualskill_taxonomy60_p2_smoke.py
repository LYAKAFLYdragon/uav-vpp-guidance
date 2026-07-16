#!/usr/bin/env python3
"""Independently verify the disposable Taxonomy60 P2 engineering smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE"
FORMAL_SOURCE_ID = "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
METHODS = (
    "fixed_head_on",
    "fixed_crossing",
    "static_oracle_gate",
    "canonical_ppo",
    "ppo_unbootstrapped_crossing",
)
SMOKE_SCENARIO = "taxonomy60_engineering_smoke_crossing_entry"
PPO_METHODS = {"canonical_ppo", "ppo_unbootstrapped_crossing"}
COMMON_STEP_FIELDS = (
    "range_m",
    "range_rate_mps",
    "ata_deg",
    "aa_deg",
    "vp_forward_bias_m",
    "vp_lateral_bias_m",
    "offset_z",
    "nz_cmd",
    "nz_g",
    "roll_rate_cmd",
    "throttle_cmd",
    "prediction_valid",
    "prediction_fallback",
)


class SmokeVerificationError(RuntimeError):
    """Raised when a disposable smoke output cannot support a P2 pass."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise SmokeVerificationError("{} is missing {}".format(context, key))
    return mapping[key]


def _run_dir(root: Path, opponent: str) -> Path:
    return root / "{}-{}".format(SOURCE_ID, opponent)


def _verify_step(step: Mapping[str, Any], *, method: str, context: str) -> Dict[str, int]:
    prediction = {"valid_steps": 0, "fallback_steps": 0}
    for key in COMMON_STEP_FIELDS:
        value = _require(step, key, context=context)
        if key in {"prediction_valid", "prediction_fallback"}:
            if not isinstance(value, bool):
                raise SmokeVerificationError("{} has non-boolean {}".format(context, key))
            prediction["valid_steps"] += int(value) if key == "prediction_valid" else 0
            prediction["fallback_steps"] += int(value) if key == "prediction_fallback" else 0
        elif not _is_finite(value):
            raise SmokeVerificationError("{} has non-finite {}".format(context, key))

    if method in PPO_METHODS:
        _require(step, "commander_requested_mode_id", context=context)
        _require(step, "commander_mode_id", context=context)
        for key in (
            "commander_crossing_initial_task_bootstrap_enabled",
            "commander_crossing_initial_task_bootstrap_active",
            "commander_crossing_pre_merge_mode_lock_active",
            "commander_mode_constraint_triggered",
        ):
            if not isinstance(_require(step, key, context=context), bool):
                raise SmokeVerificationError("{} has non-boolean {}".format(context, key))
    return prediction


def _verify_record(record: Mapping[str, Any], *, opponent: str) -> Dict[str, Any]:
    context = "{}/{}".format(opponent, record.get("controller", "unknown"))
    method = str(_require(record, "controller", context=context))
    if method not in METHODS:
        raise SmokeVerificationError("{} has unexpected method".format(context))
    if str(_require(record, "opponent_stage", context=context)) != opponent:
        raise SmokeVerificationError("{} has mismatched opponent stage".format(context))
    if str(_require(record, "scenario", context=context)) != SMOKE_SCENARIO:
        raise SmokeVerificationError("{} did not use the disposable smoke scenario".format(context))
    if record.get("backend") != "jsbsim" or record.get("strict_backend") is not True:
        raise SmokeVerificationError("{} did not use strict JSBSim".format(context))
    if bool(record.get("backend_fallback_occurred", False)):
        raise SmokeVerificationError("{} reports a backend fallback".format(context))
    if int(record.get("steps", 0)) <= 0:
        raise SmokeVerificationError("{} has no executed steps".format(context))
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise SmokeVerificationError("{} has no readable trajectory".format(context))

    valid_steps = 0
    fallback_steps = 0
    for index, step in enumerate(trajectory):
        if not isinstance(step, Mapping):
            raise SmokeVerificationError("{} trajectory step {} is not a mapping".format(context, index))
        receipt = _verify_step(step, method=method, context="{} step {}".format(context, index))
        valid_steps += receipt["valid_steps"]
        fallback_steps += receipt["fallback_steps"]

    first_step = trajectory[0]
    if method == "canonical_ppo":
        if first_step["commander_crossing_initial_task_bootstrap_enabled"] is not True:
            raise SmokeVerificationError("canonical PPO bootstrap is not enabled")
        if first_step["commander_crossing_initial_task_bootstrap_active"] is not True:
            raise SmokeVerificationError("canonical PPO bootstrap did not activate at the first step")
    if method == "ppo_unbootstrapped_crossing":
        if first_step["commander_crossing_initial_task_bootstrap_enabled"] is not False:
            raise SmokeVerificationError("unbootstrapped PPO bootstrap is still enabled")
        if first_step["commander_crossing_initial_task_bootstrap_active"] is not False:
            raise SmokeVerificationError("unbootstrapped PPO bootstrap unexpectedly activated")

    return {
        "method": method,
        "steps": int(record["steps"]),
        "prediction_valid_steps": valid_steps,
        "prediction_fallback_steps": fallback_steps,
    }


def verify_smoke_root(root: Path) -> Dict[str, Any]:
    """Check all 15 episodes without inspecting any comparative outcome field."""

    root = Path(root)
    execution_path = root / "smoke_execution_manifest.json"
    execution = _load_json(execution_path)
    if execution.get("source_id") != SOURCE_ID:
        raise SmokeVerificationError("smoke execution source ID mismatch")
    if execution.get("formal_source_id") != FORMAL_SOURCE_ID:
        raise SmokeVerificationError("smoke execution formal Source ID mismatch")

    per_opponent: Dict[str, Any] = {}
    total_records = 0
    all_prediction_valid_steps = 0
    all_prediction_fallback_steps = 0
    for opponent in OPPONENTS:
        run_dir = _run_dir(root, opponent)
        manifest = _load_json(run_dir / "run_manifest.json")
        failures = _load_json(run_dir / "failures.json")
        records = _load_json(run_dir / "aggregate" / "episode_records.json")
        contract = _load_json(run_dir / "artifact_contract.json")

        if manifest.get("status") != "completed":
            raise SmokeVerificationError("{} run did not complete".format(opponent))
        if manifest.get("paper_safe") is not False:
            raise SmokeVerificationError("{} smoke was incorrectly marked paper-safe".format(opponent))
        if manifest.get("extra", {}).get("backend") != "jsbsim":
            raise SmokeVerificationError("{} manifest did not record JSBSim".format(opponent))
        if manifest.get("extra", {}).get("total_episodes") != len(METHODS):
            raise SmokeVerificationError("{} manifest has an unexpected episode count".format(opponent))
        if not manifest.get("extra", {}).get("artifact_validation", {}).get("valid", False):
            raise SmokeVerificationError("{} artifact contract is invalid".format(opponent))
        if failures.get("failures"):
            raise SmokeVerificationError("{} run records failures".format(opponent))
        if "aggregate/episode_records.json" not in contract.get("required_files", []):
            raise SmokeVerificationError("{} contract lacks episode records".format(opponent))
        if not isinstance(records, list) or len(records) != len(METHODS):
            raise SmokeVerificationError("{} has an unexpected record universe".format(opponent))

        methods_seen = set()
        receipt_rows = []
        for record in records:
            receipt = _verify_record(record, opponent=opponent)
            methods_seen.add(receipt["method"])
            receipt_rows.append(receipt)
        if methods_seen != set(METHODS):
            raise SmokeVerificationError("{} methods do not match the frozen smoke matrix".format(opponent))
        total_records += len(receipt_rows)
        all_prediction_valid_steps += sum(row["prediction_valid_steps"] for row in receipt_rows)
        all_prediction_fallback_steps += sum(row["prediction_fallback_steps"] for row in receipt_rows)
        per_opponent[opponent] = {"records": receipt_rows, "passes": True}

    return {
        "source_id": SOURCE_ID,
        "formal_source_id": FORMAL_SOURCE_ID,
        "expected_records": len(OPPONENTS) * len(METHODS),
        "records": total_records,
        "prediction_valid_steps": all_prediction_valid_steps,
        "prediction_fallback_steps": all_prediction_fallback_steps,
        "per_opponent": per_opponent,
        "verdict": "pass",
        "paper_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_smoke_root(args.output_root)
    except (OSError, ValueError, SmokeVerificationError) as error:
        result = {
            "source_id": SOURCE_ID,
            "formal_source_id": FORMAL_SOURCE_ID,
            "verdict": "fail",
            "error": str(error),
            "paper_evidence": False,
        }
    output_path = args.output_root / "smoke_verification.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
