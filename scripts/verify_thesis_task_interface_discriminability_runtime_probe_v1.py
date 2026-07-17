#!/usr/bin/env python3
"""Independent verifier for the one-time task-interface runtime probe."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
PROBE_ID = "ENGINEERING-TASK-INTERFACE-DISCRIMINABILITY-RUNTIME-PROBE-V1-NOT-A-PAPER-SOURCE-ID"
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
PRIMARY = ("expert", "end_to_end")
METHODS = ("canonical_ppo", "ppo_unbootstrapped_crossing")
CONTEXTS = ("head_on", "crossing_feasible")
CONFIG = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_task_interface_discriminability_runtime_probe_v1_authorized.yaml"


class VerificationError(RuntimeError):
    pass


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_finite(item) for item in value)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _initial_observation(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Read either the original flat runtime receipt or a nested receipt."""
    nested = record.get("initial_observation")
    if isinstance(nested, Mapping):
        return nested
    if isinstance(record.get("feature_names"), list) and isinstance(record.get("vector"), list):
        return {"feature_names": record["feature_names"], "vector": record["vector"]}
    return None


def _verify_initial(record: Mapping[str, Any], context: str) -> None:
    initial = _initial_observation(record)
    if not isinstance(initial, Mapping):
        raise VerificationError("{} lacks initial observation receipt".format(context))
    names, vector = initial.get("feature_names"), initial.get("vector")
    if not isinstance(names, list) or not isinstance(vector, list) or len(names) != 19 or len(vector) != 19:
        raise VerificationError("{} does not have the 19-D canonical observation receipt".format(context))
    if names.count("is_crossing") != 1 or not all(_finite(value) for value in vector):
        raise VerificationError("{} has an unreadable task-bit observation receipt".format(context))
    receipt = record.get("physical_reset_receipt")
    if not isinstance(receipt, Mapping) or not _vector(receipt.get("own_position_m"), 3) or not _vector(receipt.get("target_position_m"), 3):
        raise VerificationError("{} lacks a raw JSBSim reset receipt".format(context))


def _verify_trajectory(record: Mapping[str, Any], context: str) -> None:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != 20:
        raise VerificationError("{} does not contain the frozen 20-step window".format(context))
    for index, step in enumerate(trajectory):
        if not isinstance(step, Mapping) or not _vector(step.get("vpp_action"), 3):
            raise VerificationError("{} has invalid VPP telemetry at step {}".format(context, index + 1))
        if not _vector(step.get("own_pos_m"), 3) or not _vector(step.get("target_pos_m"), 3):
            raise VerificationError("{} has invalid raw physical state at step {}".format(context, index + 1))
        for key in ("aoa_deg", "nz_cmd", "nz_actual"):
            if not _finite(step.get(key)):
                raise VerificationError("{} lacks {} at step {}".format(context, key, index + 1))
        # Backend provenance is stored once per episode, and is verified before
        # this function is called; raw telemetry steps intentionally omit it.
        valid, fallback = step.get("prediction_valid"), step.get("prediction_fallback")
        if not isinstance(valid, bool) or not isinstance(fallback, bool):
            raise VerificationError("{} has invalid prediction receipt".format(context))
        if index < 9 and (valid or not fallback):
            raise VerificationError("{} violates prediction warm-up".format(context))
        if index >= 9 and (not valid or fallback):
            raise VerificationError("{} has prediction fallback after warm-up".format(context))


def _same_except_task_bit(left: Mapping[str, Any], right: Mapping[str, Any], context: str) -> None:
    for key in ("own_position_m", "target_position_m"):
        if left["physical_reset_receipt"][key] != right["physical_reset_receipt"][key]:
            raise VerificationError("{} differs in raw reset {}".format(context, key))
    left_initial, right_initial = _initial_observation(left), _initial_observation(right)
    if left_initial is None or right_initial is None:
        raise VerificationError("{} lacks an initial observation receipt".format(context))
    if left_initial["feature_names"] != right_initial["feature_names"]:
        raise VerificationError("{} observation schemas differ".format(context))
    task_index = left_initial["feature_names"].index("is_crossing")
    differences = [index for index, (a, b) in enumerate(zip(left_initial["vector"], right_initial["vector"])) if abs(float(a) - float(b)) > 1e-9]
    if differences != [task_index] or float(left_initial["vector"][task_index]) != 0.0 or float(right_initial["vector"][task_index]) != 1.0:
        raise VerificationError("{} does not differ only at is_crossing".format(context))


def _first_difference(canonical: Mapping[str, Any], ablated: Mapping[str, Any]) -> bool:
    left, right = canonical["trajectory"][0], ablated["trajectory"][0]
    if left.get("commander_mode_id") != right.get("commander_mode_id"):
        return True
    return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(left["vpp_action"], right["vpp_action"]))


def verify(root: Path, config_path: Path = CONFIG) -> dict[str, Any]:
    root = Path(root)
    config = _load_config(config_path)
    protocol = config.get("task_interface_discriminability_runtime_probe", {})
    if protocol.get("probe_id") != PROBE_ID or protocol.get("execution_permitted") is not True:
        raise VerificationError("authorized probe contract mismatch")
    manifest = json.loads((root / "execution_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("probe_id") != PROBE_ID or manifest.get("status") != "completed" or int(manifest.get("records", 0)) != 48:
        raise VerificationError("execution manifest is incomplete")
    records = json.loads((root / "records.json").read_text(encoding="utf-8")).get("records")
    if not isinstance(records, list) or len(records) != 48:
        raise VerificationError("records file is not the 48-record probe")
    grouped: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for record in records:
        identity = record.get("probe_identity", {})
        opponent, physical, method, task = (str(identity.get(key)) for key in ("opponent", "physical_scenario_id", "method", "context"))
        context = "{}/{}/{}/{}".format(opponent, physical, method, task)
        if opponent not in OPPONENTS or method not in METHODS or task not in CONTEXTS:
            raise VerificationError("{} is outside the frozen matrix".format(context))
        if record.get("backend") != "jsbsim" or record.get("strict_backend") is not True or record.get("backend_fallback_occurred") is not False:
            raise VerificationError("{} violates strict JSBSim".format(context))
        _verify_initial(record, context)
        _verify_trajectory(record, context)
        grouped[(opponent, physical, method, task)] = record
    if len(grouped) != 48:
        raise VerificationError("probe contains duplicate identities")
    physical_ids = sorted({key[1] for key in grouped})
    if len(physical_ids) != 4:
        raise VerificationError("probe lacks four physical reset pairs")
    gate_counts: dict[str, int] = {}
    for opponent in OPPONENTS:
        different = 0
        for physical in physical_ids:
            for method in METHODS:
                _same_except_task_bit(grouped[(opponent, physical, method, "head_on")], grouped[(opponent, physical, method, "crossing_feasible")], "{}/{}".format(opponent, physical))
            canonical = grouped[(opponent, physical, "canonical_ppo", "crossing_feasible")]
            ablated = grouped[(opponent, physical, "ppo_unbootstrapped_crossing", "crossing_feasible")]
            first_canonical, first_ablated = canonical["trajectory"][0], ablated["trajectory"][0]
            if first_canonical.get("commander_crossing_initial_task_bootstrap_active") is not True or first_ablated.get("commander_crossing_initial_task_bootstrap_active") is not False:
                raise VerificationError("{} crossing bootstrap telemetry is invalid".format(opponent))
            if first_canonical.get("commander_crossing_initial_task_bootstrap_enabled") is not True or first_ablated.get("commander_crossing_initial_task_bootstrap_enabled") is not False:
                raise VerificationError("{} bootstrap enablement telemetry is invalid".format(opponent))
            different += int(_first_difference(canonical, ablated))
        gate_counts[opponent] = different
    threshold = int(protocol["runtime_gate"]["min_context_pairs_per_primary_opponent"])
    passes = all(gate_counts[opponent] >= threshold for opponent in PRIMARY)
    return {
        "probe_id": PROBE_ID,
        "verdict": "pass" if passes else "fail",
        "records": len(records),
        "first_effective_mode_or_vpp_difference_pairs": gate_counts,
        "threshold_per_primary_opponent": threshold,
        "formal_m3_m4_ablation_permitted": False,
        "next_action": "separately_design_formal_source" if passes else "stop_m3_m4_ablation",
        "boundary": "This result establishes or rejects interface discriminability only; it is not a performance result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    try:
        result = verify(args.output_root.resolve(), args.config.resolve())
    except (OSError, ValueError, KeyError, TypeError, VerificationError) as error:
        result = {"probe_id": PROBE_ID, "verdict": "fail", "records": 0, "error": str(error)}
    (args.output_root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
