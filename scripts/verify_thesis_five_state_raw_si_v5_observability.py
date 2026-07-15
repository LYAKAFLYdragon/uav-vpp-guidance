#!/usr/bin/env python3
"""Independently verify a complete V5 raw-SI observability artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.raw_si_first_pass_event import RawSIFirstPassConfig, replay_raw_si_first_pass
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_contract import (
    SOURCE_ID,
    FiveStateRawSIV5ContractError,
    build_plan,
    evaluate_observability,
    expected_pairs_by_state,
    validate_manifest,
)
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import (
    FiveStateRawSIV5RecordError,
    runtime_record,
)


class V5ArtifactVerificationError(ValueError):
    """Raised when a V5 bundle cannot prove its frozen runtime contract."""


REQUIRED_ARTIFACTS = (
    "authorized_overlay.yaml",
    "frozen_base_config.yaml",
    "frozen_observability_manifest.yaml",
    "authorization_preflight.json",
    "resolved_config.json",
    "artifact_provenance.json",
    "raw_si_observability_gate.json",
    "run_manifest.json",
    "episodes",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise V5ArtifactVerificationError("expected YAML mapping: {0}".format(path))
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V5ArtifactVerificationError("expected JSON mapping: {0}".format(path))
    return payload


def _require(path: Path) -> None:
    if not path.exists():
        raise V5ArtifactVerificationError("required artifact is missing: {0}".format(path.name))


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolved_snapshot(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_path = output_dir / "frozen_base_config.yaml"
    overlay_path = output_dir / "authorized_overlay.yaml"
    manifest_path = output_dir / "frozen_observability_manifest.yaml"
    base, overlay, manifest = _load_yaml(base_path), _load_yaml(overlay_path), _load_yaml(manifest_path)
    if _sha256_file(base_path) != str(overlay.get("execution", {}).get("base_config_sha256", "")).lower():
        raise V5ArtifactVerificationError("authorized overlay does not hash the frozen base config")
    resolved = _merge(base, {key: value for key, value in overlay.items() if key != "base_config"})
    recorded = _load_json(output_dir / "resolved_config.json")
    if resolved != recorded:
        raise V5ArtifactVerificationError("resolved config snapshot does not match frozen base plus overlay")
    return resolved, manifest, _load_json(output_dir / "artifact_provenance.json")


def _validate_ledger(ledger: Mapping[str, Any], record: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    steps = ledger.get("steps")
    summary = ledger.get("summary")
    if not isinstance(steps, list) or not steps or not isinstance(summary, Mapping):
        raise V5ArtifactVerificationError("V5 ledger lacks steps or summary")
    event_cfg = RawSIFirstPassConfig(**config["observability"]["raw_si_first_pass_event"])
    event = replay_raw_si_first_pass(steps, event_cfg)
    if bool(event is not None) != bool(record["result"].get("raw_event_reached")):
        raise V5ArtifactVerificationError("raw-SI replay disagrees with V5 record event")
    post_event_66d = 0
    for step in steps:
        if "reward" in step or step.get("history_padding_used") is not False:
            raise V5ArtifactVerificationError("V5 ledger leaked reward or history padding")
        base = np.asarray(step.get("base_observation_16d"), dtype=np.float64).reshape(-1)
        action = np.asarray(step.get("normalized_vpp_action"), dtype=np.float64).reshape(-1)
        if base.shape != (16,) or action.shape != (3,) or not np.isfinite(base).all() or not np.isfinite(action).all():
            raise V5ArtifactVerificationError("V5 ledger has invalid 16-D base or 3-D action")
        obs66 = step.get("observation_66d")
        if obs66 is not None:
            vector = np.asarray(obs66, dtype=np.float64).reshape(-1)
            if vector.shape != (66,) or not np.isfinite(vector).all() or int(step.get("history_length", -1)) != 10:
                raise V5ArtifactVerificationError("V5 ledger has invalid continuous 66-D observation")
        if bool(step.get("post_event_66d_valid")):
            if obs66 is None or not bool(step.get("raw_si_first_pass_latched")):
                raise V5ArtifactVerificationError("V5 post-event 66-D marker is inconsistent")
            post_event_66d += 1
        if step.get("taxonomy_angle_order") != ["ATA", "AA"]:
            raise V5ArtifactVerificationError("V5 taxonomy angle-order drifted")
    if post_event_66d != int(record["result"].get("post_event_66d_steps", -1)):
        raise V5ArtifactVerificationError("V5 post-event 66-D count disagrees with the ledger")
    if bool(summary.get("raw_si_event_replay_passed")) is not True:
        raise V5ArtifactVerificationError("V5 ledger does not certify raw-SI replay")


def verify_bundle(output_dir: Path) -> dict[str, Any]:
    for name in REQUIRED_ARTIFACTS:
        _require(output_dir / name)
    config, manifest, provenance = _resolved_snapshot(output_dir)
    if config.get("source_id") != SOURCE_ID or config.get("authorization", {}).get("execution_permitted") is not True:
        raise V5ArtifactVerificationError("bundle does not contain an authorized V5 configuration")
    plan = build_plan(config)
    validate_manifest(manifest)
    if provenance.get("source_id") != SOURCE_ID or provenance.get("training_reward") != "prohibited":
        raise V5ArtifactVerificationError("V5 provenance drifted from its reward-free contract")
    records: list[dict[str, Any]] = []
    expected = expected_pairs_by_state(manifest)
    for opponent in plan.opponents:
        episode_paths = sorted((output_dir / "episodes" / opponent).glob("*.json"))
        if len(episode_paths) != 60:
            raise V5ArtifactVerificationError("V5 bundle must contain 60 records for {0}".format(opponent))
        for path in episode_paths:
            payload = _load_json(path)
            record, ledger = payload.get("record"), payload.get("ledger")
            if not isinstance(record, Mapping) or not isinstance(ledger, Mapping):
                raise V5ArtifactVerificationError("V5 episode payload lacks record or ledger")
            if record.get("source_id") != SOURCE_ID or record.get("opponent") != opponent:
                raise V5ArtifactVerificationError("V5 record identity drifted")
            try:
                runtime_record(
                    opponent=opponent,
                    scenario={"metadata": record["metadata"]},
                    result=record["result"],
                    ledger=ledger,
                    runtime_evidence=record["runtime_evidence"],
                )
            except (FiveStateRawSIV5RecordError, KeyError, TypeError) as error:
                raise V5ArtifactVerificationError("V5 runtime record is invalid: {0}".format(error)) from error
            _validate_ledger(ledger, record, config)
            records.append(dict(record))
    recomputed = evaluate_observability(records, plan, expected)
    recorded_gate = _load_json(output_dir / "raw_si_observability_gate.json")
    if recorded_gate != recomputed:
        raise V5ArtifactVerificationError("V5 recorded gate differs from independent recomputation")
    run_manifest = _load_json(output_dir / "run_manifest.json")
    if run_manifest.get("record_count") != 180 or run_manifest.get("gate") != recomputed:
        raise V5ArtifactVerificationError("V5 run manifest is incomplete or inconsistent")
    return {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "verification_status": "pass",
        "output_dir": str(output_dir),
        "record_count": len(records),
        "recomputed_gate": recomputed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = verify_bundle(args.output_dir)
        status = 0
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError, FiveStateRawSIV5ContractError) as error:
        report = {"schema_version": 1, "source_id": SOURCE_ID, "verification_status": "fail", "error": str(error)}
        status = 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
