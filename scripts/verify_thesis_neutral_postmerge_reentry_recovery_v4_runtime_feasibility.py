#!/usr/bin/env python3
"""Independently verify a completed V4 runtime-feasibility artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_contract import (
    SOURCE_ID,
    NeutralPostMergeV4ContractError,
    build_plan,
    evaluate_family_coverage,
    runtime_identity_index,
    validate_feasibility_manifest,
)
from uav_vpp_guidance.evaluation.thesis_neutral_postmerge_v4_record_only import (
    NeutralPostMergeV4RecordError,
    runtime_record,
)


class V4ArtifactVerificationError(ValueError):
    """Raised when a V4 output bundle cannot independently prove its contract."""


REQUIRED_ARTIFACTS = (
    "authorized_overlay.yaml",
    "frozen_base_config.yaml",
    "frozen_feasibility_manifest.yaml",
    "authorization_preflight.json",
    "resolved_config.json",
    "artifact_provenance.json",
    "runtime_feasibility_gate.json",
    "run_manifest.json",
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
        raise V4ArtifactVerificationError(f"expected YAML mapping: {path.name}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise V4ArtifactVerificationError(f"expected JSON mapping: {path.name}")
    return payload


def _require(path: Path) -> None:
    if not path.is_file():
        raise V4ArtifactVerificationError(f"missing required artifact: {path.name}")


def _phase_replay(ledger: Mapping[str, Any]) -> bool:
    from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine

    tracker = engine.legacy.PhaseTracker()
    for step in ledger.get("steps", []):
        raw = step.get("raw_si_phase_input") if isinstance(step, Mapping) else None
        if not isinstance(raw, Mapping):
            return False
        expected = tracker.update(float(raw["range_m"]), float(raw["range_rate_mps"]))
        if expected != step.get("phase"):
            return False
    return True


def _verify_snapshots(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in REQUIRED_ARTIFACTS:
        _require(output_dir / name)
    failure = output_dir / "runner_failure_manifest.json"
    if failure.is_file():
        raise V4ArtifactVerificationError("runner_failure_manifest.json is present; V4 did not complete")
    overlay = _load_yaml(output_dir / "authorized_overlay.yaml")
    base = _load_yaml(output_dir / "frozen_base_config.yaml")
    manifest = _load_yaml(output_dir / "frozen_feasibility_manifest.yaml")
    authorization = _load_json(output_dir / "authorization_preflight.json")
    resolved = _load_json(output_dir / "resolved_config.json")
    provenance = _load_json(output_dir / "artifact_provenance.json")
    if base.get("source_id") != SOURCE_ID or base.get("status") != "preregistered_design_execution_not_authorised":
        raise V4ArtifactVerificationError("frozen base config source/status drifted")
    if resolved.get("source_id") != SOURCE_ID or resolved.get("status") != "authorized_one_time_execution":
        raise V4ArtifactVerificationError("resolved config source/status drifted")
    if not bool(resolved.get("authorization", {}).get("execution_permitted")):
        raise V4ArtifactVerificationError("resolved config did not record execution authorization")
    if "proposed_training" in resolved:
        raise V4ArtifactVerificationError("resolved V4 config contains a training block")
    feasibility = resolved.get("runtime_feasibility", {})
    if feasibility.get("training_reward") != "prohibited" or feasibility.get("environment_reward_recording") != "prohibited":
        raise V4ArtifactVerificationError("resolved V4 reward-use boundary drifted")
    if authorization.get("source_id") != SOURCE_ID or not bool(authorization.get("execution_permitted")):
        raise V4ArtifactVerificationError("authorization receipt drifted")
    if authorization.get("config_sha256") != _sha256_file(output_dir / "authorized_overlay.yaml"):
        raise V4ArtifactVerificationError("authorized overlay hash mismatch")
    if authorization.get("base_config_sha256") != _sha256_file(output_dir / "frozen_base_config.yaml"):
        raise V4ArtifactVerificationError("frozen base config hash mismatch")
    if provenance.get("source_id") != SOURCE_ID:
        raise V4ArtifactVerificationError("artifact provenance source drifted")
    if provenance.get("authorized_overlay_sha256") != authorization.get("config_sha256"):
        raise V4ArtifactVerificationError("artifact provenance overlay hash mismatch")
    if provenance.get("frozen_base_config_sha256") != authorization.get("base_config_sha256"):
        raise V4ArtifactVerificationError("artifact provenance base hash mismatch")
    manifest_hash = _sha256_file(output_dir / "frozen_feasibility_manifest.yaml")
    if resolved.get("inputs", {}).get("feasibility_manifest_sha256") != manifest_hash:
        raise V4ArtifactVerificationError("resolved config manifest hash mismatch")
    if provenance.get("frozen_feasibility_manifest_sha256") != manifest_hash:
        raise V4ArtifactVerificationError("artifact provenance manifest hash mismatch")
    if overlay.get("status") != "authorized_one_time_execution":
        raise V4ArtifactVerificationError("authorized overlay status drifted")
    return resolved, manifest, provenance, authorization


def _verify_record(record_payload: Mapping[str, Any], expected_source: str) -> dict[str, Any]:
    record = record_payload.get("record")
    ledger = record_payload.get("ledger")
    if not isinstance(record, Mapping) or not isinstance(ledger, Mapping):
        raise V4ArtifactVerificationError("episode artifact lacks record or ledger mapping")
    metadata = record.get("metadata")
    result = record.get("result")
    evidence = record.get("runtime_evidence")
    if record.get("source_id") != expected_source or not isinstance(metadata, Mapping):
        raise V4ArtifactVerificationError("episode record source or metadata drifted")
    if not isinstance(result, Mapping) or not isinstance(evidence, Mapping):
        raise V4ArtifactVerificationError("episode record result/evidence is invalid")
    if "reward" in result or result.get("training_reward") != "prohibited" or result.get("environment_reward_recording") != "prohibited":
        raise V4ArtifactVerificationError("episode record violates the reward-unobserved boundary")
    if bool(record.get("paired_delta_eligible")) or record.get("role") != "nonlearning_reward_unobserved_runtime_feasibility":
        raise V4ArtifactVerificationError("episode record role or paired-delta boundary drifted")
    if not _phase_replay(ledger):
        raise V4ArtifactVerificationError("episode raw-SI phase replay failed")
    steps = ledger.get("steps")
    if not isinstance(steps, list) or any("reward" in step for step in steps if isinstance(step, Mapping)):
        raise V4ArtifactVerificationError("episode telemetry contains a reward field")
    post_steps = [step for step in steps if isinstance(step, Mapping) and step.get("stage") == "post_handoff"]
    if bool(result.get("handoff_reached")):
        if not post_steps:
            raise V4ArtifactVerificationError("handoff record has no post-handoff telemetry")
        for step in post_steps:
            observation = step.get("observation_66d")
            action = step.get("normalized_vpp_action")
            if not isinstance(observation, list) or len(observation) != 66:
                raise V4ArtifactVerificationError("post-handoff telemetry lacks a 66-D observation")
            if not isinstance(action, list) or len(action) != 3:
                raise V4ArtifactVerificationError("post-handoff telemetry lacks a 3-D action")
    if int(result.get("valid_target_steps", 0)) > len(post_steps):
        raise V4ArtifactVerificationError("valid target steps exceed post-handoff telemetry")
    try:
        rebuilt = runtime_record(
            source_id=str(record["source_id"]),
            opponent=str(record["opponent"]),
            scenario={"metadata": metadata},
            result=result,
            ledger=ledger,
            runtime_evidence=evidence,
        )
    except (NeutralPostMergeV4RecordError, KeyError, TypeError) as error:
        raise V4ArtifactVerificationError(f"episode runtime record contract failed: {error}") from error
    if rebuilt != record:
        raise V4ArtifactVerificationError("episode record does not round-trip through the V4 serializer")
    return dict(record)


def verify_bundle(output_dir: Path) -> dict[str, Any]:
    """Read and independently validate a completed V4 output bundle."""

    output_dir = Path(output_dir)
    resolved, manifest, provenance, authorization = _verify_snapshots(output_dir)
    try:
        plan = build_plan(resolved)
        validate_feasibility_manifest(manifest)
        expected = runtime_identity_index(manifest)
    except NeutralPostMergeV4ContractError as error:
        raise V4ArtifactVerificationError(f"V4 design contract failed: {error}") from error
    episode_root = output_dir / "episodes"
    if not episode_root.is_dir():
        raise V4ArtifactVerificationError("episodes directory is missing")
    paths = sorted(episode_root.rglob("*.json"))
    records = [_verify_record(_load_json(path), SOURCE_ID) for path in paths]
    try:
        recomputed_gate = evaluate_family_coverage(records, plan, expected)
    except NeutralPostMergeV4ContractError as error:
        raise V4ArtifactVerificationError(f"V4 record universe/gate failed: {error}") from error
    recomputed_gate["training_unlocked"] = False
    stored_gate = _load_json(output_dir / "runtime_feasibility_gate.json")
    run_manifest = _load_json(output_dir / "run_manifest.json")
    if stored_gate != recomputed_gate:
        raise V4ArtifactVerificationError("stored runtime gate does not match independent recomputation")
    if run_manifest.get("source_id") != SOURCE_ID or int(run_manifest.get("record_count", -1)) != len(records):
        raise V4ArtifactVerificationError("run manifest source or record count drifted")
    if run_manifest.get("gate") != recomputed_gate:
        raise V4ArtifactVerificationError("run manifest gate does not match independent recomputation")
    if run_manifest.get("training_reward") != "prohibited" or run_manifest.get("environment_reward_recording") != "prohibited":
        raise V4ArtifactVerificationError("run manifest reward-use boundary drifted")
    return {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "verification_status": "pass",
        "output_dir": str(output_dir),
        "record_count": len(records),
        "planned_record_count": 144,
        "authorization_git_sha": authorization.get("git_sha"),
        "artifact_provenance": provenance,
        "recomputed_gate": recomputed_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = verify_bundle(args.output_dir)
        status = 0
    except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as error:
        report = {
            "schema_version": 1,
            "source_id": SOURCE_ID,
            "verification_status": "fail",
            "output_dir": str(args.output_dir),
            "error": str(error),
        }
        status = 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
