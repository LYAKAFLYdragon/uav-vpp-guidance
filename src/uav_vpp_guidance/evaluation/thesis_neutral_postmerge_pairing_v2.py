"""Manifest pairing contract for the V2 neutral/post-merge pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SOURCE_ID = "THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2"


class NeutralPostMergeV2PairingError(ValueError):
    """Raised when V2 cannot establish a unique physical pairing key."""


def payload_sha256(payload: Mapping[str, Any]) -> str:
    prepared = copy.deepcopy(dict(payload))
    if isinstance(prepared.get("integrity"), dict):
        prepared["integrity"].pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def pair_key(scenario: Mapping[str, Any]) -> str:
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        raise NeutralPostMergeV2PairingError("scenario metadata is missing")
    cell, seed, serialized = metadata.get("geometry_cell_id"), metadata.get("scenario_seed"), metadata.get("pair_key")
    if not isinstance(cell, str) or not cell or isinstance(seed, bool) or not isinstance(seed, int):
        raise NeutralPostMergeV2PairingError("scenario lacks geometry_cell_id or integer scenario_seed")
    expected = f"{cell}::seed={seed}"
    if serialized != expected:
        raise NeutralPostMergeV2PairingError("serialized pair_key does not match geometry_cell_id/scenario_seed")
    return expected


def validate_manifest(manifest: Mapping[str, Any], *, split: str, expected_count: int) -> list[str]:
    if manifest.get("source_id") != f"{SOURCE_ID}-{split.upper()}" or manifest.get("split") != split:
        raise NeutralPostMergeV2PairingError("V2 manifest identity drifted")
    if manifest.get("status") != "preregistered_design_only":
        raise NeutralPostMergeV2PairingError("V2 manifest status drifted")
    if payload_sha256(manifest) != manifest.get("integrity", {}).get("payload_sha256"):
        raise NeutralPostMergeV2PairingError("V2 manifest payload SHA drifted")
    scenarios = list(manifest.get("scenarios") or [])
    if len(scenarios) != expected_count:
        raise NeutralPostMergeV2PairingError("V2 manifest scenario count drifted")
    keys = [pair_key(item) for item in scenarios]
    if len(set(keys)) != expected_count:
        raise NeutralPostMergeV2PairingError("V2 pair keys are not unique")
    for scenario in scenarios:
        metadata = scenario["metadata"]
        if (metadata.get("initial_class"), metadata.get("taxonomy_geometry_state"), metadata.get("phase_at_reset")) != ("neutral", "neutral", "pre_merge"):
            raise NeutralPostMergeV2PairingError("V2 scenario geometry/phase contract drifted")
        if metadata.get("continuous_run_in") is not True or metadata.get("routing_enabled") is not False:
            raise NeutralPostMergeV2PairingError("V2 continuous run-in/routing contract drifted")
    return keys


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise NeutralPostMergeV2PairingError(f"expected YAML mapping: {path}")
    return payload
