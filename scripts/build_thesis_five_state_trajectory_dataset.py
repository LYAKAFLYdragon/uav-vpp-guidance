"""Freeze train-only trajectory provenance for temporal-encoder preparation.

This script deliberately does not collect trajectories.  P2 only freezes the
schema and leakage controls so that P3 cannot silently reuse dev/heldout data
or fit normalisation statistics on evaluation trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISTRIBUTION = (
    REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
)
DEFAULT_DEV = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_dev30.yaml"
DEFAULT_HELDOUT = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_heldout60.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_trajectory_dataset_provenance.yaml"
RESULT_ROOT = Path("E:/uav-vpp-guidance-thesis-five-state-v1-results")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def build_provenance(
    distribution: Mapping[str, Any], dev_manifest: Mapping[str, Any], heldout_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if distribution.get("integrity", {}).get("dev30_payload_sha256") != dev_manifest.get(
        "integrity", {}
    ).get("payload_sha256"):
        raise ValueError("Distribution is not frozen against the supplied dev30 manifest")
    if distribution.get("integrity", {}).get("heldout60_payload_sha256") != heldout_manifest.get(
        "integrity", {}
    ).get("payload_sha256"):
        raise ValueError("Distribution is not frozen against the supplied heldout60 manifest")
    return {
        "schema_version": 1,
        "source_id": "THESIS-FIVE-STATE-SHARED-INTENT-V1-TRAJECTORY-PROVENANCE",
        "lane": "noncanonical_thesis_extension",
        "family": "thesis_five_state_shared_intent_v1",
        "status": "schema_frozen_no_training_trajectories_collected",
        "collection_scope": {
            "allowed_split": "train",
            "prohibited_splits": ["dev30", "heldout60", "taxonomy30"],
            "allowed_phase": "P3_temporal_encoder_pretraining_only",
            "raw_trajectory_root": str(RESULT_ROOT / "trajectory_datasets" / "train_only_v1" / "raw"),
            "index_path": str(RESULT_ROOT / "trajectory_datasets" / "train_only_v1" / "train_episode_index.jsonl"),
            "full_raw_telemetry_retention": "prohibited_during_skill_and_high_level_training",
        },
        "record_contract": {
            "episode_id": "globally_unique_train_only_identifier",
            "source_distribution_seed": "required",
            "opponent_id": "required",
            "initial_class": "required",
            "height_condition": "required",
            "mirror_sign": "required",
            "step_index": "strictly_increasing_within_episode",
            "reset_boundary": "required",
            "observation_history_window_steps": 10,
            "future_state_or_label": "prohibited",
        },
        "normalization_contract": {
            "fit_split": "train_only",
            "fit_after_train_episode_index_freeze": True,
            "dev30_usage": "transform_only_never_fit",
            "heldout60_usage": "transform_only_never_fit",
            "statistics_artifact": str(RESULT_ROOT / "trajectory_datasets" / "train_only_v1" / "normalization_train_only.json"),
        },
        "provenance_sources": {
            "train_distribution_source_id": distribution["source_id"],
            "train_distribution_payload_sha256": distribution["integrity"]["payload_sha256"],
            "dev30_source_id": dev_manifest["source_id"],
            "dev30_payload_sha256": dev_manifest["integrity"]["payload_sha256"],
            "heldout60_source_id": heldout_manifest["source_id"],
            "heldout60_payload_sha256": heldout_manifest["integrity"]["payload_sha256"],
        },
        "stop_rule": (
            "If an episode cannot prove train-only provenance or chronological order, "
            "exclude it; do not substitute dev30, heldout60, or Taxonomy30 data."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--heldout-manifest", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    provenance = build_provenance(
        _load_yaml(args.train_distribution),
        _load_yaml(args.dev_manifest),
        _load_yaml(args.heldout_manifest),
    )
    provenance["integrity"] = {
        "builder_code_sha256": _sha256_file(Path(__file__)),
    }
    provenance["integrity"]["payload_sha256"] = _sha256_bytes(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": provenance["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
