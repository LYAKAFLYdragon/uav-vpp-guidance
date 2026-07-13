from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PROVENANCE_PATH = REPO_ROOT / "reports" / "thesis_five_state_shared_intent_v1_trajectory_dataset_provenance.yaml"
DISTRIBUTION_PATH = REPO_ROOT / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"
DEV_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_dev30.yaml"
HELDOUT_PATH = REPO_ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_heldout60.yaml"


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_trajectory_provenance_is_train_only_and_freezes_normalization_boundary():
    provenance = _load(PROVENANCE_PATH)
    distribution = _load(DISTRIBUTION_PATH)
    dev = _load(DEV_PATH)
    heldout = _load(HELDOUT_PATH)

    assert provenance["status"] == "schema_frozen_no_training_trajectories_collected"
    assert provenance["collection_scope"]["allowed_split"] == "train"
    assert set(provenance["collection_scope"]["prohibited_splits"]) == {
        "dev30", "heldout60", "taxonomy30"
    }
    assert provenance["collection_scope"]["full_raw_telemetry_retention"] == (
        "prohibited_during_skill_and_high_level_training"
    )
    assert provenance["normalization_contract"]["fit_split"] == "train_only"
    assert provenance["normalization_contract"]["dev30_usage"] == "transform_only_never_fit"
    assert provenance["normalization_contract"]["heldout60_usage"] == "transform_only_never_fit"
    assert provenance["provenance_sources"]["train_distribution_payload_sha256"] == distribution["integrity"]["payload_sha256"]
    assert provenance["provenance_sources"]["dev30_payload_sha256"] == dev["integrity"]["payload_sha256"]
    assert provenance["provenance_sources"]["heldout60_payload_sha256"] == heldout["integrity"]["payload_sha256"]
