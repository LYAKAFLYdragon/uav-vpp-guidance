from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from uav_vpp_guidance.evaluation import thesis_neutral_postmerge_pairing_v2 as pairing


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_neutral_postmerge_reentry_recovery_pilot_v2_manifests.py"
DEV = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_dev12.yaml"
HELDOUT = ROOT / "config" / "experiment" / "manifests" / "thesis_neutral_postmerge_reentry_recovery_pilot_v2_heldout24.yaml"


def _load_builder():
    spec = importlib.util.spec_from_file_location("neutral_v2_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_manifests_expose_unique_round_trip_pair_keys():
    builder = _load_builder()
    dev, heldout = yaml.safe_load(DEV.read_text(encoding="utf-8")), yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))
    assert pairing.validate_manifest(dev, split="dev12", expected_count=12) == [item["metadata"]["pair_key"] for item in dev["scenarios"]]
    assert len(pairing.validate_manifest(heldout, split="heldout24", expected_count=24)) == 24
    assert builder.payload_sha256(dev) == dev["integrity"]["payload_sha256"]
    assert all(item["physical_intersection_count"] == 0 and item["seed_intersection_count"] == 0 for item in dev["disjointness"].values())


def test_v2_pairing_fails_closed_if_serialized_key_drifts():
    manifest = yaml.safe_load(DEV.read_text(encoding="utf-8"))
    manifest["scenarios"][0]["metadata"]["pair_key"] = "incorrect"
    manifest["integrity"]["payload_sha256"] = pairing.payload_sha256(manifest)
    with pytest.raises(pairing.NeutralPostMergeV2PairingError, match="serialized pair_key"):
        pairing.validate_manifest(manifest, split="dev12", expected_count=12)
