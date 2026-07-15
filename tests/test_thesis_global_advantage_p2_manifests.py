from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_thesis_global_advantage_p2_manifests.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_global_advantage_v1_heldout240.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("global_advantage_p2_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_builder_creates_disjoint_60_cell_four_seed_manifest():
    builder = _module()
    manifest = builder.build_manifest(
        _load(builder.TRAIN_DISTRIBUTION_PATH),
        _load(builder.DEV30_PATH),
        _load(builder.HISTORICAL_HELDOUT60_PATH),
    )

    assert manifest["source_id"] == builder.SOURCE_ID
    assert manifest["split"] == "heldout240"
    assert len(manifest["scenarios"]) == 240
    assert manifest["generation_contract"]["geometry_cell_count"] == 60
    assert manifest["generation_contract"]["evaluation_seed_count_per_cell"] == 4
    assert manifest["disjointness"]["dev30_instance_intersection_count"] == 0
    assert manifest["disjointness"]["dev30_physical_intersection_count"] == 0
    assert manifest["disjointness"]["historical_heldout60_instance_intersection_count"] == 0
    assert manifest["disjointness"]["historical_heldout60_physical_intersection_count"] == 0
    assert manifest["disjointness"]["train_support_intersection_count"] == 0


def test_frozen_heldout240_payload_hash_and_roles_match_builder():
    builder = _module()
    payload = _load(MANIFEST)
    assert payload["protocol"]["selection_permitted"] is False
    assert payload["protocol"]["training_permitted"] is False
    assert payload["protocol"]["allowed_use"] == "formal_evaluation_only_after_p2_physical_reachability_pass"
    frozen_hash = payload["integrity"]["payload_sha256"]
    hashed = copy.deepcopy(payload)
    hashed["integrity"].pop("payload_sha256")
    assert builder._payload_sha256(hashed) == frozen_hash
