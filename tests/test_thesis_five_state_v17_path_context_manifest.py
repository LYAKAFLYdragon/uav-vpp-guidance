from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _load_builder():
    path = ROOT / "scripts" / "build_thesis_five_state_v17_path_context_manifest.py"
    spec = importlib.util.spec_from_file_location("v17_manifest_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v17_manifest_has_two_disjoint_packages_and_the_full_24_cell_universe():
    builder = _load_builder()
    sources = {
        label: builder.v6._load(path)
        for label, path in (*builder.v6.v5.DISJOINT_MANIFESTS, *builder.EXTRA_MANIFESTS)
    }
    manifest = builder.build_manifest(sources)
    assert manifest["source_id"] == builder.MANIFEST_SOURCE_ID
    assert len(manifest["scenarios"]) == 24
    assert {item["metadata"]["distance_speed_package"] for item in manifest["scenarios"]} == {
        "v17_path_context_a",
        "v17_path_context_b",
    }
    assert len({item["metadata"]["pair_key"] for item in manifest["scenarios"]}) == 24
    assert manifest["protocol"]["training_permitted"] is False
    assert manifest["disjointness"]["v16"]["physical_intersection_count"] == 0
    assert len(manifest["integrity"]["payload_sha256"]) == 64


def test_frozen_v17_manifest_matches_the_builder_payload():
    builder = _load_builder()
    frozen = yaml.safe_load(
        (ROOT / "config/experiment/manifests/thesis_five_state_v17_path_context_preflight24.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["source_id"] == builder.MANIFEST_SOURCE_ID
    assert len(frozen["scenarios"]) == 24
    assert frozen["integrity"]["payload_sha256"] == "1c79c22d78976b148115e39c8c77d75869afed708c2c981eb934061cc6f52351"
