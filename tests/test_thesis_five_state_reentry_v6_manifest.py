from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_five_state_reentry_v6_manifest.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_reentry_v6_control12.yaml"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("v6_manifest_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v6_manifest_is_rebuildable_and_disjoint_from_all_frozen_sources():
    builder = _load(BUILDER)
    sources = {label: builder._load(path) for label, path in builder.v5.DISJOINT_MANIFESTS}
    rebuilt = builder.build_manifest(sources)
    tracked = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))

    assert rebuilt["integrity"]["payload_sha256"] == tracked["integrity"]["payload_sha256"]
    assert len(tracked["scenarios"]) == 12
    assert {item["metadata"]["initial_class"] for item in tracked["scenarios"]} == {"head_on", "crossing_entry"}
    assert all(item["physical_intersection_count"] == 0 for item in tracked["disjointness"].values())
