from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_thesis_global_advantage_p2_b3_phase_observability_manifest.py"


def _module():
    spec = importlib.util.spec_from_file_location("p2_b3_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p2_b3_manifest_covers_taxonomy_and_stays_disjoint():
    builder = _module()
    sources = {label: yaml.safe_load(path.read_text(encoding="utf-8")) for label, path in zip(
        ("p2_b2", "heldout240", "dev30", "historical_heldout60"), builder.DISJOINT_MANIFESTS
    )}
    manifest = builder.build_manifest(sources)
    scenarios = manifest["scenarios"]

    assert manifest["source_id"] == builder.SOURCE_ID
    assert len(scenarios) == 30
    assert {item["metadata"]["initial_class"] for item in scenarios} == {
        "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"
    }
    assert {item["metadata"]["height_condition"] for item in scenarios} == {
        "own_below", "co_altitude", "own_above"
    }
    assert {item["metadata"]["mirror_sign"] for item in scenarios} == {"negative", "positive"}
    assert all(item["metadata"]["phase_at_reset"] == "pre_merge" for item in scenarios)
