from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_thesis_global_advantage_p2_b4_raw_si_manifest.py"


def _module():
    spec = importlib.util.spec_from_file_location("p2_b4_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p2_b4_manifest_is_disjoint_and_covers_five_state_grid():
    builder = _module()
    labels = ("p2_b2", "p2_b3", "heldout240", "dev30", "historical_heldout60")
    sources = {label: yaml.safe_load(path.read_text(encoding="utf-8")) for label, path in zip(labels, builder.DISJOINT_MANIFESTS)}
    manifest = builder.build_manifest(sources)

    assert manifest["source_id"] == builder.SOURCE_ID
    assert len(manifest["scenarios"]) == 30
    assert all(item["metadata"]["phase_at_reset"] == "pre_merge" for item in manifest["scenarios"])
    assert {item["metadata"]["initial_class"] for item in manifest["scenarios"]} == {
        "advantage", "head_on", "disadvantage", "neutral", "crossing_entry"
    }
