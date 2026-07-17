from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_thesis_disadvantage_pressure_exposed_postmerge_manifest.py"


def _module():
    spec = importlib.util.spec_from_file_location("pressure_manifest", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_is_disjoint_balanced_and_deterministic() -> None:
    manifest = _module().build_manifest()
    scenarios = manifest["scenarios"]
    assert manifest["source_id"].endswith("MANIFEST16")
    assert len(scenarios) == 16
    assert len({row["name"] for row in scenarios}) == 16
    assert {row["metadata"]["initial_class"] for row in scenarios} == {"disadvantage"}
    assert {row["metadata"]["mirror_sign"] for row in scenarios} == {"negative", "positive"}
    assert len({row["metadata"]["mirror_pair_id"] for row in scenarios}) == 8
    assert manifest["integrity"]["scenario_count"] == 16
    assert manifest == _module().build_manifest()
