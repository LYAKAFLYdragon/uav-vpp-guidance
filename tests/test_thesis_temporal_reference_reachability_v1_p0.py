from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


def _module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p0_manifest_is_disjoint_and_complete() -> None:
    module = _module("temporal_reference_manifest_builder", "build_thesis_temporal_reference_reachability_v1_manifest.py")
    manifest = module.build_manifest()
    module.validate_manifest(manifest)
    assert manifest["integrity"]["scenario_count"] == 12
    assert len({item["metadata"]["mirror_pair_id"] for item in manifest["scenarios"]}) == 6
    assert min(item["metadata"]["scenario_seed"] for item in manifest["scenarios"]) == 275100
    assert manifest["protocol"]["evaluation_only"] is True
    assert manifest["protocol"]["training_permitted"] is False


def test_p0_source_snapshot_validates_all_frozen_inputs(tmp_path: Path) -> None:
    module = _module("temporal_reference_source_manifest", "build_thesis_temporal_reference_reachability_v1_source_manifest.py")
    snapshot = module.build_source_manifest(require_clean_git=False)
    assert snapshot["source_id"] == module.SOURCE_ID
    assert snapshot["authorization"]["execution_permitted"] is False
    assert snapshot["temporal_reference_contract"]["duration_s"] == 8.0
    assert snapshot["temporal_reference_contract"]["template_names"] == [
        "lateral_extension",
        "vertical_extension",
        "combined_break_extension",
    ]
    output = tmp_path / "source_manifest.json"
    output.write_text(json.dumps(snapshot), encoding="utf-8")
    assert module.verify_source_manifest(output)["passes"] is True
