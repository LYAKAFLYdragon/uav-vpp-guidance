from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_thesis_global_advantage_p2_physical_preflight_manifest.py"
MANIFEST = ROOT / "config" / "experiment" / "manifests" / "thesis_global_advantage_v1_p2_physical_preflight60.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("p2_physical_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_preflight_manifest_has_all_geometry_cells_and_new_seeds():
    payload = _load(MANIFEST)
    scenarios = payload["scenarios"]
    assert payload["source_id"] == "THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT60-V1"
    assert len(scenarios) == 60
    assert len({item["metadata"]["geometry_cell_id"] for item in scenarios}) == 60
    assert all(item["metadata"]["preflight_only"] is True for item in scenarios)
    assert all(
        item["metadata"]["preflight_seed"] != item["metadata"]["source_heldout_evaluation_seed"]
        for item in scenarios
    )


def test_frozen_preflight_payload_hash_matches_builder():
    builder = _module()
    payload = _load(MANIFEST)
    expected = payload["integrity"]["payload_sha256"]
    hashed = copy.deepcopy(payload)
    hashed.pop("integrity")
    assert builder._sha256(hashed) == expected
