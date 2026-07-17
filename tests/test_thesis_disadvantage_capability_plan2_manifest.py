from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


def _module():
    path = ROOT / "scripts" / "build_thesis_disadvantage_capability_plan2_manifest.py"
    spec = importlib.util.spec_from_file_location("plan2_manifest_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan2_manifest_has_complete_mirrors_and_a_new_seed_range():
    module = _module()
    manifest = module.build_manifest()
    module.validate_manifest(manifest)

    assert manifest["integrity"]["scenario_count"] == 12
    assert len({item["metadata"]["mirror_pair_id"] for item in manifest["scenarios"]}) == 6
    assert {item["metadata"]["height_condition"] for item in manifest["scenarios"]} == {
        "own_below",
        "co_altitude",
        "own_above",
    }
    assert min(item["metadata"]["scenario_seed"] for item in manifest["scenarios"]) == 274100


def test_plan2_manifest_rejects_hash_drift():
    module = _module()
    manifest = module.build_manifest()
    manifest["scenarios"][0]["metadata"]["scenario_seed"] = -1

    try:
        module.validate_manifest(manifest)
    except ValueError as error:
        assert "integrity hash mismatch" in str(error)
    else:
        raise AssertionError("manifest hash drift must fail")
