from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _builder():
    spec = importlib.util.spec_from_file_location("temporal_corridor_manifest_builder", ROOT / "scripts" / "build_thesis_temporal_corridor_transfer_manifest.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_temporal_corridor_manifest_is_a_fresh_twelve_cell_design():
    manifest = _builder().build_manifest()
    assert manifest["source_id"].endswith("MANIFEST12")
    assert len(manifest["scenarios"]) == 12
    assert manifest["generation_contract"]["physical_intersection_count"] == 0
    assert len({item["metadata"]["pair_key"] for item in manifest["scenarios"]}) == 12


def test_temporal_corridor_design_locks_three_methods_and_two_opponent_gate():
    design = yaml.safe_load((ROOT / "config" / "experiment" / "thesis_temporal_corridor_transfer_v1_design.yaml").read_text(encoding="utf-8"))
    assert tuple(design["methods"]) == ("frozen_fixed_head_on", "moderate_left", "moderate_right")
    assert design["authorization"]["execution_permitted"] is False
    assert design["opponents"]["pooled_gate"] == "prohibited"
    assert design["gates"]["candidate_opponents_required"] == 2
    assert design["contract"]["guidance_pid_change"] == "prohibited"
