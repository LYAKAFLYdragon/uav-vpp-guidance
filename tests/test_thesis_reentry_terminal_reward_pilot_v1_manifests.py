from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_reentry_terminal_reward_pilot_v1_manifests.py"
DEV = ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_dev12.yaml"
HELDOUT = ROOT / "config/experiment/manifests/thesis_reentry_terminal_reward_pilot_v1_heldout24.yaml"


def _load():
    spec = importlib.util.spec_from_file_location("reward_pilot_manifest_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reward_pilot_manifests_are_rebuildable_and_disjoint():
    builder = _load()
    sources = builder.frozen_sources()
    dev = builder.build_manifest(split="dev12", packages=builder.DEV_PACKAGES, seed_base=160100, sources=sources)
    heldout = builder.build_manifest(split="heldout24", packages=builder.HELDOUT_PACKAGES, seed_base=160200, sources=sources)
    builder._add_recent_disjointness(dev)
    builder._add_recent_disjointness(heldout)
    tracked_dev = yaml.safe_load(DEV.read_text(encoding="utf-8"))
    tracked_heldout = yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))

    assert dev["integrity"]["payload_sha256"] == tracked_dev["integrity"]["payload_sha256"]
    assert heldout["integrity"]["payload_sha256"] == tracked_heldout["integrity"]["payload_sha256"]
    assert len(tracked_dev["scenarios"]) == 12
    assert len(tracked_heldout["scenarios"]) == 24
    assert all(item["physical_intersection_count"] == 0 for item in tracked_dev["disjointness"].values())
    assert all(item["physical_intersection_count"] == 0 for item in tracked_heldout["disjointness"].values())
