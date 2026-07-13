from __future__ import annotations

from collections import Counter
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
DEV_PATH = SCRIPTS.parent / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_dev30.yaml"
HELDOUT_PATH = SCRIPTS.parent / "config" / "experiment" / "manifests" / "thesis_five_state_shared_intent_v1_heldout60.yaml"
DISTRIBUTION_PATH = SCRIPTS.parent / "config" / "experiment" / "thesis_five_state_shared_intent_v1_train_distribution.yaml"


def _load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_dev30_and_heldout60_are_balanced_and_have_frozen_roles():
    dev = _load(DEV_PATH)
    heldout = _load(HELDOUT_PATH)

    assert dev["split"] == "dev30"
    assert heldout["split"] == "heldout60"
    assert len(dev["scenarios"]) == 30
    assert len(heldout["scenarios"]) == 60
    assert dev["protocol"]["selection_permitted"] is True
    assert heldout["protocol"]["selection_permitted"] is False
    assert heldout["protocol"]["allowed_use"] == "final_formal_evaluation_only"

    assert Counter(item["metadata"]["initial_class"] for item in dev["scenarios"]) == {
        "advantage": 6,
        "head_on": 6,
        "disadvantage": 6,
        "neutral": 6,
        "crossing_entry": 6,
    }
    assert Counter(item["metadata"]["height_condition"] for item in heldout["scenarios"]) == {
        "own_below": 20,
        "co_altitude": 20,
        "own_above": 20,
    }
    assert Counter(item["metadata"]["mirror_sign"] for item in heldout["scenarios"]) == {
        "negative": 30,
        "positive": 30,
    }
    assert Counter(item["metadata"]["distance_speed_package"] for item in heldout["scenarios"]) == {
        "heldout_unseen_far_fast_a": 30,
        "heldout_unseen_far_fast_b": 30,
    }


def test_manifests_are_signature_disjoint_from_each_other_and_training_support():
    builder = _load_script("build_thesis_five_state_manifests.py")
    dev = _load(DEV_PATH)
    heldout = _load(HELDOUT_PATH)
    distribution = _load(DISTRIBUTION_PATH)

    dev_signatures = {builder.scenario_signature(item) for item in dev["scenarios"]}
    heldout_signatures = {builder.scenario_signature(item) for item in heldout["scenarios"]}
    assert not dev_signatures.intersection(heldout_signatures)
    assert all(builder._outside_train_support(item, distribution) for item in dev["scenarios"])
    assert all(builder._outside_train_support(item, distribution) for item in heldout["scenarios"])
    assert dev["disjointness"]["dev30_heldout60_intersection_count"] == 0
    for source in heldout["disjointness"]["historical_sources"]:
        assert source["dev30_intersection_count"] == 0
        assert source["heldout60_intersection_count"] == 0


def test_quadrants_avoid_transition_band_and_crossing_is_explicitly_marked():
    dev = _load(DEV_PATH)
    for scenario in dev["scenarios"]:
        metadata = scenario["metadata"]
        target_angle = metadata["target_velocity_to_own_los_angle_deg"]
        if metadata["initial_class"] == "crossing_entry":
            assert metadata["crossing_context"] is True
            assert target_angle == pytest.approx(90.0)
            assert metadata["taxonomy_geometry_state"] == "transition"
        else:
            assert not 85.0 <= metadata["own_to_target_los_angle_deg"] <= 95.0
            assert not 85.0 <= target_angle <= 95.0
            assert metadata["taxonomy_geometry_state"] == metadata["initial_class"]


def test_frozen_manifest_payload_hashes_match_current_content():
    builder = _load_script("build_thesis_five_state_manifests.py")
    for path in (DEV_PATH, HELDOUT_PATH):
        payload = copy.deepcopy(_load(path))
        expected_hash = payload["integrity"].pop("payload_sha256")
        assert builder._payload_sha256(payload) == expected_hash
