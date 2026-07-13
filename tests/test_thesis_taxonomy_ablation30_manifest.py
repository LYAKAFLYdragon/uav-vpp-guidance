from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path
import copy

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = ROOT / "scripts" / "build_thesis_taxonomy_ablation30_manifest.py"
MANIFEST_PATH = (
    ROOT
    / "config"
    / "experiment"
    / "manifests"
    / "thesis_taxonomy_ablation30_v1.yaml"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("ablation30_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angle_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def test_builder_generates_exactly_thirty_balanced_scenarios():
    builder = _load_builder()
    manifest = builder.build_manifest()
    scenarios = manifest["scenarios"]

    assert len(scenarios) == 30
    assert Counter(s["metadata"]["initial_class"] for s in scenarios) == {
        "advantage": 6,
        "head_on": 6,
        "disadvantage": 6,
        "neutral": 6,
        "crossing_entry": 6,
    }
    assert Counter(s["metadata"]["height_condition"] for s in scenarios) == {
        "own_below": 10,
        "co_altitude": 10,
        "own_above": 10,
    }
    assert Counter(s["metadata"]["mirror_sign"] for s in scenarios) == {
        "negative": 15,
        "positive": 15,
    }


def test_full_signatures_and_ids_are_unique():
    builder = _load_builder()
    scenarios = builder.build_manifest()["scenarios"]

    assert len({scenario["name"] for scenario in scenarios}) == 30
    assert len({builder.scenario_signature(scenario) for scenario in scenarios}) == 30


def test_each_class_height_cell_is_an_exact_mirror_pair():
    builder = _load_builder()
    scenarios = builder.build_manifest()["scenarios"]
    pairs = defaultdict(list)
    for scenario in scenarios:
        pairs[scenario["metadata"]["mirror_pair_id"]].append(scenario)

    assert len(pairs) == 15
    for pair in pairs.values():
        assert len(pair) == 2
        negative = next(s for s in pair if s["metadata"]["mirror_sign"] == "negative")
        positive = next(s for s in pair if s["metadata"]["mirror_sign"] == "positive")

        own_neg = negative["own_init"]
        own_pos = positive["own_init"]
        target_neg = negative["target_init"]
        target_pos = positive["target_init"]

        assert own_neg["position_m"] == own_pos["position_m"]
        assert target_neg["position_m"][0] == pytest.approx(target_pos["position_m"][0])
        assert target_neg["position_m"][1] == pytest.approx(-target_pos["position_m"][1])
        assert target_neg["position_m"][2] == pytest.approx(target_pos["position_m"][2])
        assert own_neg["velocity_mps"] == own_pos["velocity_mps"]
        assert target_neg["velocity_mps"] == target_pos["velocity_mps"]
        assert _angle_delta_deg(own_neg["heading_deg"], -own_pos["heading_deg"]) < 1e-6
        assert _angle_delta_deg(target_neg["heading_deg"], -target_pos["heading_deg"]) < 1e-6


def test_initial_geometry_recomputes_to_the_preregistered_taxonomy():
    builder = _load_builder()
    scenarios = builder.build_manifest()["scenarios"]

    expected_quadrants = {"advantage", "head_on", "disadvantage", "neutral"}
    for scenario in scenarios:
        metadata = scenario["metadata"]
        geometry = builder.initial_geometry(scenario)
        assert geometry["initial_range_m"] == pytest.approx(
            builder.INITIAL_RANGE_M, abs=1e-5
        )
        assert geometry["altitude_diff_m"] == pytest.approx(
            builder.HEIGHT_OFFSETS_M[metadata["height_condition"]]
        )
        assert geometry["own_to_target_los_angle_deg"] == pytest.approx(
            metadata["own_to_target_los_angle_deg"], abs=1e-5
        )
        assert geometry["target_velocity_to_own_los_angle_deg"] == pytest.approx(
            metadata["target_velocity_to_own_los_angle_deg"], abs=1e-5
        )

        if metadata["initial_class"] in expected_quadrants:
            assert geometry["geometry_state"] == metadata["initial_class"]
            assert not 85.0 <= geometry["own_to_target_los_angle_deg"] <= 95.0
            assert not 85.0 <= geometry["target_velocity_to_own_los_angle_deg"] <= 95.0
            assert metadata["task_registry_key"] == "head_on"
            assert metadata["ppo_task_bit"] == 0
        else:
            assert metadata["initial_class"] == "crossing_entry"
            assert geometry["geometry_state"] == "transition"
            assert metadata["crossing_context"] is True
            assert metadata["task_registry_key"] == "crossing_feasible"
            assert metadata["ppo_task_bit"] == 1


def test_height_sign_semantics_are_target_minus_own():
    builder = _load_builder()
    scenarios = builder.build_manifest()["scenarios"]
    for scenario in scenarios:
        dz = scenario["target_init"]["position_m"][2] - scenario["own_init"][
            "position_m"
        ][2]
        assert dz == pytest.approx(
            builder.HEIGHT_OFFSETS_M[scenario["metadata"]["height_condition"]]
        )


def test_disjointness_validator_rejects_any_historical_signature_collision():
    builder = _load_builder()
    scenarios = builder.build_manifest()["scenarios"]
    collision = builder.scenario_signature(scenarios[0])

    with pytest.raises(ValueError, match="not disjoint"):
        builder.validate_disjointness(
            scenarios,
            {"SYNTHETIC-HISTORICAL": {collision}},
        )


def test_manifest_contract_records_frozen_source_and_stop_rule():
    builder = _load_builder()
    manifest = builder.build_manifest()

    assert manifest["source_id"] == "THESIS-TAXONOMY-ABLATION30-V1"
    assert manifest["frozen_source"]["git_sha"] == (
        "9a9f9bf6d68560afa81560f5260b78f318dcd9b1"
    )
    assert manifest["protocol"]["evaluation_only"] is True
    assert manifest["protocol"]["training_permitted"] is False
    assert manifest["protocol"]["episodes_total"] == 240
    assert "No training" in manifest["protocol"]["stop_rule"]
    assert math.isclose(manifest["generation_contract"]["initial_range_m"], 2600.0)


def test_frozen_manifest_matches_builder_and_records_zero_historical_overlap():
    builder = _load_builder()
    frozen = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert frozen["scenarios"] == builder.build_manifest()["scenarios"]
    assert frozen["disjointness"]["intersection_count"] == 0
    assert {source["source_id"] for source in frozen["disjointness"]["sources"]} == {
        "CAN-20260705",
        "RQ1-20260711",
        "RQ2-20260712",
        "RQ3-RQ1-20260712",
        "RQ3-RQ2-20260712",
        "RQ4-20260712",
        "IND-HEADON-20260712",
    }

    payload = copy.deepcopy(frozen)
    expected_hash = payload["integrity"].pop("payload_sha256")
    assert builder._payload_sha256(payload) == expected_hash
