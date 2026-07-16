from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import copy

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = ROOT / "scripts" / "build_thesis_dualskill_taxonomy60_manifest.py"
MANIFEST_PATH = ROOT / "config" / "experiment" / "manifests" / "thesis_dualskill_taxonomy60_v1.yaml"
DECISION_CONTRACT_PATH = (
    ROOT
    / "config"
    / "experiment"
    / "decision_contracts"
    / "thesis_dualskill_taxonomy60_no_go_v2.yaml"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("taxonomy60_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angle_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def test_builder_generates_balanced_sixty_scenarios_and_protocol():
    builder = _load_builder()
    manifest = builder.build_manifest("TEST-SHA")
    scenarios = manifest["scenarios"]

    assert manifest["source_id"] == "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
    assert manifest["frozen_source"]["generation_git_sha"] == "TEST-SHA"
    assert manifest["protocol"]["episodes_total"] == 900
    assert manifest["protocol"]["opponent_stages"] == [
        "expert",
        "end_to_end",
        "independent_ppo_vpp",
    ]
    assert len(scenarios) == 60
    assert Counter(s["metadata"]["initial_class"] for s in scenarios) == {
        "advantage": 12,
        "head_on": 12,
        "disadvantage": 12,
        "neutral": 12,
        "crossing_entry": 12,
    }
    assert Counter(s["metadata"]["height_condition"] for s in scenarios) == {
        "own_below": 20,
        "co_altitude": 20,
        "own_above": 20,
    }
    assert Counter(s["metadata"]["mirror_sign"] for s in scenarios) == {
        "negative": 30,
        "positive": 30,
    }
    assert Counter(s["metadata"]["geometry_package"] for s in scenarios) == {
        "wide_balanced": 30,
        "fast_closure": 30,
    }
    assert manifest["decision_contract"]["id"] == "THESIS-DUALSKILL-TAXONOMY60-NO-GO-V2"
    assert manifest["method_contracts"]["static_oracle_gate"]["mapping"] == {
        "crossing_entry": "frozen_crossing_specialist",
        "advantage": "frozen_head_on_specialist",
        "head_on": "frozen_head_on_specialist",
        "disadvantage": "frozen_head_on_specialist",
        "neutral": "frozen_head_on_specialist",
    }


def test_manifest_signatures_are_unique_and_each_cell_is_a_mirror_pair():
    builder = _load_builder()
    scenarios = builder.build_manifest("TEST-SHA")["scenarios"]

    assert len({scenario["name"] for scenario in scenarios}) == 60
    assert len({builder.scenario_signature(scenario) for scenario in scenarios}) == 60
    pairs = defaultdict(list)
    for scenario in scenarios:
        pairs[scenario["metadata"]["mirror_pair_id"]].append(scenario)
    assert len(pairs) == 30
    for pair in pairs.values():
        assert len(pair) == 2
        negative = next(s for s in pair if s["metadata"]["mirror_sign"] == "negative")
        positive = next(s for s in pair if s["metadata"]["mirror_sign"] == "positive")
        assert negative["target_init"]["position_m"][0] == pytest.approx(
            positive["target_init"]["position_m"][0]
        )
        assert negative["target_init"]["position_m"][1] == pytest.approx(
            -positive["target_init"]["position_m"][1]
        )
        assert negative["target_init"]["position_m"][2] == pytest.approx(
            positive["target_init"]["position_m"][2]
        )
        assert _angle_delta_deg(
            negative["own_init"]["heading_deg"], -positive["own_init"]["heading_deg"]
        ) < 1e-6
        assert _angle_delta_deg(
            negative["target_init"]["heading_deg"], -positive["target_init"]["heading_deg"]
        ) < 1e-6


def test_initial_geometry_recomputes_labels_and_preserves_legacy_task_bits():
    builder = _load_builder()
    scenarios = builder.build_manifest("TEST-SHA")["scenarios"]
    quadrants = {"advantage", "head_on", "disadvantage", "neutral"}
    for scenario in scenarios:
        metadata = scenario["metadata"]
        geometry = builder.initial_geometry(scenario)
        assert geometry["initial_range_m"] == pytest.approx(metadata["initial_range_m"])
        assert geometry["altitude_diff_m"] == pytest.approx(metadata["altitude_diff_m"])
        assert math.isclose(
            geometry["own_to_target_los_angle_deg"], metadata["own_to_target_los_angle_deg"], abs_tol=1e-5
        )
        assert math.isclose(
            geometry["target_velocity_to_own_los_angle_deg"],
            metadata["target_velocity_to_own_los_angle_deg"],
            abs_tol=1e-5,
        )
        if metadata["initial_class"] in quadrants:
            assert geometry["geometry_state"] == metadata["initial_class"]
            assert metadata["task_registry_key"] == "head_on"
            assert metadata["ppo_task_bit"] == 0
        else:
            assert geometry["geometry_state"] == "transition"
            assert metadata["task_registry_key"] == "crossing_feasible"
            assert metadata["ppo_task_bit"] == 1


def test_disjointness_rejects_colliding_historical_signature():
    builder = _load_builder()
    scenarios = builder.build_manifest("TEST-SHA")["scenarios"]
    with pytest.raises(ValueError, match="not disjoint"):
        builder.validate_disjointness(
            scenarios,
            {"SYNTHETIC-HISTORICAL": {builder.scenario_signature(scenarios[0])}},
        )


def test_git_blob_source_replays_a_historical_yaml_include_chain():
    builder = _load_builder()
    relative_path = "config/experiment/manifests/thesis_taxonomy_ablation30_v1.yaml"
    source = builder.GitConfigSource("HEAD", relative_path)

    assert builder.load_git_config_signatures(source) == builder.load_config_signatures(
        ROOT / relative_path
    )
    source_id, parsed = builder._parse_disjoint_source(
        "HISTORICAL=git:HEAD:{}".format(relative_path)
    )
    assert source_id == "HISTORICAL"
    assert isinstance(parsed, builder.GitConfigSource)


def test_cli_writes_a_manifest_from_a_clean_python_process(tmp_path):
    output = tmp_path / "taxonomy60.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--source-git-sha",
            "TEST-SHA",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    frozen = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert frozen["source_id"] == "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
    assert len(frozen["scenarios"]) == 60


def test_frozen_manifest_matches_builder_and_records_disjoint_source_files():
    builder = _load_builder()
    frozen = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_sha = frozen["frozen_source"]["generation_git_sha"]
    expected = dict(builder.build_manifest(source_sha))

    assert frozen["scenarios"] == expected["scenarios"]
    assert frozen["protocol"] == expected["protocol"]
    assert frozen["disjointness"]["intersection_count"] == 0
    assert len(frozen["disjointness"]["source_files"]) >= 40
    payload = copy.deepcopy(frozen)
    expected_hash = payload["integrity"].pop("payload_sha256")
    assert builder._payload_sha256(payload) == expected_hash


def test_no_go_contract_has_primary_opponent_gate_and_guard_confound_route():
    contract = yaml.safe_load(DECISION_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["applies_to_source"] == "THESIS-DUALSKILL-TAXONOMY60-3OPP-V1"
    assert contract["evidence_scope"]["primary_opponents"] == ["expert", "end_to_end"]
    assert contract["evidence_scope"]["supporting_opponents"] == ["independent_ppo_vpp"]
    assert contract["classification"]["mixed_interface_guard_bound"]["trigger"]
    assert contract["allowed_routes"]["paper_closure"]["prohibited"] == [
        "training",
        "tuning",
        "scenario_replacement",
        "result_driven_rerun",
    ]
