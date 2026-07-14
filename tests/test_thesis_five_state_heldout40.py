from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = ROOT / "scripts" / "build_thesis_five_state_heldout40_manifest.py"
ANALYZER_PATH = ROOT / "scripts" / "analyze_thesis_five_state_heldout40.py"
MANIFEST_PATH = ROOT / "config" / "experiment" / "manifests" / "thesis_five_state_heldout40_v1.yaml"
CONFIG_PATH = ROOT / "config" / "experiment" / "jsbsim_hrl_thesis_five_state_heldout40_v1.yaml"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _angle_delta_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def test_builder_creates_forty_balanced_new_geometries():
    builder = _module(BUILDER_PATH, "heldout40_builder")
    scenarios = builder.build_manifest()["scenarios"]

    assert len(scenarios) == 40
    assert Counter(item["metadata"]["initial_class"] for item in scenarios) == {
        "advantage": 8,
        "head_on": 8,
        "disadvantage": 8,
        "neutral": 8,
        "crossing_entry": 8,
    }
    assert Counter(item["metadata"]["height_condition"] for item in scenarios) == {
        "own_below": 20,
        "own_above": 20,
    }
    assert Counter(item["metadata"]["mirror_sign"] for item in scenarios) == {
        "negative": 20,
        "positive": 20,
    }
    assert Counter(item["metadata"]["distance_speed_package"] for item in scenarios) == {
        "heldout40_closing_a": 20,
        "heldout40_closing_b": 20,
    }
    assert len({builder.scenario_signature(item) for item in scenarios}) == 40


def test_mirror_pairs_and_taxonomy_are_physical_not_name_based():
    builder = _module(BUILDER_PATH, "heldout40_geometry")
    scenarios = builder.build_manifest()["scenarios"]
    pairs: dict[str, list[dict]] = defaultdict(list)
    for scenario in scenarios:
        pairs[scenario["metadata"]["mirror_pair_id"]].append(scenario)
        geometry = builder.initial_geometry(scenario)
        expected = scenario["metadata"]["initial_class"]
        if expected == "crossing_entry":
            assert geometry["taxonomy_geometry_state"] == "transition"
        else:
            assert geometry["taxonomy_geometry_state"] == expected
        assert geometry["initial_range_m"] == pytest.approx(
            scenario["metadata"]["initial_range_m"], abs=1e-5
        )

    assert len(pairs) == 20
    for pair in pairs.values():
        assert len(pair) == 2
        negative = next(item for item in pair if item["metadata"]["mirror_sign"] == "negative")
        positive = next(item for item in pair if item["metadata"]["mirror_sign"] == "positive")
        assert negative["target_init"]["position_m"][0] == pytest.approx(
            positive["target_init"]["position_m"][0]
        )
        assert negative["target_init"]["position_m"][1] == pytest.approx(
            -positive["target_init"]["position_m"][1]
        )
        assert _angle_delta_deg(
            negative["own_init"]["heading_deg"], -positive["own_init"]["heading_deg"]
        ) < 1e-6


def test_frozen_manifest_matches_builder_and_prior_envelopes_are_disjoint():
    builder = _module(BUILDER_PATH, "heldout40_frozen")
    frozen = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert frozen["scenarios"] == builder.build_manifest()["scenarios"]
    assert frozen["protocol"]["episodes_total"] == 480
    assert frozen["protocol"]["episode_horizon_high_level_steps"] == 160
    assert frozen["disjointness"]["intersection_count"] == 0
    assert {item["source_id"] for item in frozen["disjointness"]["sources"]} == {
        "THESIS-TAXONOMY-ABLATION30-V1",
        "THESIS-FIVE-STATE-SHARED-INTENT-V1-DEV30",
        "THESIS-FIVE-STATE-SHARED-INTENT-V1-HELDOUT60",
    }
    payload = copy.deepcopy(frozen)
    expected = payload["integrity"].pop("payload_sha256")
    assert builder._payload_sha256(payload) == expected


def test_config_freezes_assets_and_preregisters_nontraining_gate():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = config["heldout40_protocol"]
    assert protocol["evaluation_only"] is True
    assert protocol["training_permitted"] is False
    assert protocol["tuning_permitted"] is False
    assert protocol["episode_horizon_high_level_steps"] == 160
    assert protocol["opponents"] == ["expert", "end_to_end", "independent_ppo_vpp"]
    assert "MVP" not in json.dumps(protocol)
    gate = protocol["head_on_terminal_discriminability"]
    assert gate["subset_initial_class"] == "head_on"
    assert gate["minimum_resolved_ratio"] == pytest.approx(0.6)
    assert gate["failure_label"] == "terminal_win_rate_not_discriminative"
    assert config["observation"] == {
        "include_gains": True,
        "include_opponent_stage": False,
        "include_task_type": True,
    }
    assert config["scenario_manifest"]["payload_sha256"] == yaml.safe_load(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )["integrity"]["payload_sha256"]

    assets = config["frozen_assets"]["artifacts"]
    for asset in assets:
        path = Path(asset["path"])
        assert path.is_file(), asset["id"]
        assert _sha256(path) == asset["sha256"], asset["id"]
    assert config["opponent_registry"]["end_to_end"]["checkpoint_sha256"] == next(
        asset["sha256"] for asset in assets if asset["id"] == "end_to_end_neural_opponent"
    )
    assert config["opponent_registry"]["independent_ppo_vpp"]["checkpoint_sha256"] == next(
        asset["sha256"] for asset in assets if asset["id"] == "independent_ppo_vpp_opponent"
    )
    commander_modes = {mode["id"]: mode for mode in config["commander"]["modes"]}
    assert commander_modes[0]["checkpoint"] == next(
        asset["path"] for asset in assets if asset["id"] == "head_on_vpp_specialist"
    )
    assert commander_modes[1]["checkpoint"] == next(
        asset["path"] for asset in assets if asset["id"] == "crossing_vpp_specialist"
    )
    assert commander_modes[2]["checkpoint"] == commander_modes[0]["checkpoint"]
    assert commander_modes[2]["specialist_profile"] == "post_merge_recovery"


def test_feature_name_projection_is_exact_and_rejects_missing_fields():
    from uav_vpp_guidance.evaluation.feature_name_projected_opponent import (
        BASE_GEOMETRY_FEATURE_NAMES,
        project_role_reversed_base_observation,
    )

    names = ["gain_k_los", *BASE_GEOMETRY_FEATURE_NAMES, "is_crossing", "gain_k_pos"]
    vector = np.arange(len(names), dtype=np.float32)
    projected = project_role_reversed_base_observation(
        {
            "observation_vector": vector,
            "observation_schema": {
                "feature_names": names,
                "role_reversed": True,
            },
            "sentinel": "preserved",
        }
    )
    assert projected["sentinel"] == "preserved"
    assert projected["observation_schema"]["feature_names"] == list(BASE_GEOMETRY_FEATURE_NAMES)
    assert projected["observation_vector"].shape == (16,)
    assert projected["observation_vector"].tolist() == list(range(1, 17))

    with pytest.raises(ValueError, match="missing required features"):
        project_role_reversed_base_observation(
            {
                "observation_vector": np.zeros(15, dtype=np.float32),
                "observation_schema": {
                    "feature_names": list(BASE_GEOMETRY_FEATURE_NAMES[:-1]),
                    "role_reversed": True,
                },
            }
        )
    with pytest.raises(ValueError, match="role-reversed"):
        project_role_reversed_base_observation(
            {
                "observation_vector": np.zeros(16, dtype=np.float32),
                "observation_schema": {
                    "feature_names": list(BASE_GEOMETRY_FEATURE_NAMES),
                    "role_reversed": False,
                },
            }
        )


def test_frozen_neural_opponents_accept_the_adapter_base16_contract():
    from uav_vpp_guidance.evaluation.feature_name_projected_opponent import (
        BASE_GEOMETRY_FEATURE_NAMES,
        FeatureNameProjectedOpponent,
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    observation = {
        "observation_vector": np.zeros(16, dtype=np.float32),
        "observation_schema": {
            "feature_names": list(BASE_GEOMETRY_FEATURE_NAMES),
            "role_reversed": True,
        },
    }
    for opponent_name in ("end_to_end", "independent_ppo_vpp"):
        entry = config["opponent_registry"][opponent_name]
        adapter = FeatureNameProjectedOpponent(
            checkpoint_path=entry["checkpoint"], config=entry["config"], device="cpu"
        )
        action = adapter.act(observation)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))
        diagnostics = adapter.get_diagnostics()
        assert diagnostics["opponent_adapter"] == "feature_name_base16_projection"
        assert diagnostics["opponent_adapter_projection_count"] == 1


def _trajectory() -> list[dict]:
    return [
        {
            "time_s": 0.2,
            "pre_merge": True,
            "range_rate_mps": -400.0,
            "aa_deg": 20.0,
            "ata_deg": 20.0,
            "ego_in_attack_zone": False,
            "target_in_attack_zone": False,
            "vp_forward_bias_m": 5.0,
            "vp_lateral_bias_m": 2.0,
            "vp_pos_z": 5202.0,
            "ego_pos_z": 5200.0,
        },
        {
            "time_s": 0.4,
            "pre_merge": False,
            "range_rate_mps": -100.0,
            "aa_deg": 160.0,
            "ata_deg": 20.0,
            "ego_in_attack_zone": True,
            "target_in_attack_zone": False,
            "vp_forward_bias_m": 7.0,
            "vp_lateral_bias_m": 4.0,
            "vp_pos_z": 5204.0,
            "ego_pos_z": 5200.0,
        },
    ]


def _write_synthetic_run(run_dir: Path, opponent: str, scenarios: list[dict]) -> None:
    records = []
    for scenario in scenarios:
        initial_class = scenario["metadata"]["initial_class"]
        for method in (
            "canonical_ppo_high_level_policy",
            "legacy_static_oracle_task_gate",
            "head_on_vpp_specialist_no_routing",
            "crossing_vpp_specialist_no_routing",
        ):
            resolved = initial_class == "head_on" and method != "crossing_vpp_specialist_no_routing"
            records.append(
                {
                    "opponent_stage": opponent,
                    "controller": method,
                    "scenario": scenario["name"],
                    "seed": scenario["metadata"]["scenario_seed"],
                    "backend": "jsbsim",
                    "strict_backend": True,
                    "scenario_metadata": scenario["metadata"],
                    "win": resolved,
                    "loss": False,
                    "draw": not resolved,
                    "combat_reason": "timeout_hp_advantage" if resolved else "timeout_draw",
                    "ego_hp": 100.0,
                    "target_hp": 90.0 if resolved else 100.0,
                    "hp_advantage": 10.0 if resolved else 0.0,
                    "total_time_s": 0.4,
                    "trajectory": _trajectory(),
                }
            )
    aggregate = run_dir / "aggregate"
    aggregate.mkdir(parents=True)
    (aggregate / "episode_records.json").write_text(
        json.dumps({"episodes": records}), encoding="utf-8"
    )


def test_read_only_analysis_enforces_pairing_and_terminal_discriminability(tmp_path):
    analyzer = _module(ANALYZER_PATH, "heldout40_analyzer")
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dirs = {}
    for opponent in analyzer.OPPONENTS:
        run_dir = tmp_path / opponent
        _write_synthetic_run(run_dir, opponent, manifest["scenarios"])
        run_dirs[opponent] = run_dir

    output = tmp_path / "analysis"
    report = analyzer.analyze(run_dirs=run_dirs, manifest_path=MANIFEST_PATH, output_dir=output)
    assert report["n_episode_rows"] == 480
    assert all(item["passed"] for item in report["gate"]["opponents"].values())
    assert (output / "heldout40_episode_metrics.csv").is_file()
    assert (output / "heldout40_summary.csv").is_file()
    assert (output / "heldout40_terminal_discriminability_gate.json").is_file()
