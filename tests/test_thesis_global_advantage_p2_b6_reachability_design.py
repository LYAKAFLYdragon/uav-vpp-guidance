from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_thesis_global_advantage_p2_b6_reachability_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_thesis_global_advantage_p2_b6_opponent_conditional_reachability.py"
CONFIG = ROOT / "config" / "experiment" / "thesis_global_advantage_v1_p2_b6_opponent_conditional_reachability.yaml"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b6_manifest_covers_five_states_two_packages_and_is_disjoint():
    builder = _load(BUILDER, "b6_builder")
    sources = {label: builder._load_yaml(path) for label, path in builder.DISJOINT_MANIFESTS}
    manifest = builder.build_manifest(sources)
    scenarios = manifest["scenarios"]

    assert len(scenarios) == 60
    assert len({scenario["name"] for scenario in scenarios}) == 60
    assert {scenario["metadata"]["initial_class"] for scenario in scenarios} == {
        "advantage",
        "head_on",
        "disadvantage",
        "neutral",
        "crossing_entry",
    }
    assert {scenario["metadata"]["phase_at_reset"] for scenario in scenarios} == {"pre_merge"}
    assert {scenario["metadata"]["mirror_sign"] for scenario in scenarios} == {"negative", "positive"}
    assert all(item["physical_intersection_count"] == 0 for item in manifest["disjointness"].values())
    crossing = [item for item in scenarios if item["metadata"]["initial_class"] == "crossing_entry"]
    assert {item["metadata"]["taxonomy_geometry_state"] for item in crossing} == {"transition"}


def test_b6_preflight_remains_design_only_and_plans_three_opponent_records(tmp_path: Path):
    preflight = _load(PREFLIGHT, "b6_preflight")
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["status"] = "preregistered_design_execution_not_authorised"
    payload["authorization"]["execution_permitted"] = False
    design_config = tmp_path / "b6_design.yaml"
    design_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = preflight.validate_design(design_config)

    assert result["execution_permitted"] is False
    assert result["training_permitted"] is False
    assert result["scenario_count"] == 60
    assert result["planned_records"] == 180
    assert result["opponents"] == ["expert", "end_to_end", "independent_ppo_vpp"]
    assert isinstance(result["output_root_absent"], bool)
    assert isinstance(result["disk_gate_would_pass"], bool)
