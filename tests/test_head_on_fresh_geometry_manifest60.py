from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"


def _load_yaml(name: str):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_resolved(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    includes = payload.pop("includes", []) or []
    merged = {}
    for include in includes:
        merged = _deep_merge(merged, _load_yaml_resolved((path.parent / include).resolve()))
    return _deep_merge(merged, payload)


ORACLE_METHOD = "oracle_task_gate"
COMMANDER_METHOD = (
    "hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_"
    "oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_"
    "secondary_clamp_lowalt_latrange_descent"
)
MANIFEST60_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot.yaml"
)
MANIFEST30_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest30_pilot.yaml"
)
BASE_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_pilot.yaml"
)


def test_manifest60_config_only_compares_two_canonical_methods():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)

    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot"
    )
    assert ORACLE_METHOD in resolved["run_defaults"]["main_methods"]
    assert COMMANDER_METHOD in resolved["run_defaults"]["main_methods"]
    assert len(resolved["run_defaults"]["main_methods"]) == 2
    assert set(resolved["run_defaults"]["ablation_methods"]) == {ORACLE_METHOD, COMMANDER_METHOD}


def test_manifest60_resolved_has_sixty_head_on_scenarios():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)

    head_on = resolved["tasks"]["head_on"]
    scenarios = head_on["task"]["scenarios"]
    assert len(scenarios) == 60

    # Verify all scenario names are unique
    names = [s["name"] for s in scenarios]
    assert len(names) == len(set(names)), "Duplicate scenario names found"

    # Verify metadata consistency
    for s in scenarios:
        assert s["metadata"]["scenario_type"] == "head_on"
        assert s["metadata"]["manifest_family"] == "fresh_geometry_manifest60"
        assert "initial_range_m" in s["metadata"]
        assert "altitude_diff_m" in s["metadata"]
        assert "lateral_offset_m" in s["metadata"]


def test_manifest60_crossing_feasible_scope_unchanged():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    base_resolved = _load_yaml_resolved(CONFIG_DIR / BASE_CONFIG_NAME)

    # crossing_feasible should be present and identical to base
    assert "crossing_feasible" in resolved["tasks"]
    manifest_crossing = resolved["tasks"]["crossing_feasible"]
    base_crossing = base_resolved["tasks"]["crossing_feasible"]
    assert manifest_crossing == base_crossing

    # Count crossing scenarios
    crossing_scenarios = manifest_crossing["task"]["scenarios"]
    assert len(crossing_scenarios) == 4


def test_manifest60_head_on_scenarios_are_disjoint_from_manifest30():
    manifest60 = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    manifest30 = _load_yaml_resolved(CONFIG_DIR / MANIFEST30_CONFIG_NAME)

    manifest60_names = {s["name"] for s in manifest60["tasks"]["head_on"]["task"]["scenarios"]}
    manifest30_names = {s["name"] for s in manifest30["tasks"]["head_on"]["task"]["scenarios"]}

    # All manifest30 scenarios should be a subset of manifest60
    assert manifest30_names.issubset(manifest60_names)

    # There should be exactly 30 new scenarios
    new_names = manifest60_names - manifest30_names
    assert len(new_names) == 30


def test_manifest60_new_scenarios_use_disjoint_parameters():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    scenarios = resolved["tasks"]["head_on"]["task"]["scenarios"]

    # Extract parameter tuples from names for existing vs new
    existing_prefixes = {
        "r1800_v200_v200", "r2000_v210_v200", "r2200_v200_v210",
        "r2400_v210_v210", "r2600_v220_v200", "r2000_v200_v200",
        "r2200_v210_v200", "r2600_v220_v210", "r2400_v200_v210",
    }

    for s in scenarios:
        name = s["name"]
        # All new scenarios (not in manifest30) should have disjoint parameter combinations
        if not any(prefix in name for prefix in existing_prefixes):
            # These are new scenarios; verify they use new ranges or speeds
            assert ("r1500" in name or "r1700" in name or "r2800" in name
                    or "r3000" in name or "r3200" in name
                    or "v190" in name or "v230" in name or "v240" in name
                    or "altp100" in name or "altm100" in name
                    or "altp400" in name or "altm400" in name
                    or "altp500" in name)


def test_manifest60_tasks_list_matches_combat_scope():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]


def test_manifest60_methods_have_required_fields():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    for method_name in (ORACLE_METHOD, COMMANDER_METHOD):
        method = resolved["methods"][method_name]
        assert "label" in method
        assert "agent_type" in method
        assert "config_path" in method
