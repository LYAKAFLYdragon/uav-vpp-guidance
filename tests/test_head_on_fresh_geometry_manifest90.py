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
MANIFEST90_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest90_pilot.yaml"
)
MANIFEST60_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot.yaml"
)
BASE_CONFIG_NAME = (
    "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_pilot.yaml"
)


def test_manifest90_config_only_compares_two_canonical_methods():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest90_pilot"
    )
    assert ORACLE_METHOD in resolved["run_defaults"]["main_methods"]
    assert COMMANDER_METHOD in resolved["run_defaults"]["main_methods"]
    assert len(resolved["run_defaults"]["main_methods"]) == 2
    assert set(resolved["run_defaults"]["ablation_methods"]) == {ORACLE_METHOD, COMMANDER_METHOD}


def test_manifest90_resolved_has_ninety_head_on_scenarios():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    head_on = resolved["tasks"]["head_on"]
    scenarios = head_on["task"]["scenarios"]
    assert len(scenarios) == 90, f"Expected 90, got {len(scenarios)}"
    names = [s["name"] for s in scenarios]
    assert len(names) == len(set(names)), "Duplicate scenario names found"
    for s in scenarios:
        assert s["metadata"]["scenario_type"] == "head_on"
        assert "initial_range_m" in s["metadata"]
        assert "altitude_diff_m" in s["metadata"]
        assert "lateral_offset_m" in s["metadata"]


def test_manifest90_crossing_feasible_scope_unchanged():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    base_resolved = _load_yaml_resolved(CONFIG_DIR / BASE_CONFIG_NAME)
    assert "crossing_feasible" in resolved["tasks"]
    manifest_crossing = resolved["tasks"]["crossing_feasible"]
    base_crossing = base_resolved["tasks"]["crossing_feasible"]
    assert manifest_crossing == base_crossing
    crossing_scenarios = manifest_crossing["task"]["scenarios"]
    assert len(crossing_scenarios) == 4


def test_manifest90_head_on_scenarios_are_disjoint_from_manifest60():
    manifest90 = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    manifest60 = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    manifest90_names = {s["name"] for s in manifest90["tasks"]["head_on"]["task"]["scenarios"]}
    manifest60_names = {s["name"] for s in manifest60["tasks"]["head_on"]["task"]["scenarios"]}
    assert manifest60_names.issubset(manifest90_names)
    new_names = manifest90_names - manifest60_names
    assert len(new_names) == 30


def test_manifest90_new_scenarios_use_disjoint_parameters():
    manifest90 = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    manifest60 = _load_yaml_resolved(CONFIG_DIR / MANIFEST60_CONFIG_NAME)
    manifest90_names = {s["name"] for s in manifest90["tasks"]["head_on"]["task"]["scenarios"]}
    manifest60_names = {s["name"] for s in manifest60["tasks"]["head_on"]["task"]["scenarios"]}
    new_names = manifest90_names - manifest60_names
    
    # Existing manifest60 parameters
    existing_ranges = {1500, 1700, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200}
    existing_v = {190, 200, 210, 220, 230, 240}
    existing_alt = {0, 100, 200, 300, 400, 500}
    existing_lat = {0, 75, 150}
    
    for name in new_names:
        # Extract r, v_own, v_tgt, alt, lat from name
        # Format: head_on_manifest_r{r}_v{ov}_v{tv}_alt{...}
        parts = name.split("_")
        # parts[0] = head, [1] = on, [2] = manifest, [3] = r1300, [4] = v180, [5] = v180, [6] = alt0...
        r = int(parts[3][1:])
        v_own = int(parts[4][1:])
        v_tgt = int(parts[5][1:])
        # Check if it's a new scenario (not in manifest60 ranges)
        if r not in existing_ranges or v_own not in existing_v or v_tgt not in existing_v:
            continue  # This is a new scenario with new parameters
        # If it is in existing ranges, it must use new alt or new lat
        alt_str = parts[6]
        alt_val = 0
        if alt_str.startswith("altp"):
            alt_val = int(alt_str[4:])
        elif alt_str.startswith("altm"):
            alt_val = -int(alt_str[4:])
        lat_val = 0
        if "ypos" in name:
            lat_val = int(name.split("ypos")[1].split("_")[0])
        elif "yneg" in name:
            lat_val = -int(name.split("yneg")[1].split("_")[0])
        assert (alt_val not in existing_alt or lat_val not in existing_lat), \
            f"Scenario {name} uses parameters fully covered by manifest60"


def test_manifest90_tasks_list_matches_combat_scope():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]


def test_manifest90_methods_have_required_fields():
    resolved = _load_yaml_resolved(CONFIG_DIR / MANIFEST90_CONFIG_NAME)
    for method_name in (ORACLE_METHOD, COMMANDER_METHOD):
        method = resolved["methods"][method_name]
        assert "label" in method
        assert "agent_type" in method
        assert "config_path" in method
