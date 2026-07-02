from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config" / "experiment"


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


def _extract_tuples(cfg_path: Path):
    """Extract (r, v_own, v_tgt, alt, lat) tuples from a config's head_on scenarios."""
    cfg = _load_yaml_resolved(cfg_path)
    head_on = cfg["tasks"]["head_on"]["task"]["scenarios"]
    tuples = set()
    for s in head_on:
        own = s["own_init"]
        tgt = s["target_init"]
        r = int(round(float(tgt["position_m"][0])))
        y = int(round(float(tgt["position_m"][1])))
        z_own = int(round(float(own["position_m"][2])))
        z_tgt = int(round(float(tgt["position_m"][2])))
        alt = z_tgt - z_own
        v_own = int(round(float(own["velocity_mps"])))
        v_tgt = int(round(float(tgt["velocity_mps"])))
        tuples.add((r, v_own, v_tgt, alt, y))
    return tuples


# --- Config paths ---
MANIFEST30 = CONFIG_DIR / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest30_pilot.yaml"
MANIFEST60 = CONFIG_DIR / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot.yaml"
MANIFEST90 = CONFIG_DIR / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest90_pilot.yaml"
HELDOUT = CONFIG_DIR / "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout.yaml"

ORACLE_METHOD = "oracle_task_gate"
COMMANDER_METHOD = (
    "hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_"
    "oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_"
    "secondary_clamp_lowalt_latrange_descent"
)


def test_formal_heldout_config_only_compares_two_canonical_methods():
    resolved = _load_yaml_resolved(HELDOUT)
    assert resolved["experiment"]["name"] == (
        "jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_formal_heldout"
    )
    assert ORACLE_METHOD in resolved["run_defaults"]["main_methods"]
    assert COMMANDER_METHOD in resolved["run_defaults"]["main_methods"]
    assert len(resolved["run_defaults"]["main_methods"]) == 2
    assert set(resolved["run_defaults"]["ablation_methods"]) == {ORACLE_METHOD, COMMANDER_METHOD}


def test_formal_heldout_has_thirty_six_head_on_scenarios():
    resolved = _load_yaml_resolved(HELDOUT)
    scenarios = resolved["tasks"]["head_on"]["task"]["scenarios"]
    assert len(scenarios) == 36, f"Expected 36, got {len(scenarios)}"
    names = [s["name"] for s in scenarios]
    assert len(names) == len(set(names)), "Duplicate scenario names found"


def test_formal_heldout_scenarios_have_correct_manifest_family():
    resolved = _load_yaml_resolved(HELDOUT)
    for s in resolved["tasks"]["head_on"]["task"]["scenarios"]:
        assert s["metadata"]["scenario_type"] == "head_on"
        assert s["metadata"]["manifest_family"] == "formal_heldout"


def test_formal_heldout_is_disjoint_from_manifest30():
    heldout_tuples = _extract_tuples(HELDOUT)
    manifest30_tuples = _extract_tuples(MANIFEST30)
    overlap = heldout_tuples & manifest30_tuples
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping tuples with manifest30: {overlap}"


def test_formal_heldout_is_disjoint_from_manifest60():
    heldout_tuples = _extract_tuples(HELDOUT)
    manifest60_tuples = _extract_tuples(MANIFEST60)
    overlap = heldout_tuples & manifest60_tuples
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping tuples with manifest60: {overlap}"


def test_formal_heldout_is_disjoint_from_manifest90():
    heldout_tuples = _extract_tuples(HELDOUT)
    manifest90_tuples = _extract_tuples(MANIFEST90)
    overlap = heldout_tuples & manifest90_tuples
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping tuples with manifest90: {overlap}"


def test_formal_heldout_is_disjoint_from_all_pilot_manifests():
    """Union check: held-out has zero overlap with the combined set of all pilot manifests."""
    heldout_tuples = _extract_tuples(HELDOUT)
    all_pilot = _extract_tuples(MANIFEST30) | _extract_tuples(MANIFEST60) | _extract_tuples(MANIFEST90)
    overlap = heldout_tuples & all_pilot
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping tuples with any pilot manifest: {overlap}"
    assert len(heldout_tuples) == 36, f"Expected 36 unique tuples, got {len(heldout_tuples)}"


def test_formal_heldout_crossing_feasible_scope_unchanged():
    resolved = _load_yaml_resolved(HELDOUT)
    base_resolved = _load_yaml_resolved(MANIFEST30)
    assert "crossing_feasible" in resolved["tasks"]
    manifest_crossing = resolved["tasks"]["crossing_feasible"]
    base_crossing = base_resolved["tasks"]["crossing_feasible"]
    assert manifest_crossing == base_crossing
    crossing_scenarios = manifest_crossing["task"]["scenarios"]
    assert len(crossing_scenarios) == 4


def test_formal_heldout_tasks_list_matches_combat_scope():
    resolved = _load_yaml_resolved(HELDOUT)
    assert resolved["run_defaults"]["tasks"] == ["head_on", "crossing_feasible"]


def test_formal_heldout_methods_have_required_fields():
    resolved = _load_yaml_resolved(HELDOUT)
    for method_name in (ORACLE_METHOD, COMMANDER_METHOD):
        method = resolved["methods"][method_name]
        assert "label" in method
        assert "agent_type" in method
        assert "config_path" in method


def test_formal_heldout_scenarios_use_disjoint_range_values():
    """All held-out ranges must be values never used in any pilot manifest."""
    heldout_tuples = _extract_tuples(HELDOUT)
    all_pilot = _extract_tuples(MANIFEST30) | _extract_tuples(MANIFEST60) | _extract_tuples(MANIFEST90)
    pilot_ranges = {t[0] for t in all_pilot}
    for t in heldout_tuples:
        assert t[0] not in pilot_ranges, (
            f"Held-out scenario uses range {t[0]} which already appeared in a pilot manifest. "
            f"Held-out ranges must be completely new."
        )
