"""Unified ScenarioRegistry.

Stage 6H.0-F.1: All scenario definitions — candidate search, regression baseline,
smoke tests, threshold optimization — originate from this single registry.

Eliminates duplication and guarantees semantic consistency across:
    - scenario parameters
    - actual geometry after env reset
    - telemetry labels
    - evaluator classification
"""

import copy
from typing import Dict, List, Optional

from .geometry_scenarios import build_explicit_scenario, VALID_SCENARIO_TYPES


class ScenarioRegistry:
    """Central registry for all scenario definitions."""

    # Canonical scenario sets
    _REGISTRY: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, scenario: Dict, scenario_set: str = "default") -> None:
        """Register a scenario under a name and optional set tag."""
        cls._REGISTRY[name] = {
            "scenario": copy.deepcopy(scenario),
            "scenario_set": scenario_set,
        }

    @classmethod
    def get(cls, name: str) -> Optional[Dict]:
        """Retrieve a scenario dict by name."""
        entry = cls._REGISTRY.get(name)
        return copy.deepcopy(entry["scenario"]) if entry else None

    @classmethod
    def list_names(cls, scenario_set: Optional[str] = None) -> List[str]:
        """List all registered scenario names, optionally filtered by set."""
        if scenario_set is None:
            return sorted(cls._REGISTRY.keys())
        return sorted(
            k for k, v in cls._REGISTRY.items() if v["scenario_set"] == scenario_set
        )

    @classmethod
    def list_sets(cls) -> List[str]:
        """List all scenario set tags."""
        return sorted({v["scenario_set"] for v in cls._REGISTRY.values()})

    @classmethod
    def get_set(cls, scenario_set: str) -> Dict[str, Dict]:
        """Retrieve all scenarios belonging to a set."""
        return {
            name: copy.deepcopy(entry["scenario"])
            for name, entry in cls._REGISTRY.items()
            if entry["scenario_set"] == scenario_set
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._REGISTRY.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._REGISTRY

    @classmethod
    def get_regression_suite(cls) -> List[Dict]:
        """Return regression baseline scenarios."""
        return list(cls.get_set("regression_baseline").values())

    @classmethod
    def get_candidate_suite(cls) -> List[Dict]:
        """Return candidate search scenarios."""
        return list(cls.get_set("candidate_search").values())

    @classmethod
    def get_negative_suite(cls) -> List[Dict]:
        """Return negative control scenarios."""
        return list(cls.get_set("negative_control").values())

    @classmethod
    def get_smoke_suite(cls) -> List[Dict]:
        """Return smoke test scenarios."""
        return list(cls.get_set("smoke_test").values())


def _build_and_register(
    name: str,
    scenario_type: str,
    initial_range_m: float,
    ego_speed_mps: float,
    target_speed_mps: float,
    altitude_diff_m: float = 0.0,
    base_altitude_m: float = 5000.0,
    lateral_offset_m: float = 0.0,
    scenario_set: str = "default",
) -> None:
    """Helper: build explicit scenario and register it."""
    scenario = build_explicit_scenario(
        scenario_type=scenario_type,
        initial_range_m=initial_range_m,
        ego_speed_mps=ego_speed_mps,
        target_speed_mps=target_speed_mps,
        altitude_diff_m=altitude_diff_m,
        base_altitude_m=base_altitude_m,
        lateral_offset_m=lateral_offset_m,
    )
    scenario["name"] = name
    ScenarioRegistry.register(name, scenario, scenario_set=scenario_set)


def initialize_canonical_scenarios() -> None:
    """Populate the registry with all canonical scenario sets.

    This is the SINGLE source of truth for scenario definitions.
    All scripts, tests, and evaluators must draw from here.
    """
    ScenarioRegistry.clear()

    # ------------------------------------------------------------------
    # Regression baseline (Stage 6H.0-R recovered)
    # ------------------------------------------------------------------
    _build_and_register(
        "regression_neutral",
        "head_on",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="regression_baseline",
    )
    # regression_challenging uses the actual Stage 6F.5 challenging geometry
    # Geometrically head-on (aspect 180, los_from_ego ~0). Verified 5/5 success.
    ScenarioRegistry.register(
        "regression_challenging",
        {
            "name": "regression_challenging",
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 45.0,
            },
            "target_init": {
                "position_m": [1500.0, 1500.0, 5200.0],
                "velocity_mps": 210.0,
                "heading_deg": 225.0,
            },
            "metadata": {
                "scenario_type": "crossing_left",
                "initial_range_m": 2121.3,
                "altitude_diff_m": 200.0,
                "note": "Stage 6F.5 challenging geometry. Geometrically head-on. Verified 5/5.",
            },
        },
        scenario_set="regression_baseline",
    )
    # True crossing_left with significant lateral offset. Verified 5/5 success.
    ScenarioRegistry.register(
        "regression_crossing_left",
        {
            "name": "regression_crossing_left",
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 1500.0, 5200.0],
                "velocity_mps": 210.0,
                "heading_deg": 225.0,
            },
            "metadata": {
                "scenario_type": "crossing_left",
                "initial_range_m": 2121.3,
                "altitude_diff_m": 200.0,
                "note": "True crossing_left with high cross-range. Verified 5/5.",
            },
        },
        scenario_set="regression_baseline",
    )
    # crossing_right with challenging-like geometry (both aircraft angled)
    # Verified 3/3 success with audit_no_pred_final checkpoint
    ScenarioRegistry.register(
        "regression_crossing_right",
        {
            "name": "regression_crossing_right",
            "own_init": {
                "position_m": [0.0, 0.0, 5000.0],
                "velocity_mps": 200.0,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, -1500.0, 5200.0],
                "velocity_mps": 210.0,
                "heading_deg": 135.0,
            },
            "metadata": {
                "scenario_type": "crossing_right",
                "initial_range_m": 2121.3,
                "altitude_diff_m": 200.0,
                "note": "Challenging-like mirrored geometry. Verified feasible 3/3.",
            },
        },
        scenario_set="regression_baseline",
    )

    # ------------------------------------------------------------------
    # Smoke test: one scenario per geometry family
    # ------------------------------------------------------------------
    _build_and_register(
        "smoke_tail_chase",
        "tail_chase",
        initial_range_m=2000.0,
        ego_speed_mps=340.0,
        target_speed_mps=180.0,
        scenario_set="smoke_test",
    )
    _build_and_register(
        "smoke_head_on",
        "head_on",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="smoke_test",
    )
    _build_and_register(
        "smoke_crossing_left",
        "crossing_left",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="smoke_test",
    )
    _build_and_register(
        "smoke_crossing_right",
        "crossing_right",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="smoke_test",
    )
    _build_and_register(
        "smoke_offset_attack",
        "offset_attack",
        initial_range_m=1000.0,
        ego_speed_mps=200.0,
        target_speed_mps=220.0,
        lateral_offset_m=400.0,
        scenario_set="smoke_test",
    )
    _build_and_register(
        "smoke_fleeing",
        "fleeing",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="smoke_test",
    )

    # ------------------------------------------------------------------
    # Candidate search (for threshold optimization)
    # ------------------------------------------------------------------
    _build_and_register(
        "candidate_head_on_close",
        "head_on",
        initial_range_m=1500.0,
        ego_speed_mps=250.0,
        target_speed_mps=200.0,
        scenario_set="candidate_search",
    )
    _build_and_register(
        "candidate_head_on_far",
        "head_on",
        initial_range_m=3000.0,
        ego_speed_mps=250.0,
        target_speed_mps=200.0,
        scenario_set="candidate_search",
    )
    _build_and_register(
        "candidate_crossing_close",
        "crossing_left",
        initial_range_m=1500.0,
        ego_speed_mps=250.0,
        target_speed_mps=200.0,
        scenario_set="candidate_search",
    )
    # Stage 6H.1: crossing_left at 3000m is infeasible under both VPP and PN.
    # Replaced with head_on at 2500m to maintain range-coverage diversity.
    _build_and_register(
        "candidate_head_on_medium",
        "head_on",
        initial_range_m=2500.0,
        ego_speed_mps=250.0,
        target_speed_mps=200.0,
        scenario_set="candidate_search",
    )

    # ------------------------------------------------------------------
    # Negative controls
    # ------------------------------------------------------------------
    # tail_chase: VPP fails (0%), PN rescues (100%).
    # Mode-switch MUST fire and save this scenario.
    _build_and_register(
        "negative_tail_chase",
        "tail_chase",
        initial_range_m=2000.0,
        ego_speed_mps=340.0,
        target_speed_mps=180.0,
        scenario_set="negative_control",
    )
    # Far-range crossing: gate should NOT fire (range > max threshold).
    _build_and_register(
        "negative_far_crossing",
        "crossing_left",
        initial_range_m=6000.0,
        ego_speed_mps=250.0,
        target_speed_mps=200.0,
        scenario_set="negative_control",
    )
    _build_and_register(
        "negative_fleeing",
        "fleeing",
        initial_range_m=2000.0,
        ego_speed_mps=200.0,
        target_speed_mps=200.0,
        scenario_set="negative_control",
    )
    _build_and_register(
        "negative_offset_attack",
        "offset_attack",
        initial_range_m=1000.0,
        ego_speed_mps=200.0,
        target_speed_mps=220.0,
        lateral_offset_m=400.0,
        scenario_set="negative_control",
    )


# Auto-populate on import
initialize_canonical_scenarios()
