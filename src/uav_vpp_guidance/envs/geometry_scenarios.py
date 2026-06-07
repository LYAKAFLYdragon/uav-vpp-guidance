"""Explicit geometry scenario builder.

Stage 6H.0-F: Replaces the ambiguous ``aspect_angle_deg``-only builder with
scenario-type-aware construction.  Each type explicitly defines target position
and heading relative to ego, eliminating the ``aspect=180`` confusion
(target-behind-ego fleeing vs. true head-on).

Contract:
    - Ego always at [0, 0, base_altitude_m], heading 0° (flying +x).
    - ``scenario_type`` determines the geometric relationship.
    - ``aspect_angle_deg`` is computed from the resulting geometry, not an input.
    - All outputs are plain Python lists / floats (no ndarrays) for YAML safety.
"""

import math
from typing import Dict, Literal

import numpy as np

ScenarioType = Literal[
    "tail_chase",
    "head_on",
    "crossing_left",
    "crossing_right",
    "offset_pursuit",
    "offset_attack",
    "fleeing",
]

VALID_SCENARIO_TYPES = set(ScenarioType.__args__)  # type: ignore[attr-defined]

# Geometry family documentation (Stage 6H.0-F.1)
GEOMETRY_FAMILY_DOCS: dict = {
    "tail_chase": {
        "description": "Target is ahead of ego, both flying the same direction.",
        "position": "Target at [+range, 0, alt] relative to ego.",
        "target_heading": "Same as ego (0 deg).",
        "aspect_angle": "~0 deg.",
        "closure_condition": "Ego must be faster than target for positive closure.",
    },
    "head_on": {
        "description": "Target is ahead of ego, flying directly toward ego.",
        "position": "Target at [+range, 0, alt] relative to ego.",
        "target_heading": "Opposite to ego (180 deg).",
        "aspect_angle": "~180 deg.",
        "closure_condition": "Always positive closure (both aircraft closing).",
    },
    "crossing_left": {
        "description": "Target is to ego's left, crossing ego's flight path.",
        "position": "Target at [0, +range, alt] relative to ego.",
        "target_heading": "270 deg (flying -y, crossing from left to right).",
        "aspect_angle": "~90 deg.",
        "closure_condition": "Positive closure if ego has x-component toward target.",
    },
    "crossing_right": {
        "description": "Target is to ego's right, crossing ego's flight path.",
        "position": "Target at [0, -range, alt] relative to ego.",
        "target_heading": "90 deg (flying +y, crossing from right to left).",
        "aspect_angle": "~90 deg.",
        "closure_condition": "Positive closure if ego has x-component toward target.",
    },
    "offset_pursuit": {
        "description": "Target is behind ego with lateral offset; ego must perform lead turn.",
        "position": "Target at [-0.8*range, offset, alt] relative to ego.",
        "target_heading": "~30 deg (angled away from ego).",
        "aspect_angle": "~30-150 deg depending on offset.",
        "closure_condition": "Positive closure if ego turns toward target.",
    },
    "offset_attack": {
        "description": "Alias for offset_pursuit. Target behind with lateral offset.",
        "position": "Same as offset_pursuit.",
        "target_heading": "Same as offset_pursuit.",
        "aspect_angle": "Same as offset_pursuit.",
        "closure_condition": "Same as offset_pursuit.",
    },
    "fleeing": {
        "description": "Target is behind ego, moving away in the opposite direction.",
        "position": "Target at [-range, 0, alt] relative to ego.",
        "target_heading": "180 deg (opposite to ego, moving away).",
        "aspect_angle": "~180 deg.",
        "closure_condition": "Negative closure (target moving away).",
    },
}


def build_explicit_scenario(
    scenario_type: str,
    initial_range_m: float,
    ego_speed_mps: float,
    target_speed_mps: float,
    altitude_diff_m: float = 0.0,
    base_altitude_m: float = 5000.0,
    lateral_offset_m: float = 0.0,
) -> Dict:
    """Build a scenario dict with explicit geometric semantics.

    Args:
        scenario_type: One of the ScenarioType literals.
        initial_range_m: Initial straight-line distance ego ↔ target.
        ego_speed_mps: Ego airspeed (m/s).
        target_speed_mps: Target airspeed (m/s).
        altitude_diff_m: Target altitude offset relative to ego.
        base_altitude_m: Ego altitude.
        lateral_offset_m: Optional lateral offset for offset_pursuit / crossing.

    Returns:
        dict with keys ``name``, ``own_init``, ``target_init``,
        ``metadata`` (includes computed aspect, closure rate, etc.).

    Raises:
        ValueError: If ``scenario_type`` is unknown or ambiguous.
    """
    if scenario_type not in VALID_SCENARIO_TYPES:
        raise ValueError(
            f"Unknown scenario_type '{scenario_type}'. "
            f"Valid: {sorted(VALID_SCENARIO_TYPES)}"
        )

    own_init = {
        "position_m": [0.0, 0.0, float(base_altitude_m)],
        "velocity_mps": float(ego_speed_mps),
        "heading_deg": 0.0,
    }

    rng = float(initial_range_m)
    v_e = float(ego_speed_mps)
    v_t = float(target_speed_mps)

    if scenario_type == "tail_chase":
        # Target ahead of ego, same direction, ego faster → positive closing
        target_pos = [rng, 0.0, base_altitude_m + altitude_diff_m]
        target_hdg = 0.0
        note = "Target ahead, same heading. Ego must be faster for positive closure."

    elif scenario_type == "head_on":
        # Target ahead of ego, flying directly toward ego
        target_pos = [rng, 0.0, base_altitude_m + altitude_diff_m]
        target_hdg = 180.0
        note = "Target ahead, opposite heading. High closing speed."

    elif scenario_type == "crossing_left":
        # Target to ego's left, crossing ego's path
        target_pos = [0.0, rng, base_altitude_m + altitude_diff_m]
        target_hdg = 270.0  # flying -y, will cross ego's +x path
        note = "Target to left, crossing from left to right relative to ego."

    elif scenario_type == "crossing_right":
        # Target to ego's right, crossing ego's path
        target_pos = [0.0, -rng, base_altitude_m + altitude_diff_m]
        target_hdg = 90.0  # flying +y, will cross ego's +x path
        note = "Target to right, crossing from right to left relative to ego."

    elif scenario_type in ("offset_pursuit", "offset_attack"):
        # Target behind and offset; ego must perform lead turn
        offset = float(lateral_offset_m) if lateral_offset_m != 0.0 else 400.0
        target_pos = [-rng * 0.8, offset, base_altitude_m + altitude_diff_m]
        target_hdg = 30.0  # slightly angled away
        note = "Target behind with lateral offset. Ego must perform lead turn."

    elif scenario_type == "fleeing":
        # Target behind ego, moving away explicitly (opposite heading)
        target_pos = [-rng, 0.0, base_altitude_m + altitude_diff_m]
        target_hdg = 180.0  # opposite direction, behind ego → definitely moving away
        note = "Target behind ego, opposite heading. Negative closure (target moving away)."

    else:
        # Should never reach here because of VALID_SCENARIO_TYPES check
        raise ValueError(f"Unhandled scenario_type: {scenario_type}")

    target_init = {
        "position_m": [float(v) for v in target_pos],
        "velocity_mps": float(target_speed_mps),
        "heading_deg": float(target_hdg),
    }

    # Compute derived geometry metadata
    meta = _compute_metadata(own_init, target_init)
    meta["scenario_type"] = scenario_type
    meta["expected_behavior_note"] = note
    meta["initial_range_m"] = float(initial_range_m)
    meta["altitude_diff_m"] = float(altitude_diff_m)

    return {
        "name": scenario_type,
        "own_init": own_init,
        "target_init": target_init,
        "metadata": meta,
    }


def _compute_metadata(own_init: Dict, target_init: Dict) -> Dict:
    """Compute aspect angle, closure rate, and feasibility flag."""
    own_pos = np.array(own_init["position_m"][:2], dtype=float)
    tgt_pos = np.array(target_init["position_m"][:2], dtype=float)
    own_vel = np.array(
        [
            own_init["velocity_mps"] * math.cos(math.radians(own_init["heading_deg"])),
            own_init["velocity_mps"] * math.sin(math.radians(own_init["heading_deg"])),
        ],
        dtype=float,
    )
    tgt_vel = np.array(
        [
            target_init["velocity_mps"] * math.cos(math.radians(target_init["heading_deg"])),
            target_init["velocity_mps"] * math.sin(math.radians(target_init["heading_deg"])),
        ],
        dtype=float,
    )

    rel_pos = tgt_pos - own_pos
    range_m = float(np.linalg.norm(rel_pos))
    los_unit = rel_pos / max(range_m, 1e-6)

    rel_vel = tgt_vel - own_vel
    range_rate = float(np.dot(rel_vel, los_unit))  # positive = target moving away
    closure_rate = -range_rate

    # Aspect angle: absolute heading difference, wrapped to [0, 180]
    hdg_diff = abs(target_init["heading_deg"] - own_init["heading_deg"]) % 360.0
    aspect_deg = float(min(hdg_diff, 360.0 - hdg_diff))

    ttc = range_m / max(closure_rate, 1.0)
    feasible = closure_rate > 0.0 and ttc < 100.0

    return {
        "aspect_angle_deg": round(aspect_deg, 2),
        "closure_rate_mps": round(closure_rate, 2),
        "range_rate_mps": round(range_rate, 2),
        "estimated_time_to_capture_s": round(ttc, 2),
        "expected_feasible_flag": feasible,
    }


def build_scenario_from_legacy_params(
    initial_range_m: float,
    ego_speed_mps: float,
    target_speed_mps: float,
    aspect_angle_deg: float,
    altitude_diff_m: float,
    base_altitude_m: float = 5000.0,
) -> Dict:
    """Legacy wrapper that infers scenario_type from aspect_angle_deg.

    **Deprecated**: Use ``build_explicit_scenario`` for new code.
    This wrapper exists for backward compatibility with Stage 6G.5 grids.

    Raises:
        ValueError: If ``aspect_angle_deg`` is ambiguous (e.g. 180° could
            mean head_on or fleeing).
    """
    a = float(aspect_angle_deg)
    if a == 0.0:
        return build_explicit_scenario(
            "tail_chase", initial_range_m, ego_speed_mps, target_speed_mps,
            altitude_diff_m, base_altitude_m,
        )
    elif a == 90.0:
        # Legacy 90° was broadside/crossing; arbitrarily choose crossing_left
        return build_explicit_scenario(
            "crossing_left", initial_range_m, ego_speed_mps, target_speed_mps,
            altitude_diff_m, base_altitude_m,
        )
    elif a == 180.0:
        raise ValueError(
            "aspect_angle_deg=180 is ambiguous: use 'head_on' or 'fleeing' "
            "explicitly via build_explicit_scenario()."
        )
    else:
        # Best-effort fallback for intermediate angles
        if 0 < a < 90:
            return build_explicit_scenario(
                "crossing_left", initial_range_m, ego_speed_mps, target_speed_mps,
                altitude_diff_m, base_altitude_m, lateral_offset_m=initial_range_m * math.sin(math.radians(a)),
            )
        else:
            return build_explicit_scenario(
                "crossing_right", initial_range_m, ego_speed_mps, target_speed_mps,
                altitude_diff_m, base_altitude_m, lateral_offset_m=initial_range_m * math.sin(math.radians(180 - a)),
            )
