"""Geometry validation and classification utilities.

Stage 6H.0-F.1: Given an initialized scenario or live aircraft state,
compute geometric relationships and produce a human-readable classification.

This module eliminates ambiguity between:
    - scenario parameters
    - actual geometry after env reset
    - telemetry labels
    - evaluator classification
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

# Geometry families (must match ScenarioRegistry)
GEOMETRY_FAMILIES = {
    "tail_chase",
    "head_on",
    "crossing_left",
    "crossing_right",
    "offset_attack",
    "fleeing",
}

# Aspect-angle thresholds for family classification (deg)
_ASPECT_TAIL_CHASE_MAX = 30.0
_ASPECT_CROSSING_MIN = 60.0
_ASPECT_CROSSING_MAX = 120.0
_ASPECT_HEAD_ON_MIN = 150.0


def compute_relative_geometry(
    own_position_m: np.ndarray,
    own_heading_deg: float,
    own_speed_mps: float,
    target_position_m: np.ndarray,
    target_heading_deg: float,
    target_speed_mps: float,
) -> Dict:
    """Compute full relative geometry from state vectors.

    Args:
        own_position_m: Ego position [x, y, z] (m).
        own_heading_deg: Ego true heading (deg, 0 = +x).
        own_speed_mps: Ego scalar speed (m/s).
        target_position_m: Target position [x, y, z] (m).
        target_heading_deg: Target true heading (deg, 0 = +x).
        target_speed_mps: Target scalar speed (m/s).

    Returns:
        dict with keys:
            - relative_position_m
            - relative_heading_deg
            - los_angle_deg
            - los_angle_from_ego_deg
            - los_unit_vector
            - closing_velocity_mps
            - closure_rate_mps
            - range_rate_mps
            - range_m
            - aspect_angle_deg
            - cross_range_m
            - opening_closing_status
    """
    own_pos = np.asarray(own_position_m, dtype=float)
    tgt_pos = np.asarray(target_position_m, dtype=float)
    rel_pos = tgt_pos - own_pos
    range_m = float(np.linalg.norm(rel_pos))
    los_unit = rel_pos / max(range_m, 1e-6)

    # Velocity vectors in NE plane
    own_hdg_rad = math.radians(own_heading_deg)
    tgt_hdg_rad = math.radians(target_heading_deg)
    own_vel = np.array(
        [own_speed_mps * math.cos(own_hdg_rad), own_speed_mps * math.sin(own_hdg_rad), 0.0]
    )
    tgt_vel = np.array(
        [target_speed_mps * math.cos(tgt_hdg_rad), target_speed_mps * math.sin(tgt_hdg_rad), 0.0]
    )

    rel_vel = tgt_vel - own_vel
    range_rate = float(np.dot(rel_vel[:2], los_unit[:2]))  # positive = opening
    closure_rate = -range_rate

    # LOS angle from global +x axis
    los_angle_deg = float(np.degrees(np.arctan2(los_unit[1], los_unit[0])))

    # LOS angle from ego heading
    los_from_ego_deg = (los_angle_deg - own_heading_deg) % 360.0
    if los_from_ego_deg > 180.0:
        los_from_ego_deg -= 360.0

    # Relative heading
    rel_hdg = (target_heading_deg - own_heading_deg) % 360.0
    if rel_hdg > 180.0:
        rel_hdg -= 360.0

    # Aspect angle: absolute heading difference wrapped to [0, 180]
    hdg_diff = abs(target_heading_deg - own_heading_deg) % 360.0
    aspect_deg = float(min(hdg_diff, 360.0 - hdg_diff))

    # Cross-range: perpendicular distance from LOS to ego velocity vector
    ego_vel_2d = own_vel[:2]
    cross_range_m = float(
        np.linalg.norm(np.cross(np.append(ego_vel_2d, 0.0), np.append(rel_pos[:2], 0.0)))
        / max(np.linalg.norm(ego_vel_2d), 1e-6)
    ) if np.linalg.norm(ego_vel_2d) > 1e-6 else 0.0

    # Opening / closing status
    if closure_rate > 5.0:
        oc_status = "closing"
    elif closure_rate < -5.0:
        oc_status = "opening"
    else:
        oc_status = "neutral"

    return {
        "relative_position_m": rel_pos.tolist(),
        "relative_heading_deg": round(rel_hdg, 2),
        "los_angle_deg": round(los_angle_deg, 2),
        "los_angle_from_ego_deg": round(los_from_ego_deg, 2),
        "los_unit_vector": los_unit.tolist(),
        "closing_velocity_mps": round(closure_rate, 2),
        "closure_rate_mps": round(closure_rate, 2),
        "range_rate_mps": round(range_rate, 2),
        "range_m": round(range_m, 2),
        "aspect_angle_deg": round(aspect_deg, 2),
        "cross_range_m": round(cross_range_m, 2),
        "opening_closing_status": oc_status,
    }


def classify_geometry_family(geometry: Dict) -> str:
    """Classify geometry into an explicit family from computed geometry.

    Uses aspect angle + LOS bearing + closure rate to resolve ambiguities
    (e.g., head-on vs fleeing both have aspect ~180).

    Args:
        geometry: Output of compute_relative_geometry().

    Returns:
        One of GEOMETRY_FAMILIES.
    """
    aspect = geometry["aspect_angle_deg"]
    los_from_ego = geometry["los_angle_from_ego_deg"]
    closure = geometry["closure_rate_mps"]

    # Tail chase: small aspect, target ahead
    if aspect <= _ASPECT_TAIL_CHASE_MAX and abs(los_from_ego) <= _ASPECT_TAIL_CHASE_MAX:
        return "tail_chase"

    # Head-on vs fleeing vs crossing: all can have aspect near 180
    if aspect >= _ASPECT_HEAD_ON_MIN:
        cross_range = geometry.get("cross_range_m", 0.0)
        if closure > 0:
            # If significant cross-range, it's a crossing even with 180 aspect
            if cross_range > 200.0:
                return "crossing_left" if los_from_ego > 0 else "crossing_right"
            return "head_on" if abs(los_from_ego) <= 30.0 else "crossing_right"
        else:
            return "fleeing"

    # Crossing: medium aspect
    if _ASPECT_CROSSING_MIN <= aspect <= _ASPECT_CROSSING_MAX:
        # Distinguish left vs right by LOS bearing from ego
        if los_from_ego > 0:
            return "crossing_left"
        else:
            return "crossing_right"

    # Offset attack: target is behind ego with lateral offset, aspect small-to-medium,
    # and LOS from ego is > 90 deg (behind)
    if abs(los_from_ego) > 90.0 and aspect < _ASPECT_HEAD_ON_MIN:
        return "offset_attack"

    # Offset attack can also be ahead with lateral offset
    if abs(los_from_ego) > 30.0 and aspect > _ASPECT_TAIL_CHASE_MAX and aspect < _ASPECT_CROSSING_MIN:
        return "offset_attack"

    # Default fallback based on closure and LOS
    if closure < 0:
        return "fleeing"
    if abs(los_from_ego) <= 30.0:
        return "head_on"
    return "crossing_right" if los_from_ego < 0 else "crossing_left"


def validate_scenario_geometry(scenario: Dict) -> Dict:
    """Validate a scenario dict and return full geometry report.

    Args:
        scenario: dict with keys own_init, target_init.

    Returns:
        dict with:
            - geometry: full relative geometry
            - classified_family: geometry family name
            - human_readable: one-line description
            - consistency_checks: dict of pass/fail assertions
    """
    own = scenario.get("own_init", {})
    tgt = scenario.get("target_init", {})

    own_pos = np.array(own.get("position_m", [0.0, 0.0, 5000.0]))
    own_hdg = float(own.get("heading_deg", 0.0))
    own_spd = float(own.get("velocity_mps", 0.0))

    tgt_pos = np.array(tgt.get("position_m", [0.0, 0.0, 5000.0]))
    tgt_hdg = float(tgt.get("heading_deg", 0.0))
    tgt_spd = float(tgt.get("velocity_mps", 0.0))

    geo = compute_relative_geometry(own_pos, own_hdg, own_spd, tgt_pos, tgt_hdg, tgt_spd)
    family = classify_geometry_family(geo)

    # Consistency checks
    checks = {
        "range_positive": geo["range_m"] > 0,
        "aspect_in_0_180": 0.0 <= geo["aspect_angle_deg"] <= 180.0,
        "family_known": family in GEOMETRY_FAMILIES,
        "los_unit_normalized": abs(np.linalg.norm(geo["los_unit_vector"]) - 1.0) < 0.01,
    }

    # Human-readable summary
    hr = (
        f"{family}: range={geo['range_m']:.0f}m, "
        f"aspect={geo['aspect_angle_deg']:.1f}deg, "
        f"closure={geo['closure_rate_mps']:.1f}m/s, "
        f"LOS_from_ego={geo['los_angle_from_ego_deg']:.1f}deg"
    )

    return {
        "geometry": geo,
        "classified_family": family,
        "human_readable": hr,
        "consistency_checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def validate_env_state_after_reset(env) -> Dict:
    """Validate actual geometry after environment reset.

    Args:
        env: A reset environment instance (CloseRangeTrackingEnv or similar).

    Returns:
        Same format as validate_scenario_geometry().
    """
    # Extract state from env backend
    if hasattr(env, "_simple_env"):
        # Simple backend
        simple = env._simple_env
        own_state = simple.own_state
        tgt_state = simple.target_state
        own_pos = np.array(own_state.get("position_m", [0.0, 0.0, 5000.0]))
        own_hdg = float(np.degrees(own_state.get("heading_rad", 0.0)))
        own_spd = float(np.linalg.norm(own_state.get("velocity_vector_mps", [0.0, 0.0, 0.0])))
        tgt_pos = np.array(tgt_state.get("position_m", [0.0, 0.0, 5000.0]))
        tgt_hdg = float(np.degrees(tgt_state.get("heading_rad", 0.0)))
        tgt_spd = float(np.linalg.norm(tgt_state.get("velocity_vector_mps", [0.0, 0.0, 0.0])))
    elif hasattr(env, "jsbsim_env") and env.jsbsim_env is not None:
        # JSBSim backend — extract from property dictionary
        jsb = env.jsbsim_env
        own_pos = np.array([
            jsb.get_property("position/eci-x-m", 0.0),
            jsb.get_property("position/eci-y-m", 0.0),
            jsb.get_property("position/eci-z-m", 0.0),
        ])
        own_hdg = jsb.get_property("attitude/heading-true-rad", 0.0)
        own_hdg = float(np.degrees(own_hdg))
        own_spd = jsb.get_property("velocities/vt-mps", 0.0)
        # Target extraction depends on JSBSimEnv structure
        tgt_pos = np.array([0.0, 0.0, 5000.0])  # placeholder
        tgt_hdg = 0.0
        tgt_spd = 0.0
    else:
        raise RuntimeError("Cannot extract state from unknown env type")

    geo = compute_relative_geometry(own_pos, own_hdg, own_spd, tgt_pos, tgt_hdg, tgt_spd)
    family = classify_geometry_family(geo)

    checks = {
        "range_positive": geo["range_m"] > 0,
        "aspect_in_0_180": 0.0 <= geo["aspect_angle_deg"] <= 180.0,
        "family_known": family in GEOMETRY_FAMILIES,
    }

    hr = (
        f"{family}: range={geo['range_m']:.0f}m, "
        f"aspect={geo['aspect_angle_deg']:.1f}deg, "
        f"closure={geo['closure_rate_mps']:.1f}m/s"
    )

    return {
        "geometry": geo,
        "classified_family": family,
        "human_readable": hr,
        "consistency_checks": checks,
        "all_checks_pass": all(checks.values()),
    }
