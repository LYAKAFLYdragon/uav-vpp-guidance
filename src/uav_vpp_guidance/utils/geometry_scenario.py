"""Geometry scenario builder for wide-envelope sweep experiments.

.. deprecated::
    Stage 6H.0-F.1
    This module is retained for backward compatibility with Stage 6G.5A
    grid sweep experiments.  All NEW code should use
    ``uav_vpp_guidance.envs.geometry_scenarios.build_explicit_scenario``
    or the ``ScenarioRegistry`` instead.

    The legacy ``aspect_angle_deg`` parameter is AMBIGUOUS:
        - aspect=0   → tail chase (target ahead, same heading)
        - aspect=90  → crossing (target broadside, heading 90)
        - aspect=180 → AMBIGUOUS: could be head-on OR fleeing
    The explicit builder eliminates this ambiguity by requiring a
    ``scenario_type`` string.

Explicit geometry families (see ``geometry_scenarios.GEOMETRY_FAMILY_DOCS``):
    - tail_chase:     target ahead, same heading, aspect ~0
    - head_on:        target ahead, opposite heading, aspect ~180, positive closure
    - crossing_left:  target to left, crossing path, aspect ~90
    - crossing_right: target to right, crossing path, aspect ~90
    - offset_attack:  target behind with lateral offset, ego must lead-turn
    - fleeing:        target behind, opposite heading, aspect ~180, negative closure
"""

import math
import warnings
from itertools import product
from typing import Dict, List

import numpy as np


def build_geometry_scenario(
    initial_range_m: float,
    ego_speed_mps: float,
    target_speed_mps: float,
    aspect_angle_deg: float,
    altitude_diff_m: float,
    base_altitude_m: float = 5000.0,
) -> Dict:
    """Build an own_init / target_init scenario fragment from geometry parameters.

    .. deprecated::
        Use ``build_explicit_scenario`` from ``geometry_scenarios`` instead.
        This function relies on the ambiguous ``aspect_angle_deg`` convention.

    Args:
        initial_range_m: Initial distance between ownship and target.
        ego_speed_mps: Ownship speed.
        target_speed_mps: Target speed.
        aspect_angle_deg: Angle of target relative to ownship heading (0 = tail-chase,
            90 = broadside/crossing). Target heading equals this angle.
            **WARNING**: aspect=180 is ambiguous (head-on vs fleeing).
        altitude_diff_m: Target altitude offset relative to ownship.
        base_altitude_m: Ownship altitude.

    Returns:
        dict with keys ``own_init`` and ``target_init``.
    """
    warnings.warn(
        "build_geometry_scenario() is deprecated. Use build_explicit_scenario() "
        "from uav_vpp_guidance.envs.geometry_scenarios for unambiguous geometry.",
        DeprecationWarning,
        stacklevel=2,
    )
    aspect_rad = math.radians(aspect_angle_deg)

    own_init = {
        "position_m": [0.0, 0.0, base_altitude_m],
        "velocity_mps": float(ego_speed_mps),
        "heading_deg": 0.0,
    }

    target_x = float(initial_range_m) * math.cos(aspect_rad)
    target_y = float(initial_range_m) * math.sin(aspect_rad)
    target_alt = base_altitude_m + float(altitude_diff_m)

    target_init = {
        "position_m": [target_x, target_y, target_alt],
        "velocity_mps": float(target_speed_mps),
        "heading_deg": float(aspect_angle_deg),
    }

    return {"name": "geometry_sweep", "own_init": own_init, "target_init": target_init}


def compute_geometry_metadata(params: Dict) -> Dict:
    """Derive secondary geometry metrics from raw parameters using vector projection.

    Parameters expected in *params*:
        - ego_speed_mps
        - target_speed_mps
        - aspect_angle_deg
        - initial_range_m
    """
    ego = float(params["ego_speed_mps"])
    tgt = float(params["target_speed_mps"])
    aspect_rad = math.radians(float(params["aspect_angle_deg"]))
    rng = float(params["initial_range_m"])

    # LOS unit vector from ownship to target
    los_unit = np.array([math.cos(aspect_rad), math.sin(aspect_rad)], dtype=float)

    # Velocity vectors in NE plane
    own_vel = np.array([ego, 0.0], dtype=float)  # own heading = 0 deg
    tgt_vel = np.array([tgt * math.cos(aspect_rad), tgt * math.sin(aspect_rad)], dtype=float)

    # Range rate = projection of relative velocity onto LOS
    rel_vel = tgt_vel - own_vel
    range_rate = float(np.dot(rel_vel, los_unit))  # positive if target moving away
    closure_rate = -range_rate  # positive if closing

    ttc = rng / max(closure_rate, 1.0)

    # Feasibility heuristic: positive closure and enough time within max steps (~100 s)
    feasible = closure_rate > 0.0 and ttc < 100.0

    return {
        "closure_rate_mps": round(closure_rate, 2),
        "range_rate_mps": round(range_rate, 2),
        "estimated_time_to_capture_s": round(ttc, 2),
        "expected_feasible_flag": feasible,
    }


def build_full_grid(grid_def: Dict) -> List[Dict]:
    """Cartesian product of all grid axes."""
    axes = list(grid_def.keys())
    value_lists = [grid_def[a] for a in axes]
    points = []
    for combo in product(*value_lists):
        points.append(dict(zip(axes, combo)))
    return points


def sample_grid(grid_def: Dict, sample_size: int, method: str, seed: int) -> List[Dict]:
    """Sample points from the full discrete grid.

    Args:
        grid_def: Mapping axis_name -> list of values.
        sample_size: Number of points to draw.
        method: ``"random"`` or ``"latin_hypercube"``.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled parameter dictionaries.
    """
    full_grid = build_full_grid(grid_def)
    rng = np.random.default_rng(seed)
    n_total = len(full_grid)

    if sample_size >= n_total:
        return full_grid

    if method == "random":
        indices = rng.choice(n_total, size=sample_size, replace=False)
        return [full_grid[i] for i in indices]

    if method == "latin_hypercube":
        try:
            from scipy.stats import qmc
        except Exception:  # pragma: no cover
            indices = rng.choice(n_total, size=sample_size, replace=False)
            return [full_grid[i] for i in indices]

        axes = list(grid_def.keys())
        value_lists = [grid_def[a] for a in axes]
        sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
        sample = sampler.random(n=sample_size)

        points = []
        for row in sample:
            point = {}
            for axis, vals, s in zip(axes, value_lists, row):
                idx = min(int(s * len(vals)), len(vals) - 1)
                point[axis] = vals[idx]
            points.append(point)
        return points

    raise ValueError(f"Unknown sampling method: {method}")
