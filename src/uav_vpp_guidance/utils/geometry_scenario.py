"""Geometry scenario builder for wide-envelope sweep experiments.

Stage 6G.5A: Supports aspect-angle-based target placement and derived
geometry metadata (closure rate, TTC, feasibility heuristics).
"""

import math
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

    Args:
        initial_range_m: Initial distance between ownship and target.
        ego_speed_mps: Ownship speed.
        target_speed_mps: Target speed.
        aspect_angle_deg: Angle of target relative to ownship heading (0 = tail-chase,
            90 = broadside/crossing). Target heading equals this angle.
        altitude_diff_m: Target altitude offset relative to ownship.
        base_altitude_m: Ownship altitude.

    Returns:
        dict with keys ``own_init`` and ``target_init``.
    """
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
    """Derive secondary geometry metrics from raw parameters.

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

    # Closure rate along the initial LOS
    closure = ego - tgt * math.cos(aspect_rad)
    ttc = rng / max(closure, 1.0)

    # Crude feasibility heuristic: positive closure and enough time within max steps
    feasible = closure > 0.0 and ttc < 100.0

    return {
        "closure_rate_mps": round(closure, 2),
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
