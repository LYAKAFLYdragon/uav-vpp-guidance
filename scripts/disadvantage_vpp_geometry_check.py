#!/usr/bin/env python3
"""Debug: show how VPP dynamics-aware clipping behaves in the disadvantage scenario."""
import sys
from pathlib import Path
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_vpp_guidance.virtual_point.generator import VirtualPointGenerator


def heading_from_deg(deg):
    return np.array([
        math.cos(math.radians(deg)),
        math.sin(math.radians(deg)),
        0.0,
    ])


def make_disadvantage_states():
    own = {
        "position_m": np.array([0.0, 0.0, 5000.0]),
        "velocity_vector_mps": 200.0 * heading_from_deg(0.0),
    }
    target = {
        "position_m": np.array([-600.0, 400.0, 5000.0]),
        "velocity_vector_mps": 220.0 * heading_from_deg(30.0),
    }
    return own, target


def main():
    own, target = make_disadvantage_states()
    los = target["position_m"] - own["position_m"]
    print(f"Own pos: {own['position_m']}, heading: 0 deg, speed: 200 m/s")
    print(f"Target pos: {target['position_m']}, heading: 30 deg, speed: 220 m/s")
    print(f"LOS heading to target: {math.degrees(math.atan2(los[1], los[0])):.1f} deg, range: {np.linalg.norm(los):.0f} m")

    vp_config = {
        "action_dim": 3,
        "d_long_range": [-1000.0, 1000.0],
        "d_lat_range": [-600.0, 600.0],
        "d_vert_range": [-300.0, 300.0],
        "smoothing_alpha": 0.5,
        "dynamics_aware": False,
        "max_heading_rate": 0.2,
        "lookahead_steps": 5,
    }

    actions = {
        "straight_ahead": np.array([1.0, 0.0, 0.0]),   # far ahead of target along x
        "left_of_target": np.array([0.0, -1.0, 0.0]),  # lateral offset -600 m
        "right_of_target": np.array([0.0, 1.0, 0.0]),  # lateral offset +600 m
    }

    for dyn in [False, True]:
        vp_config["dynamics_aware"] = dyn
        vpg = VirtualPointGenerator(vp_config)
        print(f"\n--- dynamics_aware={dyn}, max_feasible_heading_change={math.degrees(0.2 * 0.2 * 5):.1f} deg ---")
        for name, action in actions.items():
            vp = vpg.action_to_virtual_point(
                action, own, target,
                anchor_mode="current_target",
                return_info=True,
            )
            vp_pos = vp[0]["position"]
            info = vp[1]
            los_vp = vp_pos - own["position_m"]
            hdg = math.degrees(math.atan2(los_vp[1], los_vp[0]))
            print(f"  {name}: offset={info['offset']}, VPP heading={hdg:.1f} deg, VPP pos={vp_pos}")


if __name__ == "__main__":
    main()
