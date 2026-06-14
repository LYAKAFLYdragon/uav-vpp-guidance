#!/usr/bin/env python3
"""
Demo: fly a single maneuver primitive in JSBSim.

This script is a template.  JSBSim must be installed and the project JSBSim
assets (`data/jsbsim`) must be present.  The script sets up a trimmed F-16,
executes one maneuver from the library, and logs state/commands to CSV.

Example:
    python scripts/demo_maneuver_jsbsim.py --maneuver loop --steps 3000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from uav_vpp_guidance.maneuver_library import (
    FlightState,
    InnerLoopController,
    ManeuverExecutor,
    ManeuverLibrary,
)


def build_flight_state(fdm) -> FlightState:
    """Convert JSBSim state to library FlightState."""
    # SI conversions: ft -> m, ft/s -> m/s
    FT2M = 0.3048
    return FlightState(
        t=fdm.get_sim_time(),
        position_m=np.array([
            fdm.get_property_value("position/lat-gc-deg"),
            fdm.get_property_value("position/long-gc-deg"),
            fdm.get_property_value("position/h-sl-ft") * FT2M,
        ]),
        velocity_mps=fdm.get_property_value("velocities/vt-fps") * FT2M,
        altitude_m=fdm.get_property_value("position/h-sl-ft") * FT2M,
        phi_rad=fdm.get_property_value("attitude/phi-rad"),
        theta_rad=fdm.get_property_value("attitude/theta-rad"),
        psi_rad=fdm.get_property_value("attitude/psi-rad"),
        p_rps=fdm.get_property_value("velocities/p-rad_sec"),
        q_rps=fdm.get_property_value("velocities/q-rad_sec"),
        r_rps=fdm.get_property_value("velocities/r-rad_sec"),
        nz=fdm.get_property_value("accelerations/n-z-cg-fps_sec") / 32.174 + 1.0,
        alpha_rad=fdm.get_property_value("aero/alpha-rad"),
        beta_rad=fdm.get_property_value("aero/beta-rad"),
        mach=fdm.get_property_value("velocities/mach"),
    )


def apply_command(fdm, cmd):
    fdm.set_property_value("fcs/elevator-cmd-norm", cmd.elevator)
    fdm.set_property_value("fcs/aileron-cmd-norm", cmd.aileron)
    fdm.set_property_value("fcs/rudder-cmd-norm", cmd.rudder)
    fdm.set_property_value("fcs/throttle-cmd-norm", cmd.throttle)


def set_initial_conditions(fdm, altitude_m, vt_mps, theta_deg=0.0, phi_deg=0.0, psi_deg=0.0):
    """Set JSBSim initial conditions before run_ic()."""
    M2FT = 3.28084
    fdm.set_property_value("ic/h-sl-ft", altitude_m * M2FT)
    fdm.set_property_value("ic/vt-fps", vt_mps * M2FT)
    fdm.set_property_value("ic/theta-deg", theta_deg)
    fdm.set_property_value("ic/phi-deg", phi_deg)
    fdm.set_property_value("ic/psi-deg", psi_deg)
    fdm.set_property_value("ic/lat-gc-deg", 0.0)
    fdm.set_property_value("ic/long-gc-deg", 0.0)
    fdm.set_property_value("ic/gamma-deg", 0.0)


def main():
    parser = argparse.ArgumentParser(description="JSBSim maneuver demo")
    parser.add_argument("--maneuver", type=str, default="loop",
                        choices=ManeuverLibrary.list_maneuvers())
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/maneuver_demo/log.csv"))
    parser.add_argument("--params", type=str, default="{}",
                        help="JSON dict of maneuver parameters")
    parser.add_argument("--altitude-m", type=float, default=5000.0,
                        help="Initial altitude (m)")
    parser.add_argument("--vt-mps", type=float, default=250.0,
                        help="Initial true airspeed (m/s)")
    parser.add_argument("--settle-steps", type=int, default=600,
                        help="Steps to let the aircraft settle before starting maneuver")
    args = parser.parse_args()

    import json
    params = json.loads(args.params)

    try:
        import jsbsim
    except ImportError as exc:
        raise RuntimeError(
            "JSBSim is not installed. Install it or run this on a JSBSim-capable machine."
        ) from exc

    root = Path("data/jsbsim")
    fdm = jsbsim.FGFDMExec(str(root))
    fdm.load_model("f16")
    fdm.set_dt(args.dt)
    set_initial_conditions(fdm, args.altitude_m, args.vt_mps)
    fdm.run_ic()  # run initial conditions

    maneuver = ManeuverLibrary.create(args.maneuver, params)
    executor = ManeuverExecutor(InnerLoopController())

    # Trim the aircraft for the requested flight condition.
    fdm.set_property_value("simulation/do_simple_trim", 1)
    # Run a short stabilization period while holding the trimmed controls.
    for _ in range(min(args.settle_steps, 120)):
        fdm.run()

    state = build_flight_state(fdm)
    print(f"Initial state: alt={state.altitude_m:.1f}m, vt={state.velocity_mps:.1f}m/s, "
          f"phi={np.degrees(state.phi_rad):.1f}deg, theta={np.degrees(state.theta_rad):.1f}deg")
    if not executor.select(maneuver, state):
        print(f"Maneuver {args.maneuver} cannot enter from current state.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "t", "alt_m", "vt_mps", "phi_deg", "theta_deg", "psi_deg",
            "nz", "elevator", "aileron", "rudder", "throttle",
        ])

        for step in range(args.steps):
            state = build_flight_state(fdm)
            cmd = executor.update(state, args.dt)
            apply_command(fdm, cmd)
            fdm.run()

            writer.writerow([
                step, state.t, state.altitude_m, state.velocity_mps,
                np.degrees(state.phi_rad), np.degrees(state.theta_rad),
                np.degrees(state.psi_rad), state.nz,
                cmd.elevator, cmd.aileron, cmd.rudder, cmd.throttle,
            ])

            if executor.is_idle():
                print(f"Maneuver completed at step {step}")
                break

    print(f"Demo finished. Log saved to {args.output}")


if __name__ == "__main__":
    main()
