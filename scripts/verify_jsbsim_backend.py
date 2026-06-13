"""Verify JSBSim backend integration with self-contained project data.

This script checks that:
1. The project-internal JSBSim data directory exists and contains required subdirs.
2. _resolve_jsbsim_data_dir resolves to the project-internal path by default.
3. JSBSimEnv can initialize, reset, and step an F-16 aircraft.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.uav_vpp_guidance.envs.jsbsim_env import (  # noqa: E402
    _resolve_jsbsim_data_dir,
    _JSBSimAircraft,
    JSBSimEnv,
)


def check_data_dir(data_dir: str) -> bool:
    """Check that data_dir contains aircraft/, engine/, systems/."""
    required = ["aircraft", "engine", "systems"]
    missing = [d for d in required if not os.path.isdir(os.path.join(data_dir, d))]
    if missing:
        print(f"FAIL: {data_dir} missing subdirectories: {missing}")
        return False
    f16_xml = os.path.join(data_dir, "aircraft", "f16", "f16.xml")
    if not os.path.isfile(f16_xml):
        print(f"FAIL: F-16 model not found at {f16_xml}")
        return False
    print(f"OK: JSBSim data directory valid: {data_dir}")
    return True


def check_resolution() -> bool:
    """Check that _resolve_jsbsim_data_dir returns project-internal path."""
    data_dir = _resolve_jsbsim_data_dir({})
    expected = os.path.join(PROJECT_ROOT, "data", "jsbsim")
    if data_dir and os.path.samefile(data_dir, expected):
        print(f"OK: Resolved to project-internal path: {data_dir}")
        return True
    print(f"FAIL: Expected {expected}, got {data_dir}")
    return False


def check_aircraft() -> bool:
    """Check that a single aircraft wrapper can find its data dir."""
    config = {"model": "f16", "sim_freq": 60}
    aircraft = _JSBSimAircraft("test", config, origin=(120.0, 60.0, 0.0))
    if aircraft.jsbsim_data_dir and os.path.isdir(aircraft.jsbsim_data_dir):
        print(f"OK: _JSBSimAircraft data dir: {aircraft.jsbsim_data_dir}")
        return True
    print("FAIL: _JSBSimAircraft did not resolve a valid data dir")
    return False


def check_env() -> bool:
    """Check that JSBSimEnv can reset and step."""
    config = {"model": "f16", "sim_freq": 60, "use_jsbsim": True}
    env = JSBSimEnv(config)
    env.add_aircraft("own")
    state = env.reset({"own": {"ic/h-sl-ft": 20000.0, "ic/u-fps": 800.0}})
    if "own" not in state:
        print("FAIL: JSBSimEnv reset did not return 'own' state")
        return False
    for _ in range(3):
        state = env.step(
            {
                "own": {
                    "fcs/throttle-cmd-norm": 0.8,
                    "fcs/elevator-cmd-norm": 0.0,
                    "fcs/aileron-cmd-norm": 0.0,
                    "fcs/rudder-cmd-norm": 0.0,
                }
            }
        )
    env.close()
    print(
        f"OK: JSBSimEnv reset and step succeeded; final speed={state['own']['speed_mps']:.2f} m/s"
    )
    return True


def main() -> int:
    data_dir = os.path.join(PROJECT_ROOT, "data", "jsbsim")
    checks = [
        check_data_dir(data_dir),
        check_resolution(),
        check_aircraft(),
        check_env(),
    ]
    if all(checks):
        print("\nJSBSim backend integration verified successfully.")
        return 0
    print("\nJSBSim backend integration verification FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
