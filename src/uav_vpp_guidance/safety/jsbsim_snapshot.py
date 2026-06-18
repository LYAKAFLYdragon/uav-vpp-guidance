"""Capture and replay a JSBSim aircraft state for finite-difference Jacobians.

JSBSim's Python FGFDMExec does not expose a cheap deterministic save/restore.
The most reliable way to re-initialize to the current dynamic state is to
re-`run_ic` from initial-condition properties plus a small set of writable
engine / throttle states.  This module converts a `_JSBSimAircraft` state into
an ``init_state`` dict and engine-state dict that can be applied to a freshly
loaded aircraft, giving bit-identical starting conditions for repeated
perturbation roll-outs.
"""

from typing import Any, Dict

import numpy as np


# JSBSim property names that survive run_ic and must be restored afterwards.
_WRITABLE_POST_IC_PROPS = [
    "fcs/throttle-cmd-norm",
    "fcs/throttle-pos-norm",
    "propulsion/engine/n1",
    "propulsion/engine/n2",
]


def aircraft_state_to_init_state(state: Dict[str, Any]) -> Dict[str, float]:
    """Convert an aircraft state dict into JSBSim IC properties.

    Args:
        state: State dict as returned by ``_JSBSimAircraft.get_state()``.
            Must contain ``position_lla`` (lon, lat, alt_m), ``attitude_rpy``
            (roll, pitch, yaw rad), body velocity ``velocity_body`` (u, v, w m/s),
            and ``body_rates_rps`` (p, q, r rad/s).

    Returns:
        Dict of ``ic/...`` properties that can be passed to ``reload()``.
    """
    lla = np.asarray(state["position_lla"], dtype=np.float64)
    rpy = np.asarray(state["attitude_rpy"], dtype=np.float64)
    vel_body = np.asarray(state["velocity_body"], dtype=np.float64)
    rates = np.asarray(state["body_rates_rps"], dtype=np.float64)

    # Convert altitude meters -> feet and body velocity m/s -> ft/s.
    m_to_ft = 3.28084
    ms_to_fps = 3.28084

    return {
        "ic/long-gc-deg": float(lla[0]),
        "ic/lat-geod-deg": float(lla[1]),
        "ic/h-sl-ft": float(lla[2]) * m_to_ft,
        "ic/phi-rad": float(rpy[0]),
        "ic/theta-rad": float(rpy[1]),
        "ic/psi-true-deg": float(np.degrees(rpy[2])),
        "ic/u-fps": float(vel_body[0]) * ms_to_fps,
        "ic/v-fps": float(vel_body[1]) * ms_to_fps,
        "ic/w-fps": float(vel_body[2]) * ms_to_fps,
        "ic/p-rad_sec": float(rates[0]),
        "ic/q-rad_sec": float(rates[1]),
        "ic/r-rad_sec": float(rates[2]),
    }


def capture_aircraft_state(aircraft: Any) -> Dict[str, float]:
    """Capture the writable dynamic state of a single JSBSim aircraft.

    Args:
        aircraft: ``_JSBSimAircraft`` instance.

    Returns:
        Dict mapping property name to current value.
    """
    exec_ = aircraft.jsbsim_exec
    return {p: float(exec_.get_property_value(p)) for p in _WRITABLE_POST_IC_PROPS}


def apply_aircraft_state(aircraft: Any, post_ic_state: Dict[str, float]) -> None:
    """Apply writable dynamic state after ``run_ic`` has completed.

    Args:
        aircraft: ``_JSBSimAircraft`` instance.
        post_ic_state: Values from :func:`capture_aircraft_state`.
    """
    exec_ = aircraft.jsbsim_exec
    for prop, value in post_ic_state.items():
        try:
            exec_.set_property_value(prop, float(value))
        except Exception:
            # Some JSBSim builds expose the property as read-only; ignore.
            pass


def capture_jsbsim_env(env: Any) -> Dict[str, Dict[str, float]]:
    """Capture IC + writable state for every aircraft in a JSBSimEnv.

    Args:
        env: ``JSBSimEnv`` instance.

    Returns:
        Mapping ``uid -> {"init_state": {...}, "post_ic_state": {...}}``.
    """
    return {
        uid: {
            "init_state": aircraft_state_to_init_state(ac.get_state()),
            "post_ic_state": capture_aircraft_state(ac),
        }
        for uid, ac in env._aircraft.items()
    }


def apply_jsbsim_env(env: Any, snapshot: Dict[str, Dict[str, float]]) -> None:
    """Reset a JSBSimEnv to a captured snapshot.

    This calls ``env.reset()`` with the captured ICs and then restores the
    writable post-IC properties for each aircraft.

    Args:
        env: ``JSBSimEnv`` instance (fresh or reused).
        snapshot: Snapshot from :func:`capture_jsbsim_env`.
    """
    init_states = {uid: data["init_state"] for uid, data in snapshot.items()}
    env.reset(init_states)
    for uid, data in snapshot.items():
        apply_aircraft_state(env._aircraft[uid], data["post_ic_state"])
