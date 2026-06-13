"""
JSBSim flight dynamics environment wrapper.

This module provides a thin wrapper around JSBSim FGFDMExec for single
or multiple aircraft simulation. JSBSim data is self-contained in
<project_root>/data/jsbsim/ (aircraft/, engine/, systems/).
"""

import os
import warnings
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# JSBSim Python bindings (must be installed separately)
try:
    import jsbsim
except ImportError:
    jsbsim = None
    logger.warning("jsbsim package not found. JSBSimEnv will not function.")

# ---------------------------------------------------------------------------
# Coordinate conversion utilities (migrated from legacy utils.py)
# ---------------------------------------------------------------------------

_pymap3d = None
try:
    import pymap3d
    _pymap3d = pymap3d
except ImportError:
    pass


def lla2neu(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0.0):
    """
    Convert Geodetic (lon, lat, alt) to NEU (north, east, up) relative to origin.

    Migrated from legacy LLA2NEU in utils.py.

    Args:
        lon, lat (float): geodetic longitude/latitude in degrees.
        alt (float): altitude above mean sea level in meters.
        lon0, lat0, alt0 (float): origin geodetic coordinates.

    Returns:
        np.ndarray: [north, east, up] in meters.
    """
    if _pymap3d is None:
        raise ImportError(
            "pymap3d is required for coordinate conversion. "
            "Install it via: pip install pymap3d"
        )
    n, e, d = _pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d], dtype=np.float64)


def neu2lla(n, e, u, lon0=120.0, lat0=60.0, alt0=0.0):
    """
    Convert NEU (north, east, up) to Geodetic (lon, lat, alt).

    Migrated from legacy NEU2LLA in utils.py.

    Args:
        n, e, u (float): relative position in meters.
        lon0, lat0, alt0 (float): origin geodetic coordinates.

    Returns:
        np.ndarray: [longitude, latitude, altitude] (deg, deg, m).
    """
    if _pymap3d is None:
        raise ImportError(
            "pymap3d is required for coordinate conversion. "
            "Install it via: pip install pymap3d"
        )
    lat, lon, h = _pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h], dtype=np.float64)


# ---------------------------------------------------------------------------
# JSBSim data directory resolution
# ---------------------------------------------------------------------------

def _resolve_jsbsim_data_dir(config):
    """Resolve JSBSim data directory with multiple strategies.

    Priority:
    1. config.get("jsbsim_data_dir") -- user-specified absolute path
    2. Project-relative default: <project_root>/data/jsbsim
    3. Backward compat: legacy_project_root + "envs/JSBSim/data"
    4. Environment variable: JSBSIM_ROOT + "envs/JSBSim/data"
    """
    # 1. User-specified path
    user_dir = config.get("jsbsim_data_dir")
    if user_dir and os.path.isdir(user_dir):
        return user_dir

    # 2. Project-relative default (preferred for self-contained project)
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "jsbsim")
    )
    if os.path.isdir(project_dir) and os.path.isdir(os.path.join(project_dir, "aircraft")):
        return project_dir

    # 3. Backward compat: legacy_project_root
    legacy_root = config.get("legacy_project_root", "")
    if legacy_root:
        compat_dir = os.path.join(legacy_root, "envs", "JSBSim", "data")
        if os.path.isdir(compat_dir):
            return compat_dir

    # 4. Environment variable
    env_root = os.environ.get("JSBSIM_ROOT")
    if env_root:
        env_dir = os.path.join(env_root, "envs", "JSBSim", "data")
        if os.path.isdir(env_dir):
            return env_dir

    warnings.warn(
        "JSBSim data directory not found. Expected one of:\n"
        "  1. config['jsbsim_data_dir'] (user-specified path)\n"
        "  2. <project_root>/data/jsbsim (project-relative, self-contained)\n"
        "  3. config['legacy_project_root'] + /envs/JSBSim/data (backward compat)\n"
        "  4. JSBSIM_ROOT env var + /envs/JSBSim/data\n"
        "Please run: cp -r <external_jsbsim>/data <project_root>/data/jsbsim",
        stacklevel=2,
    )
    return None


# ---------------------------------------------------------------------------
# Internal single-aircraft wrapper
# ---------------------------------------------------------------------------

class _JSBSimAircraft:
    """
    Minimal wrapper around a single JSBSim FGFDMExec instance.

    Migrated from legacy AircraftSimulator in simulatior.py.
    Removed: missile logic, team colors, bloods, multi-engine update callbacks.
    Kept: init, reload, step, state extraction.
    """

    def __init__(self, uid: str, config: dict, origin: tuple):
        """
        Args:
            uid (str): Unique aircraft identifier.
            config (dict): Aircraft-specific config (model, sim_freq, etc.).
            origin (tuple): (lon0, lat0, alt0) for NEU coordinate origin.
        """
        self.uid = uid
        self.config = config
        self.model = config.get("model", "f16")
        self.sim_freq = config.get("sim_freq", 60)
        self.dt = 1.0 / self.sim_freq
        self.origin = origin
        self.lon0, self.lat0, self.alt0 = origin
        self.jsbsim_data_dir = _resolve_jsbsim_data_dir(config)
        self.jsbsim_exec = None
        self._state = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self, init_state: Optional[Dict[str, float]] = None):
        """
        Reload the aircraft: create a fresh FGFDMExec, load model,
        apply initial conditions, init propulsion, and update state.

        Args:
            init_state (dict, optional): Mapping of JSBSim property names
                (e.g. "ic/h-sl-ft") to initial values.
        """
        self._close_exec()

        jsbsim_data_dir = self.jsbsim_data_dir
        if not jsbsim_data_dir or not os.path.isdir(jsbsim_data_dir):
            raise RuntimeError(
                f"JSBSim data directory not found: {jsbsim_data_dir}. "
                f"Please ensure <project_root>/data/jsbsim exists and contains "
                f"aircraft/, engine/, systems/ subdirectories."
            )

        self.jsbsim_exec = jsbsim.FGFDMExec(jsbsim_data_dir)
        self.jsbsim_exec.set_debug_level(0)
        self.jsbsim_exec.load_model(self.model)
        self.jsbsim_exec.set_dt(self.dt)

        # Clear legacy default initial conditions
        self._clear_default_conditions()

        # Apply user-provided initial state
        if init_state:
            for key, value in init_state.items():
                self.set_property_value(key, float(value))

        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to initialize simulation conditions.")

        # Init propulsion (engines to running state)
        propulsion = self.jsbsim_exec.get_propulsion()
        n_engines = propulsion.get_num_engines()
        for j in range(n_engines):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()

        self._update_state()

    def _close_exec(self):
        """Release the FGFDMExec reference."""
        self.jsbsim_exec = None

    def close(self):
        """Clean up."""
        self._close_exec()

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """
        Run one JSBSim integration step.

        Returns:
            bool: True if simulation continues successfully.
        """
        if self.jsbsim_exec is None:
            raise RuntimeError("Aircraft not initialized. Call reload() first.")
        result = self.jsbsim_exec.run()
        if not result:
            raise RuntimeError("JSBSim simulation step failed.")
        self._update_state()
        return result

    # ------------------------------------------------------------------
    # Property access
    # ------------------------------------------------------------------

    def set_property_value(self, name: str, value: float):
        """
        Set a JSBSim property by its string name.

        Args:
            name (str): JSBSim property name, e.g. "fcs/throttle-cmd-norm".
            value (float): Value to assign.
        """
        if self.jsbsim_exec is None:
            raise RuntimeError("Aircraft not initialized.")
        self.jsbsim_exec.set_property_value(name, float(value))

    def get_property_value(self, name: str) -> float:
        """
        Get a JSBSim property by its string name.

        Args:
            name (str): JSBSim property name.

        Returns:
            float: Current property value.
        """
        if self.jsbsim_exec is None:
            raise RuntimeError("Aircraft not initialized.")
        return self.jsbsim_exec.get_property_value(name)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _clear_default_conditions(self):
        """
        Reset common IC properties to legacy defaults.
        Migrated from AircraftSimulator.clear_defalut_condition().
        """
        defaults = {
            "ic/long-gc-deg": 120.0,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": 0.0,
            "ic/u-fps": 800.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/p-rad_sec": 0.0,
            "ic/q-rad_sec": 0.0,
            "ic/r-rad_sec": 0.0,
            "ic/roc-fpm": 0.0,
            "ic/terrain-elevation-ft": 0.0,
        }
        for k, v in defaults.items():
            self.jsbsim_exec.set_property_value(k, v)

    def _update_state(self):
        """
        Read essential properties from JSBSim and populate self._state.
        Migrated from AircraftSimulator._update_properties().
        """
        # Geodetic position
        lon = self.get_property_value("position/long-gc-deg")
        lat = self.get_property_value("position/lat-geod-deg")
        # JSBSim F-16 model does not populate position/h-sl-m reliably;
        # use position/h-sl-ft and convert to meters.
        alt_ft = self.get_property_value("position/h-sl-ft")
        alt_m = alt_ft * 0.3048

        # Attitude (rad)
        roll = self.get_property_value("attitude/roll-rad")
        pitch = self.get_property_value("attitude/pitch-rad")
        yaw = self.get_property_value("attitude/heading-true-rad")

        # Velocity (NED, m/s) — convert from fps
        fps_to_mps = 0.3048
        vn = self.get_property_value("velocities/v-north-fps") * fps_to_mps
        ve = self.get_property_value("velocities/v-east-fps") * fps_to_mps
        vd = self.get_property_value("velocities/v-down-fps") * fps_to_mps

        # True airspeed (m/s)
        vt = self.get_property_value("velocities/vt-fps") * fps_to_mps

        # NEU position relative to origin
        if _pymap3d is not None:
            n, e, u = lla2neu(lon, lat, alt_m, self.lon0, self.lat0, self.alt0)
        else:
            # Fallback local-plane approximation (accurate within ~50 km).
            # Used when pymap3d is not installed; for production, install pymap3d.
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * np.cos(np.radians(self.lat0))
            n = (lat - self.lat0) * meters_per_deg_lat
            e = (lon - self.lon0) * meters_per_deg_lon
            u = alt_m - self.alt0

        # Body-frame velocities (m/s)
        u_b = self.get_property_value("velocities/u-fps") * fps_to_mps
        v_b = self.get_property_value("velocities/v-fps") * fps_to_mps
        w_b = self.get_property_value("velocities/w-fps") * fps_to_mps

        # Body rates (rad/s) and load factor (g) — useful for control diagnosis
        try:
            p_rps = float(self.get_property_value("velocities/p-rad_sec"))
        except Exception:
            p_rps = 0.0
        try:
            q_rps = float(self.get_property_value("velocities/q-rad_sec"))
        except Exception:
            q_rps = 0.0
        try:
            r_rps = float(self.get_property_value("velocities/r-rad_sec"))
        except Exception:
            r_rps = 0.0
        try:
            nz_g = float(self.get_property_value("accelerations/Nz"))
        except Exception:
            # Fallback: estimate from vertical acceleration if Nz not available
            try:
                az = float(self.get_property_value("accelerations/a-z-ft_sec2")) * 0.3048
                nz_g = 1.0 + az / 9.80665
            except Exception:
                nz_g = 1.0

        # 统一别名字段，供上层模块（TerminationChecker、feature_builder 等）直接使用
        self._state = {
            "position_neu": np.array([n, e, u], dtype=np.float64),
            "position_m": np.array([n, e, u], dtype=np.float64),          # 兼容 simple 后端字段
            "position_lla": np.array([lon, lat, alt_m], dtype=np.float64),
            "altitude_m": float(alt_m),
            "attitude_rpy": np.array([roll, pitch, yaw], dtype=np.float64),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "yaw_rad": float(yaw),
            "velocity_ned": np.array([vn, ve, vd], dtype=np.float64),
            "velocity_vector_mps": np.array([vn, ve, -vd], dtype=np.float64),  # NED→NEU 兼容
            "velocity_body": np.array([u_b, v_b, w_b], dtype=np.float64),
            "body_rates_rps": np.array([p_rps, q_rps, r_rps], dtype=np.float64),
            "p_rps": float(p_rps),
            "q_rps": float(q_rps),
            "r_rps": float(r_rps),
            "nz_g": float(nz_g),
            "speed_mps": float(vt),
            "vt_mps": float(vt),
            "sim_time": self.jsbsim_exec.get_sim_time(),
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the current aircraft state dictionary.

        Returns:
            dict: Keys include position_neu, position_lla, attitude_rpy,
                  velocity_ned, velocity_body, vt_mps, sim_time.
        """
        return self._state.copy()


# ---------------------------------------------------------------------------
# Public multi-aircraft environment
# ---------------------------------------------------------------------------

class JSBSimEnv:
    """
    Thin wrapper around the legacy JSBSim-based flight environment.

    Responsibilities:
    - initialize JSBSim simulation for one or more aircraft
    - reset aircraft state
    - apply low-level actuator or command inputs
    - step simulation at sim_freq
    - expose aircraft state
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Environment configuration dictionary.
                Expected keys: sim_freq, jsbsim_data_dir (optional), origin (optional).
        """
        self.config = config
        self.sim_freq = config.get("sim_freq", 60)
        self.dt = 1.0 / self.sim_freq
        # Resolve JSBSim data directory (self-contained project preferred)
        self.jsbsim_data_dir = _resolve_jsbsim_data_dir(config)
        self.origin = config.get("origin", (120.0, 60.0, 0.0))
        self._aircraft: Dict[str, _JSBSimAircraft] = {}

        # Validate data directory early so tracking_env can fallback gracefully
        jsbsim_data_dir = self.jsbsim_data_dir
        if not jsbsim_data_dir or not os.path.isdir(jsbsim_data_dir):
            raise RuntimeError(
                f"JSBSim data directory not found: {jsbsim_data_dir}. "
                f"Please ensure <project_root>/data/jsbsim exists and contains "
                f"aircraft/, engine/, systems/ subdirectories."
            )

    # ------------------------------------------------------------------
    # Aircraft registry
    # ------------------------------------------------------------------

    def add_aircraft(self, uid: str, aircraft_config: Optional[dict] = None):
        """
        Register a new aircraft to be managed by this environment.

        Args:
            uid (str): Unique identifier for the aircraft.
            aircraft_config (dict, optional): Aircraft-specific config
                (e.g. model, sim_freq). Merged with env config.
        """
        cfg = {**self.config, **(aircraft_config or {})}
        self._aircraft[uid] = _JSBSimAircraft(uid, cfg, self.origin)

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    def reset(self, aircraft_states: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, dict]:
        """
        Reset all managed aircraft to the given initial states.

        Args:
            aircraft_states (dict): Mapping from uid to init_state dict.
                e.g. {"own": {"ic/h-sl-ft": 20000, "ic/psi-true-deg": 0}}.
                If an aircraft is missing, an empty init_state is used.

        Returns:
            dict: Current state for all aircraft.
        """
        aircraft_states = aircraft_states or {}
        for uid, ac in self._aircraft.items():
            init_state = aircraft_states.get(uid, {})
            ac.reload(init_state)
        return self.get_state()

    def step(self, control_inputs: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, dict]:
        """
        Execute one simulation step for all aircraft.

        Args:
            control_inputs (dict, optional): Mapping from uid to a dict of
                JSBSim property names and values to set before stepping.
                e.g. {"own": {"fcs/throttle-cmd-norm": 0.8}}.

        Returns:
            dict: Updated state for all aircraft.
        """
        # Apply control inputs
        if control_inputs:
            for uid, inputs in control_inputs.items():
                ac = self._aircraft.get(uid)
                if ac is None:
                    continue
                for prop_name, value in inputs.items():
                    ac.set_property_value(prop_name, float(value))

        # Run all aircraft
        for ac in self._aircraft.values():
            ac.step()

        return self.get_state()

    def get_state(self) -> Dict[str, dict]:
        """
        Get the current state of all managed aircraft.

        Returns:
            dict: Mapping from uid to state dictionary.
        """
        return {uid: ac.get_state() for uid, ac in self._aircraft.items()}

    def close(self):
        """Clean up all aircraft simulations."""
        for ac in self._aircraft.values():
            ac.close()
        self._aircraft.clear()
