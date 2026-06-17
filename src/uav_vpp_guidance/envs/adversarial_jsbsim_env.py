"""
Adversarial pursuit-evasion environment with two RL-controllable aircraft.

Wraps JSBSimEnv to manage a pursuer (blue) and a target/evader (red)
simultaneously, with asymmetric control pipelines:

- Pursuer: VPP action → VirtualPointGenerator → LOSRateGuidance →
  clip → filter → ActuatorDynamics → LowLevelController → JSBSim
- Target:  RL action → map-to-physical → clip → filter →
  ActuatorDynamics → LowLevelController → JSBSim

Both aircraft step together inside JSBSim each decision step.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .jsbsim_env import JSBSimEnv, neu2lla
from .observation import (
    ObservationBuilder,
    compute_relative_geometry,
    build_observation,
)
from .reward import RewardCalculator
from .termination import TerminationChecker
from .geometry_scenarios import build_explicit_scenario
from .bandit_controller import BanditManeuverController, build_bandit_flight_state
from ..expert_system.expert_vpp_policy import ExpertVPPPolicy
from ..flight_control.command_filter import MultiChannelCommandFilter
from ..flight_control.command_limiter import clip_command
from ..flight_control.low_level_controller import LowLevelController
from ..flight_control.actuator_dynamics import ActuatorDynamics
from ..guidance.los_rate_guidance import LOSRateGuidance
from ..guidance.gain_config import GuidanceGains
from ..virtual_point.generator import VirtualPointGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default physical limits (shared with CloseRangeTrackingEnv)
# ---------------------------------------------------------------------------
DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
    "nz": (-2.0, 7.0),
    "roll_rate": (-1.5, 1.5),
    "throttle": (0.0, 1.0),
}


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key from an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_scenario_attr(scenario: Any, key: str) -> Any:
    """Get a named sub-object from a scenario."""
    if scenario is None:
        return None
    if isinstance(scenario, dict):
        return scenario.get(key)
    return getattr(scenario, key, None)


# ---------------------------------------------------------------------------
# Target reward calculator
# ---------------------------------------------------------------------------

class TargetRewardCalculator:
    """
    Reward calculator for the adversarial target (evader).

    The target is rewarded for:
    - Surviving (small per-step bonus)
    - Increasing distance from the pursuer (escaping bonus)
    - Maintaining energy (speed bonus)
    and penalized for:
    - Being captured (terminal)
    - Crashing (terminal)
    - Going out of bounds (terminal)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        reward_cfg = cfg.get("target_reward", cfg.get("reward", {}))
        self.w_survival: float = float(reward_cfg.get("w_survival", 0.01))
        self.w_escaping: float = float(reward_cfg.get("w_escaping", 0.1))
        self.w_energy: float = float(reward_cfg.get("w_energy", 0.05))
        self.w_range: float = float(reward_cfg.get("w_range", 0.5))
        self.terminal_captured: float = float(reward_cfg.get("terminal_captured", -200.0))
        self.terminal_crash: float = float(reward_cfg.get("terminal_crash", -300.0))
        self.terminal_oob: float = float(reward_cfg.get("terminal_oob", -200.0))
        self.terminal_timeout: float = float(reward_cfg.get("terminal_timeout", 0.0))
        self.ref_speed: float = 400.0
        self.ref_range_rate: float = 200.0
        self.ref_range: float = 3000.0

    def reset(self) -> None:
        """Reset internal state (stateless calculator)."""
        pass

    def compute(
        self,
        own_state: Dict[str, Any],
        pursuer_state: Dict[str, Any],
        rel_state: Dict[str, Any],
        termination_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute target (evader) reward.

        Args:
            own_state: Target (own) aircraft state.
            pursuer_state: Pursuer aircraft state.
            rel_state: Relative geometry dict from target perspective
                (compute_relative_geometry(target_state, pursuer_state)).
            termination_info: Dict with keys ``is_success``, ``is_crash``,
                ``is_out_of_bounds``, ``is_timeout``.

        Returns:
            Tuple of (total_reward, reward_terms_dict).
        """
        # Range components (from target perspective)
        range_m = float(rel_state.get("range_m", 3000.0))
        range_rate = float(rel_state.get("range_rate_mps", 0.0))
        # Note: from target perspective, positive range_rate means pursuer is
        # moving away (target is escaping), which is good for the target.

        # Target's own speed
        own_vel = own_state.get("velocity_vector_mps",
                                own_state.get("velocity_ned", np.zeros(3)))
        own_speed = float(np.linalg.norm(np.asarray(own_vel)))

        terms: Dict[str, float] = {}

        # Survival bonus (per step)
        terms["survival"] = self.w_survival

        # Range reward (farther is better for target)
        terms["range"] = self.w_range * min(range_m / self.ref_range, 1.0)

        # Escaping bonus (positive range_rate from target perspective = escaping)
        terms["escaping"] = self.w_escaping * max(0.0, min(range_rate / self.ref_range_rate, 1.0))

        # Energy maintenance (keep speed up for maneuverability)
        terms["energy"] = self.w_energy * min(own_speed / self.ref_speed, 1.0)

        total = sum(terms.values())

        # Terminal rewards
        if termination_info is not None:
            if termination_info.get("is_success"):
                # Pursuer captured the target → target loses
                terms["terminal"] = self.terminal_captured
            elif termination_info.get("is_crash"):
                terms["terminal"] = self.terminal_crash
            elif termination_info.get("is_out_of_bounds"):
                terms["terminal"] = self.terminal_oob
            elif termination_info.get("is_timeout"):
                # Target survived the full episode → small positive
                terms["terminal"] = self.terminal_timeout
            else:
                terms["terminal"] = 0.0
            total += terms.get("terminal", 0.0)

        return total, terms


# ---------------------------------------------------------------------------
# Adversarial JSBSim Environment
# ---------------------------------------------------------------------------

class AdversarialJSBSimEnv:
    """
    Two-agent JSBSim environment for adversarial pursuit-evasion.

    Manages two F-16 aircraft with asymmetric control pipelines:
    - **Pursuer**: full VPP → LOS guidance → LowLevelController pipeline
    - **Target**:  direct RL → LowLevelController pipeline

    Compatible with existing configuration conventions.

    Usage::

        env = AdversarialJSBSimEnv(config)
        p_obs, t_obs = env.reset()
        p_action, t_action = ...  # from RL agents
        (p_obs, t_obs), (p_rew, t_rew), terminated, truncated, info = env.step(p_action, t_action)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Full environment + guidance configuration. Expected keys:
                ``env``, ``guidance``, ``virtual_point``, ``limits``,
                ``reward``, ``target_reward``, ``actuator_dynamics``.
        """
        self.config = config
        self.env_config = config.get("env", {})
        self.sim_freq: int = int(self.env_config.get("sim_freq", 60))
        self.decision_freq: int = int(self.env_config.get("decision_freq", 5))
        self.max_steps: int = int(self.env_config.get("max_high_level_steps", 512))
        self._sim_steps_per_decision: int = max(1, self.sim_freq // self.decision_freq)
        self._high_level_dt: float = float(self.env_config.get("high_level_dt", 1.0 / self.decision_freq))

        # ---- JSBSim backend with two F-16 aircraft ----
        self.jsbsim_env = JSBSimEnv(self.env_config)
        self._pursuer_uid = "pursuer"
        self._target_uid = "target"
        self.jsbsim_env.add_aircraft(self._pursuer_uid, {"model": "f16"})
        self.jsbsim_env.add_aircraft(self._target_uid, {"model": "f16"})

        # ---- Pursuer guidance pipeline ----
        vp_config = config.get("virtual_point", config.get("guidance", {}).get("virtual_point", {}))
        self._vp_generator = VirtualPointGenerator(vp_config)
        guidance_config = config.get("guidance", {})
        self._pursuer_guidance = LOSRateGuidance(guidance_config)
        self._pursuer_gains = GuidanceGains(**guidance_config.get("gains", {}))

        # ---- Low-level controllers (one per aircraft) ----
        llc_config = guidance_config.get("gains", {})
        self._pursuer_llc = LowLevelController(llc_config)
        self._target_llc = LowLevelController(llc_config)

        # ---- Command filters (one per aircraft) ----
        filter_alpha = float(llc_config.get("alpha_filter", 0.3))
        self._pursuer_cmd_filter = MultiChannelCommandFilter(alpha=filter_alpha)
        self._target_cmd_filter = MultiChannelCommandFilter(alpha=filter_alpha)

        # ---- Actuator dynamics (one per aircraft) ----
        actuator_config = config.get("actuator_dynamics", {})
        self._pursuer_actuator = ActuatorDynamics(actuator_config, dt=self._high_level_dt)
        self._target_actuator = ActuatorDynamics(actuator_config, dt=self._high_level_dt)

        # ---- Action limits ----
        limits = config.get("limits", {})
        self._nz_min = float(limits.get("nz_min", DEFAULT_LIMITS["nz"][0]))
        self._nz_max = float(limits.get("nz_max", DEFAULT_LIMITS["nz"][1]))
        self._rr_min = float(limits.get("roll_rate_min", DEFAULT_LIMITS["roll_rate"][0]))
        self._rr_max = float(limits.get("roll_rate_max", DEFAULT_LIMITS["roll_rate"][1]))
        self._th_min = float(limits.get("throttle_min", DEFAULT_LIMITS["throttle"][0]))
        self._th_max = float(limits.get("throttle_max", DEFAULT_LIMITS["throttle"][1]))

        # ---- Observation builders ----
        self._pursuer_obs_builder = ObservationBuilder(config)
        self._target_obs_builder = ObservationBuilder(config)

        # ---- Reward calculators ----
        self._pursuer_reward = RewardCalculator(config)
        self._target_reward = TargetRewardCalculator(config)

        # ---- Termination checker (shared, symmetric criteria) ----
        self._termination_checker = TerminationChecker(self.env_config)

        # ---- Random number generator ----
        seed = self.config.get("experiment", {}).get("seed")
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # ---- Controller configuration ----
        self._pursuer_controller_type = self.env_config.get("pursuer_controller_type", "rl")
        self._target_controller_type = self.env_config.get("target_controller_type", "rl")
        self._pursuer_controller = None
        self._target_controller = None
        if self._pursuer_controller_type == "expert_vpp":
            self._pursuer_controller = ExpertVPPPolicy(self.config.get("expert_vpp", {}))
        if self._target_controller_type == "maneuver_library":
            self._target_controller = BanditManeuverController(self.env_config.get("bandit", {}))
        elif self._target_controller_type == "expert_vpp":
            raise NotImplementedError("target_controller_type='expert_vpp' is not supported yet")

        # ---- Episode state ----
        self.current_step: int = 0
        self._sim_time_s: float = 0.0
        self._episode_count: int = 0

        logger.info(
            "AdversarialJSBSimEnv initialized: sim_freq=%d decision_freq=%d max_steps=%d",
            self.sim_freq,
            self.decision_freq,
            self.max_steps,
        )

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario: Any = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset both aircraft and return initial observations.

        Args:
            scenario: Optional scenario object with ``own_init`` and
                ``target_init`` attributes (``own`` = pursuer, ``target`` = target).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (pursuer_obs, target_obs), each a dict with at least
            ``observation_vector`` (np.ndarray, shape (16,)).
        """
        self.current_step = 0
        self._sim_time_s = 0.0
        self._episode_count += 1

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset sub-components
        self._pursuer_reward.reset()
        self._target_reward.reset()
        self._termination_checker.reset()
        self._pursuer_obs_builder.reset()
        self._target_obs_builder.reset()
        self._pursuer_llc.reset()
        self._target_llc.reset()
        self._pursuer_cmd_filter.reset()
        self._target_cmd_filter.reset()
        self._pursuer_actuator.reset()
        self._target_actuator.reset()

        # Build default non-overlapping scenario when none is provided
        if scenario is None:
            scenario = build_explicit_scenario(
                scenario_type="tail_chase",
                initial_range_m=2000.0,
                ego_speed_mps=274.0,
                target_speed_mps=250.0,
                base_altitude_m=6096.0,
            )

        # Build JSBSim initial conditions from the scenario
        own_sc = _get_scenario_attr(scenario, "own_init")
        target_sc = _get_scenario_attr(scenario, "target_init")
        pursuer_init = self._scenario_to_jsbsim_init(own_sc)
        target_init = self._scenario_to_jsbsim_init(target_sc)

        # Also accept flat dict scenarios
        if isinstance(scenario, dict):
            if "pursuer_init" in scenario:
                pursuer_init = self._scenario_to_jsbsim_init(scenario["pursuer_init"])
            if "target_init" in scenario:
                target_init = self._scenario_to_jsbsim_init(scenario["target_init"])
            # Fallback to own_init/target_init naming
            if "pursuer_init" not in scenario and "own_init" in scenario:
                pursuer_init = self._scenario_to_jsbsim_init(scenario["own_init"])

        states = self.jsbsim_env.reset({
            self._pursuer_uid: pursuer_init,
            self._target_uid: target_init,
        })

        # Build initial observations
        pursuer_state = states[self._pursuer_uid]
        target_state = states[self._target_uid]

        # Initialize non-RL target controller with the starting geometry
        if self._target_controller is not None:
            alpha, beta, mach = self._get_aero_props(self._target_uid)
            fs = build_bandit_flight_state(target_state, alpha, beta, mach)
            self._target_controller.reset(
                own_state=pursuer_state,
                bandit_state=target_state,
                bandit_flight_state=fs,
                sim_time=self._sim_time_s,
            )

        pursuer_obs = self._build_pursuer_obs(pursuer_state, target_state)
        target_obs = self._build_target_obs(target_state, pursuer_state)

        return pursuer_obs, target_obs

    def step(
        self,
        pursuer_action: np.ndarray,
        target_action: np.ndarray,
    ) -> Tuple[
        Dict[str, Any],   # pursuer_obs
        Dict[str, Any],   # target_obs
        float,            # pursuer_reward
        float,            # target_reward
        bool,             # terminated
        bool,             # truncated
        Dict[str, Any],   # info
    ]:
        """
        Execute one decision step for both agents.

        Args:
            pursuer_action: Normalized VPP offset, shape (3,), values in [-1, 1].
            target_action: Normalized flight commands, shape (3,), values in [-1, 1]
                mapping to [nz, roll_rate, throttle].

        Returns:
            Tuple of:
            - pursuer_obs: Pursuer observation dict.
            - target_obs: Target observation dict.
            - pursuer_reward: Pursuer reward (float).
            - target_reward: Target reward (float).
            - terminated: Whether episode terminated.
            - truncated: Whether episode was truncated.
            - info: Dict with detailed metrics.
        """
        self.current_step += 1
        self._sim_time_s += self._high_level_dt

        # 1. Get pre-step states
        pursuer_state_pre, target_state_pre = self._get_current_states()

        # 1b. Expert pursuer override (replaces the incoming RL action)
        if self._pursuer_controller_type == "expert_vpp":
            rel_pre = compute_relative_geometry(pursuer_state_pre, target_state_pre)
            pursuer_action = self._pursuer_controller.get_action(
                pursuer_state_pre, target_state_pre, rel_pre
            )

        # 2. Compute pursuer's guidance command (VPP → LOS guidance)
        # The pursuer action is a 3D VPP offset
        vp = self._compute_virtual_point(pursuer_action, pursuer_state_pre, target_state_pre)
        raw_p_cmd = self._pursuer_guidance.compute_command(
            pursuer_state_pre, target_state_pre, vp, self._pursuer_gains
        )

        # 3. Pursuer command pipeline: clip → filter → actuator → LLC
        p_clipped = clip_command(raw_p_cmd, self.config.get("limits", {}))
        p_filtered = self._pursuer_cmd_filter.filter(p_clipped)
        p_actuated = self._pursuer_actuator.step(p_filtered)
        p_jsbsim = self._pursuer_llc.compute_actuator(p_actuated, pursuer_state_pre)
        p_props = {k: v for k, v in p_jsbsim.items() if k.startswith("fcs/")}

        # 4. Target command pipeline: RL or maneuver library
        t_info: Dict[str, Any] = {}
        if self._target_controller_type == "maneuver_library":
            alpha, beta, mach = self._get_aero_props(self._target_uid)
            fs = build_bandit_flight_state(target_state_pre, alpha, beta, mach)
            cmd, t_info = self._target_controller.update(
                own_state=pursuer_state_pre,
                bandit_state=target_state_pre,
                bandit_flight_state=fs,
                sim_time=self._sim_time_s,
                dt=self._high_level_dt,
            )
            t_props = {
                "fcs/elevator-cmd-norm": float(cmd.elevator),
                "fcs/aileron-cmd-norm": float(cmd.aileron),
                "fcs/rudder-cmd-norm": float(cmd.rudder),
                "fcs/throttle-cmd-norm": float(cmd.throttle),
            }
            t_filtered = None
        else:
            t_physical = self._map_target_action_to_physical(target_action)
            t_clipped = clip_command(t_physical, self.config.get("limits", {}))
            t_filtered = self._target_cmd_filter.filter(t_clipped)
            t_actuated = self._target_actuator.step(t_filtered)
            t_jsbsim = self._target_llc.compute_actuator(t_actuated, target_state_pre)
            t_props = {k: v for k, v in t_jsbsim.items() if k.startswith("fcs/")}

        # 5. Step JSBSim with both aircraft controls simultaneously
        control_inputs = {
            self._pursuer_uid: p_props,
            self._target_uid: t_props,
        }
        for _ in range(self._sim_steps_per_decision):
            self.jsbsim_env.step(control_inputs)

        # 6. Get post-step states
        pursuer_state_post, target_state_post = self._get_current_states()

        # Defensive guard: if JSBSim diverged and produced NaN/inf state,
        # treat the episode as a crash for both agents and return a safe
        # placeholder observation. This prevents NaN gradients from poisoning
        # the PPO update.
        if not self._state_is_finite(pursuer_state_post) or not self._state_is_finite(target_state_post):
            logger.warning(
                "Non-finite aircraft state detected at step %d (sim_time=%.2f); "
                "terminating episode as crash.",
                self.current_step, self._sim_time_s,
            )
            term_info = {
                "reason": "crash",
                "is_success": False,
                "is_crash": True,
                "is_timeout": False,
                "is_out_of_bounds": False,
                "success_hold_steps": 0,
            }
            p_reward = float(self._pursuer_reward.terminal_crash)
            t_reward = float(self._target_reward.terminal_crash)
            return (
                self._zero_observation(),
                self._zero_observation(),
                p_reward,
                t_reward,
                True,   # terminated
                False,  # truncated
                {"termination": term_info},
            )

        # 7. Compute relative geometry (from pursuer's perspective for termination)
        rel_state = compute_relative_geometry(pursuer_state_post, target_state_post)

        # 8. Check termination
        terminated, term_info = self._termination_checker.check(
            pursuer_state_post,
            target_state_post,
            rel_state,
            self.current_step,
        )
        truncated = bool(term_info.get("is_timeout", False))
        terminated = bool(
            term_info.get("is_success", False)
            or term_info.get("is_crash", False)
            or term_info.get("is_out_of_bounds", False)
        )

        # 9. Compute pursuer reward (using existing RewardCalculator interface)
        terminal_reward = 0.0
        if term_info.get("is_success"):
            terminal_reward = self._pursuer_reward.terminal_success
        elif term_info.get("is_crash"):
            terminal_reward = self._pursuer_reward.terminal_crash
        elif term_info.get("is_timeout") or term_info.get("is_out_of_bounds"):
            terminal_reward = self._pursuer_reward.terminal_failure

        p_info = {
            "own_state": pursuer_state_post,
            "target_state": target_state_post,
            "relative_state": rel_state,
            "command": p_filtered,
            "terminal_reward": terminal_reward,
        }
        p_reward, p_terms = self._pursuer_reward.compute(p_info)

        # Target reward uses swapped perspective
        t_rel = compute_relative_geometry(target_state_post, pursuer_state_post)
        t_reward, t_terms = self._target_reward.compute(
            target_state_post,
            pursuer_state_post,
            t_rel,
            term_info,
        )

        # 10. Build observations
        p_obs = self._build_pursuer_obs(pursuer_state_post, target_state_post)
        t_obs = self._build_target_obs(target_state_post, pursuer_state_post)

        # 11. Assemble info
        info: Dict[str, Any] = {
            "pursuer_state": pursuer_state_post,
            "target_state": target_state_post,
            "relative_state": rel_state,
            "termination": term_info,
            "pursuer_reward_terms": p_terms,
            "target_reward_terms": t_terms,
            "pursuer_command": p_filtered,
            "target_command": t_filtered,
            "step": self.current_step,
            "sim_time_s": self._sim_time_s,
            "pursuer_controller_type": self._pursuer_controller_type,
            "target_controller_type": self._target_controller_type,
        }
        if self._target_controller_type == "maneuver_library":
            info["target_maneuver"] = t_info.get("bandit_maneuver")

        return (p_obs, t_obs, p_reward, t_reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_states(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get current states of both aircraft."""
        states = self.jsbsim_env.get_state()
        return states[self._pursuer_uid], states[self._target_uid]

    def _get_aero_props(self, uid: str) -> Tuple[float, float, float]:
        """Return (alpha_rad, beta_rad, mach) for the requested aircraft."""
        ac = self.jsbsim_env._aircraft[uid]
        return (
            float(ac.get_property_value("aero/alpha-rad")),
            float(ac.get_property_value("aero/beta-rad")),
            float(ac.get_property_value("velocities/mach")),
        )

    @staticmethod
    def _state_is_finite(state: Dict[str, Any]) -> bool:
        """Return True if all numeric fields in an aircraft state are finite."""
        if state is None:
            return False
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    return False
            elif isinstance(value, (int, float)):
                if not np.isfinite(value):
                    return False
        return True

    def _zero_observation(self) -> Dict[str, Any]:
        """Return a safe placeholder observation used after a JSBSim divergence."""
        return {
            "observation_vector": np.zeros(16, dtype=np.float32),
            "relative_state": {},
            "own_state": {},
            "target_state": {},
        }

    def _compute_virtual_point(
        self,
        action: np.ndarray,
        own_state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Compute virtual point from VPP action."""
        # Get anchor point (target position)
        anchor_pos = target_state.get("position_neu", target_state.get("position_m"))
        if anchor_pos is None:
            anchor_pos = np.array([2000.0, 0.0, 5000.0])
        anchor = {"position_neu": np.asarray(anchor_pos, dtype=np.float64)}

        return self._vp_generator.action_to_virtual_point(
            action, own_state, anchor
        )

    def _map_target_action_to_physical(self, action: np.ndarray) -> Dict[str, float]:
        """
        Map normalized target action [-1,1]^3 to physical commands.

        Same mapping as CloseRangeTrackingEnv.step() lines 700-710.
        """
        nz = float(action[0]) * (self._nz_max - self._nz_min) / 2.0 + (self._nz_max + self._nz_min) / 2.0
        roll_rate = float(action[1]) * (self._rr_max - self._rr_min) / 2.0 + (self._rr_max + self._rr_min) / 2.0
        throttle = float(action[2]) * (self._th_max - self._th_min) / 2.0 + (self._th_max + self._th_min) / 2.0
        return {
            "nz_cmd": nz,
            "roll_rate_cmd": roll_rate,
            "throttle_cmd": throttle,
        }

    def _build_pursuer_obs(
        self,
        pursuer_state: Dict[str, Any],
        target_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build pursuer observation (pursuer is own, target is enemy)."""
        obs_vec = self._pursuer_obs_builder.build(
            pursuer_state, target_state, None, None, None
        )
        rel = compute_relative_geometry(pursuer_state, target_state)
        return {
            "observation_vector": obs_vec,
            "relative_state": rel,
            "own_state": pursuer_state,
            "target_state": target_state,
        }

    def _build_target_obs(
        self,
        target_state: Dict[str, Any],
        pursuer_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build target observation (target is own, pursuer is enemy)."""
        obs_vec = self._target_obs_builder.build(
            target_state, pursuer_state, None, None, None
        )
        rel = compute_relative_geometry(target_state, pursuer_state)
        return {
            "observation_vector": obs_vec,
            "relative_state": rel,
            "own_state": target_state,
            "target_state": pursuer_state,
        }

    # ------------------------------------------------------------------
    # Scenario → JSBSim initial conditions
    # ------------------------------------------------------------------

    @staticmethod
    def _default_jsbsim_init(heading_deg: float = 0.0) -> Dict[str, float]:
        """Default JSBSim initial conditions for an F-16 at ~6096m altitude."""
        return {
            "ic/long-gc-deg": 120.0,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": float(heading_deg),
            "ic/u-fps": 900.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }

    def _scenario_to_jsbsim_init(self, aircraft_init: Any) -> Dict[str, float]:
        """
        Convert scenario init to JSBSim init_state dict.

        Converts NEU position_m to JSBSim geodetic initial conditions
        relative to the environment origin.
        """
        pos = _get_attr(aircraft_init, "position_m", np.array([0.0, 0.0, 5000.0]))
        vel_mps = _get_attr(aircraft_init, "velocity_mps", 243.8)  # ~800 fps
        heading_deg = _get_attr(aircraft_init, "heading_deg", 0.0)
        pitch_deg = _get_attr(aircraft_init, "pitch_deg", 0.0)
        roll_deg = _get_attr(aircraft_init, "roll_deg", 0.0)

        h_sl_ft = (
            float(pos[2]) / 0.3048
            if hasattr(pos, "__len__") and len(pos) > 2
            else 20000.0
        )
        psi_deg = float(heading_deg)
        u_fps = float(vel_mps) / 0.3048

        result: Dict[str, float] = {
            "ic/h-sl-ft": h_sl_ft,
            "ic/psi-true-deg": psi_deg,
            "ic/u-fps": u_fps,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/theta-deg": float(pitch_deg),
            "ic/phi-deg": float(roll_deg),
        }

        # Convert NEU horizontal position to geodetic
        origin = getattr(self.jsbsim_env, "origin", (120.0, 60.0, 0.0))
        lon0, lat0, alt0 = origin
        try:
            lon_deg, lat_deg, _alt_m = neu2lla(
                float(pos[0]), float(pos[1]), float(pos[2]), lon0, lat0, alt0
            )
            result["ic/long-gc-deg"] = float(lon_deg)
            result["ic/lat-geod-deg"] = float(lat_deg)
        except Exception as exc:
            # Keep default origin if coordinate conversion fails
            logger.warning(
                "Coordinate conversion failed for position %s: %s", pos, exc
            )
            result["ic/long-gc-deg"] = 120.0
            result["ic/lat-geod-deg"] = 60.0

        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release JSBSim resources."""
        if self.jsbsim_env is not None:
            self.jsbsim_env.close()
