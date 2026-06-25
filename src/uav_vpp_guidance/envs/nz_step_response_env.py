"""
Nz step response environment for measuring low-level flight-controller
step-response characteristics.

The target flies straight at constant speed.  The guidance/controller is
bypassed — nz_cmd is injected directly with a pre-configured step sequence:

- Phase 0 (0-2 s, 10 steps): nz_cmd = 1.0 g  (trim)
- Phase 1 (2-7 s, 25 steps): nz_cmd = 5.0 g  (step up)
- Phase 2 (7-12 s, 25 steps): nz_cmd = 1.0 g (step down)
- Phase 3 (12-17 s, 25 steps): nz_cmd = -1.0 g (negative g, key reversal)
- Phase 4 (17-20 s, 15 steps): nz_cmd = 1.0 g  (return to trim)

Total: 100 steps = 20 s at dt = 0.2.

Each step records nz_cmd and actual nz.  Per-phase recovery_delay (steps)
and overshoot_pct metrics are computed in info every step.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Tuple

import numpy as np

from .tracking_env import CloseRangeTrackingEnv


class NzStepResponseEnv(CloseRangeTrackingEnv):
    """
    Low-level step-response measurement task.

    This env directly injects nz_cmd into the controller bypassing the full
    policy/VPP/guidance pipeline.  The purpose is to compare step-response
    characteristics across different low-level controller implementations.
    """

    def __init__(self, config: dict, opponent_policy=None, opponent_config=None):
        super().__init__(
            config,
            opponent_policy=opponent_policy,
            opponent_config=opponent_config,
        )
        self.task_cfg = config.get("task", {}).get("nz_step_response", {})

        # Parse step sequence from config (task.nz_step_response.step_sequence).
        self._step_sequence_cfg: List[dict] = self.task_cfg.get(
            "step_sequence",
            [
                {"duration_s": 2.0, "nz_cmd": 1.0},
                {"duration_s": 5.0, "nz_cmd": 5.0},
                {"duration_s": 5.0, "nz_cmd": 1.0},
                {"duration_s": 5.0, "nz_cmd": -1.0},
                {"duration_s": 3.0, "nz_cmd": 1.0},
            ],
        )

        # Pre-compute per-step nz_cmd lookup.
        self._nz_cmd_lookup: List[float] = self._build_nz_cmd_lookup()

        # Per-step tracking.
        self._nz_cmd_hist: List[float] = []
        self._actual_nz_hist: List[float] = []
        self._nz_hist_valid: List[bool] = []
        self._phase_transitions: List[int] = []  # step indices where phases change

        # Phase metrics: computed at the end of each phase.
        self._phase_metrics: dict = {}
        self._current_phase: int = 0
        self._phase_step_counter: int = 0
        self._phase_peak_nz: float = 0.0
        self._phase_settled_steps: int = -1
        self._phase_overshoot_pct: float = 0.0
        self._phase_recovery_delay_steps: int = 0

    # ------------------------------------------------------------------
    # nz_cmd lookup
    # ------------------------------------------------------------------

    def _build_nz_cmd_lookup(self) -> List[float]:
        """Pre-compute per-step nz_cmd from the configured step sequence."""
        dt = float(self.env_config.get("high_level_dt", 0.2))
        lookup: List[float] = []
        self._phase_transitions = []
        for phase_idx, phase in enumerate(self._step_sequence_cfg):
            duration_s = float(phase["duration_s"])
            nz_cmd = float(phase["nz_cmd"])
            n_steps = max(1, int(round(duration_s / dt)))
            if phase_idx == 0:
                self._phase_transitions.append(0)
            else:
                self._phase_transitions.append(len(lookup))
            lookup.extend([nz_cmd] * n_steps)
        return lookup

    def _get_nz_cmd_for_current_step(self) -> float:
        """Return the pre-computed nz_cmd for the current step index."""
        step_idx = self.current_step
        if step_idx < len(self._nz_cmd_lookup):
            return self._nz_cmd_lookup[step_idx]
        # Past end of sequence: hold last value.
        return self._nz_cmd_lookup[-1] if self._nz_cmd_lookup else 1.0

    def _get_current_phase(self) -> int:
        """Return current phase index (0-based) based on step count."""
        step_idx = self.current_step
        phase = 0
        for i, start_step in enumerate(self._phase_transitions):
            if step_idx >= start_step:
                phase = i
        return phase

    # ------------------------------------------------------------------
    # Gym-like interface overrides
    # ------------------------------------------------------------------

    def reset(self, scenario=None, seed=None) -> dict:
        """Reset with a task-specific default scenario if none is provided."""
        if scenario is None:
            scenario = self._build_default_scenario(seed)
        return super().reset(scenario=scenario, seed=seed)

    # ------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------

    def _task_reset(self, seed: Optional[int] = None) -> None:
        """Initialize per-episode history and phase tracking."""
        self._nz_cmd_hist = []
        self._actual_nz_hist = []
        self._nz_hist_valid = []
        self._phase_metrics = {}
        self._current_phase = 0
        self._phase_step_counter = 0
        self._phase_peak_nz = 0.0
        self._phase_settled_steps = -1
        self._phase_overshoot_pct = 0.0
        self._phase_recovery_delay_steps = 0

    def _task_get_target_state(self, backend_target_state: dict, own_state: dict) -> dict:
        """Return a straight-flying target state ahead of own aircraft."""
        base_alt = float(self.task_cfg.get("base_altitude_m", 5000.0))
        target_speed = float(self.task_cfg.get("own_speed_mps", 250.0))
        # Target flies straight, 1500 m ahead.
        pos = np.array([1500.0, 0.0, base_alt], dtype=float)
        return {
            "position_m": pos.copy(),
            "position_neu": pos.copy(),
            "velocity_vector_mps": np.array([target_speed, 0.0, 0.0], dtype=float),
            "heading_deg": 0.0,
            "speed_mps": target_speed,
        }

    def _task_pre_step(self, own_state: dict, target_state: dict) -> Tuple[dict, dict]:
        """Store the current target state for reference; no modification needed."""
        return own_state, target_state

    def step(self, action=None):
        """Override step() to inject nz_cmd directly via command_override.

        The action parameter (normally VPP offsets from a policy) is ignored.
        """
        nz_cmd = self._get_nz_cmd_for_current_step()

        # Record nz_cmd before stepping.
        self._nz_cmd_hist.append(nz_cmd)

        command_override = {
            "nz_cmd": nz_cmd,
            "roll_rate_cmd": 0.0,
            "throttle_cmd": 0.7,
        }
        return super().step(action=np.zeros(3), command_override=command_override)

    def _task_post_step(
        self,
        own_state_post: dict,
        target_state_post: dict,
        info: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, bool, bool, dict]:
        """Record actual nz and compute per-phase step-response metrics."""
        # Extract actual nz from JSBSim state.
        actual_nz = float(
            own_state_post.get(
                "nz_g",
                own_state_post.get("nz", own_state_post.get("load_factor", np.nan)),
            )
        )
        self._actual_nz_hist.append(actual_nz)

        # Determine which phase we are in.
        phase = self._get_current_phase()
        if phase != self._current_phase:
            # Finalize metrics for the completed phase.
            self._finalize_phase_metrics(self._current_phase)
            # Start new phase.
            self._current_phase = phase
            self._phase_step_counter = 0
            self._phase_peak_nz = actual_nz
            self._phase_settled_steps = -1
            self._phase_overshoot_pct = 0.0
            self._phase_recovery_delay_steps = 0

        self._phase_step_counter += 1

        # Update peak for overshoot calculation.
        if self._current_phase >= 1:
            # Only meaningful for step-up/down phases.
            if abs(actual_nz) > abs(self._phase_peak_nz):
                self._phase_peak_nz = actual_nz

        # Compute per-phase metrics for this step.
        nz_cmd = self._nz_cmd_hist[-1] if self._nz_cmd_hist else 1.0

        # Overshoot: how much actual nz exceeds commanded nz in the step-up phase.
        overshoot_pct = 0.0
        if nz_cmd > 1.5 and actual_nz > nz_cmd:
            overshoot_pct = (actual_nz - nz_cmd) / max(abs(nz_cmd - 1.0), 0.01) * 100.0

        recovery_delay = 0
        # Recovery delay is computed per-phase at phase transitions.
        # For the current step, report the previous phase's final metrics.
        prev_phase_metrics = self._phase_metrics.get(
            f"phase_{self._current_phase - 1}", {}
        ) if self._current_phase > 0 else {}

        # Populate info with step-response diagnostics.
        info["nz_cmd"] = nz_cmd
        info["actual_nz"] = actual_nz
        info["nz_step_response_phase"] = self._current_phase
        info["nz_step_response_phase_step"] = self._phase_step_counter
        info["nz_step_response_overshoot_pct"] = overshoot_pct
        info["nz_step_response_recovery_delay"] = prev_phase_metrics.get(
            "recovery_delay_steps", 0
        )
        info["nz_step_response_peak_nz"] = self._phase_peak_nz

        # Disable default proximity success termination.
        if info.get("is_success") and not info.get("combat_outcome"):
            terminated = False
            truncated = False
            info["is_success"] = False
            info["reason"] = None

        # Stall protection.
        speed = float(
            own_state_post.get("speed_mps", 250.0)
            or np.linalg.norm(own_state_post.get("velocity_vector_mps", [0.0, 0.0, 0.0]))
        )
        min_safe = float(self.task_cfg.get("min_safe_speed_mps", 150.0))
        if speed < min_safe:
            terminated = True
            truncated = False
            info["termination_reason"] = "stall"
            info["reason"] = "stall"
            info["is_crash"] = True

        return reward, terminated, truncated, info

    def _finalize_phase_metrics(self, phase_idx: int) -> None:
        """Compute and store final metrics for a completed phase."""
        if phase_idx < 0:
            return

        nz_cmds = [self._nz_cmd_lookup[min(p, len(self._nz_cmd_lookup) - 1)]
                    for p in range(
                        self._phase_transitions[phase_idx]
                        if phase_idx < len(self._phase_transitions)
                        else len(self._nz_cmd_lookup),
                        min(
                            (self._phase_transitions[phase_idx + 1]
                             if phase_idx + 1 < len(self._phase_transitions)
                             else len(self._nz_cmd_lookup)),
                            len(self._nz_cmd_lookup),
                        ),
                    )]
        peak_nz = self._phase_peak_nz
        phase_nz_cmd = nz_cmds[0] if nz_cmds else 1.0

        # Compute overshoot percentage for step-up phases.
        overshoot_pct = 0.0
        if phase_nz_cmd > 1.5:
            overshoot_pct = max(0.0, (peak_nz - phase_nz_cmd) / max(abs(phase_nz_cmd - 1.0), 0.01) * 100.0)
        elif phase_nz_cmd < 0.5:
            # Negative-g phase: undershoot is when nz goes below commanded.
            overshoot_pct = max(0.0, (phase_nz_cmd - peak_nz) / max(abs(phase_nz_cmd - 1.0), 0.01) * 100.0)

        # Recovery delay: number of steps from phase start until nz settles
        # within 10% band around the commanded value.
        recovery_delay_steps = self._phase_step_counter  # default if never settles
        settle_band = 0.1 * max(abs(phase_nz_cmd - 1.0), 0.5)
        phase_actuals = self._actual_nz_hist[-self._phase_step_counter:] if self._phase_step_counter > 0 else []
        for i, actual in enumerate(phase_actuals):
            if abs(actual - phase_nz_cmd) <= settle_band:
                # Check sustained: next 3 steps also within band.
                sustained = True
                for j in range(1, min(4, len(phase_actuals) - i)):
                    if abs(phase_actuals[i + j] - phase_nz_cmd) > settle_band:
                        sustained = False
                        break
                if sustained:
                    recovery_delay_steps = i + 1
                    break

        self._phase_metrics[f"phase_{phase_idx}"] = {
            "phase_idx": phase_idx,
            "nz_cmd": phase_nz_cmd,
            "overshoot_pct": round(overshoot_pct, 4),
            "recovery_delay_steps": recovery_delay_steps,
            "peak_nz": round(peak_nz, 4),
            "steps": self._phase_step_counter,
        }

    # ------------------------------------------------------------------
    # Termination override: use absolute bounds, not range-to-target
    # ------------------------------------------------------------------

    def _check_done(self, own_state, target_state, rel_state):
        """Check only altitude, absolute XY bounds, and timeout."""
        term_info = {
            "reason": None,
            "is_success": False,
            "is_crash": False,
            "is_timeout": False,
            "is_out_of_bounds": False,
        }

        altitude_m = float(own_state.get("altitude_m", 5000.0))
        pos = np.asarray(own_state.get("position_m", own_state.get("position_neu", [0.0, 0.0, 5000.0])))
        xy_limit = float(self.env_config.get("xy_limit_m", 20000.0))
        min_alt = float(self.env_config.get("min_altitude_m", 500.0))
        max_alt = float(self.env_config.get("max_altitude_m", 15000.0))
        max_steps = int(self.env_config.get("max_high_level_steps", 100))

        if altitude_m < min_alt or altitude_m > max_alt:
            term_info["reason"] = "crash"
            term_info["is_crash"] = True
            return True, False, term_info

        if len(pos) >= 2 and (abs(pos[0]) > xy_limit or abs(pos[1]) > xy_limit):
            term_info["reason"] = "out_of_bounds"
            term_info["is_out_of_bounds"] = True
            return True, False, term_info

        if self.current_step >= max_steps:
            term_info["reason"] = "timeout"
            term_info["is_timeout"] = True
            return False, True, term_info

        return False, False, term_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_states(self, noisy: bool = False):
        """Override to inject the straight-flying reference target state."""
        own_state, _ = super()._get_current_states(noisy=noisy)
        target_state = self._task_get_target_state({}, own_state)
        return own_state, target_state

    def _build_default_scenario(self, seed):
        """Build a default straight-flying scenario."""
        base_alt = float(self.task_cfg.get("base_altitude_m", 5000.0))
        own_speed = float(self.task_cfg.get("own_speed_mps", 250.0))
        return {
            "name": "nz_step_response_default",
            "own_init": {
                "position_m": [0.0, 0.0, base_alt],
                "velocity_mps": own_speed,
                "heading_deg": 0.0,
            },
            "target_init": {
                "position_m": [1500.0, 0.0, base_alt],
                "velocity_mps": own_speed,
                "heading_deg": 0.0,
            },
        }

    def get_step_response_summary(self) -> dict:
        """Return a summary of step-response metrics across all phases."""
        # Finalize the last phase if not already done.
        if self._current_phase >= 0 and f"phase_{self._current_phase}" not in self._phase_metrics:
            self._finalize_phase_metrics(self._current_phase)

        return {
            "phase_metrics": dict(self._phase_metrics),
            "nz_cmd_hist": list(self._nz_cmd_hist),
            "actual_nz_hist": list(self._actual_nz_hist),
            "total_steps": len(self._nz_cmd_hist),
        }
