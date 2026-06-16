"""Rule-based tactical maneuver selector for the bandit (red) aircraft."""
from __future__ import annotations

from collections import deque
from typing import Dict, Any, Optional

import numpy as np

from ..maneuver_library import ManeuverLibrary
from ..maneuver_library.base import Maneuver
from .situation_evaluator import SituationEvaluator


# Safe default parameters for each maneuver when used by the bandit.
# These are chosen to keep the F-16 inside a reasonable flight envelope.
DEFAULT_MANEUVER_PARAMS: Dict[str, Dict[str, Any]] = {
    "straight_level": {"duration_s": 10.0},
    "coordinated_turn": {"heading_change_deg": 90.0, "bank_angle_deg": 45.0, "velocity_ref_mps": 220.0},
    "dive": {"gamma_deg": 20.0, "target_altitude_m": 3000.0, "velocity_ref_mps": 300.0},
    "loop": {"entry_speed_mps": 350.0, "nz_target": 3.0, "throttle": 1.0},
    "barrel_roll": {"roll_rate_dps": 60.0, "nz_target": 1.2},
    "high_yoyo": {"bank_angle_deg": 45.0, "pitch_up_deg": 25.0, "min_altitude_m": 3000.0},
    "low_yoyo": {"bank_angle_deg": 45.0, "dive_angle_deg": 20.0, "min_altitude_m": 2000.0},
    "scissors": {"cycles": 2, "bank_angle_deg": 45.0, "turn_angle_deg": 60.0},
    "split_s": {"entry_speed_mps": 350.0, "nz_pull": 5.0, "min_altitude_m": 500.0},
    "immelmann": {"entry_speed_mps": 350.0, "nz_pull": 2.0, "min_altitude_m": 2000.0},
    # --- dogfight-specific maneuvers (P1 loaded by default) ---
    "break_turn": {
        "bank_angle_deg": 70.0,
        "heading_change_deg": 135.0,
        "velocity_ref_mps": 240.0,
        "throttle": 1.0,
        "min_altitude_m": 1500.0,
        "min_speed_mps": 150.0,
    },
    "displacement_roll": {
        "roll_target_deg": 135.0,
        "roll_rate_dps": 90.0,
        "hold_time_s": 1.5,
        "nz_ref": 1.3,
        "throttle": 0.85,
        "min_altitude_m": 1500.0,
        "min_speed_mps": 150.0,
    },
    # --- additional dogfight maneuvers (registered but not loaded by default) ---
    "vertical_scissors": {
        "bank_angle_deg": 60.0,
        "turn_angle_deg": 90.0,
        "climb_angle_deg": 20.0,
        "dive_angle_deg": 20.0,
        "cycles": 3,
        "throttle": 0.7,
        "min_altitude_m": 1500.0,
        "min_speed_mps": 150.0,
    },
    "defensive_spiral": {
        "bank_angle_deg": 65.0,
        "dive_angle_deg": 15.0,
        "throttle": 0.8,
        "min_altitude_m": 1000.0,
        "min_speed_mps": 160.0,
        "max_duration_s": 8.0,
    },
    "extension": {
        "dive_angle_deg": 5.0,
        "throttle": 1.0,
        "velocity_ref_mps": 350.0,
        "max_duration_s": 6.0,
        "min_altitude_m": 1500.0,
        "min_speed_mps": 160.0,
    },
    "jink": {
        "bank_angle_deg": 45.0,
        "roll_rate_dps": 120.0,
        "cycles": 3,
        "nz_ref": 1.3,
        "throttle": 0.9,
        "min_altitude_m": 1500.0,
        "min_speed_mps": 150.0,
    },
}


_DIFFICULTY_ALLOWED = {
    "easy": ["straight_level", "coordinated_turn", "barrel_roll"],
    "medium": [
        "straight_level",
        "coordinated_turn",
        "barrel_roll",
        "dive",
        "high_yoyo",
        "low_yoyo",
        "scissors",
    ],
    "hard": list(DEFAULT_MANEUVER_PARAMS.keys()),
}


# Keys that control safety envelopes and should NOT be randomized.
_SAFETY_KEYS = {
    "min_altitude_m",
    "min_speed_mps",
    "max_alpha_rad",
    "max_alpha_deg",
}

# Sensible clamp ranges for randomized maneuver parameters.
_PERTURB_BOUNDS: Dict[str, tuple] = {
    "bank_angle_deg": (5.0, 90.0),
    "roll_target_deg": (5.0, 180.0),
    "heading_change_deg": (30.0, 180.0),
    "turn_angle_deg": (30.0, 180.0),
    "pitch_up_deg": (5.0, 60.0),
    "dive_angle_deg": (5.0, 45.0),
    "climb_angle_deg": (5.0, 45.0),
    "gamma_deg": (5.0, 45.0),
    "roll_rate_dps": (10.0, 180.0),
    "nz_ref": (0.8, 7.0),
    "nz_target": (0.8, 7.0),
    "nz_pull": (0.8, 7.0),
    "velocity_ref_mps": (120.0, 500.0),
    "entry_speed_mps": (200.0, 500.0),
    "target_speed_mps": (120.0, 500.0),
    "max_speed_mps": (150.0, 600.0),
    "throttle": (0.2, 1.0),
    "duration_s": (0.5, 30.0),
    "hold_time_s": (0.2, 5.0),
    "max_duration_s": (1.0, 20.0),
    "cycles": (1, 6),
}

# Broad tactical class used by the anti-repetition heuristic.
_CLASS_MAP: Dict[str, str] = {
    "straight_level": "neutral",
    "coordinated_turn": "neutral",
    "dive": "escape",
    "loop": "escape",
    "barrel_roll": "defensive",
    "high_yoyo": "energy",
    "low_yoyo": "offensive",
    "scissors": "defensive",
    "split_s": "offensive",
    "immelmann": "offensive",
    "break_turn": "defensive",
    "displacement_roll": "defensive",
    "vertical_scissors": "defensive",
    "defensive_spiral": "defensive",
    "extension": "escape",
    "jink": "defensive",
}


class BanditManeuverSelector:
    """Select maneuvers for the red aircraft based on relative situation.

    The selector is intentionally simple and interpretable: it maps situation
    features to maneuver names via a small set of IF-THEN rules.  Difficulty
    controls the allowed maneuver set, reaction time, and noise.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.difficulty = str(self.config.get("difficulty", "medium")).lower()
        self.reaction_time_s = float(self.config.get("reaction_time_s", 3.0))
        self.selector_noise = float(self.config.get("selector_noise", 0.0))
        self.min_altitude_m = float(self.config.get("min_altitude_m", 1500.0))
        self.min_speed_mps = float(self.config.get("min_speed_mps", 120.0))

        allowed = self.config.get("allowed_maneuvers")
        if allowed is None:
            allowed = _DIFFICULTY_ALLOWED.get(self.difficulty, _DIFFICULTY_ALLOWED["medium"])
        self.allowed_maneuvers = list(allowed) if allowed is not None else ["straight_level"]
        if not self.allowed_maneuvers:
            self.allowed_maneuvers = ["straight_level"]

        self.evaluator = SituationEvaluator(self.config.get("situation", {}))
        self.rng = np.random.default_rng(self.config.get("seed", None))

        # Anti-repetition: keep the last N selected maneuver names.
        self.history_window = int(self.config.get("history_window", 3))
        self.history_switch_prob = float(self.config.get("history_switch_prob", 0.5))
        self._history: deque[str] = deque(maxlen=self.history_window)

        # Parameter perturbation: vary maneuver numeric parameters per execution
        # to make the expert system less exploitable.
        self.param_perturb_scale = float(self.config.get("param_perturb_scale", 0.1))

        # Emergency interruption thresholds.
        self.emergency_range_m = float(self.config.get("emergency_range_m", 1000.0))
        self.emergency_closure_mps = float(self.config.get("emergency_closure_mps", 80.0))
        self.emergency_maneuvers = list(
            self.config.get(
                "emergency_maneuvers",
                [
                    "break_turn",
                    "displacement_roll",
                    "vertical_scissors",
                    "barrel_roll",
                    "scissors",
                    "coordinated_turn",
                ],
            )
        )

        self._last_select_t = -1e9
        self._current_maneuver_name: Optional[str] = None

        # Temporal stability: minimum time a maneuver must run before it can be
        # replaced by a non-emergency switch.
        self.min_dwell_time_s = float(self.config.get("min_dwell_time_s", 2.0))
        # Cooldown: after leaving a maneuver, wait this long before selecting it
        # again. Reduces exploitable repetition and manic switching.
        self.maneuver_cooldown_s = float(self.config.get("maneuver_cooldown_s", 5.0))
        self._maneuver_cooldown_expiry: Dict[str, float] = {}
        self._current_maneuver_start_t: float = -1e9

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_initial(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        sim_time: float = 0.0,
    ) -> Optional[Maneuver]:
        """Pick the first maneuver after reset."""
        situation = self.evaluator.evaluate(own_state, bandit_state)
        name = self._rule_select(situation, bandit_state)
        name = self._apply_temporal_constraints(name, sim_time, force=True)
        name = self._select_with_history(name)
        self._last_select_t = sim_time
        self._current_maneuver_start_t = sim_time
        self._current_maneuver_name = name
        self._record_history(name)
        return self._create_maneuver(name)

    def select(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        current_maneuver_name: Optional[str],
        elapsed_in_maneuver: float,
        sim_time: float,
    ) -> Optional[Maneuver]:
        """Decide whether to switch maneuvers.

        Returns a new Maneuver instance if a switch is requested, otherwise None.
        """
        self._current_maneuver_name = current_maneuver_name

        # Respect reaction time and minimum dwell time to avoid manic switching.
        if (
            sim_time - self._last_select_t < self.reaction_time_s
            and elapsed_in_maneuver < 5.0
        ):
            return None
        if elapsed_in_maneuver < self.min_dwell_time_s:
            return None

        situation = self.evaluator.evaluate(own_state, bandit_state)
        desired = self._rule_select(situation, bandit_state)

        # Stochastic noise: with probability selector_noise pick a random
        # allowed maneuver instead of the optimal one.
        if self.rng.random() < self.selector_noise:
            desired = self.rng.choice(self.allowed_maneuvers)

        desired = self._apply_temporal_constraints(desired, sim_time, force=False)
        desired = self._select_with_history(desired)

        if desired == current_maneuver_name:
            return None

        self._mark_cooldown(current_maneuver_name, sim_time)
        self._last_select_t = sim_time
        self._current_maneuver_start_t = sim_time
        self._current_maneuver_name = desired
        self._record_history(desired)
        return self._create_maneuver(desired)

    def select_emergency(
        self,
        own_state: Dict[str, Any],
        bandit_state: Dict[str, Any],
        sim_time: float = 0.0,
    ) -> Optional[Maneuver]:
        """Bypass reaction time and pick an emergency defensive maneuver.

        Called by the controller when the opponent is inside the emergency
        envelope (e.g., very close, high closure rate, on the tail).
        """
        situation = self.evaluator.evaluate(own_state, bandit_state)
        name = self._emergency_maneuver(situation)
        name = self._apply_temporal_constraints(name, sim_time, force=True)
        name = self._select_with_history(name)
        self._mark_cooldown(self._current_maneuver_name, sim_time)
        self._last_select_t = sim_time
        self._current_maneuver_start_t = sim_time
        self._current_maneuver_name = name
        self._record_history(name)
        return self._create_maneuver(name)

    def is_emergency(self, situation: Dict[str, Any]) -> bool:
        """Return True if the current situation warrants an emergency maneuver."""
        range_m = float(situation.get("range_m", float("inf")))
        closure_rate = float(situation.get("closure_rate_mps", 0.0))
        own_tail = bool(situation.get("own_on_bandit_tail", False))
        return (
            own_tail
            and range_m < self.emergency_range_m
            and closure_rate > self.emergency_closure_mps
        )

    # ------------------------------------------------------------------
    # Internal rules
    # ------------------------------------------------------------------

    def _rule_select(self, situation: Dict[str, Any], bandit_state: Dict[str, Any]) -> str:
        alt = float(bandit_state.get("altitude_m", 0.0))
        speed = float(bandit_state.get("speed_mps", 0.0))

        # Safety envelopes override everything.
        if alt < self.min_altitude_m + 500.0 or speed < self.min_speed_mps + 20.0:
            return "straight_level"

        zone = situation["weapon_zone"]
        own_tail = situation["own_on_bandit_tail"]
        bandit_tail = situation["bandit_on_own_tail"]
        energy_adv = situation["bandit_energy_advantage"]
        closure_rate = float(situation.get("closure_rate_mps", 0.0))

        # Defensive: bandit is in front of ownship (own on tail).
        if own_tail:
            if zone in ("ideal", "merge"):
                # Very close merge with high closure: vertical scissors to force
                # an overshoot by trading altitude for turn rate.
                if (
                    zone == "merge"
                    and closure_rate > 80.0
                    and self._allowed("vertical_scissors")
                ):
                    return "vertical_scissors"
                # Close-range hard defense: break away from the threat HUD.
                if self._allowed("break_turn"):
                    return "break_turn"
                if energy_adv and self._allowed("high_yoyo"):
                    return "high_yoyo"
                if self._allowed("displacement_roll"):
                    return "displacement_roll"
                # Last-ditch spiral defense when the opponent has an energy
                # advantage and we have altitude to spend.
                if (
                    not energy_adv
                    and alt >= self.min_altitude_m + 1500.0
                    and self._allowed("defensive_spiral")
                ):
                    return "defensive_spiral"
                if self._allowed("barrel_roll"):
                    return "barrel_roll"
                return "scissors" if self._allowed("scissors") else "coordinated_turn"
            if zone in ("max_range", "outside"):
                # If we have an energy advantage, extend rather than just dive.
                if energy_adv and self._allowed("extension"):
                    return "extension"
                if self._allowed("dive"):
                    return "dive"
                return "coordinated_turn"
            return "coordinated_turn"

        # Offensive: bandit is behind ownship.
        if bandit_tail:
            if zone in ("ideal", "merge"):
                if self._allowed("low_yoyo"):
                    return "low_yoyo"
                if self._allowed("split_s"):
                    return "split_s"
                return "coordinated_turn"
            if zone == "max_range" and self._allowed("dive"):
                return "dive"
            return "coordinated_turn"

        # Neutral / cold side: turn toward the opponent to generate a fight.
        if self._allowed("scissors") and zone in ("ideal", "merge"):
            return "scissors"
        return "coordinated_turn"

    def _allowed(self, name: str) -> bool:
        return name in self.allowed_maneuvers

    def _record_history(self, name: str):
        """Record a maneuver selection for anti-repetition."""
        self._history.append(name)

    def _mark_cooldown(self, name: Optional[str], sim_time: float):
        """Put a maneuver on cooldown after leaving it."""
        if name is None or self.maneuver_cooldown_s <= 0.0:
            return
        self._maneuver_cooldown_expiry[name] = sim_time + self.maneuver_cooldown_s

    def _is_in_cooldown(self, name: str, sim_time: float) -> bool:
        """Return True if ``name`` is currently on cooldown."""
        if self.maneuver_cooldown_s <= 0.0:
            return False
        expiry = self._maneuver_cooldown_expiry.get(name, -1e9)
        return sim_time < expiry

    def _apply_temporal_constraints(
        self, desired: str, sim_time: float, force: bool = False
    ) -> str:
        """Respect maneuver cooldowns. If ``force`` is True, ignore cooldown.

        Falls back to the same-tactical-class alternatives not on cooldown,
        then to any allowed maneuver not on cooldown, then finally to the
        fallback ``straight_level``.
        """
        if force or not self._is_in_cooldown(desired, sim_time):
            return desired

        desired_class = _CLASS_MAP.get(desired, "neutral")
        alternatives = [
            m
            for m in self.allowed_maneuvers
            if m != desired
            and not self._is_in_cooldown(m, sim_time)
            and _CLASS_MAP.get(m, "neutral") == desired_class
        ]
        if alternatives:
            return self.rng.choice(alternatives)

        # Broaden search to any allowed maneuver not on cooldown.
        alternatives = [
            m
            for m in self.allowed_maneuvers
            if m != desired and not self._is_in_cooldown(m, sim_time)
        ]
        if alternatives:
            return self.rng.choice(alternatives)

        # Last resort: straight and level is always safe and never on cooldown.
        return "straight_level"

    def _select_with_history(self, desired: str) -> str:
        """Return a maneuver name, avoiding exact repetition when possible.

        If ``desired`` is in the recent history and there is another allowed
        maneuver of the same tactical class not in the history, switch to it
        with probability ``history_switch_prob``.
        """
        if desired not in self._history:
            return desired
        if self.history_switch_prob <= 0.0:
            return desired
        if self.rng.random() >= self.history_switch_prob:
            return desired

        desired_class = _CLASS_MAP.get(desired, "neutral")
        alternatives = [
            m
            for m in self.allowed_maneuvers
            if m != desired
            and m not in self._history
            and _CLASS_MAP.get(m, "neutral") == desired_class
        ]
        if not alternatives:
            return desired
        return self.rng.choice(alternatives)

    def _emergency_maneuver(self, situation: Dict[str, Any]) -> str:
        """Pick the highest-priority allowed emergency maneuver."""
        if not self.is_emergency(situation):
            return "straight_level"
        for name in self.emergency_maneuvers:
            if self._allowed(name):
                return name
        return "straight_level"

    def _perturb_params(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of params with numeric values slightly perturbed."""
        if self.param_perturb_scale <= 0.0:
            return dict(params)
        perturbed = dict(params)
        for key, value in perturbed.items():
            if key in _SAFETY_KEYS:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scale = self.param_perturb_scale
            new_value = float(value) * float(self.rng.uniform(1.0 - scale, 1.0 + scale))
            lo, hi = _PERTURB_BOUNDS.get(key, (-float("inf"), float("inf")))
            perturbed[key] = float(np.clip(new_value, lo, hi))
        return perturbed

    def _create_maneuver(self, name: str) -> Maneuver:
        params = DEFAULT_MANEUVER_PARAMS.get(name, {})
        params = self._perturb_params(name, params)
        return ManeuverLibrary.create(name, params)
