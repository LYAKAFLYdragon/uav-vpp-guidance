"""
ExpertVPPPolicy

Main expert policy that outputs normalized Δp in [-1, 1].

Pipeline:
  own_state + target_state + rel_state
    -> SituationEvaluator
    -> RuleEngine
    -> ExpertVPPActionGenerator
    -> normalized action [dx_norm, dy_norm, dz_norm]
"""

import numpy as np

from .situation_evaluator import SituationEvaluator
from .rule_engine import RuleEngine, ManeuverIntent


class ExpertVPPPolicy:
    """
    Rule-driven virtual-pursuit-point policy.

    Outputs normalized 3-D action compatible with VirtualPointGenerator.
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): Expert VPP configuration.
        """
        self.config = config or {}
        self.situation_evaluator = SituationEvaluator(self.config.get("situation", {}))
        self.rule_engine = RuleEngine(self.config.get("situation", {}))

        # Action mapping for each intent (target-velocity-frame normalized offsets)
        default_actions = {
            ManeuverIntent.PURE_PURSUIT: np.array([0.0, 0.0, 0.0], dtype=np.float64),
            ManeuverIntent.LEAD_PURSUIT: np.array([0.5, 0.0, 0.0], dtype=np.float64),
            ManeuverIntent.LAG_PURSUIT: np.array([-0.5, 0.0, 0.0], dtype=np.float64),
            ManeuverIntent.HIGH_ANGLE_CUT: np.array([0.3, 0.5, 0.0], dtype=np.float64),
            ManeuverIntent.EXTEND: np.array([-0.8, 0.3, 0.0], dtype=np.float64),
            ManeuverIntent.ALTITUDE_RECOVER: np.array([0.0, 0.0, 0.8], dtype=np.float64),
        }
        action_cfg = self.config.get("action", {})
        self.action_map = {}
        for intent, default in default_actions.items():
            arr = np.array(action_cfg.get(intent.lower(), default), dtype=np.float64)
            self.action_map[intent] = arr

        # History for diagnostics
        self._history = []

    def get_action(self, own_state, target_state, rel_state):
        """
        Compute expert action.

        Args:
            own_state (dict): Own aircraft state.
            target_state (dict): Target aircraft state.
            rel_state (dict): Relative geometry from compute_relative_geometry.

        Returns:
            np.ndarray: Normalized action shape=(3,) in [-1, 1].
        """
        # 1. Evaluate situation
        situation = self.situation_evaluator.evaluate(own_state, target_state, rel_state)

        # 2. Select intent
        intent_result = self.rule_engine.select_intent(situation, rel_state)
        intent = intent_result["intent"]
        rule_id = intent_result["rule_id"]
        fallback_reason = intent_result["fallback_reason"]

        # 3. Generate action in target-velocity frame, then convert to world NEU
        action_tv = self.action_map.get(intent, np.zeros(3, dtype=np.float64)).copy()

        # Resolve lateral sign for HIGH_ANGLE_CUT based on relative geometry
        if intent == ManeuverIntent.HIGH_ANGLE_CUT:
            action_tv[1] = self._resolve_lateral_sign(own_state, target_state, rel_state)

        # Convert target-velocity-frame offset to world NEU frame
        action_world = self._tv_frame_to_world(action_tv, target_state)

        # Clip to [-1, 1]
        action_world = np.clip(action_world, -1.0, 1.0)

        # Assemble diagnostics
        diagnostics = {
            "expert_enabled": True,
            "expert_tactical_state": situation["tactical_state"],
            "expert_maneuver_intent": intent,
            "expert_situation_score": float(situation["situation_score"]),
            "expert_angle_score": float(situation["scores"]["angle"]),
            "expert_range_score": float(situation["scores"]["range"]),
            "expert_energy_score": float(situation["scores"]["energy"]),
            "expert_altitude_score": float(situation["scores"]["altitude"]),
            "expert_closure_score": float(situation["scores"]["closure"]),
            "expert_action_x": float(action_world[0]),
            "expert_action_y": float(action_world[1]),
            "expert_action_z": float(action_world[2]),
            "expert_rule_id": rule_id,
            "expert_fallback_reason": fallback_reason,
        }
        self._history.append(diagnostics)

        return action_world

    def get_last_diagnostics(self):
        """Return diagnostics from the most recent get_action call."""
        if not self._history:
            return {}
        return self._history[-1]

    def reset_history(self):
        """Clear diagnostic history."""
        self._history.clear()

    @staticmethod
    def _resolve_lateral_sign(own_state, target_state, rel_state):
        """
        Decide left/right cut direction for HIGH_ANGLE_CUT.
        Choose the side that reduces aspect angle faster.
        """
        own_pos = _get_position(own_state)
        target_pos = _get_position(target_state)
        own_vel = _get_velocity(own_state)
        target_vel = _get_velocity(target_state)

        # Horizontal relative position
        rel_pos_h = target_pos - own_pos
        rel_pos_h[2] = 0.0

        # Target heading in horizontal plane
        target_heading = np.arctan2(target_vel[1], target_vel[0]) if np.linalg.norm(target_vel[:2]) > 1e-6 else 0.0

        # Own heading in horizontal plane
        own_heading = np.arctan2(own_vel[1], own_vel[0]) if np.linalg.norm(own_vel[:2]) > 1e-6 else 0.0

        # Relative bearing from target to own (horizontal)
        bearing = np.arctan2(-rel_pos_h[1], -rel_pos_h[0]) - target_heading
        bearing = _normalize_angle(bearing)

        # If own is on target's right side, cut to target's left (positive y in TV frame)
        # We want to move toward the side that closes the angle faster.
        # Simple heuristic: turn toward the side where the target is moving away from.
        sign = 1.0 if bearing >= 0 else -1.0
        return sign * 0.5

    @staticmethod
    def _tv_frame_to_world(action_tv, target_state):
        """
        Convert action from target-velocity frame to world NEU frame.

        Target-velocity frame:
          x = target velocity direction (horizontal)
          y = horizontal right of target velocity
          z = up
        """
        target_vel = _get_velocity(target_state)
        speed_h = float(np.linalg.norm(target_vel[:2]))
        if speed_h < 1e-6:
            # Target stationary or near-vertical: fall back to world frame
            return action_tv.copy()

        heading = np.arctan2(target_vel[1], target_vel[0])
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Rotation about z-axis only (horizontal plane)
        # [x_world]   [cos  -sin  0] [x_tv]
        # [y_world] = [sin   cos  0] [y_tv]
        # [z_world]   [ 0     0   1] [z_tv]
        x_w = action_tv[0] * cos_h - action_tv[1] * sin_h
        y_w = action_tv[0] * sin_h + action_tv[1] * cos_h
        z_w = action_tv[2]

        return np.array([x_w, y_w, z_w], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_position(state):
    pos = state.get("position_neu")
    if pos is None:
        pos = state.get("position_m")
    if pos is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(pos, dtype=np.float64)


def _get_velocity(state):
    vel = state.get("velocity_ned")
    if vel is None:
        vel = state.get("velocity_vector_mps")
    if vel is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vel, dtype=np.float64)


def _normalize_angle(angle):
    angle = angle % (2.0 * np.pi)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return angle
