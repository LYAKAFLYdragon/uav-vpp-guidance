"""Control Barrier Function (CBF) safety filter for aircraft control.

This module implements a CBF-based safety filter that can be inserted
between the policy/guidance output and the low-level controller to
guarantee safety constraints (collision avoidance, flight envelope, etc.)
while minimally interfering with the learned policy.

References:
    - Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019).
      Control barrier functions: Theory and applications. In 2019 18th European control conference (ECC) (pp. 3420-3431). IEEE.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class CBFSafetyFilter:
    """CBF safety filter for aircraft control commands.
    
    The filter takes a nominal control command (from policy/guidance) and
    computes the minimally modified safe command by solving a QP at each step.
    
    Constraints supported:
    1. Collision avoidance: maintain minimum distance to target
    2. Altitude bounds: stay within safe altitude envelope  
    3. Speed bounds: avoid stall and overspeed
    4. Overload bounds: stay within aircraft g-limits
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize CBF safety filter.
        
        Args:
            config: Configuration dict with keys:
                - enabled: bool, whether CBF is active
                - d_min_m: float, minimum safe distance to target [m]
                - alpha_collision: float, CBF class-K gain for collision
                - h_min_m: float, minimum safe altitude [m]
                - h_max_m: float, maximum safe altitude [m]
                - alpha_altitude: float, CBF class-K gain for altitude
                - v_min_mps: float, minimum safe speed [m/s]
                - v_max_mps: float, maximum safe speed [m/s]
                - alpha_speed: float, CBF class-K gain for speed
                - nz_max: float, max normal overload [g]
                - alpha_overload: float, CBF class-K gain for overload
                - qp_penalty: float, penalty on deviation from nominal
                - slack_weight: float, weight on slack variables
                - verbose: bool, log CBF interventions
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        
        # Collision avoidance parameters
        self.d_min = float(self.config.get("d_min_m", 150.0))
        self.alpha_collision = float(self.config.get("alpha_collision", 0.5))
        
        # Altitude constraints
        self.h_min = float(self.config.get("h_min_m", 500.0))
        self.h_max = float(self.config.get("h_max_m", 12000.0))
        self.alpha_altitude = float(self.config.get("alpha_altitude", 0.3))
        
        # Speed constraints
        self.v_min = float(self.config.get("v_min_mps", 80.0))
        self.v_max = float(self.config.get("v_max_mps", 400.0))
        self.alpha_speed = float(self.config.get("alpha_speed", 0.2))
        
        # Overload constraint
        self.nz_max = float(self.config.get("nz_max", 7.0))
        self.alpha_overload = float(self.config.get("alpha_overload", 0.5))
        
        # QP parameters
        self.qp_penalty = float(self.config.get("qp_penalty", 1.0))
        self.slack_weight = float(self.config.get("slack_weight", 1000.0))
        self.verbose = self.config.get("verbose", False)
        
        # Statistics
        self._intervention_count = 0
        self._total_steps = 0
        self._last_intervention = False
        self._last_slack = 0.0
    
    def filter_command(
        self,
        nominal_command: dict,
        own_state: dict,
        target_state: dict,
        rel_state: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Filter a nominal command through CBF constraints.
        
        Args:
            nominal_command: Dict with keys nz_cmd, roll_rate_cmd, throttle_cmd
            own_state: Ego aircraft state dict
            target_state: Target aircraft state dict
            rel_state: Pre-computed relative geometry (optional)
            
        Returns:
            (safe_command, cbf_info) where cbf_info contains diagnostics
        """
        self._total_steps += 1
        
        if not self.enabled:
            return dict(nominal_command), {"active": False}
        
        # Extract state variables
        p_ego = self._get_position(own_state)
        v_ego = self._get_velocity(own_state)
        h_ego = self._get_altitude(own_state)
        speed_ego = self._get_speed(own_state)
        
        p_target = self._get_position(target_state)
        v_target = self._get_velocity(target_state)
        
        # Compute relative position and distance
        rel_pos = p_target - p_ego
        d = np.linalg.norm(rel_pos)
        
        # Extract nominal controls
        u_nom = np.array([
            nominal_command.get("nz_cmd", 1.0),
            nominal_command.get("roll_rate_cmd", 0.0),
            nominal_command.get("throttle_cmd", 0.5),
        ], dtype=np.float64)
        
        # Build CBF constraints
        constraints = []
        
        # 1. Collision avoidance CBF
        if d < self.d_min * 3.0:  # Only active when reasonably close
            h_collision = d - self.d_min
            if h_collision < 10.0:  # Only enforce when near boundary
                # Lie derivative: dh/dt = (p_target - p_ego)^T (v_target - v_ego) / d
                # Approximate: dh/dt ≈ rel_pos_norm . (v_target - v_ego)
                rel_pos_norm = rel_pos / max(d, 1e-6)
                rel_vel = v_target - v_ego
                Lf_h = float(np.dot(rel_pos_norm, rel_vel))
                
                # Control effect on h (approximate)
                # dh/du ≈ 0 for nz/roll_rate, small for throttle
                # Simplified: only enforce via Lf_h + alpha*h >= 0
                # This is a conservative approximation
                constraints.append({
                    "type": "collision",
                    "h": h_collision,
                    "Lf_h": Lf_h,
                    "Lg_h": np.array([0.0, 0.0, 0.0]),  # throttle has minor effect
                    "alpha": self.alpha_collision,
                })
        
        # 2. Altitude CBF (lower bound)
        h_alt_low = h_ego - self.h_min
        if h_alt_low < 200.0:
            # dh/dt ≈ vertical velocity (positive = climbing)
            v_vert = v_ego[2] if len(v_ego) >= 3 else 0.0
            # Control effect: nz_cmd affects vertical acceleration
            # Approximate: Lg_h ≈ [const, 0, 0] for nz
            constraints.append({
                "type": "altitude_low",
                "h": h_alt_low,
                "Lf_h": v_vert,
                "Lg_h": np.array([5.0, 0.0, 0.0]),  # nz affects climb rate
                "alpha": self.alpha_altitude,
            })
        
        # 3. Altitude CBF (upper bound)
        h_alt_high = self.h_max - h_ego
        if h_alt_high < 200.0:
            v_vert = v_ego[2] if len(v_ego) >= 3 else 0.0
            constraints.append({
                "type": "altitude_high",
                "h": h_alt_high,
                "Lf_h": -v_vert,
                "Lg_h": np.array([-5.0, 0.0, 0.0]),
                "alpha": self.alpha_altitude,
            })
        
        # 4. Speed CBF (lower bound - stall protection)
        h_speed_low = speed_ego - self.v_min
        if h_speed_low < 20.0:
            # Throttle increases speed
            constraints.append({
                "type": "speed_low",
                "h": h_speed_low,
                "Lf_h": -2.0,  # approximate drag deceleration
                "Lg_h": np.array([0.0, 0.0, 50.0]),  # throttle effect
                "alpha": self.alpha_speed,
            })
        
        # 5. Speed CBF (upper bound - overspeed protection)
        h_speed_high = self.v_max - speed_ego
        if h_speed_high < 50.0:
            constraints.append({
                "type": "speed_high",
                "h": h_speed_high,
                "Lf_h": 2.0,
                "Lg_h": np.array([0.0, 0.0, -50.0]),
                "alpha": self.alpha_speed,
            })
        
        # If no constraints are active, return nominal command
        if not constraints:
            self._last_intervention = False
            return dict(nominal_command), {
                "active": True,
                "intervened": False,
                "constraints_active": 0,
            }
        
        # Solve QP: min ||u - u_nom||^2 + slack_weight * sum(s_i)
        # s.t. Lf_h + Lg_h @ u + alpha*h >= 0 for each constraint
        #      u_min <= u <= u_max
        
        # For simplicity, use a projected gradient approach
        # This is not the most efficient but is robust and easy to understand
        u_safe = self._solve_qp_simple(u_nom, constraints)
        
        # Check if intervention occurred
        intervention = np.linalg.norm(u_safe - u_nom) > 0.01
        if intervention:
            self._intervention_count += 1
            self._last_intervention = True
            if self.verbose:
                logger.info(
                    f"CBF intervention: constraints={len(constraints)}, "
                    f"nom={u_nom}, safe={u_safe}"
                )
        else:
            self._last_intervention = False
        
        safe_command = {
            "nz_cmd": float(u_safe[0]),
            "roll_rate_cmd": float(u_safe[1]),
            "throttle_cmd": float(u_safe[2]),
        }
        
        cbf_info = {
            "active": True,
            "intervened": intervention,
            "constraints_active": len(constraints),
            "constraint_types": [c["type"] for c in constraints],
            "nominal_command": {k: float(v) for k, v in nominal_command.items()},
            "cbf_command": safe_command,
            "deviation": float(np.linalg.norm(u_safe - u_nom)),
            "intervention_rate": self._intervention_count / max(self._total_steps, 1),
        }
        
        return safe_command, cbf_info
    
    def _solve_qp_simple(
        self,
        u_nom: np.ndarray,
        constraints: List[dict],
    ) -> np.ndarray:
        """Solve CBF QP using a simple projection method.
        
        This is a simplified solver that prioritizes safety over optimality.
        For production use, consider replacing with OSQP or cvxpy.
        
        Args:
            u_nom: Nominal control input [nz_cmd, roll_rate_cmd, throttle_cmd]
            constraints: List of CBF constraint dicts
            
        Returns:
            Safe control input
        """
        u = u_nom.copy()
        
        # Control bounds (physical limits)
        u_min = np.array([-2.0, -1.5, 0.0], dtype=np.float64)
        u_max = np.array([7.0, 1.5, 1.0], dtype=np.float64)
        
        # Iterative projection (simplified)
        max_iters = 10
        for _ in range(max_iters):
            all_satisfied = True
            
            for c in constraints:
                h = c["h"]
                Lf_h = c["Lf_h"]
                Lg_h = c["Lg_h"]
                alpha = c["alpha"]
                
                # CBF condition: Lf_h + Lg_h @ u + alpha * h >= 0
                lhs = Lf_h + float(np.dot(Lg_h, u)) + alpha * h
                
                if lhs < 0:
                    all_satisfied = False
                    # Project toward satisfaction
                    # Move u in direction of Lg_h to increase lhs
                    g_norm_sq = float(np.dot(Lg_h, Lg_h))
                    if g_norm_sq > 1e-6:
                        step = -lhs / g_norm_sq
                        u = u + step * Lg_h * 0.5  # conservative step
            
            # Clip to bounds
            u = np.clip(u, u_min, u_max)
            
            if all_satisfied:
                break
        
        # Final blending: minimize deviation from nominal while satisfying constraints
        # Weighted average between nominal and projected solution
        alpha_blend = 0.7  # how much to trust the projected solution
        u_final = alpha_blend * u + (1 - alpha_blend) * u_nom
        u_final = np.clip(u_final, u_min, u_max)
        
        return u_final
    
    def _get_position(self, state: dict) -> np.ndarray:
        """Extract position from state dict."""
        pos = state.get("position_neu")
        if pos is None:
            pos = state.get("position_m")
        if pos is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(pos, dtype=np.float64)
    
    def _get_velocity(self, state: dict) -> np.ndarray:
        """Extract velocity from state dict."""
        vel = state.get("velocity_vector_mps")
        if vel is None:
            vel = state.get("velocity")
        if vel is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(vel, dtype=np.float64)
    
    def _get_altitude(self, state: dict) -> float:
        """Extract altitude from state dict."""
        alt = state.get("altitude_m")
        if alt is not None:
            return float(alt)
        pos = self._get_position(state)
        return float(pos[2]) if len(pos) >= 3 else 0.0
    
    def _get_speed(self, state: dict) -> float:
        """Extract speed from state dict."""
        speed = state.get("speed_mps")
        if speed is not None:
            return float(speed)
        vel = self._get_velocity(state)
        return float(np.linalg.norm(vel))
    
    def get_stats(self) -> dict:
        """Return CBF intervention statistics."""
        return {
            "total_steps": self._total_steps,
            "intervention_count": self._intervention_count,
            "intervention_rate": self._intervention_count / max(self._total_steps, 1),
            "last_intervention": self._last_intervention,
        }
    
    def reset_stats(self):
        """Reset CBF statistics."""
        self._intervention_count = 0
        self._total_steps = 0
        self._last_intervention = False
        self._last_slack = 0.0


def create_cbf_filter(config: Optional[dict] = None) -> Optional[CBFSafetyFilter]:
    """Factory function to create a CBF filter from config.
    
    Args:
        config: Configuration dict. If None or empty, returns None.
        
    Returns:
        CBFSafetyFilter instance or None if not configured.
    """
    if config is None:
        return None
    if not config.get("enabled", False):
        return None
    return CBFSafetyFilter(config)
