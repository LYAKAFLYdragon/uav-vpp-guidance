"""
Control Barrier Function (CBF) QP safety filter for VPP-based guidance.

The filter uses a simplified point-mass model and a numerically estimated
Jacobian to enforce a minimum separation distance between the pursuer and
target.  If the RL action already satisfies the CBF condition it is passed
through unchanged; otherwise a small 3-D QP is solved to find the minimal
safe correction.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .jacobian_estimator import PointMassJacobianEstimator

from scipy.optimize import minimize

# NOTE: osqp/cvxpy are intentionally NOT imported at module top.  In this
# environment importing osqp after jsbsim has been loaded causes an access
# violation inside the osqp C extension.  The default QP backend is therefore
# scipy SLSQP, which is fully deterministic and fast enough for a 3-D problem.

# osqp is not available at import time in the JSBSim workflow; keep the flag
# defined so that solver-dispatch logic stays self-consistent.
HAS_OSQP = False

logger = logging.getLogger(__name__)


class CBFQPFilter:
    """
    Control Barrier Function (CBF) safety filter for VPP-based guidance.

    Uses a simplified point-mass model.  The control influence matrix ``K``
    (Jacobian of pursuer acceleration w.r.t. VPP offset) is estimated
    numerically on demand, only when the first-order CBF condition is
    violated.
    """

    def __init__(
        self,
        d_min: float = 500.0,
        gamma: float = 1.0,
        gamma1: float = 2.0,
        gamma2: float = 1.0,
        eps_jacobian: float = 0.01,
        use_second_order: bool = True,
        solver: str = "auto",
        slack_penalty: float = 1e3,
        jacobian_estimator: Optional[Any] = None,
        estimate_target_accel: bool = True,
        dt: Optional[float] = None,
        constraint_tol: float = 1e-6,
    ) -> None:
        """
        Args:
            d_min: Minimum safe distance (m).
            gamma: Class-K gain for the first-order CBF check.
            gamma1, gamma2: Gains for the second-order CBF condition.
            eps_jacobian: Perturbation size for numerical Jacobian estimation.
            use_second_order: If True, use the second-order CBF QP filter when
                the first-order condition is violated.  If False, fall back to
                a first-order projection (less conservative).
            solver: QP solver backend.  ``"auto"`` uses scipy by default.
                ``"osqp"`` is available but should not be used together with
                the JSBSim backend due to a C-extension compatibility issue.
            slack_penalty: Penalty weight for the constraint slack variable.
            jacobian_estimator: Callable ``(env, state, action, eps, dt) -> K``.
                If None, a :class:`PointMassJacobianEstimator` is used.
            estimate_target_accel: If True, estimate target acceleration from
                consecutive target velocities stored in the filter.
            dt: High-level decision interval (s).  If None, it is read from
                ``env`` at runtime.
            constraint_tol: Tolerance for declaring the QP constraint satisfied.
        """
        self.d_min = float(d_min)
        self.gamma = float(gamma)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.eps_jacobian = float(eps_jacobian)
        self.use_second_order = bool(use_second_order)
        self.solver = solver if solver != "auto" else "scipy"
        self.slack_penalty = float(slack_penalty)
        self.estimate_target_accel = bool(estimate_target_accel)
        self.dt = dt
        self.constraint_tol = float(constraint_tol)

        self._jacobian_estimator = jacobian_estimator or PointMassJacobianEstimator()

        # Stateful target-velocity buffer for finite-difference acceleration.
        self._prev_target_velocity: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state (call at the beginning of each episode)."""
        self._prev_target_velocity = None

    def check(
        self,
        state: Dict[str, Any],
        action_rl: np.ndarray,
        env: Any,
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Check the CBF condition and return a safe action.

        Args:
            state: Current state dict with keys ``"pursuer"`` and ``"target"``,
                each containing at least ``"position"`` and ``"velocity"``.
                Alternatively, a dict with ``"own"`` / ``"target"`` is accepted.
            action_rl: Raw RL action, shape (3,), expected in [-1, 1].
            env: Environment instance used for numerical Jacobian estimation.

        Returns:
            ``(is_safe, action_filtered, info)``.
        """
        action_rl = np.asarray(action_rl, dtype=np.float64).reshape(3)
        t_start = time.perf_counter()

        pos_o, vel_o, pos_t, vel_t = self._extract_state(state)
        r_vec = pos_o - pos_t
        r = float(np.linalg.norm(r_vec))
        r_hat = r_vec / (r + 1e-12)
        v_rel = vel_o - vel_t

        h = r - self.d_min
        h_dot = float(np.dot(r_hat, v_rel))

        # First-order CBF check (cheap, no Jacobian needed).
        first_order_ok = h_dot + self.gamma * h >= -self.constraint_tol

        info: Dict[str, Any] = {
            "h": h,
            "h_dot": h_dot,
            "range_m": r,
            "d_min": self.d_min,
            "first_order_ok": bool(first_order_ok),
            "active": False,
            "qp_solved": False,
            "fallback": False,
            "solve_time_ms": 0.0,
            "constraint_value": 0.0,
            "solver": None,
        }

        if first_order_ok:
            info["filtered_action"] = action_rl.copy()
            info["solve_time_ms"] = (time.perf_counter() - t_start) * 1e3
            return True, action_rl.copy(), info

        # First-order condition violated -> need control authority.
        info["active"] = True

        # Estimate target acceleration.
        a_t = self._estimate_target_acceleration(vel_t, env)

        # Estimate control Jacobian (only when needed).
        K = self._estimate_jacobian(env, state, action_rl)
        A = r_hat @ K  # shape (3,)

        # Curvature term in h_ddot.
        v_rel_sq = float(np.dot(v_rel, v_rel))
        radial_comp = float(np.dot(r_hat, v_rel))
        curvature = (v_rel_sq - radial_comp ** 2) / (r + 1e-12)

        # Linear CBF constraint: A @ a >= b.
        b = (
            float(np.dot(r_hat, a_t))
            - curvature
            - self.gamma1 * h_dot
            - self.gamma2 * h
        )

        info["h_ddot_base"] = float(A @ action_rl - np.dot(r_hat, a_t) + curvature)
        info["cbf_rhs"] = b

        # If the RL action already satisfies the second-order condition, pass.
        if A @ action_rl + self.constraint_tol >= b:
            info["filtered_action"] = action_rl.copy()
            info["constraint_value"] = float(A @ action_rl - b)
            info["solve_time_ms"] = (time.perf_counter() - t_start) * 1e3
            return True, action_rl.copy(), info

        # Solve QP (or fallback).
        action_filtered, solve_info = self._solve_qp(action_rl, A, b)
        info.update(solve_info)

        h_ddot_filtered = float(A @ action_filtered - np.dot(r_hat, a_t) + curvature)
        info["h_ddot_filtered"] = h_ddot_filtered
        info["constraint_value"] = float(A @ action_filtered - b)
        info["filtered_action"] = action_filtered.copy()

        is_safe = info["constraint_value"] >= -self.constraint_tol
        info["solve_time_ms"] = (time.perf_counter() - t_start) * 1e3
        return is_safe, action_filtered, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_state(
        state: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract pursuer/target positions and velocities."""
        if "pursuer" in state and "target" in state:
            own = state["pursuer"]
            target = state["target"]
        elif "own" in state and "target" in state:
            own = state["own"]
            target = state["target"]
        else:
            # Assume state is a flat dict with direct keys.
            own = state
            target = state

        def _get(obj, pos_key, vel_key):
            pos = obj.get(pos_key) if isinstance(obj, dict) else getattr(obj, pos_key, None)
            vel = obj.get(vel_key) if isinstance(obj, dict) else getattr(obj, vel_key, None)
            return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)

        pos_o, vel_o = _get(own, "position", "velocity")
        pos_t, vel_t = _get(target, "position", "velocity")
        return pos_o, vel_o, pos_t, vel_t

    def _estimate_jacobian(
        self, env: Any, state: Dict[str, Any], base_action: np.ndarray
    ) -> np.ndarray:
        """Estimate K = ∂v_o / ∂a using the configured estimator."""
        dt = self._get_dt(env)
        K = self._jacobian_estimator(env, state, base_action, self.eps_jacobian, dt)
        return np.asarray(K, dtype=np.float64).reshape(3, 3)

    def _estimate_target_acceleration(
        self, vel_t: np.ndarray, env: Any
    ) -> np.ndarray:
        """Estimate target acceleration by finite difference."""
        if not self.estimate_target_accel:
            return np.zeros(3, dtype=np.float64)

        vel_t = np.asarray(vel_t, dtype=np.float64)
        if self._prev_target_velocity is None:
            self._prev_target_velocity = vel_t.copy()
            return np.zeros(3, dtype=np.float64)

        dt = self._get_dt(env)
        if dt <= 0:
            dt = 0.2
        a_t = (vel_t - self._prev_target_velocity) / dt
        self._prev_target_velocity = vel_t.copy()
        return a_t

    def _get_dt(self, env: Any) -> float:
        """Resolve the high-level decision interval."""
        if self.dt is not None:
            return float(self.dt)
        if hasattr(env, "_high_level_dt"):
            return float(env._high_level_dt)
        if hasattr(env, "env_config"):
            return float(env.env_config.get("high_level_dt", 0.2))
        return 0.2

    # ------------------------------------------------------------------
    # QP solvers
    # ------------------------------------------------------------------

    def _solve_qp(
        self,
        action_rl: np.ndarray,
        A: np.ndarray,
        b: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve the 3-D QP safety filter.

        Constraint: ``A @ a >= b``; bounds ``-1 <= a <= 1``.
        """
        solvers_to_try = []
        if self.solver == "auto":
            # cvxpy is not implemented in this dispatch; default to scipy directly.
            solvers_to_try = ["scipy"]
        else:
            solvers_to_try = [self.solver]

        for solver_name in solvers_to_try:
            if solver_name == "osqp" and not HAS_OSQP:
                continue
            try:
                action, solve_info = self._solve_qp_with_solver(
                    action_rl, A, b, solver_name
                )
                return action, solve_info
            except Exception as exc:  # pragma: no cover
                logger.debug("QP solver %s failed: %s", solver_name, exc)

        # Fallback projection.
        action_fb = self._fallback_projection(action_rl, A, b)
        return action_fb, {
            "qp_solved": False,
            "fallback": True,
            "solver": "projection_fallback",
            "constraint_value": float(A @ action_fb - b),
        }

    def _solve_qp_with_solver(
        self,
        action_rl: np.ndarray,
        A: np.ndarray,
        b: float,
        solver_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dispatch to a specific QP solver."""
        if solver_name == "osqp":
            return self._solve_qp_osqp(action_rl, A, b)
        if solver_name == "scipy":
            return self._solve_qp_scipy(action_rl, A, b)
        raise ValueError(f"Unknown solver: {solver_name}")

    def _solve_qp_osqp(
        self, action_rl: np.ndarray, A: np.ndarray, b: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve the QP using OSQP (direct Python interface)."""
        import osqp
        import scipy.sparse as sp

        # min 0.5 a^T (2I) a - 2 a_rl^T a
        P = sp.csc_matrix(2.0 * np.eye(3))
        q = -2.0 * action_rl.astype(np.float64)

        # Constraints: A@a >= b, -1 <= a <= 1.
        # Stack: [ A  ]       [ b ] <= [ A  ] a <= [ inf ]
        #        [  I ]       [-1 ] <= [  I ] a <= [  1  ]
        #        [ -I ]       [-1 ] <= [ -I ] a <= [  1  ]
        A_mat = sp.vstack(
            [
                sp.csr_matrix(A.reshape(1, 3)),
                sp.eye(3),
                -sp.eye(3),
            ]
        ).tocsc()
        l = np.concatenate(
            [
                np.array([b], dtype=np.float64),
                -np.ones(3, dtype=np.float64),
                -np.ones(3, dtype=np.float64),
            ]
        )
        u = np.concatenate(
            [
                np.array([np.inf], dtype=np.float64),
                np.ones(3, dtype=np.float64),
                np.ones(3, dtype=np.float64),
            ]
        )

        m = osqp.OSQP()
        m.setup(
            P=P,
            q=q,
            A=A_mat,
            l=l,
            u=u,
            alpha=1.6,
            verbose=False,
            warm_starting=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=4000,
        )

        t0 = time.perf_counter()
        result = m.solve()
        solve_time = time.perf_counter() - t0

        if result.info.status_val not in (1, 2):  # 1=solved, 2=solved_inaccurate
            raise RuntimeError(f"osqp status: {result.info.status}")

        action = np.clip(result.x.astype(np.float64), -1.0, 1.0)
        return action, {
            "qp_solved": True,
            "fallback": False,
            "solver": "osqp",
            "solve_time_ms": solve_time * 1e3,
            "constraint_value": float(A @ action - b),
            "osqp_status": result.info.status,
        }

    def _solve_qp_scipy(
        self, action_rl: np.ndarray, A: np.ndarray, b: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Try hard-constrained QP first.
        def objective(a):
            diff = a - action_rl
            return float(diff @ diff)

        def jac(a):
            return 2.0 * (a - action_rl)

        cons = {"type": "ineq", "fun": lambda a: A @ a - b, "jac": lambda a: A}
        bounds = [(-1.0, 1.0)] * 3

        t0 = time.perf_counter()
        res = minimize(
            objective,
            action_rl.copy(),
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        solve_time = time.perf_counter() - t0

        if res.success and res.x is not None:
            action = np.clip(res.x.astype(np.float64), -1.0, 1.0)
            constraint_value = float(A @ action - b)
            if constraint_value >= -self.constraint_tol:
                return action, {
                    "qp_solved": True,
                    "fallback": False,
                    "solver": "scipy_slsqp",
                    "solve_time_ms": solve_time * 1e3,
                    "constraint_value": constraint_value,
                }

        # Hard constraint infeasible or failed -> use slack formulation.
        return self._solve_qp_scipy_slack(action_rl, A, b)

    def _solve_qp_scipy_slack(
        self, action_rl: np.ndarray, A: np.ndarray, b: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve with a slack variable to guarantee feasibility."""
        lam = self.slack_penalty

        def objective(x):
            a = x[:3]
            s = x[3]
            diff = a - action_rl
            return float(diff @ diff) + lam * s * s

        def jac(x):
            a = x[:3]
            s = x[3]
            g = np.zeros(4)
            g[:3] = 2.0 * (a - action_rl)
            g[3] = 2.0 * lam * s
            return g

        # Constraint: A@a - b + s >= 0
        cons = {
            "type": "ineq",
            "fun": lambda x: A @ x[:3] - b + x[3],
            "jac": lambda x: np.concatenate([A, [1.0]]),
        }
        bounds = [(-1.0, 1.0)] * 3 + [(0.0, None)]
        x0 = np.concatenate([action_rl.copy(), [0.0]])

        t0 = time.perf_counter()
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        solve_time = time.perf_counter() - t0

        if not res.success or res.x is None:
            raise RuntimeError(f"scipy slack QP failed: {res.message}")

        action = np.clip(res.x[:3].astype(np.float64), -1.0, 1.0)
        slack = float(res.x[3])
        return action, {
            "qp_solved": True,
            "fallback": False,
            "solver": "scipy_slsqp_slack",
            "solve_time_ms": solve_time * 1e3,
            "constraint_value": float(A @ action - b),
            "slack": slack,
        }

    def _fallback_projection(
        self, action_rl: np.ndarray, A: np.ndarray, b: float
    ) -> np.ndarray:
        """
        Analytical projection of ``action_rl`` onto the half-space ``A@a >= b``
        followed by box clipping.  Ignores the box during projection but clips
        afterwards, so it is a conservative fallback.
        """
        A = np.asarray(A, dtype=np.float64)
        deficit = b - A @ action_rl
        norm_sq = float(A @ A)
        if norm_sq < 1e-12:
            return np.clip(action_rl, -1.0, 1.0)

        alpha = max(0.0, deficit / norm_sq)
        action = action_rl + alpha * A
        return np.clip(action, -1.0, 1.0)
