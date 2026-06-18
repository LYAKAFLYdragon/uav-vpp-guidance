"""
Unit tests for the CBF QP safety filter.

Tests cover:
- Geometry helpers (h, h_dot).
- First/second-order CBF activation logic.
- QP solver consistency (cvxpy vs scipy) and fallback behaviour.
- Numerical Jacobian accuracy on a simple point-mass model.
- Smoke integration with CloseRangeTrackingEnv (simple backend).
"""

import sys

import numpy as np
import pytest


def _make_cbf_filter(**kwargs):
    import sys

    sys.path.insert(0, "src")
    from uav_vpp_guidance.safety import CBFQPFilter

    defaults = {
        "d_min": 500.0,
        "gamma": 1.0,
        "gamma1": 2.0,
        "gamma2": 1.0,
        "solver": "auto",
    }
    defaults.update(kwargs)
    return CBFQPFilter(**defaults)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_compute_h():
    cbf = _make_cbf_filter()
    state = {
        "pursuer": {
            "position": np.array([0.0, 0.0, 5000.0]),
            "velocity": np.array([200.0, 0.0, 0.0]),
        },
        "target": {
            "position": np.array([800.0, 0.0, 5000.0]),
            "velocity": np.array([180.0, 0.0, 0.0]),
        },
    }
    is_safe, action, info = cbf.check(state, np.zeros(3), None)
    assert info["range_m"] == pytest.approx(800.0, abs=1e-6)
    assert info["h"] == pytest.approx(300.0, abs=1e-6)
    assert is_safe
    np.testing.assert_array_equal(action, np.zeros(3))


def test_compute_h_dot():
    cbf = _make_cbf_filter()
    state = {
        "pursuer": {
            "position": np.array([0.0, 0.0, 5000.0]),
            "velocity": np.array([200.0, 0.0, 0.0]),
        },
        "target": {
            "position": np.array([800.0, 0.0, 5000.0]),
            "velocity": np.array([100.0, 0.0, 0.0]),
        },
    }
    _, _, info = cbf.check(state, np.zeros(3), None)
    # r = own - target = [-800, 0, 0], r_hat = [-1, 0, 0];
    # h_dot = r_hat^T (v_o - v_t) = -1 * (200 - 100) = -100
    assert info["h_dot"] == pytest.approx(-100.0, abs=1e-6)


def test_first_order_safe_passes_without_qp():
    cbf = _make_cbf_filter()
    # Far away and opening.
    state = {
        "pursuer": {
            "position": np.array([0.0, 0.0, 5000.0]),
            "velocity": np.array([250.0, 0.0, 0.0]),
        },
        "target": {
            "position": np.array([2000.0, 0.0, 5000.0]),
            "velocity": np.array([150.0, 0.0, 0.0]),
        },
    }
    is_safe, action, info = cbf.check(state, np.array([0.5, 0.0, 0.0]), None)
    assert is_safe
    assert not info["active"]
    assert info["solve_time_ms"] < 1.0  # purely geometric check
    np.testing.assert_allclose(action, np.array([0.5, 0.0, 0.0]), atol=1e-12)


def test_first_order_unsafe_triggers_qp():
    cbf = _make_cbf_filter(d_min=500.0, gamma1=2.0, gamma2=1.0)
    # Close and rapidly closing head-on geometry.
    state = {
        "pursuer": {
            "position": np.array([0.0, 0.0, 5000.0]),
            "velocity": np.array([300.0, 0.0, 0.0]),
        },
        "target": {
            "position": np.array([600.0, 0.0, 5000.0]),
            "velocity": np.array([-200.0, 0.0, 0.0]),
        },
    }
    # Provide a fake environment with a simple known Jacobian so the QP runs.
    class FakeEnv:
        env_config = {"high_level_dt": 0.2}

    fake_env = FakeEnv()

    # Identity Jacobian: K = I  -> changing action directly changes acceleration.
    def identity_jacobian(env, state, action, eps, dt):
        return np.eye(3)

    cbf._jacobian_estimator = identity_jacobian
    is_safe, action, info = cbf.check(state, np.array([0.0, 0.0, 0.0]), fake_env)

    assert info["active"]
    assert info["qp_solved"] or info["fallback"]
    # The filtered action should push the pursuer away from the target,
    # i.e. the longitudinal component should become negative (decelerate / turn back).
    assert action[0] < -0.1
    assert np.all(np.abs(action) <= 1.0)


# ---------------------------------------------------------------------------
# QP solvers
# ---------------------------------------------------------------------------


def _random_qp_problem(rng, dim=3):
    """Generate a feasible random QP: min ||a - a0||^2 s.t. A@a >= b, -1<=a<=1."""
    a0 = rng.uniform(-1, 1, size=dim)
    A = rng.normal(size=dim)
    # Make b feasible by choosing b slightly below A @ a0.
    b = float(A @ a0) - rng.uniform(0, 2)
    return a0, A, b


@pytest.mark.skipif(
    "jsbsim" in sys.modules,
    reason="osqp C extension crashes when imported after jsbsim in this environment",
)
def test_qp_solver_consistency():
    cbf = _make_cbf_filter(solver="scipy")
    rng = np.random.default_rng(42)
    for _ in range(100):
        a0, A, b = _random_qp_problem(rng)
        a_sci, info_sci = cbf._solve_qp_scipy(a0, A, b)
        assert info_sci["qp_solved"]
        assert A @ a_sci >= b - 1e-4

    # Compare against osqp only when it can be safely imported.
    try:
        import osqp  # noqa: F401
    except Exception:
        pytest.skip("osqp not importable in this environment")
    rng2 = np.random.default_rng(42)
    for _ in range(100):
        a0, A, b = _random_qp_problem(rng2)
        a_osqp, info_osqp = cbf._solve_qp_osqp(a0, A, b)
        a_sci, info_sci = cbf._solve_qp_scipy(a0, A, b)
        assert info_osqp["qp_solved"]
        assert info_sci["qp_solved"]
        np.testing.assert_allclose(a_osqp, a_sci, atol=1e-4)
        assert A @ a_osqp >= b - 1e-4
        assert A @ a_sci >= b - 1e-4


def test_infeasible_fallback():
    cbf = _make_cbf_filter(solver="scipy")
    # Constraint that cannot be satisfied inside [-1, 1]^3.
    a0 = np.array([1.0, 1.0, 1.0])
    A = np.array([1.0, 0.0, 0.0])
    b = 2.0
    action, info = cbf._solve_qp(a0, A, b)
    assert np.all(np.abs(action) <= 1.0)
    # Slack formulation should produce a feasible (possibly violated) action.
    assert info["qp_solved"] or info["fallback"]


def test_fallback_projection():
    cbf = _make_cbf_filter()
    a0 = np.array([0.5, 0.0, 0.0])
    A = np.array([1.0, 0.0, 0.0])
    b = 0.8
    action = cbf._fallback_projection(a0, A, b)
    # deficit = 0.3 -> projection to a0 + 0.3 * A = 0.8, clipped stays 0.8.
    np.testing.assert_allclose(action, np.array([0.8, 0.0, 0.0]), atol=1e-9)


def test_solve_time_budget():
    cbf = _make_cbf_filter(solver="auto")
    rng = np.random.default_rng(7)
    times = []
    for _ in range(1000):
        a0, A, b = _random_qp_problem(rng)
        _, info = cbf._solve_qp(a0, A, b)
        times.append(info["solve_time_ms"])
    mean_time = float(np.mean(times))
    max_time = float(np.max(times))
    assert mean_time < 1.0, f"mean QP time {mean_time:.3f} ms exceeds 1 ms budget"
    assert max_time < 10.0, f"max QP time {max_time:.3f} ms unexpectedly large"


# ---------------------------------------------------------------------------
# Jacobian estimator
# ---------------------------------------------------------------------------


def test_jacobian_numerical_accuracy():
    """Estimator should match a brute-force finite difference on the same model."""
    import sys

    sys.path.insert(0, "src")
    from uav_vpp_guidance.envs.simple_point_mass_env import SimplePointMassEnv
    from uav_vpp_guidance.safety import PointMassJacobianEstimator

    # Dummy environment whose guidance command is exactly the action.
    class DummyEnv:
        env_config = {"decision_freq": 5}

    def dummy_compute_guidance_command(env, own_state, target_state, action):
        return {
            "nz_cmd": float(action[0]),
            "roll_rate_cmd": float(action[1]),
            "throttle_cmd": float(action[2]),
        }

    estimator = PointMassJacobianEstimator()
    # Monkey-patch the command computation for this test.
    estimator._compute_guidance_command = dummy_compute_guidance_command

    own_pos = np.array([0.0, 0.0, 5000.0])
    own_vel = np.array([200.0, 0.0, 0.0])
    target_pos = np.array([1000.0, 0.0, 5000.0])
    target_vel = np.array([150.0, 0.0, 0.0])

    state = {
        "pursuer": {"position": own_pos, "velocity": own_vel},
        "target": {"position": target_pos, "velocity": target_vel},
    }

    base_action = np.array([0.0, 0.0, 0.5])
    eps = 0.01
    dt = 0.2

    K_est = estimator(DummyEnv(), state, base_action, eps, dt)

    # Brute-force reference using the same SimplePointMassEnv.
    pm = SimplePointMassEnv(DummyEnv.env_config)

    def _init():
        pm.reset(
            own_init={
                "position_m": own_pos.copy(),
                "velocity_vector_mps": own_vel.copy(),
                "heading_rad": 0.0,
                "altitude_m": 5000.0,
            },
            target_init={
                "position_m": target_pos.copy(),
                "velocity_vector_mps": target_vel.copy(),
                "heading_rad": 0.0,
                "altitude_m": 5000.0,
            },
        )

    _init()
    v_base = pm.own_state["velocity_vector_mps"].copy()
    K_ref = np.zeros((3, 3))
    for j in range(3):
        _init()
        a = base_action.copy()
        a[j] += eps
        pm.step(
            own_command={
                "nz_cmd": float(a[0]),
                "roll_rate_cmd": float(a[1]),
                "throttle_cmd": float(a[2]),
            }
        )
        K_ref[:, j] = (pm.own_state["velocity_vector_mps"] - v_base) / (eps * dt)

    rel_err = np.linalg.norm(K_est - K_ref) / (np.linalg.norm(K_ref) + 1e-12)
    assert rel_err < 0.1, f"Jacobian relative error {rel_err:.3f}"


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


def test_cbf_with_simple_tracking_env():
    """CBF filter can be attached to CloseRangeTrackingEnv with simple backend."""
    import sys

    sys.path.insert(0, "src")
    from pathlib import Path
    from uav_vpp_guidance.envs.tracking_env import CloseRangeTrackingEnv
    from uav_vpp_guidance.utils.config import load_yaml_config, merge_config

    config_path = Path("config/experiment/train_no_prediction_vpp_ppo.yaml")
    base = load_yaml_config(config_path)
    includes = base.pop("includes", [])
    merged = {}
    for inc in includes:
        inc_full = config_path.parent / inc
        if inc_full.exists():
            merged = merge_config(merged, load_yaml_config(inc_full))
    config = merge_config(merged, base)
    config["env"]["backend"] = "simple"
    config["env"]["use_jsbsim"] = False
    config["env"]["max_high_level_steps"] = 20
    config["cbf"] = {
        "enabled": True,
        "params": {"d_min": 500.0, "gamma1": 2.0, "gamma2": 1.0},
    }

    env = CloseRangeTrackingEnv(config)
    assert env._use_cbf

    obs = env.reset(seed=0, scenario="favorable")
    for _ in range(10):
        action = np.zeros(3, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        assert "cbf" in info
        assert info["cbf"] is not None
        assert np.all(np.abs(info["cbf"]["filtered_action"]) <= 1.0)
        if terminated or truncated:
            break
    env.close()


# ---------------------------------------------------------------------------
# Run-if-main guard for quick manual invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
