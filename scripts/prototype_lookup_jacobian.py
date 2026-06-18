"""Prototype for an offline lookup-table Jacobian (Scheme 2).

This script generates a small dataset of JSBSim one-step roll-outs over a grid
of engagement geometries and VPP actions, fits a KNN regressor that maps
(state-features, action) -> pursuer acceleration, and reports prediction
accuracy on a held-out test set.  It is a quick proof-of-concept for the
offline system-identification idea, not a production CBF component.

Example:
    python scripts/prototype_lookup_jacobian.py --output outputs/cbf_adversarial/lookup_prototype
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from uav_vpp_guidance.envs.adversarial_jsbsim_env import AdversarialJSBSimEnv
from uav_vpp_guidance.envs.geometry_scenarios import build_explicit_scenario
from uav_vpp_guidance.envs.observation import compute_relative_geometry
from uav_vpp_guidance.safety.jacobian_estimator import (
    JSBSimFiniteDifferenceJacobianEstimator,
    PointMassJacobianEstimator,
)


def _minimal_config() -> Dict[str, Any]:
    """Return a minimal AdversarialJSBSimEnv config for data generation."""
    return {
        "env": {
            "sim_freq": 60,
            "decision_freq": 5,
            "max_high_level_steps": 512,
            "high_level_dt": 0.2,
        },
        "guidance": {
            "mode": "los_rate",
            "gains": {
                "k_los": 1.0,
                "k_damp": 0.2,
                "k_roll": 1.0,
                "k_speed": 0.2,
                "k_energy": 0.1,
                "alpha_filter": 0.3,
            },
        },
        "virtual_point": {"enabled": True, "mode": "normal", "anchor_mode": "current_target"},
        "limits": {
            "nz_min": -2.0,
            "nz_max": 7.0,
            "roll_rate_min": -1.5,
            "roll_rate_max": 1.5,
            "throttle_min": 0.0,
            "throttle_max": 1.0,
        },
        "reward": {},
        "target_reward": {},
        "actuator_dynamics": {"enabled": False},
    }


def _make_env() -> AdversarialJSBSimEnv:
    cfg = _minimal_config()
    # Avoid importing/loading any neural target controller.
    cfg["env"]["target_controller_type"] = "rl"
    return AdversarialJSBSimEnv(cfg)


def _scenario_grid() -> List[Dict[str, Any]]:
    """Build a small grid of engagement geometries."""
    scenarios = []
    scenario_types = ["tail_chase", "head_on", "offset_pursuit"]
    ranges = [1500.0, 2500.0, 4000.0]
    ego_speeds = [220.0, 280.0]
    target_speeds = [180.0, 240.0]
    offsets = [0.0, 500.0] if "offset_pursuit" in scenario_types else [0.0]
    for st in scenario_types:
        for r in ranges:
            for v_e in ego_speeds:
                for v_t in target_speeds:
                    for lat_off in offsets:
                        if st != "offset_pursuit" and lat_off > 0:
                            continue
                        sc = build_explicit_scenario(
                            scenario_type=st,
                            initial_range_m=r,
                            ego_speed_mps=v_e,
                            target_speed_mps=v_t,
                            base_altitude_m=6096.0,
                            lateral_offset_m=lat_off,
                        )
                        scenarios.append(sc)
    return scenarios


def _action_grid() -> List[np.ndarray]:
    """Cartesian grid of 3-D VPP actions in [-1, 1]^3."""
    vals = np.linspace(-0.8, 0.8, 3)
    actions = []
    for a0 in vals:
        for a1 in vals:
            for a2 in vals:
                actions.append(np.array([a0, a1, a2], dtype=np.float64))
    return actions


def _extract_features(
    own_state: Dict[str, Any], target_state: Dict[str, Any], action: np.ndarray
) -> np.ndarray:
    """Return a low-dimensional feature vector for the lookup table."""
    rel = compute_relative_geometry(own_state, target_state)
    return np.array(
        [
            float(rel.get("range_m", 0.0)),
            float(rel.get("range_rate_mps", 0.0)),
            float(np.degrees(rel.get("ata_rad", 0.0))),
            float(np.degrees(rel.get("aa_rad", 0.0))),
            float(own_state.get("speed_mps", 0.0)),
            float(target_state.get("speed_mps", 0.0)),
            float(action[0]),
            float(action[1]),
            float(action[2]),
        ],
        dtype=np.float64,
    )


def _simulate_one_step(env: AdversarialJSBSimEnv, scenario: Dict[str, Any], action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one high-level step and return (features, v_before, v_after, accel)."""
    env.reset(scenario=scenario)
    own_before, target_before = env._get_current_states()
    v_before = np.asarray(own_before.get("velocity_vector_mps", own_before.get("velocity_ned")), dtype=np.float64)
    # Target action is zero (constant-velocity target) for the dataset.
    env.step(np.asarray(action, dtype=np.float64), np.zeros(3, dtype=np.float64))
    own_after, _ = env._get_current_states()
    v_after = np.asarray(own_after.get("velocity_vector_mps", own_after.get("velocity_ned")), dtype=np.float64)
    dt = env._high_level_dt
    accel = (v_after - v_before) / dt
    features = _extract_features(own_before, target_before, action)
    return features, v_before, v_after, accel


def generate_dataset(output_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) dataset. If max_samples > 0, cap at that many samples."""
    env = _make_env()
    scenarios = _scenario_grid()
    actions = _action_grid()
    print(f"Generating data: {len(scenarios)} scenarios x {len(actions)} actions = {len(scenarios)*len(actions)} max samples")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total = len(scenarios) * len(actions)
    t0 = time.time()
    for i, sc in enumerate(scenarios):
        for j, act in enumerate(actions):
            if max_samples > 0 and len(X_list) >= max_samples:
                break
            try:
                features, _, _, accel = _simulate_one_step(env, sc, act)
                X_list.append(features)
                y_list.append(accel)
            except Exception as exc:
                print(f"  skip sc={i} act={j}: {exc}")
        if max_samples > 0 and len(X_list) >= max_samples:
            break
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(scenarios)} scenarios done, {len(X_list)} samples, {elapsed:.1f}s")

    env.close()
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "lookup_X.npy", X)
    np.save(output_dir / "lookup_y.npy", y)
    return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray, test_fraction: float = 0.2) -> Dict[str, float]:
    """Train KNN regressor and report test-set metrics."""
    n = len(X)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(n * (1.0 - test_fraction))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = KNeighborsRegressor(n_neighbors=5, weights="distance")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2": float(r2_score(y_test, y_pred)),
        "mae_mps2": float(mean_absolute_error(y_test, y_pred)),
        "rmse_mps2": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
    }
    return metrics, model, X_test, y_test


def evaluate_jacobian_estimate(
    model: KNeighborsRegressor,
    env: AdversarialJSBSimEnv,
    scenario: Dict[str, Any],
    base_action: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    """Compare lookup-table Jacobian against JSBSim finite difference on one state."""
    env.reset(scenario=scenario)
    own_before, target_before = env._get_current_states()
    features_base = _extract_features(own_before, target_before, base_action)
    v_before = np.asarray(own_before.get("velocity_vector_mps", own_before.get("velocity_ned")), dtype=np.float64)

    # Lookup-table finite-difference Jacobian.
    eps = 0.01
    v_base_pred = model.predict(features_base.reshape(1, -1))[0] * dt + v_before
    K_lookup = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        a_pert = np.asarray(base_action, dtype=np.float64).copy()
        a_pert[j] += eps
        features_pert = _extract_features(own_before, target_before, a_pert)
        v_pert_pred = model.predict(features_pert.reshape(1, -1))[0] * dt + v_before
        K_lookup[:, j] = (v_pert_pred - v_base_pred) / (eps * dt)

    # Ground-truth JSBSim finite-difference Jacobian.
    state = {
        "pursuer": {
            "position": own_before.get("position_m", own_before.get("position_neu")),
            "velocity": own_before.get("velocity_vector_mps", own_before.get("velocity_ned")),
        },
        "target": {
            "position": target_before.get("position_m", target_before.get("position_neu")),
            "velocity": target_before.get("velocity_vector_mps", target_before.get("velocity_ned")),
        },
    }
    K_fd = JSBSimFiniteDifferenceJacobianEstimator()(env, state, base_action, eps, dt)

    rel_err = np.linalg.norm(K_lookup - K_fd) / (np.linalg.norm(K_fd) + 1e-12)
    return {
        "K_lookup": K_lookup.tolist(),
        "K_fd": K_fd.tolist(),
        "relative_error": float(rel_err),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/cbf_adversarial/lookup_prototype")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap dataset size for quick tests.")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse existing lookup_X.npy/lookup_y.npy.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    X_path = output_dir / "lookup_X.npy"
    y_path = output_dir / "lookup_y.npy"

    if args.skip_generation and X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"Loaded cached dataset: X.shape={X.shape}, y.shape={y.shape}")
    else:
        X, y = generate_dataset(output_dir, max_samples=args.max_samples)
        print(f"Dataset: X.shape={X.shape}, y.shape={y.shape}")

    print("Training KNN regressor...")
    metrics, model, X_test, y_test = train_and_evaluate(X, y)
    print("Test metrics:", metrics)

    print("Comparing lookup Jacobian to JSBSim FD on a few hold-out geometries...")
    env = _make_env()
    test_scenarios = [
        build_explicit_scenario("tail_chase", 2200.0, 250.0, 200.0, base_altitude_m=6096.0),
        build_explicit_scenario("head_on", 3500.0, 260.0, 210.0, base_altitude_m=6096.0),
        build_explicit_scenario("offset_pursuit", 1800.0, 240.0, 190.0, base_altitude_m=6096.0, lateral_offset_m=400.0),
    ]
    jac_results = []
    for sc in test_scenarios:
        res = evaluate_jacobian_estimate(
            model, env, sc, np.array([0.2, -0.1, 0.0], dtype=np.float64), env._high_level_dt
        )
        jac_results.append({"scenario": sc.get("name", "unknown"), **res})
        print(f"  {sc.get('name')}: rel_err={res['relative_error']:.3f}")
    env.close()

    summary = {
        "dataset_shape": list(X.shape),
        "metrics": metrics,
        "jacobian_tests": jac_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
