# Classical Prediction VPP Integration

## Overview

This document describes the integration of classical trajectory predictors
(Constant Velocity / Constant Acceleration) into the Virtual Pursuit Point (VPP)
guidance framework.

**Key principle**: CV/CA are classical prediction anchors, not machine-learning
models. Their purpose is to provide a computationally cheap, physically
interpretable baseline for the VPP anchor. We do **not** claim that prediction
always improves performance over the no-prediction baseline; rather, we provide
a mechanism to evaluate the hypothesis in a reproducible way.

## Predictor Models

### ConstantVelocityPredictor

- **Formula**: `pred_pos = pos + vel * T_lookahead`
- **Inputs**: `position_neu` or `position_m`, `velocity_ned` or `velocity_vector_mps`
- **Fallback**: if velocity is missing, returns current position (`fallback=True`)
- **Output**: absolute predicted position

### ConstantAccelerationPredictor

- **Formula**: `pred_pos = pos + vel * T + 0.5 * acc * T^2`
- **Inputs**: same position/velocity fields as CV
- **Acceleration estimation**: derived from feature-history velocity differences
- **Fallback rules**:
  1. History length < 3 → fallback to CV
  2. Non-finite acceleration estimate → fallback to CV
  3. Missing velocity → return current position
- **NaN/Inf protection**: any non-finite acceleration triggers CV fallback

## TrajectoryPredictorAdapter

The adapter connects the predictor to the environment step loop:

1. `update(own_state, target_state, relative_state)` — builds a feature vector
   and pushes it into the `TrajectoryStateBuffer`
2. `predict(current_target_state)` — queries the predictor with history + current state
3. Handles output-mode translation (relative displacement vs absolute position)
4. On exception, falls back to `ConstantVelocityPredictor`

## Environment Integration

`CloseRangeTrackingEnv` supports three configurations:

| `trajectory_prediction.enabled` | `virtual_point.anchor_mode` | Behavior |
|--------------------------------|----------------------------|----------|
| `false` | `current_target` | No predictor involved; VP anchored at current target position |
| `true` | `predicted_target` | Adapter updates each step; VP anchored at predicted target position |
| `true` | `current_target` | Adapter runs but VP ignores prediction (for ablation) |

When enabled, `env.step()` info contains:
- `prediction_enabled`
- `predictor_type`
- `prediction_valid`
- `prediction_fallback_reason`
- `predicted_target_position`
- `prediction_error_m`

## Evaluation: Scenario-Wise Comparison

All evaluations report three methods side-by-side:

1. **no_prediction** — baseline with `anchor_mode=current_target`
2. **cv_prediction** — `ConstantVelocityPredictor` with `anchor_mode=predicted_target`
3. **ca_prediction** — `ConstantAccelerationPredictor` with `anchor_mode=predicted_target`

This comparison is **scenario-wise**: the same random seeds and scenario
parameters are used across all three methods so that differences are due to
the predictor, not initialization luck.

### Unified Metrics Fields

| Field | Description |
|-------|-------------|
| `method` | `no_prediction`, `cv_prediction`, or `ca_prediction` |
| `scenario` | Scenario name or `all` for aggregated |
| `seed` | Seed identifier or `all` for aggregated |
| `episodes` | Number of evaluated episodes |
| `instant_success_rate` | Proportion of episodes ending in success |
| `score_win_rate` | Proportion where ego score > target score |
| `mean_return` | Average episode return |
| `mean_final_range_m` | Average final range |
| `mean_final_ata_deg` | Average final ATA |
| `prediction_rmse_m` | Root-mean-square prediction error (m) |
| `prediction_fallback_rate` | Rate of prediction fallback steps |
| `timeout_rate` | Episode timeout rate |
| `crash_rate` | Episode crash rate |
| `out_of_bounds_rate` | Episode out-of-bounds rate |

## Smoke Test Commands

```bash
# CV prediction training smoke
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
    --config config/experiment/train_vpp_ppo_cv.yaml --smoke

# CA prediction training smoke
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo \
    --config config/experiment/train_vpp_ppo_ca.yaml --smoke

# Three-method comparison evaluation
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison \
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml \
    --backend simple --episodes 3 --seeds 0 1
```

## Scope and Limitations

- **Current stage**: validation is performed in the `SimplePointMass` / 3DoF
  simplified environment. This is sufficient to verify the prediction mechanism,
  buffer update, feature building, and anchor translation.
- **JSBSim high-fidelity validation** is intentionally left to a later stage.
  The JSBSim data dependency (`<JSBSIM_ROOT>/envs/JSBSim/data`) is
  not available in CI, so JSBSim tests are skipped when the directory is missing.
- **No claim of superiority**: CV/CA may or may not outperform the no-prediction
  baseline depending on scenario. The framework is designed to measure this, not
  to assume it.

## Test Coverage

See `tests/test_classical_prediction.py` for:
- CV/CA formula correctness
- Field compatibility (`position_neu` / `position_m`, `velocity_ned` / `velocity_vector_mps`)
- Fallback behavior and `fallback_reason` stability
- NaN/Inf protection in CA acceleration estimation
- Adapter update/predict chain
- Environment prediction anchor integration
- Info field presence (`prediction_valid`, `predicted_target_position`, etc.)
