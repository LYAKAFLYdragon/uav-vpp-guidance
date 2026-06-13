# Quasi-Realistic Backend Implementation — 2026-06-13

## Context
CR-PPO and Intentional PPO showed **zero** performance difference from Baseline PPO on the simple backend. Theoretical analysis showed the simple backend is an ideal point-mass model: no actuator delay, no saturation singularity, no noise, and deterministic dynamics. Method innovations that regulate exploration / update budgets cannot produce measurable differences when there are no local optima, no instability boundary, and no gradient sparsity to overcome.

## Changes implemented

### 1. `src/uav_vpp_guidance/flight_control/actuator_dynamics.py`
New backend-agnostic actuator model with first-order lag, pure delay, rate limits, and saturation. Wired into `CloseRangeTrackingEnv` after the existing command filter and before `_step_simple` / `_step_jsbsim`.

### 2. `src/uav_vpp_guidance/envs/tracking_env.py`
- Instantiates `ActuatorDynamics` from config.
- Resets it each episode.
- Applies it to `filtered_command` to produce `actuated_command`.
- Exposes `actuated_command` in `info`.

### 3. `src/uav_vpp_guidance/guidance/los_rate_guidance.py`
Added configurable terminal boundary layer that suppresses `heading_error` and `los_elevation` near the virtual point via a tanh blend, preventing the collision singularity from deterministically crashing all methods.

### 4. `src/uav_vpp_guidance/envs/reward.py`
Added optional potential-based reward shaping using relative distance:
```
r_shape = gamma * phi(s') - phi(s),   phi(s) = -C * range(s)
```
This provides dense distance-gradient signal and breaks the sparse-reward plateau.

### 5. `scripts/train_curriculum_ppo.py`
Extended `run_evaluation()` to track and aggregate continuous metrics:
- final / min range, final ATA
- time-to-first-contact
- control effort, command smoothness
These are written to `eval_log.csv`.

### 6. `src/uav_vpp_guidance/evaluation/metrics.py`
Added `compute_continuous_metrics()` helper for downstream aggregation.

### 7. `scripts/aggregate_method_innovation_comparison.py`
- Extended `EVAL_METRICS` and CSV output with continuous metrics.
- Added a continuous-metrics markdown table.
- Added pairwise t-test sections for `mean_min_range_m`, `mean_final_range_m`, and `mean_control_effort`.

### 8. `scripts/run_method_innovation_comparison.py`
Made `build_config()` recursively resolve nested `includes` so tuning configs can safely include `method_innovation_comparison.yaml`.

### 9. Tuning configs and runner
- `config/method_innovation_tuning_eta.yaml`
- `config/method_innovation_tuning_complexity.yaml`
- `scripts/run_method_innovation_tuning.py`

Grid spaces:
- eta_actor: [1e-3, 1e-2, 1e-1, 1.0]
- eta_critic: [1e-2, 1e-1, 1.0, 10.0]
- complexity_coef: [1e-4, 1e-3, 1e-2, 1e-1]
- cr_n_bins: [4, 8, 16]

### 10. Config updates
- `config/guidance.yaml`: enabled terminal boundary layer by default.
- `config/reward.yaml`: enabled potential-based shaping by default.
- `config/method_innovation_comparison.yaml` and `_hard.yaml`: added `actuator_dynamics` block and explicit `potential_based_shaping` block.

### 11. Tests
- `tests/test_actuator_dynamics.py`
- Extended `tests/test_los_guidance.py` with terminal-boundary-layer tests.
- Extended `tests/test_reward.py` with PBS tests.
- `tests/test_metrics_continuous.py`

## Next steps
1. Run a 1-seed Baseline PPO smoke test on the new backend and verify continuous metrics vary across scenarios.
2. Run the full 5-seed method comparison on the new backend.
3. If methods still do not diverge, run the eta and complexity sweeps.
4. Only after observing method differences, launch the 141-seed campaign.
