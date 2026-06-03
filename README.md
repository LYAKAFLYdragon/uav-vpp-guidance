# uav-vpp-guidance

UAV Virtual Pursuit Point Guidance with LOS-rate Guidance and Strategy-Gain Bilevel Optimization

## Project Goal

This project aims to build a reproducible, extensible, and paper-ready engineering framework for close-range UAV dynamic tracking, featuring:

- **Virtual Pursuit Point (VPP) policy**: A learned policy outputs normalized parameters that define a virtual pursuit point relative to the target aircraft.
- **LOS-rate guidance law**: Generate normal overload and roll-rate commands based on line-of-sight angular rate.
- **Proportional Navigation (True 3D PN)**: Classical PN guidance with filtered LOS-rate estimation for terminal-phase stability.
- **Hybrid Guidance**: Automatic switching or continuous blending between geometric LOS-rate and PN based on engagement range / energy state.
- **Command Post-Processor**: Final saturation, terminal-phase protection, load-roll coordination, and energy compensation before actuator mapping.
- **Regret-minimized strategy-gain bilevel optimization**: Alternating optimization of the pursuit policy (strategy) and the guidance gains.
- **Target trajectory prediction**: Predict the target's future position (`f_ψ`) so the VPP anchor shifts from the current position to the predicted future position.

## Relationship with Legacy Project

- **Legacy project path**: `E:\CloseAirCombat_control`
- The legacy project is used **only as a reference library**.
- **No new research code will be added to the legacy project.**
- Historical experiment results, ACMI files, CSVs, PNGs, and temporary scripts are **not migrated** into this repository.

> ⚠️ **Important**: Smoke-test results (short training runs with 512 steps) and random-policy evaluations **must not be cited as final performance conclusions** in the paper. All quantitative claims require full multi-seed training (≥200k steps) and statistical comparison across scenarios.

## Directory Structure

```
uav-vpp-guidance/
├── config/                 # YAML configuration files
│   ├── env.yaml            # Simulation and environment settings
│   ├── ppo.yaml            # PPO hyperparameters
│   ├── guidance.yaml       # Guidance law and virtual point parameters
│   ├── gain_space.yaml     # Gain search space and optimizer settings
│   ├── reward.yaml         # Reward weights
│   ├── trajectory_prediction.yaml  # Target trajectory prediction settings
│   └── experiment/         # Experiment-specific configs
├── src/uav_vpp_guidance/   # Main Python package
│   ├── envs/               # JSBSim wrapper, tracking env, scenarios
│   ├── flight_control/     # Low-level controller, command filter/limiter
│   ├── virtual_point/      # VPP generation, pursuit priors, smoothing
│   ├── guidance/           # LOS-rate guidance, PN, hybrid, command post-processor, gain config
│   ├── trajectory_prediction/  # Target trajectory prediction (LSTM/GRU/Transformer)
│   ├── agents/             # PPO/SAC agents, networks, replay buffer
│   ├── gain_optimizer/     # CEM, PBT, regret, bilevel trainer
│   ├── training/           # Training entry points
│   ├── evaluation/         # Monte Carlo, metrics, ablation
│   └── utils/              # Config, seed, logger, checkpoint, plotting
├── scripts/                # PowerShell launch scripts
├── tests/                  # Unit tests
├── experiments/            # Experiment output folders (gitignored subdirs)
├── outputs/                # Global output folder (gitignored)
├── docs/                   # Documentation
└── legacy_notes/           # Migration notes
```

## Quick Start

Install in editable mode:

```powershell
pip install -e .
```

Verify import:

```powershell
python -c "import uav_vpp_guidance; print('ok')"
```

### Guidance Mode Selection

Select the guidance law via `guidance.mode` in your config:

```yaml
guidance:
  mode: los_rate          # "los_rate" | "proportional_navigation" | "hybrid"
  gains:
    k_los: 1.0
    k_roll: 1.0
    k_speed: 0.2
  params:
    distance_scale_m: 2000.0
    navigation_constant: 3.0
    hybrid_mode: range      # "range" | "energy" | "blended"
    range_threshold_m: 3000.0
```

### Capture Radius Mechanism

When the own aircraft approaches the Virtual Pursuit Point (distance < `capture_radius_m`, default 50 m), the guidance law automatically blends commands toward a safe hold state to avoid the distance singularity:

- `roll_rate_cmd` is attenuated linearly to 0 as distance → 0.
- `nz_cmd` blends toward `base_nz` (typically 1.0 g, level flight).
- Throttle remains on speed-hold logic.

Configure in `guidance.yaml`:

```yaml
guidance:
  params:
    capture_radius_m: 50.0
    enable_internal_clip: true
    enable_internal_filter: false
```

**Trade-offs**: Larger capture radius produces smoother terminal behavior but may delay fine-positioning. Smaller radius preserves responsiveness but increases numerical risk near the singularity.

Enable optional command post-processing:

```yaml
guidance:
  post_process:
    enabled: true
    enable_terminal_protection: true
    terminal_range_m: 500.0
    enable_energy_compensation: false
    enable_load_roll_coordination: false
```

### Console Scripts

After `pip install -e .`, the following entry points are available:

```powershell
# Training
uav-vpp-train-fixed-gain
uav-vpp-train-gain-only
uav-vpp-train-bilevel
uav-vpp-train-no-prediction
uav-vpp-train-no-prediction-ppo
uav-vpp-train-prediction-ppo

# Evaluation
uav-vpp-eval-no-prediction
uav-vpp-eval-prediction-comparison
uav-vpp-eval-stage6b
```

### Minimal Run Commands

Train VPP policy with fixed gains:

```powershell
.\scripts\train_fixed_gain.ps1
```

Freeze policy and optimize gains only:

```powershell
.\scripts\train_gain_only.ps1
```

Run proposed bilevel training:

```powershell
.\scripts\train_bilevel.ps1
```

Run Monte Carlo evaluation:

```powershell
.\scripts\eval_monte_carlo.ps1
```

Run Stage 6B benchmark (smoke test):

```powershell
.\scripts\run_stage6b_simple_benchmark.ps1 -Smoke
# or
python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark `
    --config config/experiment/benchmark_simple_prediction_comparison.yaml --smoke
```

## No-Prediction VPP Baseline

**No-Prediction VPP Baseline** is a runnable baseline that closes the full RL loop without trajectory prediction:

- `trajectory_prediction.enabled=false`
- `virtual_point.anchor_mode=current_target`
- No LSTM/GRU/KF/IMM predictors involved
- Validates the VPP autonomous decision framework itself

### Smoke Rollout

```powershell
python -m uav_vpp_guidance.training.train_no_prediction_vpp `
    --config config/experiment/no_prediction_vpp.yaml --smoke
```

### Evaluation

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction `
    --config config/experiment/no_prediction_vpp.yaml
```

### Rule-Based Baseline

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_no_prediction `
    --config config/experiment/rule_based_pursuit_baseline.yaml `
    --rule-mode pure_pursuit
```

See [docs/no_prediction_vpp_baseline.md](docs/no_prediction_vpp_baseline.md) for details.

## No-Prediction VPP PPO Training

Trains a PPO policy to autonomously output virtual pursuit point offsets Δp:

```powershell
# Smoke test
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml --smoke

# Full training
python -m uav_vpp_guidance.training.train_no_prediction_vpp_ppo `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml
```

### Policy Evaluation

```powershell
# Simple backend
python -m uav_vpp_guidance.evaluation.evaluate_policy `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt `
    --backend simple --episodes 10 --seeds 0 1 2 --save-trajectories

# JSBSim backend
python -m uav_vpp_guidance.evaluation.evaluate_policy `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --checkpoint outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt `
    --backend jsbsim --episodes 2 --seeds 0 --save-trajectories
```

### Training Curves

```powershell
python -m uav_vpp_guidance.visualization.plot_training_curves `
    --log-dir outputs/experiments/no_prediction_vpp_ppo/logs `
    --output outputs/experiments/no_prediction_vpp_ppo/figures
```

See [docs/no_prediction_vpp_ppo_training.md](docs/no_prediction_vpp_ppo_training.md) for details.

## Stage 6A: Classical CV/CA Prediction VPP Integration

Extends the No-Prediction baseline with Constant Velocity (CV) and Constant Acceleration (CA) trajectory predictors. The virtual pursuit point anchor shifts from `current_target` to `predicted_target`:

```
Pos_Virtual = Pos_Target_Pred + Δp
```

### Training

```powershell
# CV Prediction
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
    --config config/experiment/train_vpp_ppo_cv.yaml --smoke

# CA Prediction
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
    --config config/experiment/train_vpp_ppo_ca.yaml --smoke
```

### Prediction Comparison Evaluation

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison `
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml `
    --backend simple --episodes 10 --seeds 0 1 2 --save-trajectories
```

### Prediction Comparison Plots

```powershell
python -m uav_vpp_guidance.visualization.plot_prediction_comparison `
    --metrics outputs/tables/prediction_comparison/simple/prediction_metrics.csv `
    --trajectories outputs/trajectories/prediction_comparison/simple `
    --output outputs/figures/prediction_comparison/simple
```

See [docs/classical_prediction_vpp_integration.md](docs/classical_prediction_vpp_integration.md) for details.

## Trajectory Prediction Module

The trajectory prediction module (`trajectory_prediction/`) upgrades the VPP anchor from the target's **current position** to the target's **predicted future position**:

```
Pos_Virtual = Pos_T_pred + Δp
```

### Supported Predictors

| Model | Description | Status |
|---|---|---|
| `ConstantVelocityPredictor` | Physics baseline: `Pos + Vel * T` | ✅ |
| `LSTMTrajectoryPredictor` | Stacked LSTM + MLP head | ✅ Skeleton |
| `GRUTrajectoryPredictor` | Stacked GRU + MLP head | ✅ Skeleton |
| Transformer | Temporal attention encoder | 🔜 Interface reserved |

### Anchor Modes

- `current_target`: VPP anchored at target current position (legacy behavior).
- `constant_velocity`: VPP anchored at `Pos_T + Vel * T_lookahead`.
- `predicted_target`: VPP anchored at model-predicted future position.

### Configuration

See `config/trajectory_prediction.yaml` for hyperparameters:
- `history_len`: past frames used for prediction (default 10).
- `lookahead_time_s`: prediction horizon (default 1.0 s).
- `model.type`: `lstm`, `gru`, or future `transformer`.

## Experiment Naming Convention

| ID | Name | Description |
|---|---|---|
| 001 | baseline_fixed_gain | Fixed gains, no virtual point |
| 002 | fixed_gain_vpp | Fixed gains with virtual pursuit point |
| 003 | gain_only | Frozen VPP policy + CEM gain optimization |
| 004 | bilevel | Proposed strategy-gain bilevel optimization |
| 005 | ablation | Ablation studies (no regret, no gain obs, no safety penalty) |

## Current Migration Status

See `docs/legacy_mapping.md` and `legacy_notes/files_to_migrate.md` for detailed migration plans.

### Phase 1 (Completed): Framework
- [x] Clean project structure
- [x] Core class skeletons with clear interfaces
- [x] Configuration system (YAML)
- [x] Unit test skeletons
- [x] Legacy mapping documentation

### Phase 2 (Completed): P1 Core Migration
- [x] JSBSim minimal closed-loop wrapper
- [x] Fixed-gain pursuit baseline (env skeleton)
- [x] Virtual pursuit point generator (interface)

### Phase 3 (Completed): Trajectory Prediction
- [x] Trajectory prediction module framework
- [x] Constant velocity / LSTM / GRU predictors
- [x] Predictor adapter integrated with VPP generator
- [ ] Episode-based supervised training
- [ ] Transformer predictor

### Phase 4 (Completed): No-Prediction VPP Baseline
- [x] SimplePointMassEnv for smoke testing
- [x] CloseRangeTrackingEnv full closed loop
- [x] LOSRateGuidance with command limiter/filter
- [x] RewardCalculator with range/angle/safety/saturation/smooth terms
- [x] TerminationChecker with success/crash/timeout/out_of_bounds
- [x] RuleBasedPursuitPolicy (pure/lag/lead)
- [x] Smoke rollout and evaluation scripts
- [x] All tests passing

### Phase 5 (Completed): No-Prediction VPP PPO Baseline
- [x] PPO agent with MLP Actor-Critic
- [x] Rollout buffer with GAE
- [x] Training loop with evaluation and checkpointing
- [x] Policy evaluation on both simple and JSBSim backends
- [x] Training curve plotting
- [x] All tests passing

### Phase 6A (Completed): Classical CV/CA Prediction VPP Integration
- [x] Constant Velocity predictor baseline
- [x] Constant Acceleration predictor baseline
- [x] Predictor adapter with buffer update / feature build / prediction chain
- [x] Environment prediction anchor integration (predicted_target)
- [x] Ablation: No-Prediction vs CV vs CA on SimplePointMass

### Phase 6B (Completed): Full Simple-Backend Benchmark
- [x] Fixed-scenario benchmark (favorable / neutral / disadvantage / challenging)
- [x] Multi-seed statistical comparison (bootstrap CI, paired delta)
- [x] Automated summary.md generation with terminal-phase stability metrics
- [x] Smoke vs full benchmark runner
- [x] Command variance and limit-exceedance tracking in terminal phase
- [ ] Full multi-seed training (≥200k steps) for paper-grade results

### Phase 7 (In Progress): JSBSim High-Fidelity Validation

**Status**: Guidance diversity smoke-tested on JSBSim F-16. No NaN/Inf issues.

Quick comparison (random policy, 3 seeds × 3 episodes):

| Mode | Success | Term NZ Var | Term Roll Var | NZ Exceed | Roll Exceed |
|------|---------|-------------|---------------|-----------|-------------|
| los_rate | 33.3% | 0.0362 | ~0 | 0.00% | 0.00% |
| proportional_navigation | 33.3% | 0.0717 | ~0 | 0.00% | 0.00% |
| hybrid | 33.3% | 0.0717 | ~0 | 0.00% | 0.00% |

Run full comparison (requires JSBSim backend):
```powershell
python scripts/eval_jsbsim_guidance_comparison.py --seeds 0 1 2 --episodes 3 --require-backend jsbsim
```

- [x] Guidance mode ablation smoke test (geometric vs PN vs hybrid)
- [x] Terminal-phase command stability metrics on JSBSim
- [ ] Full JSBSim dynamics and scenario migration
- [ ] LSTM/GRU predictor training and integration
- [ ] Gain-only CEM optimization
- [ ] Strategy-gain bilevel training
- [ ] Terminal-phase command saturation analysis with high-fidelity actuator model

> **Warning**: Smoke benchmark results are for mechanism validation only and must
> not be presented as final paper conclusions. Full runs with sufficient seeds
> and episodes are required for statistical claims.
