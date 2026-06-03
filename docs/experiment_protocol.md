# Experiment Protocol

This document defines the standardized experiment protocol for the UAV-VPP-Guidance project. It serves as the canonical reference for naming conventions, configuration structure, and phase definitions on the `main` branch.

## Phase Definitions

| Phase | Name | Scope | Backend | Status on main |
|-------|------|-------|---------|----------------|
| P1 | Framework | Core class skeletons, config system, unit tests | — | Stable |
| P2 | Core Migration | JSBSim minimal closed loop, fixed-gain baseline | JSBSim | Stable |
| P3 | No-Prediction Baseline | SimplePointMassEnv, VPP loop without prediction | Simple | Stable |
| P4 | No-Prediction PPO | PPO policy training for VPP offsets | Simple / JSBSim | Stable |
| P5 | Classical Prediction | CV/CA predictors, predictor adapter | Simple | Stable |
| P6A | Prediction Hardening | Input validation, NaN guards, anchor tests | Simple | Stable |
| P6B | Simple Benchmark | Multi-seed benchmark: no_pred vs CV vs CA | Simple | Stable |
| P7 | JSBSim Validation | High-fidelity dynamics, guidance ablation, LSTM | JSBSim | Planned |

> **Note**: Phase 7 will receive guidance-diversity features (True PN, Hybrid, CommandPostProcessor) via a future PR from `feature/los-guidance-deep-hardening`. The `main` branch protocol is designed to accommodate these additions without structural changes.

## Experiment Naming Convention

Experiments are numbered sequentially and stored under `experiments/` and `outputs/experiments/`.

| ID | Directory | Description |
|----|-----------|-------------|
| 001 | `baseline_fixed_gain` | Fixed gains, no virtual point |
| 002 | `fixed_gain_vpp` | Fixed gains with virtual pursuit point |
| 003 | `gain_only` | Frozen VPP policy + CEM gain optimization |
| 004 | `bilevel` | Proposed strategy-gain bilevel optimization |
| 005 | `ablation` | Ablation studies (no regret, no gain obs, no safety) |

## Configuration Hierarchy

```
config/
├── env.yaml              # Simulation settings (dt, freq, termination thresholds)
├── ppo.yaml              # PPO hyperparameters
├── guidance.yaml         # Guidance law parameters (gains, limits)
├── reward.yaml           # Reward weights
├── gain_space.yaml       # Gain search space for CEM
├── trajectory_prediction.yaml  # Predictor hyperparameters
└── experiment/
    ├── no_prediction_vpp.yaml              # P3 baseline
    ├── no_prediction_vpp_jsbsim.yaml       # P3 baseline (JSBSim)
    ├── train_no_prediction_vpp_ppo.yaml    # P4 training
    ├── train_vpp_ppo_cv.yaml               # P5 CV training
    ├── train_vpp_ppo_ca.yaml               # P5 CA training
    ├── benchmark_simple_prediction_comparison.yaml  # P6B benchmark
    └── proposed_bilevel.yaml               # P4 bilevel training
```

### Backend Selection

Every experiment config should explicitly declare its backend:

```yaml
# Simple backend (3DoF point mass, fast, for smoke tests and PPO training)
backend: simple
env:
  use_jsbsim: false

# JSBSim backend (F-16 high-fidelity, slow, for validation)
backend: jsbsim
env:
  use_jsbsim: true
  aircraft_model: f16
  legacy_project_root: "E:/CloseAirCombat_control"
```

## Smoke Test Policy

All new experiment configs **must** support a `--smoke` flag that runs a minimal version (≤2 episodes, 1 seed) for CI validation. Smoke tests should complete in <30 seconds on the simple backend.

## Seed Management

1. The master seed is set in the config (`experiment.seed`).
2. Per-episode seeds are derived deterministically: `ep_seed = seed * 10000 + episode_idx`.
3. `uav_vpp_guidance.utils.seed.set_seed()` must be called before each seed block.
4. For paper experiments, use at least 3 seeds (preferably 5).
