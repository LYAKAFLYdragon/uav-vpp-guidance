# Benchmark Plan

This document outlines the planned benchmarks for Phases 6B and 7 on `main`.

## Phase 6B: Simple-Backend Prediction Comparison (Current)

**Goal**: Compare no-prediction, CV-prediction, and CA-prediction anchors on the SimplePointMass backend.

**Runner**: `uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark`

**Config**: `config/experiment/benchmark_simple_prediction_comparison.yaml`

**Metrics**:
- Success rate, crash rate, timeout rate, out-of-bounds rate
- Mean return ± std
- Mean final range / ATA
- Prediction RMSE and fallback rate
- Per-scenario breakdown

**Planned runs**:
| Run | Episodes | Seeds | Scenarios | Purpose |
|-----|----------|-------|-----------|---------|
| Smoke | 2 | 0 | favorable, neutral | CI validation |
| Small | 3 | 0,1 | all | Quick trend check |
| Full | 20 | 0-4 | all | Paper statistics |

## Phase 7: JSBSim High-Fidelity Validation (Planned)

**Goal**: Validate the best-performing simple-backend configuration on JSBSim F-16 dynamics.

**Pre-requisites**:
1. Merge `feature/los-guidance-deep-hardening` (guidance diversity + post-processing).
2. Train a policy checkpoint on simple backend.
3. Load checkpoint and evaluate on JSBSim backend.

**Benchmarks**:

### 7.1 Backend Consistency Check
Run the same scenarios on both backends with identical seeds and compare:
- Success rate delta
- Mean return delta
- Terminal range consistency

### 7.2 Guidance Mode Ablation
After feature merge, compare the three guidance modes:
- `los_rate` (geometric)
- `proportional_navigation` (True PN)
- `hybrid` (range/energy/blended switching)

Metrics:
- Terminal-phase command variance
- Limit exceedance rate
- Energy bleed rate
- Success rate by engagement geometry

### 7.3 Predictor Ablations
- No prediction (baseline)
- Constant Velocity
- Constant Acceleration
- LSTM (after training)
- GRU (after training)

### 7.4 Training Scalability
- Small: 10k steps (smoke)
- Medium: 100k steps (tuning)
- Full: 200k+ steps (paper)

## Output Directory Convention

```
outputs/
├── benchmark/
│   └── stage6b_simple_prediction/
│       ├── prediction_metrics.json
│       ├── prediction_metrics.csv
│       ├── scenario_metrics.csv
│       ├── summary.md
│       └── run_metadata.json
├── experiments/
│   └── {experiment_name}/
│       ├── checkpoints/
│       ├── logs/
│       ├── figures/
│       └── trajectories/
└── tables/
    └── {backend}_{scenario_set}/
```

## Console Scripts

After `pip install -e .`:

```bash
# Stage 6B
uav-vpp-eval-stage6b --config config/experiment/benchmark_simple_prediction_comparison.yaml --smoke

# Stage 7 (planned, requires trained checkpoint)
uav-vpp-eval-jsbsim-sanity --checkpoint path/to/best.pt --episodes 10
```
