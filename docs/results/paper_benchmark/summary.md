# UAV VPP Guidance — Paper Benchmark Report

**Date**: 2026-06-07 21:12
**Git Commit**: `fa9dbb2` (dirty=True, branch=main)
**Backend**: simple
**Config**: config/experiment/stage6f5_feasible_geometry.yaml
**Methods**: no_prediction, cv_prediction, ca_prediction, gain_only
**Scenarios**: all
**Seeds**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Allow Random Smoke**: False
**Run Manifest**: `docs\results\paper_benchmark\run_manifest.json`

## Benchmark Type

- ✅ **PAPER-SAFE BENCHMARK**: All checkpoints and gains loaded successfully.

## Results Summary

### no_prediction
- Success Rate: 75.00% (60/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 2665cdd9a6bf06c8
- Method Override: no_prediction
- Prediction Mode: no_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt
- Checkpoint Source: config_method
- Checkpoint Exists: True

### cv_prediction
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 23eeb651a7aef4b1
- Method Override: cv_prediction
- Prediction Mode: cv_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/vpp_ppo_cv_prediction/checkpoints/best.pt
- Checkpoint Source: config_method
- Checkpoint Exists: True

### ca_prediction
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 7f4f55ca48bc9eea
- Method Override: ca_prediction
- Prediction Mode: ca_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/vpp_ppo_ca_prediction/checkpoints/best.pt
- Checkpoint Source: config_method
- Checkpoint Exists: True

### gain_only
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 6f1c4aa9384daff0
- Method Override: no_prediction
- Prediction Mode: no_prediction
- Guidance Mode: los_rate
- Gain Source: cem
- Checkpoint Path (Final): outputs/audit_no_pred_final/checkpoints/best.pt
- Checkpoint Source: methods_default
- Checkpoint Exists: True
- Gains Path: outputs/gain_only_cem/cem_results.json
- Gains Exists: True
- Gains Schema Valid: True
- Loaded Gains: {'k_los': 1.8675406897443583, 'k_pos': 0.10000000149011612, 'k_roll': 0.31127766966193593, 'k_speed': 0.29957846042934183}
- Note: Same policy as no_prediction but with CEM-optimized gains

## Statistical Comparison
See `tables/comparison_table.md` for paired t-test and Cohen's d.

## Figures
See `figures/` directory.

## Reproducibility
Exact command used to produce this run:

```bash
scripts/run_paper_benchmark.py --config config/experiment/stage6f5_feasible_geometry.yaml --backend simple --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios all --methods no_prediction cv_prediction ca_prediction gain_only --output-dir docs/results/paper_benchmark
```
