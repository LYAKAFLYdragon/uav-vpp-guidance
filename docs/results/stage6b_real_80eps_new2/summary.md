# UAV VPP Guidance — Paper Benchmark Report

**Date**: 2026-06-08 07:26
**Git Commit**: `0c83e0d` (dirty=True, branch=main)
**Backend**: simple
**Config**: config/experiment/stage6f5_feasible_geometry.yaml
**Methods**: no_prediction, cv_prediction, ca_prediction
**Scenarios**: all
**Seeds**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Allow Random Smoke**: False
**Run Manifest**: `docs\results\stage6b_real_80eps_new2\run_manifest.json`

## Benchmark Type

- ✅ **PAPER-SAFE BENCHMARK**: All checkpoints and gains loaded successfully.

## Results Summary

### no_prediction
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 2665cdd9a6bf06c8
- Method Override: no_prediction
- Prediction Mode: no_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt
- Checkpoint Source: cli_checkpoint_map
- Checkpoint Exists: True

### cv_prediction
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 23eeb651a7aef4b1
- Method Override: cv_prediction
- Prediction Mode: cv_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/stage6b_cv_s1/checkpoints/best.pt
- Checkpoint Source: cli_checkpoint_map
- Checkpoint Exists: True

### ca_prediction
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 7f4f55ca48bc9eea
- Method Override: ca_prediction
- Prediction Mode: ca_prediction
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/stage6b_ca_s1/checkpoints/best.pt
- Checkpoint Source: cli_checkpoint_map
- Checkpoint Exists: True

## Statistical Comparison
See `tables/comparison_table.md` for paired t-test and Cohen's d.

## Figures
See `figures/` directory.

## Reproducibility
Exact command used to produce this run:

```bash
scripts/run_paper_benchmark.py --config config/experiment/stage6f5_feasible_geometry.yaml --backend simple --methods no_prediction cv_prediction ca_prediction --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios all --output-dir docs/results/stage6b_real_80eps_new2 --checkpoint-map no_prediction=outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt --checkpoint-map cv_prediction=outputs/experiments/stage6b_cv_s1/checkpoints/best.pt --checkpoint-map ca_prediction=outputs/experiments/stage6b_ca_s1/checkpoints/best.pt
```
