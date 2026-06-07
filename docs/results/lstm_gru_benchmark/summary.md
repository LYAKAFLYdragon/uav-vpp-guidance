# UAV VPP Guidance — Paper Benchmark Report

**Date**: 2026-06-07 23:36
**Git Commit**: `36809d7` (dirty=True, branch=main)
**Backend**: simple
**Config**: config/experiment/stage6f5_feasible_geometry.yaml
**Methods**: no_prediction, lstm_frozen, gru_frozen
**Scenarios**: all
**Seeds**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
**Allow Random Smoke**: False
**Run Manifest**: `docs\results\lstm_gru_benchmark\run_manifest.json`

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

### lstm_frozen
- Success Rate: 75.00% (60/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: 5ad75adb22e51d4f
- Method Override: lstm_frozen
- Prediction Mode: lstm_frozen
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/vpp_ppo_lstm_frozen/checkpoints/best.pt
- Checkpoint Source: config_method
- Checkpoint Exists: True

### gru_frozen
- Success Rate: 62.50% (50/80)
- Config Path: config/experiment/stage6f5_feasible_geometry.yaml
- Resolved Config Hash: abfa91249f2a5323
- Method Override: gru_frozen
- Prediction Mode: gru_frozen
- Guidance Mode: los_rate
- Gain Source: default
- Checkpoint Path (Final): outputs/experiments/vpp_ppo_gru_frozen/checkpoints/best.pt
- Checkpoint Source: config_method
- Checkpoint Exists: True

## Statistical Comparison
See `tables/comparison_table.md` for paired t-test and Cohen's d.

## Figures
See `figures/` directory.

## Reproducibility
Exact command used to produce this run:

```bash
scripts/run_paper_benchmark.py --config config/experiment/stage6f5_feasible_geometry.yaml --backend simple --methods no_prediction lstm_frozen gru_frozen --seeds 0 1 2 3 4 5 6 7 8 9 --scenarios all --output-dir docs/results/lstm_gru_benchmark
```
