# Stage 6B: Simple-Backend Prediction Benchmark

## Objective

Stage 6B provides a **reproducible benchmark framework** for comparing
no-prediction, constant-velocity (CV), and constant-acceleration (CA)
prediction anchors on the SimplePointMass backend.

> **Important**: The goal is to evaluate **mechanisms and comparative trends**,
> not to claim final performance superiority. Smoke or small-run results must
> not be presented as paper conclusions without full statistical validation.

## Methodology

### Three Methods

| Method | `trajectory_prediction.enabled` | `predictor_type` | `anchor_mode` |
|--------|--------------------------------|------------------|---------------|
| no_prediction | `false` | — | `current_target` |
| cv_prediction | `true` | `constant_velocity` | `predicted_target` |
| ca_prediction | `true` | `constant_acceleration` | `predicted_target` |

### Scenario-Wise Comparison

All three methods are evaluated on the **same scenarios and seeds**:

- **favorable**: Ego behind target with speed advantage
- **neutral**: Head-on encounter at similar speed
- **disadvantage**: Target behind ego with speed advantage
- **challenging**: High lateral offset with crossing trajectory

This ensures that observed differences are due to the prediction mechanism,
not initialization luck.

### Statistical Reporting

- **Aggregated metrics**: overall mean ± std across all episodes
- **Per-scenario metrics**: breakdown by scenario type
- **Pairwise delta**: CV/CA relative to no-prediction baseline
- **Bootstrap CI**: confidence intervals for mean return (when raw episodes available)

## Usage

### Smoke Test (CI)

```bash
python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
    --config config/experiment/benchmark_simple_prediction_comparison.yaml \
    --smoke
```

### Small Run (Quick Validation)

```bash
python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
    --config config/experiment/benchmark_simple_prediction_comparison.yaml \
    --episodes 3 --seeds 0 1 --scenarios favorable neutral
```

### Full Benchmark

```bash
python -m uav_vpp_guidance.evaluation.run_stage6b_simple_benchmark \
    --config config/experiment/benchmark_simple_prediction_comparison.yaml \
    --episodes 20 --seeds 0 1 2 3 4 \
    --scenarios favorable neutral disadvantage challenging
```

### PowerShell Script

```powershell
.\scripts\run_stage6b_simple_benchmark.ps1 -Episodes 20 -Seeds 0,1,2,3,4
```

## Output Files

All outputs are written to `outputs/benchmark/stage6b_simple_prediction/`:

| File | Description |
|------|-------------|
| `prediction_metrics.csv` | Overall aggregated metrics per method |
| `prediction_metrics.json` | Full metrics including per-scenario and per-seed breakdowns |
| `scenario_metrics.csv` | Per-scenario metrics for all methods |
| `summary.md` | Human-readable summary with tables and deltas |

## Unified Metrics Fields

| Field | Description |
|-------|-------------|
| `method` | `no_prediction`, `cv_prediction`, or `ca_prediction` |
| `scenario` | Scenario name or `all` |
| `seed` | Seed identifier or `all` |
| `episodes` | Number of evaluated episodes |
| `instant_success_rate` | Proportion ending in success |
| `score_win_rate` | Proportion where ego score > target score |
| `mean_return` | Average episode return |
| `mean_final_range_m` | Average final range |
| `mean_final_ata_deg` | Average final ATA |
| `prediction_rmse_m` | Prediction RMSE (m) |
| `prediction_fallback_rate` | Rate of prediction fallback steps |
| `timeout_rate` | Episode timeout rate |
| `crash_rate` | Episode crash rate |
| `out_of_bounds_rate` | Episode out-of-bounds rate |
| `terminal_nz_cmd_variance` | Variance of `nz_cmd` in terminal phase (last 20% steps) |
| `terminal_roll_rate_cmd_variance` | Variance of `roll_rate_cmd` in terminal phase |
| `terminal_throttle_cmd_variance` | Variance of `throttle_cmd` in terminal phase |
| `terminal_nz_limit_exceedance_rate` | Fraction of terminal steps where `nz_cmd` exceeds limits |
| `terminal_roll_rate_limit_exceedance_rate` | Fraction of terminal steps where `roll_rate_cmd` exceeds limits |
| `terminal_mean_range_m` | Mean range during terminal phase |

## Scope and Limitations

- **Backend**: SimplePointMass / 3DoF only. JSBSim high-fidelity validation is
  Stage 7.
- **Policy**: Random (untrained) PPO agent. Trained policies are Stage 6C/7.
- **Guidance mode**: Default is `los_rate`. The benchmark runner accepts any valid
  `guidance.mode` (`los_rate`, `proportional_navigation`, `hybrid`) via config.
- **Claim restriction**: CV/CA are classical baselines. Their performance may
  vary by scenario. The benchmark measures this variation rather than assuming
  universal improvement.
- **Statistical power**: Smoke runs use 2 episodes × 1 seed. Full runs should
  use ≥20 episodes × ≥5 seeds for reliable conclusions.

## Test Coverage

See `tests/test_stage6b_benchmark.py` for:
- Config loading and structure validation
- Smoke execution and output file presence
- Unified CSV field completeness
- JSON method coverage
- Summary markdown content
- Statistical comparison robustness (NaN handling, bootstrap CI, paired delta)
