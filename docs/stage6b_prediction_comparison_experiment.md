# Stage 6B: Prediction Comparison Experiment

**Goal**: Determine whether CV/CA predicted anchors improve VPP performance over the No-Prediction baseline under identical training budgets (200k steps, multi-seed), and identify which scenarios benefit.

**Status**: Scripts production-ready for remote execution. Do NOT run full training on local machine due to hardware stability concerns.

---

## 1. Overview

Three methods are compared:

| Method | Predictor | Anchor Mode | Config |
|--------|-----------|-------------|--------|
| No-Prediction | None | `current_target` | `train_no_prediction_vpp_ppo.yaml` |
| CV-Prediction | Constant Velocity | `predicted_target` | `train_vpp_ppo_cv.yaml` |
| CA-Prediction | Constant Acceleration | `predicted_target` | `train_vpp_ppo_ca.yaml` |

All methods use identical PPO hyperparameters, observation space, action space, and reward function. The only difference is whether the virtual point anchor is the current target position or a predicted future target position.

---

## 2. Training

### 2.1 Single-seed training (smoke test)

```powershell
# No-Prediction
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
    --config config/experiment/train_no_prediction_vpp_ppo.yaml `
    --smoke

# CV-Prediction
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
    --config config/experiment/train_vpp_ppo_cv.yaml `
    --smoke

# CA-Prediction
python -m uav_vpp_guidance.training.train_prediction_vpp_ppo `
    --config config/experiment/train_vpp_ppo_ca.yaml `
    --smoke
```

### 2.2 Full multi-seed training (run on remote machine)

```powershell
.\scripts\train_stage6b.ps1
```

This runs 3 seeds (0, 1, 2) x 3 methods = 9 training runs at 200k steps each.

**Expected outputs per run:**
- `checkpoints/best.pt` — best policy by evaluation return
- `checkpoints/last.pt` — final policy
- `episode_train_log.csv` — per-episode statistics
- `update_train_log.csv` — per-PPO-update statistics
- `eval_log.csv` — periodic evaluation statistics

---

## 3. Evaluation

### 3.1 Random scenario evaluation

Evaluate all methods with their respective checkpoints on randomly sampled scenarios:

```powershell
# Set checkpoint paths (edit defaults in script or use env vars)
$env:NO_PRED_CKPT = "outputs/experiments/stage6b_no_pred_s0/checkpoints/best.pt"
$env:CV_CKPT      = "outputs/experiments/stage6b_cv_s0/checkpoints/best.pt"
$env:CA_CKPT      = "outputs/experiments/stage6b_ca_s0/checkpoints/best.pt"

.\scripts\eval_stage6b.ps1 -backend simple -episodes 50 -scenarioSet all
```

### 3.2 Fixed scenario evaluation

Evaluate on the same fixed scenarios for direct comparison:

```powershell
.\scripts\eval_stage6b.ps1 -backend simple -episodes 50 -scenarioSet fixed
```

This uses scenarios: `favorable`, `neutral`, `disadvantage`, `challenging`.

### 3.3 Per-method checkpoint override via CLI

You can also run evaluation directly without the wrapper script:

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_prediction_comparison `
    --config config/experiment/evaluate_vpp_prediction_comparison.yaml `
    --method-checkpoint no_prediction=path/to/no_pred_best.pt `
    --method-checkpoint cv_prediction=path/to/cv_best.pt `
    --method-checkpoint ca_prediction=path/to/ca_best.pt `
    --backend simple `
    --episodes 50 `
    --seeds 0 1 2 `
    --scenarios favorable neutral disadvantage challenging `
    --output-dir outputs/tables/stage6b_simple_fixed
```

**Key arguments:**
- `--method-checkpoint method=path` — per-method checkpoint (repeatable)
- `--scenarios` — fixed scenario names (omit for random sampling)
- `--save-trajectories` — save per-episode trajectory CSVs

**Output files:**
- `prediction_metrics.json` — full metrics with per-scenario breakdown
- `prediction_metrics.csv` — scalar metrics per method
- `{method}_scenario_metrics.csv` — per-scenario metrics per method
- `trajectories/{method}/` — per-episode trajectory CSVs (if `--save-trajectories`)

---

## 4. Plotting

### 4.1 Overall comparison

```powershell
.\scripts\plot_stage6b.ps1 -backend simple -scenarioSet all
```

Generates:
- `comparison_success_rate.png`
- `comparison_score_win_rate.png`
- `comparison_final_range.png`
- `comparison_final_ata.png`
- `termination_distribution_comparison.png`

### 4.2 Per-scenario comparison

If `prediction_metrics.json` is available (auto-detected), the script also generates:
- `scenario_success_rate.png` — side-by-side bars per scenario
- `scenario_score_win_rate.png`
- `scenario_final_range.png`
- `scenario_final_ata.png`
- `scenario_min_range.png`
- `scenario_prediction_error.png`

### 4.3 Manual plotting

```powershell
python -m uav_vpp_guidance.visualization.plot_prediction_comparison `
    --metrics outputs/tables/stage6b_simple_all/prediction_metrics.csv `
    --metrics-json outputs/tables/stage6b_simple_all/prediction_metrics.json `
    --output outputs/figures/stage6b_simple_all
```

---

## 5. JSBSim Sanity Check

Load a simple-trained checkpoint into the JSBSim backend and run a small number of episodes to check stability:

```powershell
.\scripts\eval_jsbsim_sanity.ps1 `
    -checkpoint outputs/experiments/stage6b_cv_s0/checkpoints/best.pt `
    -config config/experiment/train_vpp_ppo_cv.yaml `
    -episodes 5
```

Or run directly:

```powershell
python -m uav_vpp_guidance.evaluation.evaluate_jsbsim_sanity `
    --config config/experiment/train_vpp_ppo_cv.yaml `
    --checkpoint outputs/experiments/stage6b_cv_s0/checkpoints/best.pt `
    --episodes 5 `
    --output-dir outputs/tables/jsbsim_sanity/cv
```

**Sanity checks performed:**
- Success / crash / stall / OOB / timeout rates
- Control saturation rate
- Mean control surface deflections (elevator, aileron, rudder)
- Warnings if rates exceed thresholds

---

## 6. Expected Outcomes

### 6.1 Before training (random policy)
- All methods: ~0% success, ~100% OOB or timeout
- Prediction errors large but finite for CV/CA

### 6.2 After 200k-step training (hypotheses)
- **No-Prediction**: Moderate success on favorable/neutral; struggles with crossing/challenging scenarios
- **CV-Prediction**: Better than no-pred on crossing/challenging scenarios where target motion is predictable; may over-predict on maneuvering targets
- **CA-Prediction**: Best on scenarios with acceleration (turns, climbs); may be noisy on constant-velocity segments due to finite-difference acceleration estimation

### 6.3 Scenario-specific expectations

| Scenario | Expected Best Method | Rationale |
|----------|---------------------|-----------|
| favorable | No-Prediction or CV | Target ahead, low maneuver; prediction adds little value |
| neutral | CV or CA | Head-on closure; timing matters, prediction helps |
| disadvantage | CA | Target behind with speed advantage; ownship must anticipate target motion |
| challenging | CA or CV | High lateral offset, crossing; prediction of turn direction critical |

---

## 7. Troubleshooting

### High saturation rate on JSBSim
- Expected for untrained policies
- If persists after training, check low-level controller gains in `config/guidance.yaml`
- Consider reducing `d_long_range` / `d_lat_range` to lower virtual point distances

### Prediction error is NaN
- This is expected during online evaluation because the true future target position is not stored in the environment state
- For offline error analysis, use trajectory CSVs and align timestamps with the ground-truth target trajectory

### OOB rate stays high after training
- Check that action bounds are `[-1, 1]` in policy config (P0 fix)
- Verify `VirtualPointGenerator._rescale` is mapping correctly
- Review reward shaping: `w_safety` and `terminal_failure` should penalize OOB

### Different methods have identical behavior
- Check that `trajectory_prediction.enabled` and `virtual_point.anchor_mode` are set correctly per method config
- Verify that predictor is actually being called in `tracking_env.py` step()

---

## 8. Checklist for Remote Execution

- [ ] Copy repository to remote machine
- [ ] Install dependencies: `pip install -e .`
- [ ] Verify JSBSim installation if using JSBSim backend
- [ ] Run smoke tests: `python -m pytest tests/ -x -q`
- [ ] Set `$env:PYTHONPATH` if needed
- [ ] Run `train_stage6b.ps1` (may take several hours)
- [ ] Record checkpoint paths
- [ ] Run `eval_stage6b.ps1` with recorded checkpoints
- [ ] Run `plot_stage6b.ps1`
- [ ] Run `eval_jsbsim_sanity.ps1` for each method (optional)
