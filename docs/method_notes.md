# Method Notes

## Virtual Pursuit Point (VPP)

The policy outputs a 5-dimensional normalized action vector, mapped to:
- Longitudinal offset `d_long`
- Lateral offset `d_lat`
- Vertical offset `d_vert`
- Prediction time `tau_pred`
- Speed bias `speed_bias`

These parameters define a virtual point relative to the target aircraft, which the own aircraft attempts to track.

## Guidance Laws

### LOS-Rate Guidance (Geometric)

The default guidance law computes:
- Normal overload command `nz_cmd` from LOS elevation and distance scaling.
- Roll-rate command `roll_rate_cmd` from heading error and current roll damping.
- Throttle command from speed-error feedback.

Strengths: intuitive geometric interpretation, good for midcourse pursuit.
Weaknesses: can produce high command variance in terminal phase when distance → 0.

**Numerical Hardening**: The implementation uses:
- `EPS = 1e-9` with `d_safe = max(d, EPS)` to avoid division-by-zero.
- `np.arctan2` instead of `np.arcsin` for elevation (no endpoint singularity at ±90°).
- `_stable_angle_diff` via `atan2(sin, cos)` for robust heading-error wrapping, even after 1000π cumulative drift.
- **Capture radius blending**: when `d < capture_radius_m` (default 50 m), commands fade to safe hold (`nz → base_nz`, `roll_rate → 0`).
- NaN/Inf fallback: if any command becomes non-finite, the system instantly outputs safe hold commands.

### Proportional Navigation (True 3D PN)

Classical PN guidance with LOS-rate estimation via filtered numerical differentiation:
- `a_cmd = N * Vc * d(lambda)/dt` perpendicular to the LOS.
- Decomposed into `nz_cmd` (vertical acceleration) and `roll_rate_cmd` (heading turn).

Strengths: theoretically optimal for intercept, smoother terminal-phase commands.
Weaknesses: requires LOS-rate filtering; sensitive to measurement noise.

### Hybrid Guidance

Switches or blends between geometric LOS-rate and PN based on engagement conditions:
- **Range mode**: pure PN for long range (> threshold), pure LOS for short range.
- **Energy mode**: switches to LOS when speed drops below threshold (energy protection).
- **Blended mode**: continuous linear interpolation across a transition zone.

Recommended for robustness: leverages PN efficiency in midcourse and geometric precision in terminal phase.

### Command Post-Processor

Optional final processing layer (enabled via `guidance.post_process.enabled`):
- **Terminal-phase protection**: scales down aggressive commands when range < threshold.
- **Load-roll coordination**: reduces roll rate when `nz_cmd` nears its limit.
- **Energy compensation**: boosts throttle when high g-load or low speed is detected.
- **Saturation**: clips all commands to configured limits.

## Strategy-Gain Bilevel Optimization

Outer loop: optimize guidance gains via CEM using regret.
Inner loop: train VPP policy with fixed gains via PPO.

Alternation continues until convergence or budget exhaustion.

## Terminal-Phase Behavior Comparison

| Aspect | Geometric LOS-Rate | True PN | Hybrid (Blended) |
|--------|-------------------|---------|------------------|
| Midcourse efficiency | Moderate | High | High (PN dominant) |
| Terminal smoothness | Can oscillate | Smoother | Smoother (LOS damping) |
| Energy awareness | No | No | Yes (energy mode) |
| Limit exceedance | Higher near capture | Lower | Lowest |
| Tuning complexity | Low | Medium (filter alpha, N) | Medium |

## Trajectory Prediction

### Coordinate System Convention

All trajectory prediction internal logic uses **NEU** (North-East-Up) exclusively:
- **NEU**: z-axis points **up**.
- **NED**: z-axis points **down** (legacy JSBSim convention).

Conversion rules (centralized in `coordinate_utils.py`):
- `velocity_ned=[vn, ve, vd]` → NEU `[vn, ve, -vd]`
- `acceleration_ned=[an, ae, ad]` → NEU `[an, ae, -ad]`

This fixes a previous bug where CV/CA predictors using NED velocity with NEU position would subtract vertical speed incorrectly (e.g., descending target predicted as ascending).

### Strict Predictor Initialization

Neural predictors (LSTM/GRU) with a configured `checkpoint_path` default to `strict_predictor_init=True`:
- Missing or corrupted checkpoints raise `RuntimeError` before training starts.
- Prevents silent fallback to untrained random weights.
- When `strict_predictor_init=False`, initialization failures emit a warning and disable prediction.

### Fallback Semantics

`TrajectoryPredictorAdapter` supports four `fallback_mode` values:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `constant_velocity` | `Pos + Vel * T` physics baseline | Safe default |
| `constant_acceleration` | `Pos + Vel * T + 0.5 * Acc * T^2` | Better when acceleration is available |
| `current_target` | Return current target position | Conservative, no extrapolation |
| `none` | Re-raise exception | Strict mode for debugging |

When fallback is triggered, `info` contains:
- `fallback=True`
- `fallback_mode`: which fallback was used
- `fallback_reason`: why the primary predictor failed
- `fallback_model`: model name of the fallback predictor
- `prediction_valid=False`

### Device Resolution

`device_utils.py` provides safe CPU/CUDA device selection:
- `resolve_torch_device(device_str, allow_fallback=True)`: resolves "cuda" → CPU if CUDA unavailable, with optional warning or strict raise.
- `load_checkpoint_to_model(model, ckpt_path, device_str, allow_device_fallback, strict)`: loads checkpoint with `map_location` set to resolved device.

### Predictor Health Metrics

During PPO training (`train_prediction_vpp_ppo.py`), the episode log tracks per-episode:
- `prediction_valid_rate`: fraction of steps with valid neural prediction.
- `fallback_rate`: fraction of steps where fallback was activated.
- `predictor_init_failed_count`: steps where predictor initialization failed.

The smoke summary JSON (`smoke_summary.json`) includes:
- `predictor_type`: "lstm", "gru", "constant_velocity", etc.
- `prediction_enabled`: true/false
- `prediction_valid_rate`: aggregated valid prediction rate
- `fallback_rate`: aggregated fallback rate
- `predictor_init_failed`: true if initialization failed at any point

### Supported Predictors

| Model | Description | Status |
|---|---|---|
| `ConstantVelocityPredictor` | Physics baseline: `Pos + Vel * T` | ✅ |
| `ConstantAccelerationPredictor` | Physics baseline with acceleration | ✅ |
| `LSTMTrajectoryPredictor` | Stacked LSTM + MLP head | ✅ Complete |
| `GRUTrajectoryPredictor` | Stacked GRU + MLP head | ✅ Complete |
| Transformer | Temporal attention encoder | 🔜 Interface reserved |


## Stage 6E.2: Predictor Telemetry & Reproducibility

### Checkpoint Strict Key

Canonical key: `checkpoint_strict`
Legacy alias: `strict_checkpoint` (still supported for backward compatibility)

If both keys are present with different values, a `ValueError` is raised to prevent silent misconfiguration.

### Fallback Phase Semantics

`fallback_phase` classifies why a fallback was triggered:

| Phase | Meaning |
|-------|---------|
| `warmup` | Neural predictor buffer not yet full (early episode steps) |
| `runtime_failure` | Primary predictor threw an exception during inference |
| `init_failure` | Predictor initialization failed at env construction |
| `configured_current_target` | Anchor mode was already `current_target` (no prediction attempted) |
| `none` | `fallback_mode=none` and primary predictor failed (re-raises) |

`post_warmup_fallback_rate` excludes warmup steps, giving a cleaner metric of runtime predictor reliability.

### Prediction Error Calculation

`PredictionErrorTracker` implements delayed error evaluation:
1. At simulation time `t`, a prediction `Pos_pred(t+T)` is registered with lookahead `T`.
2. When simulation reaches `t+T`, the tracker compares `Pos_pred(t+T)` against the actual target position.
3. `prediction_error_m = ||Pos_pred - Pos_actual||`.

This avoids the common pitfall of comparing a prediction against the *current* position (which is always wrong by construction for moving targets).

Output metrics:
- `latest_prediction_error_m`: most recent matured error
- `mean_prediction_error_m`: average over all matured errors in the episode
- `median_prediction_error_m`: median over all matured errors
- `prediction_error_count`: number of matured evaluations

### PPO Observability Fields

Per-episode CSV log (`episode_train_log.csv`) includes:
- `prediction_valid_rate`, `fallback_rate`
- `post_warmup_fallback_rate`, `warmup_fallback_rate`, `runtime_fallback_rate`
- `mean_prediction_error_m`, `prediction_error_count`

Smoke summary JSON (`smoke_summary.json`) aggregates:
- All rates above (mean across episodes)
- `predictor_init_failed`: true if any step reported init failure

### Checkpoint Reproducibility

Each trained neural predictor should have a manifest entry:

```yaml
entries:
  - model_type: lstm
    checkpoint_path: outputs/trajectory_prediction/best_model.pt
    sha256: "..."
    training_config:
      config_path: config/experiment/train_vpp_ppo_lstm_frozen.yaml
      git_commit: "36ae1cd"
    inference_params:
      history_len: 10
      lookahead_time_s: 1.0
      coordinate_frame: neu
```

Verify with:
```powershell
python scripts/verify_checkpoint_manifest.py --manifest config/trajectory_prediction/checkpoint_manifest.yaml
```

### Config Validation

Use `validate_tp_config(config, on_unknown="warn")` to catch:
- Invalid `predictor_type`, `fallback_mode`, `anchor_mode`
- Missing `checkpoint_path` when `strict_predictor_init=True` for LSTM/GRU
- Invalid `device` string
- Non-bool `checkpoint_strict`
- Unknown keys (warn or raise depending on `on_unknown`)
