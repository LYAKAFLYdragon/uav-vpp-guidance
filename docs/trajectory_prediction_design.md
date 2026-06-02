# Trajectory Prediction Module Design

## 1. Module Purpose

The trajectory prediction module (`f_ψ`) upgrades the virtual pursuit point (VPP) anchor from the **target's current position** to the **target's predicted future position**:

```
Old: Pos_Virtual = Pos_T_current + Δp
New: Pos_Virtual = Pos_T_pred    + Δp
```

By predicting where the target will be in `T_lookahead` seconds, the own aircraft can proactively maneuver toward a more favorable interception geometry, reducing terminal miss distance and improving pursuit efficiency.

## 2. Inputs and Outputs

### Inputs
- **History sequence**: past `K` frames of target/relative state features (16-dim per frame).
- **Current target state**: position, velocity, attitude (used as the baseline for relative displacement models).

### Outputs
- **Predicted target position** `Pos_T_pred`: absolute position in NEU coordinates.
- **Predicted variance** `pred_var`: optional uncertainty estimate (used for observation augmentation or risk-aware guidance).
- **Info dict**: fallback flags, model type, anchor mode.

## 3. History Window

- **Buffer**: `TrajectoryStateBuffer` maintains a fixed-length sliding window (`deque` with `maxlen=history_len`).
- **Length**: configurable (default 10 frames).
- **Sample rate**: aligned with the high-level decision frequency (default 5 Hz).
- **Padding**: when history is insufficient, repeat-first or zero padding is applied so the model always receives a fixed-size tensor.

## 4. Prediction Models

### Baseline — Constant Velocity
- Assumes target velocity remains constant over the prediction horizon.
- Zero trainable parameters, minimal computation.
- Used as fallback when learned models are unavailable or history is insufficient.

### Learned — LSTM / GRU
- Input: `[batch, history_len, input_dim]`
- Output: `[batch, 3]` relative displacement (future position minus current position).
- Architecture: stacked RNN (LSTM or GRU) → last hidden state → MLP head.
- Variance head: optional; uses `softplus` to ensure positive variance.
- Extensibility: the `BaseTrajectoryPredictor` interface allows future replacement with Transformer-based models.

### Future — Transformer
- Reserved interface: replace `nn.LSTM`/`nn.GRU` with a temporal Transformer encoder.
- No architectural changes required in the adapter or VPP generator.

## 5. Connection to Virtual Pursuit Point Generator

```
VirtualPointGenerator
  ├── anchor_mode = "current_target"       → Pos_T_current
  ├── anchor_mode = "constant_velocity"    → Pos_T_current + v*T
  └── anchor_mode = "predicted_target"     → TrajectoryPredictorAdapter.predict()
                                              → Pos_T_pred

Pos_Virtual = anchor_pos + policy_offset[Δx, Δy, Δz]
```

The `TrajectoryPredictorAdapter` wires together:
1. `TrajectoryStateBuffer` — stores historical frames.
2. `feature_builder` — constructs 16-dim normalized feature vectors.
3. `BaseTrajectoryPredictor` — performs the actual prediction.

## 6. Training

- **Supervised learning**: offline training on collected episode logs.
- **Dataset**: `TrajectoryPredictionDataset` samples `(history_seq, target_disp)` pairs.
- **Losses**:
  - `position_mse_loss`: direct position regression.
  - `relative_displacement_mse_loss`: displacement regression (current default).
  - `gaussian_nll_loss`: probabilistic regression with uncertainty.
- **Trainer**: `TrajectoryPredictorTrainer` manages epochs, validation, and checkpointing.
- **Freeze during RL**: the predictor can be frozen (`requires_grad=False`) while the VPP policy is trained via PPO, preventing instability.

## 7. Ablation Plan

| Variant | Description |
|---|---|
| No prediction | `anchor_mode = current_target` (baseline) |
| Constant velocity | `anchor_mode = constant_velocity` (physics-only) |
| LSTM prediction | `anchor_mode = predicted_target` + LSTM |
| GRU prediction | `anchor_mode = predicted_target` + GRU |
| Prediction in obs | `add_prediction_to_observation = true` |
| Prediction + uncertainty | `predict_variance = true` + uncertainty penalty |
