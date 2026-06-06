# Stage 6H.0-R: Stage 6F Historical Success Baseline

**Export date**: 2026-06-06T08:58:45.032346

## 1. Source Config

- **File**: `config\experiment\stage6f5_feasible_geometry.yaml`
- **Experiment**: stage6f5_feasible_geometry
- **Backend**: simple
- **JSBSim**: False

## 2. Success Criteria

- **success_range_m**: 900.0
- **success_ata_deg**: 25.0
- **success_hold_time_s**: 0.2
- **hysteresis_range_m**: 950.0
- **hysteresis_ata_deg**: 30.0
- **max_range_m**: 12000.0
- **episode horizon**: 512 steps × 0.2s = 102.4s

## 3. Scenarios

| Scenario | Range (m) | Ego (m/s) | Target (m/s) | Closure (m/s) | Aspect | Expected |
|---|---|---|---|---|---|---|
| favorable | 800.0 | 250.0 | 180.0 | 70.0 | 0.0° vs 0.0° | True |
| neutral | 2000.0 | 200.0 | 200.0 | 400.0 | 0.0° vs 180.0° | True |
| disadvantage | 721.1 | 200.0 | 220.0 | 30.0 | 0.0° vs 30.0° | marginal |
| challenging | 2121.3 | 200.0 | 210.0 | 350.0 | 45.0° vs 225.0° | True |

## 4. Checkpoints

| Method | Checkpoint Path | Exists | Size | MD5 |
|---|---|---|---|---|
| no_prediction | `outputs\experiments\no_prediction_vpp_ppo\checkpoints\best.pt` | ❌ MISSING | N/A | N/A |
| cv_prediction | `outputs\experiments\vpp_ppo_cv_prediction\checkpoints\best.pt` | ✅ | 246317 | 83d1990abbdb2b9d... |
| ca_prediction | `outputs\experiments\vpp_ppo_ca_prediction\checkpoints\best.pt` | ✅ | 246317 | 9db8f4bfeff95df5... |
| lstm_frozen | `outputs\experiments\vpp_ppo_lstm_frozen\checkpoints\best.pt` | ✅ | 248301 | ebaeecb9f416b0c1... |
| gru_frozen | `outputs\experiments\vpp_ppo_gru_frozen\checkpoints\best.pt` | ✅ | 248301 | 869c189190bd3236... |

## 5. Critical Observations

### 5.1 Favorable geometry uses 800m range, not 2000m
Stage 6F.5A `favorable` has `initial_range_m: 800.0` with ego 250 m/s vs target 180 m/s.
This is a **close-range tail-chase with strong speed advantage**, not a long-range intercept.

### 5.2 Original no_prediction checkpoint is MISSING
`outputs/experiments/no_prediction_vpp_ppo/checkpoints/best.pt` does not exist.
Current 6H.0-lite uses `no_prediction_vpp_ppo_seed0/checkpoints/best.pt` instead.
This is a **checkpoint drift** that may affect VPP policy behavior.

### 5.3 All Stage 6F scenarios have small initial range or high closure rate
- favorable: 800m, closure 70 m/s
- neutral: 2000m head-on, closure 400 m/s
- disadvantage: ~721m, crossing with lateral offset
- challenging: ~2121m, crossing with high closure

None of these are `aspect≥30°` with `range≥1200m` in the sense tested by 6H.0 baseline search.
The 6H.0 search grid may be **too sparse or too large-range** to capture feasible non-tail-chase geometries.

## 6. Paper-Safe Note

> Results limited to documented configs. No claim is made about universal feasibility.
> The discrepancy between Stage 6F success and 6H.0 baseline search failure requires further audit.
