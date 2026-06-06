# Stage 6H.0-F: Formal Regression Baseline Set

**Date**: 2026-06-06
**Source**: Stage 6F.5 replay with current code + available checkpoints
**Purpose**: Provide validated non-tail-chase VPP baselines for formal threshold search acceptance criteria.

---

## 1. Baseline Scenarios

### 1.1 Challenging (Head-on / Crossing)

| Parameter | Value |
|---|---|
| scenario_type | head_on |
| initial_range_m | 2121.3 |
| ego_speed_mps | 200.0 |
| target_speed_mps | 210.0 |
| aspect_angle_deg | 180.0 |
| altitude_diff_m | 200.0 |

**Replay result**: 100% success across all 5 methods (no_prediction, cv, ca, lstm, gru).
**Why included**: This is a genuine non-tail-chase VPP success. Mode-switch gate (aspect threshold ~15°) will NOT activate here, so it serves as a pure VPP regression check.

### 1.2 Neutral (Head-on)

| Parameter | Value |
|---|---|
| scenario_type | head_on |
| initial_range_m | 2000.0 |
| ego_speed_mps | 200.0 |
| target_speed_mps | 200.0 |
| aspect_angle_deg | 180.0 |
| altitude_diff_m | 0.0 |

**Replay result**:
- no_prediction: 0% (crash)
- cv_prediction: 100%
- ca_prediction: 100%
- lstm_frozen: 100%
- gru_frozen: 100%

**Why included with method filter**: no_prediction fails, but prediction-enabled methods succeed. Baseline includes only methods with ≥80% success.

---

## 2. Excluded Scenarios

| Scenario | Reason |
|---|---|
| favorable | 0% success all methods — not a positive baseline |
| disadvantage | 0% success all methods — not a positive baseline |

---

## 3. Baseline Rows

| scenario_id | method | baseline_success_rate | guidance_mode | checkpoint |
|---|---|---|---|---|
| challenging | no_prediction | 100% | los_rate | no_prediction_vpp_ppo_seed0 |
| challenging | cv_prediction | 100% | los_rate | vpp_ppo_cv_prediction |
| challenging | ca_prediction | 100% | los_rate | vpp_ppo_ca_prediction |
| challenging | lstm_frozen | 100% | los_rate | vpp_ppo_lstm_frozen |
| challenging | gru_frozen | 100% | los_rate | vpp_ppo_gru_frozen |
| neutral | cv_prediction | 100% | los_rate | vpp_ppo_cv_prediction |
| neutral | ca_prediction | 100% | los_rate | vpp_ppo_ca_prediction |
| neutral | lstm_frozen | 100% | los_rate | vpp_ppo_lstm_frozen |
| neutral | gru_frozen | 100% | los_rate | vpp_ppo_gru_frozen |

---

## 4. Usage in Formal Threshold Search

```bash
python scripts/run_stage6h0_lite_threshold_search.py \
  --mode formal \
  --regression-baseline-file docs/results/stage6h0f_formal_regression_baseline.csv \
  --sample-size 60 \
  --sampling-method latin_hypercube \
  --seed 0 \
  --output-dir outputs/stage6h0f_threshold_formal_lhs60
```

Acceptance criteria:
- candidate_success_rate ≥ 95%
- regression_degradation ≤ 5 percentage points (vs this baseline)
- false_activation_rate ≤ 5%

---

> **Paper-safe note**: Baseline derived from replay, not new experiments. Scope limited to tested scenarios and checkpoints.
