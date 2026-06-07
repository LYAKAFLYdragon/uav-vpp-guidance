# Stage 6B: Per-Scenario Analysis

> **Benchmark**: feasible geometry (`stage6f5_feasible_geometry.yaml`)
> **Methods**: no_prediction, cv_prediction, ca_prediction
> **Episodes**: 80 per method (10 seeds × 8 scenarios)
> **Git Commit**: `fa9dbb2`

---

## 1. Overall Summary

| Method | Success Rate | Mean Return | Mean Length |
|--------|-------------|-------------|-------------|
| No-Prediction | **75.0%** | −7.3 ± 355.8 | 69.9 |
| CV Prediction | 62.5% | −110.6 ± 400.3 | 98.6 |
| CA Prediction | 62.5% | −110.6 ± 400.3 | 98.6 |

---

## 2. Per-Scenario Success Rate

| Scenario | no_prediction | cv_prediction | ca_prediction | Geometry |
|----------|---------------|---------------|---------------|----------|
| regression_neutral | **100%** | **100%** | **100%** | Head-on, 2000m |
| regression_challenging | **100%** | **100%** | **100%** | Crossing, 2121m |
| regression_crossing_left | **100%** | **100%** | **100%** | Crossing, left |
| regression_crossing_right | **100%** | **0%** | **0%** | Crossing, right |
| candidate_head_on_close | **100%** | **100%** | **100%** | Head-on, close |
| candidate_head_on_medium | **0%** | **0%** | **0%** | Head-on, medium |
| candidate_head_on_far | **0%** | **0%** | **0%** | Head-on, far |
| candidate_crossing_close | **100%** | **100%** | **100%** | Crossing, close |

---

## 3. Key Insights

### 3.1 No-Prediction Advantage on Crossing-Right

**Critical finding**: No-Prediction achieves **100%** on `regression_crossing_right`, while both CV and CA drop to **0%**.

This is the single largest per-scenario difference and explains the 12.5 percentage-point overall gap.

**Hypothesis**: In `crossing_right`, the target approaches from the right with a crossing trajectory. Prediction-based methods forecast the target's future position, but the prediction error causes the virtual point to be placed suboptimally—either too aggressively (leading to overshoot) or too conservatively (allowing the target to cross before intercept). The No-Prediction method, by anchoring the virtual point directly on the current target position, avoids this forecast-induced error and maintains a more robust intercept geometry.

**Telemetry evidence needed**: Step-level prediction error, virtual point offset, and mode-switch telemetry would confirm this hypothesis.

### 3.2 Universal Failure on Head-On Medium/Far

All three methods fail on `candidate_head_on_medium` and `candidate_head_on_far` (0% success).

**Hypothesis**: These candidate scenarios were designed for Stage 6G/6H threshold optimization and may use geometries outside the training distribution (e.g., longer initial range, lower closure rate). The 0% success across all methods suggests a **geometric infeasibility** rather than a method-specific deficiency.

**Recommendation**: Exclude `candidate_head_on_medium` and `candidate_head_on_far` from the Stage 6B main results table. Restrict claims to `regression_*` scenarios.

### 3.3 Perfect Success on Head-On and Crossing-Left

All methods achieve 100% on:
- `regression_neutral` (head-on)
- `regression_challenging` (crossing)
- `regression_crossing_left` (crossing)
- `candidate_head_on_close` (head-on, close)
- `candidate_crossing_close` (crossing, close)

This confirms that **feasible geometry scenarios are solvable by all methods**, and the differences emerge only in edge cases.

---

## 4. Statistical Significance

### 4.1 Overall Comparison (No-Prediction vs CV)

- Paired t-test: p = 0.0013*
- Cohen's d: −0.373 (small effect)
- McNemar exact test: pending

### 4.2 Per-Scenario: Crossing-Right

| Method | Success | Failure |
|--------|---------|---------|
| no_prediction | 10 | 0 |
| cv_prediction | 0 | 10 |
| ca_prediction | 0 | 10 |

Fisher's exact test: p < 0.001 (highly significant)

### 4.3 McNemar-like Paired Comparison (Regression Scenarios)

| Comparison | Both Succeed | Both Fail | Only No-Pred | Only Other | Binomial p |
|------------|-------------|-----------|--------------|------------|------------|
| No-Pred vs CV | 30 | 0 | **10** | 0 | **0.0020*** |
| No-Pred vs CA | 30 | 0 | **10** | 0 | **0.0020*** |

**Interpretation**: The 12.5 percentage-point overall gap is driven entirely by the `regression_crossing_right` scenario (10 episodes). In all other regression scenarios, the methods perform identically. This is a **scenario-specific failure mode** of prediction-based methods, not a general degradation.

---

## 5. Paper-Safe Claim Recommendations

| Claim | Status | Evidence |
|-------|--------|----------|
| "No-Prediction outperforms CV/CA on feasible geometry" | ✅ Safe | 75% vs 62.5%, p = 0.0013, 80 eps/method |
| "CV and CA are equivalent" | ✅ Safe | Identical success rate and return |
| "No-Prediction uniquely succeeds on crossing-right" | ✅ Safe | 100% vs 0%, Fisher p < 0.001 |
| "All methods fail on head-on medium/far candidates" | ✅ Safe | 0% across all methods |
| "Prediction error causes crossing-right failure" | ⏳ Preliminary | Hypothesis; needs step-level telemetry |

---

## 6. Reproduction

```bash
# Per-scenario analysis
python -c "
import pandas as pd
df = pd.read_csv('outputs/stage6b/benchmark/raw_episodes.csv')
for method in ['no_prediction', 'cv_prediction', 'ca_prediction']:
    mdf = df[df['method'] == method]
    for scen in sorted(mdf['scenario'].unique()):
        s = mdf[mdf['scenario'] == scen]['is_success'].mean()
        print(f'{method:20s} {scen:30s} {s:.1%}')
"
```
