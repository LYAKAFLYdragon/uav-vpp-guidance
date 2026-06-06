# Stage 6H.0-lite: Mode-Switch Threshold Optimization Plan

**Status**: Pre-unblocked (pending near-threshold robustness smoke acceptance)  
**Date**: 2026-06-06  
**Gating stage**: 6G.5D-R  
**Scope**: Optimize mode-switch gate thresholds; NOT full bilevel gain optimization.

---

## 1. Objective

Find the optimal mode-switch gate thresholds that:
1. Maintain ≥95% success rate on confirmed candidate geometries (pt20/pt29/pt38).
2. Do not degrade performance on Stage 6F feasible geometries by >5 percentage points.
3. Minimize false activations (negative controls should not trigger latch).
4. Produce auditable telemetry for every mode-switch decision.

---

## 2. Why Threshold Optimization, Not Full Bilevel

Stage 6G.5D proved that:
- **VPP offset is the root cause** of tail-chase failure (~500m norm pushes virtual point away).
- **Latched PN mode-switch rescues** the geometry by bypassing VPP when gate conditions are met.
- The **gate threshold space** (aspect, range, closing speed) is small and interpretable.

Before committing to full bilevel (joint VPP policy + gain optimization), we must:
1. Confirm the threshold space is stable across near-threshold geometries.
2. Verify no regression on non-tail-chase feasible geometries.
3. Lock the threshold config as a **hard-coded rescue mechanism** before asking a bilevel optimizer to tune it.

---

## 3. Search Space

| Parameter | Values | Rationale |
|---|---|---|
| `aspect_enter_threshold_deg` | 10, 15, 20, 25 | Lower = earlier activation but more false positives; higher = tighter but may miss borderline tail-chase |
| `aspect_exit_threshold_deg` | 20, 30, 45, `null` | Exit hysteresis; `null` = episode latch (current default) |
| `range_enter_m` | 1500, 2000, 2500, 3000 | Maximum range for gate activation |
| `closing_speed_enter_mps` | 80, 120, 160 | Minimum closing speed required |
| `hold_policy` | `episode_latch`, `min_hold_2s`, `hysteresis_exit` | Default `episode_latch` is simplest and already proven |
| `guidance_mode_when_active` | `proportional_navigation` | Fixed; no reason to vary |
| `fallback_mode` | `los_rate`, `hybrid` | Only relevant for `mode_switch_vpp_elsewhere` variant |

**Total combinations**: 4 × 4 × 4 × 3 × 3 × 2 = **1,152 configs**

This is small enough for grid search (not requiring bilevel optimization).

---

## 4. Evaluation Protocol

### 4.1 Geometries

| Category | Geometries | Episodes | Purpose |
|---|---|---|---|
| Candidate success | pt20, pt29, pt38 | 10 per point × 3 seeds = 90 | Primary objective: maintain ≥95% success |
| Near-threshold | aspect 10°, 15°, 20°; range 1800, 2400 | 5 per point × 3 seeds = 90 | Sensitivity analysis |
| Negative control | aspect 60°, 90°; low ego speed; low closing speed | 5 per point × 3 seeds = 90 | False activation rate |
| Regression | Stage 6F favorable | 10 × 3 seeds = 30 | Non-tail-chase feasibility check |

### 4.2 Variants per Config

For each threshold config, evaluate:
- `mode_switch_vpp_elsewhere` (VPP+LOS normally, PN when gate active)
- `mode_switch_pn_no_vpp` (PN normally, PN when gate active — control)

### 4.3 Metrics

1. **Primary**: Success rate on candidate geometries
2. **Secondary**: Success rate on near-threshold geometries
3. **Guard**: False activation rate on negative controls (should be 0%)
4. **Regression**: Success rate on Stage 6F favorable (should not drop >5pp vs baseline)
5. **Audit**: `mode_switch_effective` telemetry coverage = 100%

---

## 5. Acceptance Criteria Before Real 6H

| Criterion | Threshold | Measurement |
|---|---|---|
| Candidate points remain stable | ≥95% success | pt20/pt29/pt38 across 3 seeds |
| Negative controls do not regress | ≤5% false activation | aspect 60°/90°, low speed scenarios |
| Stage 6F feasible geometries do not degrade | ≤5pp drop | favorable geometry vs VPP+LOS baseline |
| All mode-switch decisions are auditable | 100% coverage | telemetry contains `mode_switch_requested`, `effective`, `reason`, `virtual_point_source` |
| No random policy fallback allowed | 0 episodes | All episodes use trained PPO checkpoint |

**If all criteria pass**: Threshold config is locked. Proceed to Stage 6H (full bilevel) with mode-switch as a hard-coded rescue layer.

**If criteria fail**: Adjust search space or investigate why thresholds are unstable. Do NOT proceed to 6H.

---

## 6. Implementation Plan

### Step 1: Runner (Stage 6H.0-lite)
- `scripts/run_stage6h0_lite_threshold_optimization.py`
- Grid search over search space
- Parallelize by config index
- Output: `threshold_search_results.csv`, `threshold_search_summary.json`

### Step 2: Analysis
- `scripts/analyze_stage6h0_threshold_search.py`
- Pareto frontier: candidate success vs false activation rate
- Select best config

### Step 3: Validation
- Re-run best config with 10 episodes × 3 seeds on all geometries
- McNemar comparison vs baseline (VPP+LOS without mode-switch)

### Step 4: Lock Config
- Write `config/experiment/stage6h0_locked_mode_switch.yaml`
- Update all downstream runners to use locked config

---

## 7. Relation to Full Bilevel (Stage 6H)

Stage 6H.0-lite is **NOT** bilevel optimization. It is:
- A grid search over a 4-parameter threshold space.
- A robustness gate that must pass before bilevel is justified.

Stage 6H (full) will:
- Jointly optimize VPP policy gains + mode-switch thresholds.
- Use bilevel formulation (outer: threshold + gains; inner: episode success).
- Require 6H.0-lite acceptance criteria as initialization constraints.

---

## 8. Paper-Safe Claim Constraints

Allowed after 6H.0-lite:
- "Mode-switch thresholds X/Y/Z achieve ≥95% success on tested tail-chase candidates while maintaining performance on feasible non-tail-chase geometries."

NOT allowed:
- "Threshold optimization solves tail-chase generally"
- "Bilevel optimization is validated"
- "VPP is universally harmful"

---

*Last updated: 2026-06-06 | Stage 6G.5D-R in progress*
