# Stage 6G.5B: Direct-Track / Pure-PN Probe Plan

> **Status**: In Progress. Required because Stage 6G.5A found **no feasible VPP geometry candidates** (40 points, 120 episodes, 0% success).
>
> **Rationale**: If direct-track / pure-PN also fails, the limitation is likely in the guidance/control stack or simple backend envelope, not the VPP abstraction itself.

---

## 1. Motivation

Current architecture:

```text
Policy → VPP offset (Δp) → Guidance law (LOS-rate/PN/hybrid) → Command (Nz, roll rate, throttle)
```

Stage 6G.5A showed 0% success across 40 sampled wide-geometry points. The open question is:

> **Is the failure caused by the VPP abstraction (policy offset noise, destabilization), or by the underlying guidance-law / backend envelope?**

To answer this, we bypass the VPP layer entirely and test pure guidance-law variants:

- **Direct target LOS-rate**: No VPP offset; virtual point = target current position.
- **Pure PN without VPP**: Direct proportional navigation on target position.
- **Pure pursuit**: Velocity vector always points at target.

---

## 2. Minimal Implementation Points

### 2.1 Config Flag

Add to `config/guidance.yaml`:

```yaml
guidance:
  direct_track_mode: false   # If true, ignore policy action and set virtual_point = target_position
```

### 2.2 Environment Change (`tracking_env.py`)

In `step()`, before calling `VirtualPointGenerator`:

```python
if self.env_config.get("guidance", {}).get("direct_track_mode", False):
    # Override: virtual point is exactly the anchor position (target or predicted target)
    virtual_point_m = anchor_position_m
else:
    # Normal VPP path: policy action → offset
    virtual_point_m = self.virtual_point_generator.action_to_virtual_point(
        action, anchor_position_m, own_state, target_state
    )
```

**Estimated change**: 5–10 lines in `tracking_env.py`.

### 2.3 Runner Extension

`scripts/run_stage6g5b_direct_track_smoke.py` should:

1. Load the same geometry points as Stage 6G.5A (from CSV/JSON or regenerate with same seed).
2. Run multiple variants side-by-side:
   - `vpp_trained_ppo`: Baseline with checkpoint.
   - `direct_target_los`: `direct_track_mode=true` + LOS-rate.
   - `pure_pn_no_vpp`: `direct_track_mode=true` + PN.
3. Compare success rates and output `direct_track_vs_vpp_comparison.csv`.

---

## 3. Expected Outcomes & Interpretation

| Outcome | Interpretation | Next Step |
|---|---|---|
| Direct-track success > 20%, VPP success ≈ 0% | VPP abstraction is harmful in tail-chase | Redesign VPP policy or bypass in tail-chase geometries |
| Direct-track success ≈ VPP success ≈ 0% | Limitation is geometric/guidance-law or backend envelope, not VPP-specific | Proceed to wider geometry sweep (Stage 6G.5A-Wide+) or guidance architecture redesign |
| Direct-track success < VPP success | VPP abstraction helps; failure is elsewhere | Investigate geometry boundary or backend constraints |

---

## 4. Paper-Safe Wording

- ✅ "No feasible candidates were found in the tested Stage 6G.5A 40-point geometry sample."
- ❌ Do **not** claim "tail-chase is universally infeasible" — the sample is a subset of 324 combinations.
- ✅ "Direct-track probe Stage 6G.5B evaluates whether the VPP layer contributes to the observed failure."

---

## 5. Caveats

- `direct_track_mode` is only meaningful when `anchor_mode == current_target`. If prediction is enabled, "direct track" on a predicted anchor is still a form of VPP.
- This probe does **not** replace a proper PN guidance-law implementation; it is a minimal bypass to test the VPP hypothesis.
- No training required; only evaluation.

---

## 6. Acceptance Criteria for Implementation

- [x] `guidance.direct_track_mode` config parsed without error.
- [x] `tracking_env.py` respects the flag and skips policy action → offset conversion.
- [x] Smoke runner produces `direct_track_vs_vpp_comparison.csv`.
- [x] Unit test confirms `direct_track_mode=true` yields `virtual_point == target_position` (within tolerance).


## 7. Stage 6G.5C Results: Candidate Confirmation & VPP Failure Diagnosis

### 7.1 Execution

- **Candidate points**: pt20, pt29, pt38 (from 6G.5B pure-PN successes)
- **Episodes**: 450 (3 points × 5 variants × 3 seeds × 10 episodes)
- **Variants tested**:
  1. vpp_trained_ppo_los (baseline)
  2. direct_target_los (no VPP, LOS-rate)
  3. pure_pn_no_vpp (no VPP, PN)
  4. hybrid_no_vpp (no VPP, hybrid)
  5. vpp_policy_pn_guidance (VPP, PN)

### 7.2 Results

| Variant | Success Rate | Crash Rate | Mean Min Range (m) |
|---|---|---|---|
| vpp_trained_ppo_los | 0% | 100% | 1710 |
| direct_target_los | 0% | 100% | 1684 |
| **pure_pn_no_vpp** | **100%** | **0%** | **11.0** |
| hybrid_no_vpp | 0% | 100% | 1640 |
| vpp_policy_pn_guidance | 0% | 100% | 1334 |

### 7.3 Interpretation

| Finding | Conclusion |
|---|---|
| pure_pn_no_vpp = 100% cross-seed stable | Tail-chase IS feasible in this narrow high-energy geometry subset under pure PN |
| direct_target_los = 0% | LOS-rate guidance is a primary bottleneck, independent of VPP |
| vpp_policy_pn_guidance = 0% | VPP offset/policy anchor is harmful even when paired with PN |
| hybrid_no_vpp = 0% | Hybrid switching/blending does not rescue tail-chase |

### 7.4 Decision Gate

- **Bilevel remains BLOCKED** — no VPP-based variant shows success.
- **Next step**: Stage 6G.5D — PN/VPP mechanism redesign or mode-switch probe.
  - Option A: Automatic mode-switch to pure PN when aspect ≈ 0° and range < threshold.
  - Option B: Tail-chase-specific VPP offset constraints (clip offset magnitude, enforce forward bias).
  - Option C: Train a new VPP policy with PN guidance instead of LOS-rate.

### 7.5 Paper-Safe Claims

- ✅ "In the tested high-energy tail-chase subset (ego 340 m/s, range 2000 m, aspect 0°), pure proportional navigation without VPP achieved 100% cross-seed success, while all VPP-based and LOS-rate variants achieved 0% success."
- ❌ Do not claim "tail-chase is universally feasible under PN" — only 3 geometries tested.
- ❌ Do not claim "VPP is always harmful" — harm observed only in the tested tail-chase subset.
