# Stage 6G.5B: Direct-Track / Pure-PN Probe Plan

> **Status**: Design-only. Not implemented until Stage 6G.5A geometry smoke identifies a feasible region.
>
> **Blocked by**: `bilevel_unblocked_candidate == true` from `run_stage6g5_geometry_smoke.py`.

---

## 1. Motivation

Current architecture:

```text
Policy → VPP offset (Δp) → Guidance law (LOS-rate/PN/hybrid) → Command (Nz, roll rate, throttle)
```

If the VPP abstraction itself is harmful in tail-chase (e.g., adds unnecessary offset noise, destabilizes the terminal phase), a **pure guidance-law** approach might succeed where the VPP policy fails:

- **Pure PN**: Track target directly with proportional navigation, no VPP offset.
- **Pure pursuit**: Always point velocity vector at the instantaneous target position.
- **Pure LOS-rate**: No virtual point; use line-of-sight geometry directly.

This probe evaluates whether bypassing the VPP layer rescues tail-chase feasibility.

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

`scripts/run_stage6g5_geometry_smoke.py` (or a dedicated `run_stage6g5b_direct_track_smoke.py`) should:

1. Load the same geometry grid as Stage 6G.5A.
2. Set `guidance.direct_track_mode: true` in the resolved config.
3. Run the same sampled points with `no_prediction` method.
4. Compare success rates: VPP vs. direct-track.

---

## 3. Expected Outcomes & Interpretation

| Outcome | Interpretation | Next Step |
|---|---|---|
| Direct-track success > 20%, VPP success ≈ 0% | VPP abstraction is harmful in tail-chase | Redesign VPP policy or bypass in tail-chase geometries |
| Direct-track success ≈ VPP success ≈ 0% | Limitation is geometric/guidance-law, not VPP-specific | Proceed to wider geometry sweep or guidance redesign (§5 of Stage 6G.5 plan) |
| Direct-track success < VPP success | VPP abstraction helps; gains or geometry are the real bottleneck | Focus on bilevel gain optimization once feasible geometry is found |

---

## 4. Caveats

- `direct_track_mode` is only meaningful when `anchor_mode == current_target`. If prediction is enabled, "direct track" on a predicted anchor is still a form of VPP.
- This probe does **not** replace a proper PN guidance-law implementation; it is a minimal bypass to test the VPP hypothesis.
- No training required; only evaluation.

---

## 5. Acceptance Criteria for Implementation

- [ ] `guidance.direct_track_mode` config parsed without error.
- [ ] `tracking_env.py` respects the flag and skips policy action → offset conversion.
- [ ] Smoke runner produces `direct_track_vs_vpp_comparison.csv`.
- [ ] Unit test confirms `direct_track_mode=true` yields `virtual_point == target_position` (within tolerance).
