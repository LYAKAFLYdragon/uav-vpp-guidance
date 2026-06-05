# Stage 6G.5: Wide Geometry Sweep & True Oracle Design Plan

## Status

📝 Design-only stage. No full experiments until Stage 6G.4R merge gate is cleared.

---

## 1. Motivation

Stage 6G.4 smoke found 0% success across:
- True-velocity CV oracle anchor
- Rule-based pursuit (500m/1000m lead)
- 6 terminal-control ablation variants
- Small geometry sweep (16 combos: range 400–800m, ego_speed 150–200, target_speed 150–180)

**Open question**: Is tail-chase fundamentally infeasible under the current guidance architecture, or does the feasible region lie outside the tested envelope?

This plan designs the experiments needed to answer that question without committing to full runs prematurely.

---

## 2. Wider Geometry Sweep

### 2.1 Rationale

Current smoke envelope was too small and too tail-chase-centric. A wider sweep should test:
- **Speed advantage**: ego_speed significantly exceeding target_speed (closing geometry)
- **Larger initial ranges**: giving the guidance law more time to converge
- **Non-tail-chase aspect angles**: head-on, crossing, lag pursuit
- **Bidirectional altitude differences**

### 2.2 Proposed Parameter Grid

| Parameter | Values | Rationale |
|---|---|---|
| initial_range_m | 1200, 2000, 3200 | More convergence time |
| ego_speed_mps | 220, 280, 340 | Speed advantage over typical target (120–200) |
| target_speed_mps | 120, 160, 200 | Slower targets should be easier to catch |
| aspect_angle_deg | 0 (tail), 30, 60, 90 | Tail-chase may be hardest; head-on/crossing may be feasible |
| altitude_diff_m | -500, 0, +500 | Test altitude separation effects |

**Grid size**: 3 × 3 × 3 × 4 × 3 = **324 combinations**

**Smoke policy**: Sample a Latin-hypercube or random subset of 30–50 points for initial smoke. Only expand to full grid if smoke shows promise.

### 2.3 Scenario Generator Extension

Current `scenario_template` + `geometry_sweeps` in `stage6g3_geometry_feasibility.yaml` only supports tail-chase (both aircraft heading 0°).

**Needed extension**: Add `aspect_angle_deg` support:
- Own heading = 0° (fixed)
- Target heading = `aspect_angle_deg` relative to own
- Target position placed at `initial_range_m` along the appropriate LOS

Implementation: modify `run_stage6g4_smoke_probes.py::run_geometry_feasibility()` to compute target position from polar coordinates (range, aspect_angle).

---

## 3. True Future Oracle (Not CV Oracle)

### 3.1 Problem with Current Oracle

Stage 6G.4's `oracle_future_position` uses:
```python
anchor_pos = target_pos + target_vel * lookahead_time_s
```

This is a **constant-velocity (CV) oracle using true velocity**, not a true future-ground-truth oracle. If the target maneuvers (e.g., sinusoidal weaving), the oracle still has error.

### 3.2 True Oracle Options

**Option A: Replay-based future truth**
- Pre-generate a full trajectory for the target
- At each step, look up the target's actual future position `lookahead_time_s` ahead
- Requires deterministic scenario generation + replay buffer

**Option B: Analytic future truth**
- For constant-velocity scenarios: current implementation is exact
- For sinusoidal/weaving scenarios: integrate target motion analytically
- More complex but no replay needed

**Option C: Accept CV oracle as upper bound**
- Document clearly: "true-velocity CV oracle provides an upper bound on prediction improvement for non-maneuvering targets"
- If CV oracle cannot rescue tail-chase, no prediction method can (for CV targets)
- If target maneuvers, this does not rule out prediction helping in other scenarios

**Recommendation**: Implement Option A (replay-based) for the `constant_velocity` target mode first. This gives true future ground truth for the simplest case. Document that maneuvering targets require Option B or C caveats.

---

## 4. Pure PN / Pure Pursuit Without VPP

### 4.1 Rationale

Current architecture: Policy → VPP offset → Guidance law → Command.

If VPP abstraction itself is harmful in tail-chase (e.g., adds unnecessary offset noise), a pure guidance-law approach might succeed:
- **Pure PN**: Track target directly, no VPP
- **Pure pursuit**: Always point velocity vector at target
- **Pure LOS-rate**: No virtual point, just line-of-sight geometry

### 4.2 Implementation Sketch

Add a `guidance.direct_track_mode` config flag:
- `False` (default): Use VPP policy output
- `True`: Ignore policy action, set virtual point = target position (or use pure PN)

This is a 5–10 line change in `tracking_env.py` and requires no new training.

---

## 5. Guidance Architecture Redesign Ideas

If wide geometry sweep + true oracle + pure PN all fail, the limitation is architectural. Potential redesigns:

| Idea | Hypothesis | Complexity |
|---|---|---|
| Terminal altitude hold | Altitude drift causes OOB; hold altitude in terminal phase | Low |
| Pursuit geometry mode switch | Switch from lead to lag to pure pursuit based on range/ATA | Medium |
| Energy-aware guidance | Explicitly manage kinetic + potential energy during approach | Medium |
| Tail-chase-specific capture envelope | Relax success criteria for tail-chase (e.g., 1200m / 45°) | Low (config) |
| Bilevel gain optimization | Optimize guidance gains per scenario | High |

**Constraint**: Do not implement any redesign until Stage 6G.5 smoke results show at least one feasible configuration.

---

## 6. Success Criteria for Stage 6G.5

| Criterion | Threshold | Action if met |
|---|---|---|
| Any geometry combo > 20% success | Wider sweep identifies feasible region | Expand that region, characterize boundary |
| True oracle > 20% success (CV targets) | Prediction can help | Investigate predictor improvements |
| Pure PN > 20% success | VPP abstraction is harmful | Redesign VPP policy or bypass in tail-chase |
| All variants 0% success | Likely architectural infeasibility | Proceed to redesign (§5) or relax criteria |

---

## 7. Bilevel Status

**Still blocked.**

Bilevel gain optimization is only valuable if:
1. There exists a feasible geometry (otherwise optimizing gains is noise-fitting)
2. The failure is gain-sensitive (otherwise bilevel has no leverage)

Stage 6G.5 must first establish feasibility before bilevel is considered.
