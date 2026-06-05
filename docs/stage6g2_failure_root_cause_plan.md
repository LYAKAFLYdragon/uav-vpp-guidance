# Stage 6G.2 Failure Root-Cause Plan

## Context

Stage 6G.1 established that the tail-chase/stern-conversion dead zone is **not specific to LOS-rate guidance**: proportional navigation and hybrid guidance also show 0% success in all four dead-zone scenarios. The remaining question is **why** — is it the VPP policy action space, the guidance law architecture, the terminal-phase protection, or the fundamental pursuit geometry?

Stage 6G.2 designs four minimal probe classes to isolate the root cause. **No full experiments are to be run yet** — this document provides configs, templates, and smoke tests only.

---

## 1. Oracle VPP Anchor Probe

**Question**: If the VPP anchor were placed at the target's *future true position* (oracle prediction), would success rate improve?

**Rationale**: Stage 6G.1 uses frozen GRU predictor and no-prediction (current target position). If even an oracle predictor fails, the limitation is not prediction quality but guidance geometry or policy action space.

**Implementation**: Add an `oracle_prediction` method to `evaluate_prediction_comparison.py` that computes:

```python
target_future_pos = target_pos + target_vel * lookahead_time
anchor = target_future_pos  # perfect prediction, zero error
```

**Config template**: `config/experiment/stage6g2_oracle_vpp_anchor.yaml`

```yaml
methods:
  oracle_prediction:
    type: oracle
    lookahead_time_s: 1.0
    description: "Perfect future-position oracle; isolates prediction contribution"
  no_prediction:
    type: baseline
    description: "Current target position (existing baseline)"

guidance:
  mode: los_rate  # start with one guidance law; if oracle fails, expand to PN/hybrid

scenarios:
  favorable:
    max_range_m: 12000
    # ... (same as stage6f5_feasible_geometry)
  weaving_pursuit:
    max_range_m: 12000
    # ... (same as stage6f5_maneuvering_target)
```

**Smoke test**: 1 episode, 1 seed, verify oracle anchor != current position.

**Expected outcomes**:
- If oracle improves success → prediction is a bottleneck (but not the only one).
- If oracle still 0% → limitation is guidance geometry or policy action space.

---

## 2. Guidance-Only Oracle Policy Probe

**Question**: If we bypass the learned VPP policy entirely and use a rule-based VPP placement (e.g., pure pursuit with fixed lead angle), do any guidance laws succeed?

**Rationale**: The learned PPO policy was trained on head-on and crossing geometries. It may generate VPP offsets that are actively harmful in tail-chase.

**Implementation**: Add a `rule_based_pursuit` method that computes VPP as:

```python
# Pure pursuit: place VPP ahead of target along LOS
los_unit = (target_pos - own_pos) / distance
vpp = target_pos + los_unit * lead_distance_m
```

With `lead_distance_m` swept over `[0, 100, 500, 1000, 2000]`.

**Config template**: `config/experiment/stage6g2_rule_based_pursuit.yaml`

```yaml
methods:
  rule_based_pursuit:
    type: rule_based
    lead_distance_m: 500
    description: "Fixed lead-angle pursuit; bypasses learned policy"
  no_prediction:
    type: baseline
```

**Smoke test**: Verify VPP position changes with lead_distance.

**Expected outcomes**:
- If rule-based succeeds → learned policy is harmful in tail-chase.
- If rule-based also fails → limitation is guidance-law geometry, not policy.

---

## 3. Terminal Protection Ablation Probe

**Question**: Do capture-radius blending, altitude hold, roll/nz limiters, or energy compensation mask or cause the failure?

**Rationale**: Stage 6G.1 episodes crash at ~11,300 m in favorable. The capture-radius blending (`d < 50 m → safe hold`) should not activate at 11 km. However, terminal-phase protection or load-roll coordination may scale commands down prematurely.

**Implementation**: Run probe with `post_process.enabled = false` and `guidance.params.capture_radius_m = 0` (disabled) vs. default.

**Config template**: `config/experiment/stage6g2_terminal_protection_ablation.yaml`

```yaml
variants:
  - name: "default"
    guidance:
      post_process:
        enabled: true
      params:
        capture_radius_m: 50
  - name: "no_post_process"
    guidance:
      post_process:
        enabled: false
      params:
        capture_radius_m: 50
  - name: "no_capture_radius"
    guidance:
      post_process:
        enabled: true
      params:
        capture_radius_m: 0
  - name: "minimal_protection"
    guidance:
      post_process:
        enabled: false
      params:
        capture_radius_m: 0

guidance:
  mode: los_rate

scenarios:
  favorable:
    max_range_m: 12000
```

**Smoke test**: Verify config override propagates to guidance constructor.

**Expected outcomes**:
- If disabling protection improves success → protection is overly conservative.
- If no change → protection is not the root cause.

---

## 4. Geometry Feasibility Boundary Probe

**Question**: At what initial conditions does tail-chase become feasible?

**Rationale**: The current scenarios use fixed initial positions. A small change (e.g., reduce initial altitude difference, increase initial range, reduce target speed advantage) might cross a feasibility threshold.

**Implementation**: Parameter sweep over:

| Parameter | Base Value | Sweep Range |
|---|---|---|
| Initial range | 2000 m | 1000, 2000, 4000, 8000 m |
| Initial altitude difference | 0 m | -500, 0, +500 m |
| Ego speed | 150 m/s | 120, 150, 180, 220 m/s |
| Target speed advantage | 40 m/s | 0, 20, 40, 60 m/s |
| Aspect angle | 180° (pure tail-chase) | 150°, 165°, 180° |

**Config template**: `config/experiment/stage6g2_geometry_feasibility.yaml`

```yaml
sweeps:
  - parameter: initial_range_m
    values: [1000, 2000, 4000, 8000]
  - parameter: initial_altitude_diff_m
    values: [-500, 0, 500]
  - parameter: ego_speed_mps
    values: [120, 150, 180, 220]
  - parameter: target_speed_advantage_mps
    values: [0, 20, 40, 60]
  - parameter: aspect_angle_deg
    values: [150, 165, 180]

guidance:
  mode: los_rate

methods:
  no_prediction:
    type: baseline
```

**Smoke test**: 1 episode per parameter combination, verify scenario generator accepts parameters.

**Expected outcomes**:
- Identify a feasibility boundary (e.g., tail-chase feasible if ego_speed > target_speed + 20 m/s).
- If no combination succeeds → tail-chase is fundamentally infeasible under current guidance architecture.

---

## Test Plan

For each probe class, create a smoke test in `tests/test_stage6g2_probe_templates.py`:

1. Config template loads without YAML errors.
2. Config validation passes (`validate_tp_config` or equivalent).
3. Dry-run produces expected scenario/method matrix.
4. Smoke run (1 episode) completes and writes `prediction_metrics.json`.
5. No checkpoint required for oracle/rule-based methods (they bypass learned policy).

---

## Integration with Existing Pipeline

- Reuse `run_stage6g_guidance_limitation_probe.py` infrastructure: `--guidance-modes`, `--scenarios`, `--smoke`, artifact generation.
- Add `--probe-type {oracle, rule_based, ablation, geometry}` flag to select probe class.
- Reuse `validate_stage6g_probe_outputs.py` for output contract verification.
- Reuse `analyze_stage6g_failure_root_cause.py` for post-hoc analysis.

---

## Success Criteria for Stage 6G.2

| Criterion | Target |
|---|---|
| All 4 probe configs validated | 100% pass in `tests/test_stage6g2_probe_templates.py` |
| Smoke tests run in < 5 min | Each probe class ≤ 1 min |
| No code duplication | Reuse existing probe runner, validator, analyzer |
| Paper-safe claim isolation | Each probe isolates exactly one hypothesis |

## Do Not Do (Explicitly Out of Scope)

- **Do not** run full 720-episode experiments for Stage 6G.2 probes.
- **Do not** add new neural network training.
- **Do not** modify JSBSim backend.
- **Do not** claim "tail-chase is impossible" — claim only "within tested geometry and guidance architecture."
