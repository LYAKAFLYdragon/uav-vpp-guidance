# Stage 6G.3: Oracle & Terminal-Protection Feasibility Gate

## Objective

Decompose the "all guidance laws 0% success" finding from Stage 6G.1/6G.2 into specific, testable failure hypotheses:

1. **Prediction hypothesis**: Would perfect prediction (oracle VPP anchor) improve success?
2. **Policy hypothesis**: Would a rule-based VPP placement (bypassing learned PPO policy) improve success?
3. **Protection hypothesis**: Does terminal-phase protection (capture radius, altitude hold, limiters) cause or mask failure?
4. **Geometry hypothesis**: At what initial conditions does tail-chase become feasible?

**Constraint**: Smoke runs and contract tests only. No full 720-episode experiments until a hypothesis shows promise.

---

## 1. Oracle VPP Anchor Probe (6G.3A)

**Config**: `config/experiment/stage6g3_oracle_vpp_anchor.yaml`

**Implementation**: Added `anchor_mode="oracle_future_position"` to `VirtualPointGenerator`.
Computes anchor as `target_pos + target_vel * lookahead_time_s` using true velocity.

**Expected behavior**:
- If oracle improves success → prediction error is a bottleneck.
- If oracle still 0% → limitation is not prediction quality but guidance geometry or policy action space.

**Smoke test**:
```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --guidance-modes los_rate \
    --scenarios favorable \
    --methods oracle_vpp_anchor \
    --smoke \
    --output-dir outputs/stage6g3_oracle_smoke
```

**Status**: ✅ Config created, anchor mode implemented, tests pass.

**Stage 6G.4 smoke result**: Rule-based pursuit (500m and 1000m lead) produced **0% success** (50% crash). Geometric direction check passed (anchor ahead of target along LOS). Failure is **not caused by the learned PPO policy**.

**Stage 6G.4 smoke result**: Oracle anchor (perfect prediction) produced **0% success** (50% crash, 50% OOB). Failure is **not a prediction bottleneck**.

---

## 2. Rule-Based VPP Pursuit Probe (6G.3B)

**Config**: `config/experiment/stage6g3_rule_based_pursuit.yaml`

**Implementation**: Added `anchor_mode="rule_based_pursuit"` to `VirtualPointGenerator`.
Computes anchor as `target_pos + los_unit * lead_distance_m`.

**Expected behavior**:
- If rule-based succeeds → learned PPO policy is harmful in tail-chase.
- If rule-based also fails → limitation is guidance-law geometry, not policy.

**Smoke test**:
```bash
python scripts/run_stage6g_guidance_limitation_probe.py \
    --guidance-modes los_rate \
    --scenarios favorable \
    --methods rule_based_pursuit_500m \
    --smoke \
    --output-dir outputs/stage6g3_rulebased_smoke
```

**Status**: ✅ Config created, anchor mode implemented, tests pass.

---

## 3. Terminal Protection Ablation Probe (6G.3C)

**Config**: `config/experiment/stage6g3_terminal_protection_ablation.yaml`

**Variants**:
- `default`: capture_radius_m=50, post_process enabled
- `no_capture_radius`: capture_radius_m=0
- `no_post_process`: post_process disabled
- `minimal`: both disabled

**Expected behavior**:
- If disabling protection improves success → protection is overly conservative.
- If no change → protection is not the root cause.

**Smoke test**:
```bash
# Requires manual config editing or probe runner extension to vary protection params
python scripts/run_stage6g_guidance_limitation_probe.py \
    --guidance-modes los_rate \
    --scenarios favorable \
    --smoke \
    --output-dir outputs/stage6g3_protection_smoke
```

**Status**: ✅ Base config created and fixed for correct guidance param structure. Smoke runner supports per-run config overrides.

**Stage 6G.4 smoke result**: All 6 terminal control variants (baseline, no_capture_radius, no_terminal_protection, no_post_process, no_energy_comp, no_load_roll_coord) produced **0% success** (50% crash, 50% OOB). Failure is **not caused by any protective/limiting mechanism**.

---

## 4. Geometry Feasibility Boundary Probe (6G.3D)

**Config**: `config/experiment/stage6g3_geometry_feasibility.yaml`

**Sweeps**:
- `initial_range_m`: [400, 800, 1600, 3200]
- `ego_speed_mps`: [150, 200, 250, 300]
- `target_speed_mps`: [150, 180, 220]
- `altitude_diff_m`: [-500, 0, 500]

**Expected behavior**:
- Identify feasibility boundary (e.g., tail-chase feasible if ego_speed > target_speed + 20 m/s).
- If no combination succeeds → tail-chase is fundamentally infeasible under current guidance architecture.

**Status**: ✅ Config template created. Smoke runner generates scenarios from sweep parameters.

**Stage 6G.4 smoke result**: 16-point grid sweep (initial_range × ego_speed × target_speed × altitude_diff) produced **0/16 success** (all crash). No feasible boundary was found within the tested envelope.

---

## Telemetry Contract

**New module**: `src/uav_vpp_guidance/evaluation/telemetry_schema_validator.py`

Validates that episode data contains required fields for root-cause analysis:
- **Core**: scenario, method, guidance_mode, seeds, success/crash/OOB/timeout, reason
- **Terminal phase**: min_range_m, time_to_first_advantage_s, advantage_hold_time_s, VPP shift
- **Prediction**: prediction_valid_rate, prediction_fallback_rate, mean_prediction_error_m
- **Command saturation**: nz_cmd_max/mean/saturation_rate/modification_rate
- **Altitude/energy**: min/max/final altitude, altitude_loss_rate, energy_proxy

**Status (Stage 6G.4)**: Per-step telemetry aggregation is **now emitted** by `evaluate_prediction_comparison.py`. Command saturation and altitude/energy fields are computed as episode-level aggregates from the per-step simulation loop.

---

## Test Coverage

- `tests/test_stage6g3_anchor_modes.py`: 6 tests (oracle velocity, oracle fallback, rule-based placement, rule-based zero distance, env oracle mode, env rule-based mode)
- `tests/test_telemetry_schema_validator.py`: 9 tests (core pass, core fail, terminal phase missing, command saturation missing, empty episodes, homogeneous episodes, unavailable categories, report rendering)

**Total suite**: 597 passed (was 582 before Stage 6G.3 additions).

---

## McNemar Pairing Validator

**New module**: `src/uav_vpp_guidance/evaluation/mcnemar_pairing_validator.py`

- Strict key-aligned pairing: scenario, method, guidance_mode, training_seed, eval_seed, episode_index
- Shuffle-resistant: pairs by canonical key, not by row order
- Dimension exclusion: exclude `method` when comparing methods, exclude `guidance_mode` when comparing guidance laws
- Missing-key detection: fails if pairing fields are absent

**Tests**: `tests/test_mcnemar_pairing_validator.py` — 13 tests.

---

## Artifact Contract Validator

**Script**: `scripts/validate_stage6g_probe_outputs.py`

Validated against hardened full run (`run_20260605_103449`):
- ✅ 12 cells completed
- ✅ 720 episodes
- ✅ 3 guidance modes × 4 scenarios × 2 methods × 3 seeds × 10 episodes
- ✅ All required artifacts present
- ✅ Guidance mode consistency: requested == resolved == effective
- ✅ No duplicate episode keys
- ✅ Manifest cross-checks pass

---

## Paper-Safe Claim Status

| Claim | Status | Reason |
|---|---|---|
| Neural > Classical in feasible geometries | ✅ Paper-safe | Supported by Stage 6F evidence |
| GRU > LSTM in weaving_headon | ❌ Not paper-safe | Cross-seed strict consistency insufficient |
| CA vs CV practically negligible | ❌ Not paper-safe | No new evidence |
| Tail-chase failure not LOS-rate-specific | ✅ Within Stage 6G.1 scope | 0% across 3 guidance laws × 4 scenarios |
| Tail-chase root cause identified | ⚠️ Partial (Stage 6G.4) | Oracle, rule-based, terminal ablation all 0% success. Most likely: geometric infeasibility under current guidance architecture. Needs wider sweep or guidance redesign to confirm. |
| PN/hybrid ineffective for tail-chase | ❌ Not paper-safe | Only tested under current VPP/policy/protection stack |

---

## Next Steps

1. **Run oracle smoke probe**: ✅ Done — 0% success.
2. **Run rule-based smoke probe**: ✅ Done — 0% success, direction correct.
3. **Extend probe runner**: ✅ Done — `run_stage6g4_smoke_probes.py` supports all 4 probes.
4. **Add per-step telemetry**: ✅ Done — episode-level aggregates now emitted.
5. **Geometry sweep runner**: ✅ Done — 16-point grid executed.
6. **Next**: Widen geometry sweep (ego_speed >> target_speed, larger ranges) or test guidance redesign (pure PN, pure pursuit without VPP).
6. **Bilevel decision gate**: Only enter bilevel retraining if Stage 6G.3 shows a specific failure mode (e.g., protection too conservative) that gain tuning could remedy.
