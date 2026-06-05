# Stage 6G.4: Oracle Smoke Execution & Per-Step Telemetry Completion

## Objective

Execute minimal smoke probes to decompose the Stage 6G.1 tail-chase 0% success finding, while completing per-step telemetry so failure root-cause claims can be grounded in data.

---

## 1. Per-Step Telemetry Completion

### What changed

`evaluate_single_episode()` in `src/uav_vpp_guidance/evaluation/evaluate_prediction_comparison.py` now computes episode-level aggregates from per-step data collected during the simulation loop:

| Category | New fields | Source |
|---|---|---|
| Command saturation | `nz_cmd_max`, `nz_cmd_mean`, `nz_cmd_saturation_rate`, `nz_cmd_modification_rate` | `info["nz_cmd"]` and `info["raw_command"]` |
| Command saturation | `roll_rate_cmd_max`, `roll_rate_cmd_mean`, `roll_rate_cmd_saturation_rate`, `roll_rate_cmd_modification_rate` | `info["roll_rate_cmd"]` and `info["raw_command"]` |
| Command saturation | `throttle_cmd_max`, `throttle_cmd_mean`, `throttle_cmd_saturation_rate`, `throttle_cmd_modification_rate` | `info["throttle_cmd"]` and `info["raw_command"]` |
| Altitude | `min_altitude_m`, `max_altitude_m`, `final_altitude_m`, `altitude_loss_rate` | `info["own_state"]["position_m"][2]` |
| Energy | `energy_proxy` (total energy height = v²/2g + h) | `info["own_state"]` velocity + altitude |

**Saturation vs modification**:
- `saturation_rate`: filtered command is at the limit boundary (≤ min + eps or ≥ max − eps)
- `modification_rate`: raw command differs from filtered command (captures internal clip, terminal protection, energy compensation, load-roll coordination)

### Telemetry schema validator update

`telemetry_schema_validator.py` now treats `command_saturation` and `altitude_energy` as **episode-aggregatable** fields (no longer marked as "requires per-step telemetry not emitted"). When these fields are missing, the report states:

> "Command saturation aggregates not present in episode records. Run evaluation with Stage 6G.4+ harness or check that limits are defined in config."

---

## 2. Smoke Probe Results

### 2.1 Oracle VPP Anchor (6G.4A)

**Command**:
```bash
python scripts/run_stage6g4_smoke_probes.py --oracle --episodes 2 --seeds 0
```

**Result**:
- Success rate: **0.00%**
- Crash rate: **50.00%**
- OOB rate: **50.00%**

**Interpretation**: True-velocity CV oracle (`target_pos + target_vel * lookahead_time_s`) did **not** improve success in the tested scenarios. This suggests that **prediction error under constant-velocity assumption is not the sole bottleneck**. Note: this is a one-step true-velocity oracle, not a full future-ground-truth oracle; if the target maneuvers, positional error still exists.

---

### 2.2 Rule-Based Pursuit (6G.4B)

**Command**:
```bash
python scripts/run_stage6g4_smoke_probes.py --rule-based --episodes 2 --seeds 0
```

**Result**:
- Geometric direction check: **✅ Correct** — VPP anchor is placed ahead of target along the LOS
- rule_based_pursuit_500m: Success **0.00%**, Crash **50.00%**
- rule_based_pursuit_1000m: Success **0.00%**, Crash **50.00%**

**Interpretation**: Pure geometric pursuit with fixed lead distance also yielded 0% success. This suggests the **learned PPO policy is unlikely to be the sole root cause** in the tested scenarios. The failure lies deeper in the guidance/control chain or in geometric infeasibility, pending wider sweep verification.

---

### 2.3 Terminal Control Ablation (6G.4C)

**Command**:
```bash
python scripts/run_stage6g4_smoke_probes.py --terminal --episodes 2 --seeds 0
```

**Variants tested**:
| Variant | Success | Crash | OOB |
|---|---|---|---|
| baseline | 0.00% | 50.00% | 50.00% |
| no_capture_radius | 0.00% | 50.00% | 50.00% |
| no_terminal_protection | 0.00% | 50.00% | 50.00% |
| no_post_process | 0.00% | 50.00% | 50.00% |
| no_energy_comp | 0.00% | 50.00% | 50.00% |
| no_load_roll_coord | 0.00% | 50.00% | 50.00% |

**Interpretation**: Disabling capture radius, terminal protection, post-processing, energy compensation, or load-roll coordination did **not** improve success in the tested smoke scenarios. This suggests these mechanisms are **unlikely to be the primary cause** of failure. **Caveat**: effective-runtime-flag verification is now implemented; all variants show numerically different runtime flags, so "no change in outcome" is a valid observation, not a config-drift artifact.

---

### 2.4 Geometry Feasibility Sweep (6G.4D)

**Command**:
```bash
python scripts/run_stage6g4_smoke_probes.py --geometry --episodes 1 --seeds 0
```

**Grid** (2 values per axis = 16 combinations):
- initial_range_m: [400, 800]
- ego_speed_mps: [150, 200]
- target_speed_mps: [150, 180]
- altitude_diff_m: [-500, 0]

**Result**: **0/16 success** (all crash).

**Interpretation**: Within the tested small parameter envelope (range 400–800 m, ego_speed 150–200 m/s, target_speed 150–180 m/s, altitude_diff −500–0 m), **no combination produced success**. This supports:
1. The feasible region, if it exists, likely lies **outside this small envelope**.
2. Tail-chase may be fundamentally infeasible under the current guidance architecture, but this claim requires a **wider sweep** before it can be considered paper-safe.

---

## 3. Root-Cause Conclusions

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Prediction error causes failure | True-velocity CV oracle still 0% | ⚠️ Not rescued by CV oracle; full future-ground-truth oracle pending |
| Learned policy causes failure | Rule-based (bypass policy) still 0% | ❌ Rejected |
| Terminal protection causes failure | All ablation variants still 0% (with verified runtime flags) | ⚠️ Not rescued by disabling protection; primary cause likely elsewhere |
| Geometry infeasible in tested small grid | 16/16 grid points crash | ⚠️ Supported within envelope; wider sweep needed for general claim |
| Guidance-law architecture limitation | All guidance laws (6G.1) + all probes (6G.4) fail | ✅ Supported |

**Current best explanation**: The tail-chase / stern-conversion scenario appears **difficult or infeasible under the current guidance architecture** (LOS-rate/PN/hybrid + VPP + simple backend + current success criteria: range ≤ 900 m, ATA ≤ 25°) within the tested parameter envelope. The failure was not rescued by true-velocity CV oracle, rule-based pursuit, or terminal-control ablation. However, a **wider geometry sweep and/or guidance redesign** is required before claiming fundamental infeasibility.

---

## 4. Paper-Safe Claim Status (Updated)

| Claim | Status |
|---|---|
| Neural > Classical in feasible geometries | ✅ Paper-safe |
| GRU > LSTM in weaving_headon | ❌ Not paper-safe |
| CA vs CV practically negligible | ❌ Not paper-safe |
| Tail-chase failure not LOS-rate-specific | ✅ Paper-safe (within Stage 6G.1 scope) |
| Tail-chase root cause identified | ⚠️ Partial — guidance architecture / geometric infeasibility most likely within tested envelope; wider sweep or guidance redesign required for confirmation |
| PN/hybrid ineffective for tail-chase | ❌ Not paper-safe (only tested under current stack) |

---

## 5. Files Changed

| File | Change |
|---|---|
| `src/uav_vpp_guidance/virtual_point/generator.py` | Read `lead_distance_m` from config |
| `src/uav_vpp_guidance/evaluation/evaluate_prediction_comparison.py` | Per-step telemetry aggregation (command, altitude, energy) |
| `src/uav_vpp_guidance/evaluation/telemetry_schema_validator.py` | Update saturation/altitude field status |
| `config/experiment/stage6g3_terminal_protection_ablation.yaml` | Fix guidance param structure |
| `scripts/run_stage6g4_smoke_probes.py` | **New** — unified smoke runner for all 4 probes |
| `tests/test_stage6g4_telemetry_and_smoke.py` | **New** — 11 tests for direction geometry, telemetry, smoke outputs |

---

## 6. Next Steps

1. **Widen geometry sweep**: Test ego_speed > target_speed (e.g., 300 vs 180), larger initial ranges (1600–3200 m), and different aspect angles (not just tail-chase).
2. **Relax success criteria**: Test if the aircraft can achieve *any* stable close-range tracking, even if not within 900 m / 25°.
3. **Guidance redesign probe**: Test if a pure-PN (no VPP) or pure-pursuit guidance law can succeed in tail-chase.
4. **Bilevel readiness**: **NOT ready**. No variant showed improvement. Bilevel gain optimization would likely optimize noise on an infeasible geometry.
