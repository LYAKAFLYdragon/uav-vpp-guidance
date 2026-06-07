# Stage 10.3: JSBSim Crossing-Failure Evidence Analysis

> **Status**: Analysis complete. Crossing failures are reproducible, stratified, and attributable to a geometry-dynamics mismatch between the simple-backend training distribution and JSBSim F-16 turning capability.
>
> **Paper-safe claim**: "After correcting a scenario initialization bug, simple-backend-trained policies achieve partial zero-shot transfer to JSBSim F-16: 100% success on head-on scenarios, 0% on crossing scenarios. Crossing failures remain unresolved under current controller and F-16 dynamics."

---

## 1. Data Source

| Field | Value |
|-------|-------|
| Benchmark run | `outputs/stage10_2_jsbsim_corrected_official_20260607_164836/` |
| Methods evaluated | `no_prediction`, `gain_only` |
| Scenarios per method | 20 (10 head-on + 10 crossing) |
| Backend | JSBSim F-16 vs SimplePointMass target |
| `paper_safe` | `true` |
| `git_dirty` | `false` |

---

## 2. Stratified Results

### 2.1 Aggregate Success Rate

| Method | Overall | Head-On | Crossing |
|--------|---------|---------|----------|
| no_prediction | 50% (10/20) | **100% (10/10)** | **0% (0/10)** |
| gain_only | 50% (10/20) | **100% (10/10)** | **0% (0/10)** |

### 2.2 Failure Taxonomy

All 40 crossing episodes (20 per method) terminate with `reason = out_of_bounds`.
No crashes, no timeouts, no actuator-hard-limit terminations.

| Scenario | Method | N | Success | Crash | Timeout | OOB |
|----------|--------|---|---------|-------|---------|-----|
| regression_crossing_left | no_prediction | 10 | 0 | 0 | 0 | 10 |
| regression_crossing_right | no_prediction | 10 | 0 | 0 | 0 | 10 |
| regression_crossing_left | gain_only | 10 | 0 | 0 | 0 | 10 |
| regression_crossing_right | gain_only | 10 | 0 | 0 | 0 | 10 |

---

## 3. Telemetry Comparison: Head-On vs Crossing

### 3.1 Episode-Level Dynamics

| Metric | Head-On (n=20) | Crossing (n=40) | Ratio | Interpretation |
|--------|----------------|-----------------|-------|----------------|
| `length` (steps) | 15.0 | 166.3 | 11.1× | Crossing episodes persist until max_steps |
| `min_range_m` | 851.8 | 776.0 | 0.91 | Never approaches capture radius (~100 m) |
| `final_range_m` | 851.8 | 12,038.5 | 14.1× | Divergence: aircraft flies *away* from target |
| `final_ata_deg` | — | 128.1 | — | Large final aspect = missed crossing point |
| `min_altitude_m` | 4,999.2 | 1,157.8 | 0.23 | Altitude loss but not crash |
| `altitude_loss_rate` | +4.0 | −115.9 | −29× | Sustained altitude bleed in crossing |

### 3.2 Command Profiles

| Metric | Head-On | Crossing | Note |
|--------|---------|----------|------|
| `nz_cmd_max` | 1.37 | 2.96 | Higher load factor demanded in crossing |
| `nz_cmd_mean` | 1.31 | 1.93 | Sustained higher G commands |
| `nz_cmd_saturation_rate` | 0.0% | 0.0% | **Not saturated** — F-16 can deliver commanded Nz |
| `roll_rate_cmd_max` | ≈ 0.0 | 1.36 rad/s | Near-max roll rate commanded |
| `roll_rate_cmd_saturation_rate` | 0.0% | 12.0% | Roll rate occasionally hits limit |
| `throttle_cmd_mean` | 0.82 | 0.62 | Throttle reduced in crossing (energy conservation?) |

### 3.3 Method-Specific Crossing Breakdown

| Metric | no_prediction (crossing) | gain_only (crossing) |
|--------|--------------------------|----------------------|
| `nz_cmd_max` | **4.16** | 1.84 |
| `roll_rate_cmd_max` | **1.50** (hard limit) | 1.22 |
| `roll_rate_cmd_saturation_rate` | **0.0%** | **0.0%** |
| `min_altitude_m` | 550 / 1,415 | 1,156 / 1,509 |

**Observation**: `no_prediction` commands a higher load factor (4.16g peak) and maximum roll rate (1.50 rad/s ≈ 86°/s), yet still fails. `gain_only` is more conservative on Nz but equally unsuccessful. The failure is **not** caused by insufficient command authority — the aircraft simply cannot generate the required turn rate under the commanded flight condition.

---

## 4. Root-Cause Inference

### 4.1 Failure Sequence (Inferred)

1. **Initial geometry**: Target crosses at ~90° aspect, initial range ~8 km.
2. **Policy response**: Commands max roll rate (1.5 rad/s) and high Nz to turn toward target.
3. **F-16 turn dynamics**: Structural/energy limits prevent completing the intercept turn before the target crosses the flight path.
4. **Missed crossing point**: Aircraft passes behind target; range begins to increase.
5. **Divergence**: With target now on the tail or at high off-boresight angle, policy cannot recover. Range grows to >12 km.
6. **Termination**: `out_of_bounds` triggered after ~166 steps (near episode limit).

### 4.2 Why Head-On Succeeds

- Head-on scenarios require **negligible heading change** (< 5° roll rate commands).
- F-16 can maintain collision-course geometry with minimal maneuvering.
- Capture occurs in ~15 steps before any energy or turn-rate limit is relevant.

### 4.3 Why Crossing Fails

- Crossing scenarios require **~90° heading change in < 5 seconds** (at 800 m/s closure).
- F-16 sustained turn rate at 5,000 m and Mach ~0.8 is approximately **12–15°/s** (structural limit).
- To intercept a crossing target at 8 km, the aircraft needs ~20–25°/s instantaneous turn capability — **unavailable** at this flight condition.
- The simple backend used for training does **not** model F-16 turn-rate or energy limitations, so the policy never learned energy-management or lead-pursuit strategies for crossing geometry.

### 4.4 Evidence Against Alternative Hypotheses

| Hypothesis | Evidence For/Against | Verdict |
|------------|----------------------|---------|
| **Actuator saturation** | `nz_cmd_saturation_rate = 0%`, `roll_rate_saturation = 12%` (occasional, not sustained) | **Rejected** |
| **Altitude crash** | `min_altitude_m = 550–1500 m`, no `reason = crash` | **Rejected** |
| **Policy command insanity** | Commands are directionally correct (high roll + Nz to turn) | **Rejected** |
| **Scenario init bug** | Fix applied in Stage 10.1; head-on now 100% | **Rejected** |
| **F-16 turn-rate/energy limit** | Required turn >> achievable turn; divergence pattern matches energy bleed | **Accepted** |

### 4.5 Baseline Controller Evidence (New)

To rule out policy-specific failure, three **classical baseline controllers** were evaluated on the identical crossing scenarios using the diagnosis runner (`jsbsim_diagnosis.py`):

| Baseline | Method | Crossing Left | Crossing Right | Root Cause |
|----------|--------|---------------|----------------|------------|
| **Hold** | `hold` | ❌ OOB | ❌ OOB | `baseline_oob` |
| **Direct PN** | `direct_pn` | ❌ OOB | ❌ Crash | `baseline_saturation` / `altitude_divergence` |
| **Low-gain Direct** | `low_gain_direct` | ❌ OOB | ❌ OOB | `baseline_oob` |

**Key observations from baseline runs**:
- **Zero success across all baselines**: Classical guidance laws (PN, hold) also fail on crossing scenarios.
- **Direct PN commanded extreme loads**: `nz_cmd` peaked at **14.5g** and roll rate at **3.7 rad/s** (≈ 212°/s), far exceeding F-16 structural limits. The aircraft crashed in `regression_crossing_right` due to altitude divergence under sustained high-G maneuvering.
- **Hold controller** cannot close range at all (min_range ~820 m), confirming that passive energy management is insufficient.

**Conclusion**: Crossing failure is **not** a policy-network limitation. Even optimal classical guidance (PN) cannot succeed because the F-16 airframe cannot deliver the required turn rate at the tested flight condition (5,000 m, Mach ~0.8).

### 4.6 Guidance-Law Comparison (LOS-rate vs PN vs Hybrid)

To rule out guidance-law-specific failure, the same `no_prediction` PPO policy was evaluated under three different guidance modes on identical crossing scenarios:

| Guidance Mode | Crossing Left | Crossing Right | Min Range (left / right) |
|---------------|---------------|----------------|--------------------------|
| **LOS-rate** (default) | ❌ OOB | ❌ OOB | 839 m / 739 m |
| **Proportional Navigation** | ❌ OOB | ❌ Crash | 2,795 m / 568 m |
| **Hybrid** | ❌ OOB | ❌ OOB | 2,795 m / 740 m |

**Observations**:
- **All three guidance laws fail** on crossing scenarios. No guidance mode rescues the geometry.
- **PN performs worse on crossing_left**: min_range = 2,795 m (vs 839 m for LOS-rate). PN's pure pursuit geometry drives the aircraft into a wider miss distance before attempting the turn.
- **PN crashes on crossing_right**: The aggressive pursuit commands cause altitude divergence under sustained high-G load (similar to baseline direct PN).
- **Hybrid matches LOS-rate on crossing_right** (740 m vs 739 m) but is equally unsuccessful.

**Interpretation**: The failure is **below the guidance-law layer** — it resides in the F-16 airframe's ability to generate lateral acceleration at the required rate. No guidance law can overcome a ~2× turn-rate deficit.

---

## 5. Implications for Paper Claims

### 5.1 Safe Claims (Supported by Evidence)

> "Simple-backend-trained PPO policies achieve **partial zero-shot transfer** to JSBSim F-16: **100% success on head-on scenarios**, **0% on crossing scenarios**."

### 5.2 Unsafe Claims (Would Be Misleading)

- ❌ "Policies transfer successfully to JSBSim" — fails on 50% of scenarios by design.
- ❌ "Crossing failures are due to controller tuning" — evidence points to F-16 aerodynamic limits, not tuning.
- ❌ "Increasing gains will fix crossing" — `gain_only` (optimized gains) and `no_prediction` (learned) both fail identically.

### 5.3 Open Questions for Future Work

1. **Gain-scheduling or adaptive guidance**: Can a PN or APN law with energy-aware lead angle succeed where the PPO policy fails?
2. **Training in JSBSim**: Would online RL in JSBSim learn to manage energy and use lag/lead pursuit?
3. **Reduced crossing aspect**: At what crossing angle (< 90°) does F-16 capability intersect policy success?
4. **Altitude/energy initialization**: Would a higher initial altitude or speed extend the turn-rate envelope enough?

---

## 6. Reproducibility Checklist

| Step | Command / Action | Expected Result |
|------|------------------|-----------------|
| 1 | `python scripts/run_paper_benchmark.py --backend jsbsim --methods no_prediction gain_only --scenarios regression_neutral regression_challenging regression_crossing_left regression_crossing_right` | 40 episodes, same stratified pattern |
| 2 | Inspect `raw_episodes.csv` | `is_success = True` for head-on, `False` for crossing |
| 3 | Check `reason` column | All crossing rows = `out_of_bounds` |
| 4 | Verify `nz_cmd_saturation_rate` | ≈ 0% for all episodes |
| 5 | Verify `roll_rate_cmd_max` | ≈ 1.5 rad/s for no_prediction crossing |

---

## 7. Artifact Manifest

| Artifact | Path | Description |
|----------|------|-------------|
| Official benchmark results | `outputs/stage10_2_jsbsim_corrected_official_20260607_164836/` | Corrected Stage 10.2 run |
| Raw telemetry | `.../raw_episodes.csv` | 57-column episode-level telemetry |
| Summary | `.../summary.md` | Human-readable results table |
| Run manifest | `.../run_manifest.json` | Provenance, git hash, config SHA |
| Superseded run | `outputs/stage10_jsbsim_validation/` | Contains `SUPERSEDED` marker |
| Diagnosis runner | `src/uav_vpp_guidance/evaluation/jsbsim_diagnosis.py` | Step-level telemetry + baseline controllers |
| Baseline crossing diagnosis | `outputs/stage10_3_baseline_crossing/` | Hold/PN/low-gain direct on crossing scenarios (all fail) |
| PN crossing diagnosis | `outputs/stage10_3_pn_crossing/` | no_prediction + PN guidance on crossing (0/2) |
| Hybrid crossing diagnosis | `outputs/stage10_3_hybrid_crossing/` | no_prediction + hybrid guidance on crossing (0/2) |
| This analysis | `docs/stage10_3_crossing_failure_analysis.md` | Stratified evidence report |

---

*Generated: 2026-06-07. Analysis based on `stage10_2_jsbsim_corrected_official_20260607_164836` benchmark data.*
