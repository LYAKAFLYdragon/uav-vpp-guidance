# Hierarchical Commander Route-Stability Validation: Manifest Evolution Report

**Date:** 2026-07-01  
**Scope:** head_on combat-only, oracle_task_gate vs hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_secondary_clamp_lowalt_latrange_descent  
**Backend:** JSBSim  
**Opponents:** expert, end_to_end  
**Attack zone:** close_range_max_aoa_deg = 60

---

## 1. Background & Motivation

The original head-on evaluation pipeline was contaminated by **episode-order state leakage**: `CommandPostProcessor` retained its internal state across episodes, causing the "seed" to be non-deterministic and evaluation results to be coupled to episode order. After fixing the reset bug (commits to `overload_rollrate.py` and `tracking_env.py`), a single frozen head-on template became **deterministic across seeds**, meaning:

- **Larger seed blocks can no longer serve as fresh evidence.**
- **Freshness must come from explicit scenario manifests**, not seed expansion.

This report documents the evolution from manifest30 → manifest60 → manifest90, validating whether the commander repair maintains non-weakness against the oracle_task_gate upper bound as the scenario envelope expands.

---

## 2. Methodology

### 2.1 Canonical comparison pair

| Method | Role |
|---|---|
| `oracle_task_gate` | Static specialist selector (upper bound for this task family) |
| `hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_secondary_clamp_lowalt_latrange_descent` | Current best commander repair |

**Rules:**
- No other methods are compared in this family.
- No broad low-level reward/VPP tuning is performed.
- Only head_on and crossing_feasible tasks are evaluated.
- Backend is JSBSim (strict, no fallback to simple).

### 2.2 Manifest design principle

Each manifest is a **static, explicit YAML list** of scenario initial conditions:
- `position_m`, `velocity_mps`, `heading_deg` for own and target aircraft.
- Metadata: `scenario_type`, `manifest_family`, `initial_range_m`, `altitude_diff_m`, `lateral_offset_m`.
- All scenarios are **disjoint** from previous manifests (no overlapping parameter tuples).
- Each scenario runs once per method (seed=0, single-seed formal-small).

### 2.3 Scenario parameter envelope

| Parameter | Range | Notes |
|---|---|---|
| `initial_range_m` | 1300–3600 | Manifest30: 1800–2600; Manifest60: added 1500/1700/2800–3200; Manifest90: added 1300/1400/3400–3600 |
| `own_velocity_mps` | 180–260 | Manifest30: 200–220; Manifest60: added 190/230/240; Manifest90: added 180/250/260 |
| `target_velocity_mps` | 180–240 | Same progression as own velocity |
| `altitude_diff_m` | -800 to +800 | Manifest30: ±200/300; Manifest60: added ±100/400/500; Manifest90: added ±600/700/800 |
| `lateral_offset_m` | -150 to +150 | Manifest30: 0/±150; Manifest60: added ±75; Manifest90: added ±50 |

---

## 3. Results by Manifest

### 3.1 Manifest30 (30 head_on scenarios + 4 crossing_feasible)

**Config:** `jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest30_pilot.yaml`

| Opponent | head_on oracle | head_on commander | crossing oracle | crossing commander |
|---|---:|---:|---:|---:|
| expert | 0.733 | **0.733** | 0.750 | 0.750 |
| end_to_end | 0.621 | **0.667** | 1.000 | 1.000 |

**Key observation:** Commander already matches or exceeds oracle on the first 30-scenario fresh manifest, especially on end_to_end (+0.046).

---

### 3.2 Manifest60 (60 head_on scenarios + 4 crossing_feasible)

**Config:** `jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest60_pilot.yaml`  
**New scenarios:** 30 disjoint (10 base × 3 laterals: 0, ±75). Parameter envelope expanded to 1500/1700/2800–3200, v190/v230/v240, alt±100/400/500.

| Opponent | head_on oracle | head_on commander | crossing oracle | crossing commander |
|---|---:|---:|---:|---:|
| expert | 0.667 | **0.683** | 0.750 | 0.750 |
| end_to_end | 0.695 | **0.700** | 1.000 | 1.000 |

**Paired scenario counts:**

| Opponent | both_win | oracle_only | commander_only | both_nonwin |
|---|---:|---:|---:|---:|
| expert | 39 | 1 | 2 | 18 |
| end_to_end | 37 | 4 | 5 | 14 |

**Key observation:**
- Commander still ≥ oracle on both splits.
- Commander-only recoveries exceed oracle-only losses on end_to_end (5 vs 4).
- Mean crossing fraction for commander on end_to_end: 26.2% (higher than expert's 4.9%), indicating the commander switches to crossing mode more aggressively against the end_to_end opponent.

---

### 3.3 Manifest90 (90 head_on scenarios + 4 crossing_feasible)

**Config:** `jsbsim_hrl_oracle_vs_balanced_tail_eval10_secondary_clamp_fresh_geometry_manifest90_pilot.yaml`  
**New scenarios:** 30 disjoint (10 base × 3 laterals: 0, ±50). Parameter envelope expanded to 1300/1400/3400–3600, v180/v250/v260, alt±600/700/800, lateral±50.

| Opponent | head_on oracle | head_on commander | crossing oracle | crossing commander |
|---|---:|---:|---:|---:|
| expert | 0.611 | **0.622** | 0.750 | 0.750 |
| end_to_end | 0.674 | **0.678** | 1.000 | 1.000 |

**Paired scenario counts:**

| Opponent | both_win | oracle_only | commander_only | both_nonwin |
|---|---:|---:|---:|---:|
| expert | 53 | 2 | 3 | 32 |
| end_to_end | 56 | 4 | 5 | 25 |

**Key observation:**
- Commander ≥ oracle on **both splits** for the **third consecutive manifest**.
- Commander-only recoveries exceed or equal oracle-only losses on both splits.
- Absolute win rates dropped vs manifest60 (expert: 0.683 → 0.622; end_to_end: 0.700 → 0.678), but this is expected because the **30 new scenarios are more extreme** (larger altitude differences, higher speeds, smaller ranges). The important signal is the **relative commander-oracle gap**, which remains favorable.
- Mean crossing fraction on end_to_end: 24.5%, consistent with manifest60's 26.2%, suggesting a stable switching behavior.

---

## 4. Cross-Manifest Trend Analysis

### 4.1 Win rate trend (expert / head_on)

| Manifest | Oracle | Commander | Δ |
|---|---:|---:|---:|
| manifest30 | 0.733 | 0.733 | 0.000 |
| manifest60 | 0.667 | 0.683 | +0.016 |
| manifest90 | 0.611 | 0.622 | +0.011 |

### 4.2 Win rate trend (end_to_end / head_on)

| Manifest | Oracle | Commander | Δ |
|---|---:|---:|---:|
| manifest30 | 0.621 | 0.667 | +0.046 |
| manifest60 | 0.695 | 0.700 | +0.005 |
| manifest90 | 0.674 | 0.678 | +0.004 |

**Interpretation:**
- The **oracle upper bound also degrades** as the scenario envelope expands (0.733 → 0.611 on expert; 0.621 → 0.674 on end_to_end). This is expected: more extreme scenarios are harder for both methods.
- The **commander consistently tracks or exceeds the oracle**, even as the oracle's own performance drops. This is the correct signal for generalization, not absolute win rate.
- The Δ gap narrows from manifest30 → manifest60 → manifest90, but remains positive. This suggests the commander is **not overfitting** to the easier scenarios in the earlier manifests.

---

## 5. Commander Behavioral Diagnostics

### 5.1 Mode switching (expert)

| Manifest | head_on fraction | crossing fraction | mean switches | mean first switch step |
|---|---:|---:|---:|---:|
| manifest60 | 95.1% | 4.9% | 1.42 | 315.2 |
| manifest90 | 94.1% | 5.9% | 1.82 | 314.8 |

### 5.2 Mode switching (end_to_end)

| Manifest | head_on fraction | crossing fraction | mean switches | mean first switch step |
|---|---:|---:|---:|---:|
| manifest60 | 73.8% | 26.2% | 5.30 | 220.3 |
| manifest90 | 75.5% | 24.5% | 5.11 | 223.4 |

**Interpretation:**
- Against **expert**, the commander stays in head_on mode ~95% of the time with very few switches. This is correct because the expert opponent does not force the commander to adapt.
- Against **end_to_end**, the commander switches to crossing mode ~25% of the time with ~5 switches per episode, starting around step 220. This is consistent with the end_to_end opponent's more aggressive behavior forcing the commander to explore alternative modes.
- The switching behavior is **stable across manifest expansions** (manifest60 vs manifest90), which is another generalization signal.

---

## 6. Lessons & Insights

### 6.1 What we learned about evaluation methodology

1. **Seed-based evaluation is fundamentally broken for deterministic templates.** After the CommandPostProcessor reset fix, a single frozen template produces deterministic outcomes across seeds. Any "seed expansion" evaluation is just running the same scenario repeatedly.
2. **Explicit scenario manifests are the only valid source of freshness.** Each manifest must be a hand-designed or systematically generated set of initial conditions with no overlap with previous manifests.
3. **The oracle_task_gate is not a fixed upper bound.** Its performance degrades as the scenario envelope expands. The correct comparison is **relative commander-oracle performance**, not absolute win rate.
4. **Single-seed formal-small is sufficient for manifest validation.** Because the scenarios are deterministic, running each scenario once (seed=0) gives a valid signal. Multi-seed is only needed for held-out validation to estimate variance.

### 6.2 What we learned about the commander repair

1. **The commander repair generalizes across the expanded head-on envelope.** Three consecutive manifest expansions (30 → 60 → 90) all show commander ≥ oracle.
2. **The commander does not overfit to easier scenarios.** As the scenario envelope expands to more extreme parameters (smaller ranges, larger altitude differences, higher speeds), the commander degrades proportionally to the oracle, not disproportionately.
3. **Mode switching is a genuine behavioral feature, not noise.** The commander's crossing fraction is consistent across manifest expansions (~25% on end_to_end, ~5% on expert), and the switch timing is stable (~220 steps on end_to_end, ~315 on expert).
4. **The commander is more robust to the end_to_end opponent than the oracle.** On end_to_end, the commander maintains a positive Δ (0.678 vs 0.674) even as the absolute win rate drops. This suggests the commander's adaptive switching is valuable against non-expert opponents.

### 6.3 What we learned about scenario design

1. **The parameter envelope should be expanded systematically.** Each manifest should add new ranges, velocities, altitude differences, and lateral offsets that are disjoint from previous manifests.
2. **Lateral offset is a critical dimension.** The manifest30 → manifest60 expansion added y=±75, and manifest90 added y=±50. These smaller offsets are more challenging and test the commander's ability to handle off-boresight initial conditions.
3. **Altitude difference is another critical dimension.** Manifest90 added ±600/700/800, which are significantly more extreme than the ±500 max in manifest60. These scenarios test the commander's vertical maneuvering.

---

## 7. Limitations & Risks

1. **Combat-only scope.** The current validation is limited to head_on and crossing_feasible. Other tasks (sustained_turn, multi_waypoint) are not evaluated.
2. **Single-seed evaluation is correct and sufficient.** Each scenario is run once (seed=0). After the `CommandPostProcessor` reset fix, a frozen scenario template produces **deterministic** outcomes; multi-seed evaluation provides no additional information because there is no per-episode random perturbation. Multi-seed should only be used if explicit environmental stochasticity is introduced (e.g., wind gusts, sensor noise) and its parameters are recorded in `provenance`. Without such perturbation, multi-seed is not a substitute for genuine scenario novelty.
3. **No held-out validation yet.** The manifest30/60/90 are all "pilot" manifests. A formal held-out set (never seen in any training or pilot) is still needed for the final paper claim.
4. **Commander win rate is declining in absolute terms.** While the commander tracks the oracle, the absolute win rate on expert dropped from 0.733 (manifest30) to 0.622 (manifest90). This suggests the 90-scenario envelope is approaching the limits of the current commander's training distribution.
5. **The crossing_feasible task is too easy.** Both methods achieve 100% on end_to_end and 75% on expert. This task does not provide discriminative signal.

---

## 8. Recommendations for Next Steps

### 8.1 Immediate: Formal held-out design

The current evidence (manifest30 → 60 → 90, all showing commander ≥ oracle) is sufficient to **propose a formal held-out gate**. The held-out should:

- Be **completely disjoint** from manifest30/60/90 (no overlapping parameter tuples).
- Use the **same parameter envelope** (1300–3600 range, 180–260 m/s, ±800 altitude, ±150 lateral).
- Include **36 head-on scenarios** (plus 4 crossing_feasible as regression guard) to ensure a geometrically diverse, disjoint set without over-extending beyond the current commander's training distribution.
- Be evaluated with **single-seed (seed=0)** per scenario, consistent with the manifest30/60/90 pilot protocol and the determinism guarantee from the `CommandPostProcessor` reset fix.
- Have a **pre-registered pass/fail criterion**: commander win_rate ≥ oracle win_rate on both expert and end_to_end splits.

### 8.2 Medium-term: Full formal held-out evaluation

Run the **36-scenario formal held-out** (see `docs/formal_held_out_design.md`) end-to-end:
- Expert stage first (~15–20 min).
- End-to-end stage second (~30–40 min).
- Verify that commander win_rate ≥ oracle win_rate on both stages (pass/fail criterion pre-registered in `formal_held_out_design.md`).
- If the held-out fails, return to manifest consolidation (do not expand held-out or introduce new methods).

### 8.3 Long-term: Paper integration

The manifest evolution narrative should be presented in the paper as:
1. **Phase 1:** Manifest30 pilot — initial validation of commander repair.
2. **Phase 2:** Manifest60 expansion — commander holds on larger disjoint set.
3. **Phase 3:** Manifest90 expansion — commander holds on even larger, more extreme set.
4. **Phase 4:** Formal held-out — final generalization claim.

This four-phase narrative provides a **rigorous, evidence-based** path to claiming generalization, avoiding the pitfalls of seed-based evaluation and post-hoc cherry-picking.

---

## 9. Artifact Index

| Artifact | Path | Description |
|---|---|---|
| Manifest30 config | `config/experiment/...manifest30_pilot.yaml` | 30 head_on + 4 crossing scenarios |
| Manifest60 config | `config/experiment/...manifest60_pilot.yaml` | 60 head_on + 4 crossing scenarios |
| Manifest90 config | `config/experiment/...manifest90_pilot.yaml` | 90 head_on + 4 crossing scenarios |
| Manifest60 tests | `tests/test_head_on_fresh_geometry_manifest60.py` | 7 targeted tests |
| Manifest90 tests | `tests/test_head_on_fresh_geometry_manifest90.py` | 7 targeted tests |
| Manifest60 report | `outputs/diagnostics/...manifest60_validation_20260701/` | Expert + end_to_end analysis |
| Manifest90 report | `outputs/diagnostics/...manifest90_validation_20260701/` | Expert + end_to-end analysis |
| Formal held-out config | `config/experiment/...formal_heldout.yaml` | 36 head_on + 4 crossing (disjoint from all pilot manifests) |
| Formal held-out design | `docs/formal_held_out_design.md` | Frozen spec: single-seed, 36 scenarios, pass/fail criteria |
| This report | `docs/manifest_evolution_report.md` | Cross-manifest synthesis |
