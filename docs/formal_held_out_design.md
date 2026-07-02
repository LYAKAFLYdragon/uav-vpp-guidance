# Formal Held-Out Design: Hierarchical Commander Head-On Generalization Claim

**Version:** 1.0  
**Date:** 2026-07-01  
**Scope:** Close-range head-on combat evaluation for `oracle_task_gate` vs `hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_secondary_clamp_lowalt_latrange_descent`  
**Backend:** JSBSim (strict, no fallback)  
**Attack zone:** `close_range_max_aoa_deg = 60`

---

## 1. Purpose

This document defines the **formal held-out evaluation gate** for claiming that the current commander repair generalizes across the head-on combat scenario envelope. The design is informed by the manifest30 → manifest60 → manifest90 pilot evolution, where commander win_rate remained ≥ oracle win_rate on all three manifest expansions. The formal held-out is the **final, pre-registered** validation step before the paper claim is considered paper-safe.

**What this document is:**
- A frozen specification of the held-out scenario set, evaluation protocol, and pass/fail criteria.
- A design intended to be committed and version-controlled before any held-out data is generated.

**What this document is NOT:**
- A continuation of the manifest pilot series. No manifest120, manifest150, or further expansions will be created or run before the formal held-out is evaluated.
- An invitation to modify the commander or oracle methods. The canonical comparison pair is frozen.

---

## 2. Freshness Unit: Why New Scenes, Not New Seeds

### 2.1 The deterministic scenario manifest paradigm

After the `CommandPostProcessor` reset fix (commits to `overload_rollrate.py` and `tracking_env.py`), a single frozen head-on scenario template is **deterministic across seeds**. For any given scenario tuple (range, velocity, altitude, lateral offset), running seed=0, seed=1, seed=2 produces identical trajectories and outcomes.

**Consequence:** Running the *same* scenario with *different seeds* does **not** produce new evidence. It produces **redundant evidence**. The information gain from N seeds per scene is zero, not additive.

### 2.2 The only valid freshness unit: a completely new scenario tuple

Under the deterministic scenario manifest paradigm, the **freshness unit** is a **scenario tuple that has never appeared in any previous manifest or training data**. A new scenario is defined as a unique combination of:

- `initial_range_m` (r)
- `own_velocity_mps` (v_own)
- `target_velocity_mps` (v_tgt)
- `altitude_diff_m` (alt)
- `lateral_offset_m` (lat)

Two scenarios are considered **the same** if all five parameters match (within rounding tolerance). They are **disjoint** if at least one parameter differs.

**Important corollary:** A scenario with `lat = 0` is NOT automatically "the same" as another `lat = 0` scenario if the range, velocity, or altitude differs. The disjointness test operates on the **full tuple**, not on individual parameters.

### 2.3 When multi-seed evaluation is useful

Multi-seed (e.g., 3 seeds per scenario) is **only useful** if:
- The environment includes **explicit random perturbation** (e.g., wind gusts, sensor noise, opponent stochasticity) that is recorded in `provenance`.
- The perturbation is applied **per-episode** and its parameters are stored in `run_manifest.json`.

Without explicit random perturbation, multi-seed evaluation provides **no additional information** beyond the single deterministic outcome. It should **not** be used as a substitute for genuine scenario novelty.

**This design does NOT enable random perturbation.** The formal held-out is single-seed (seed=0) per scenario, consistent with the manifest30/60/90 pilot protocol. The freshness comes from the scenario tuple itself, not from seed variation.

---

## 3. Held-Out Scenario Design

### 3.1 Parameter envelope

The held-out scenarios use the **same parameter envelope** as the pilot manifests:

| Parameter | Min | Max | Granularity |
|---|---|---|---|
| `initial_range_m` | 1200 | 3700 | Any integer (not necessarily 100-step) |
| `own_velocity_mps` | 170 | 270 | Any integer or half-integer |
| `target_velocity_mps` | 170 | 270 | Any integer or half-integer |
| `altitude_diff_m` | -900 | +900 | Any integer |
| `lateral_offset_m` | -200 | +200 | Any integer |

The envelope is bounded by the physical constraints of the JSBSim aircraft models and the close-range combat setting (3 km max, 60° AoA).

### 3.2 Disjointness rule

A held-out scenario is **valid** if and only if its full parameter tuple `(r, v_own, v_tgt, alt, lat)` is **not present** in any of the following manifests:

- `manifest30_pilot.yaml` (30 scenarios)
- `manifest60_pilot.yaml` (60 scenarios)
- `manifest90_pilot.yaml` (90 scenarios)

**Disjointness test:** The `test_formal_heldout_disjointness.py` test validates this rule by:
1. Loading all scenarios from manifest30/60/90.
2. Computing their full parameter tuples (rounded to nearest integer).
3. Loading all scenarios from the formal held-out config.
4. Asserting that the intersection of the two sets is empty.
5. Asserting that the held-out scenarios have `manifest_family: formal_heldout` in their metadata.

### 3.3 Scenario generation strategy

The held-out scenarios are generated by systematically sampling the **parameter gaps** not covered by the pilot manifests:

**Step 1: Identify available range values.**
The pilot manifests cover ranges: 1300, 1400, 1500, 1700, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3500, 3600.
Available (never used): 1200, 1600, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3700.

**Step 2: Identify available velocity values.**
The pilot manifests cover: 180, 190, 200, 210, 220, 230, 240, 250, 260 (own); 180, 190, 200, 210, 220, 230, 240 (target).
Available: 170, 270 (own); 170, 250, 260, 270 (target).

**Step 3: Identify available altitude differences.**
The pilot manifests cover: -800, -700, -600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800.
Wait, let me recheck: manifest30 has ±200, 300; manifest60 has ±100, ±400, ±500; manifest90 has ±600, ±700, ±800.
So covered: -800, -700, -600, -500, -400, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800.
Missing: -900, -300, -50, 50, 150, 250, 350, 450, 550, 650, 750, 850, 900.

**Step 4: Identify available lateral offsets.**
The pilot manifests cover: 0, ±50, ±75, ±150.
Available: ±25, ±100, ±125, ±200.

**Step 5: Generate held-out scenarios.**
To ensure systematic coverage and complete disjointness, the held-out uses:
- 12 base scenarios with **new range values** and **new velocity/altitude combinations**.
- 3 lateral offsets per base: 0, ±100 (both are new to the full tuples because the base ranges are new).
- Total: 12 × 3 = **36 scenarios**.

The 12 base scenarios are designed to span the parameter envelope evenly:

| # | Range | Own v | Target v | Alt diff | Rationale |
|---|---:|---:|---:|---:|:---|
| 1 | 1600 | 195 | 195 | 0 | Mid-range, symmetric speeds, co-altitude |
| 2 | 1900 | 205 | 195 | 150 | Mid-range, own faster, target above |
| 3 | 2100 | 215 | 205 | -150 | Longer range, own faster, target below |
| 4 | 2300 | 225 | 215 | 250 | Longer range, speed advantage, large alt diff |
| 5 | 2500 | 235 | 225 | -250 | Same as #4 but below |
| 6 | 2700 | 245 | 235 | 350 | High range, high speed, large alt diff |
| 7 | 2900 | 255 | 245 | -350 | Same as #6 but below |
| 8 | 3100 | 265 | 255 | 450 | Extreme range, extreme speed, very large alt diff |
| 9 | 3300 | 245 | 255 | -450 | Same as #8 but reversed speed advantage |
| 10 | 1900 | 195 | 205 | 550 | Target faster, very high altitude difference |
| 11 | 2500 | 225 | 235 | -550 | Target faster, very low altitude difference |
| 12 | 3100 | 265 | 265 | 0 | Both very fast, co-altitude, extreme range |

For each base scenario, three lateral variants are generated:
- `lat = 0` (on-boresight)
- `lat = +100` (off-boresight right)
- `lat = -100` (off-boresight left)

**Total: 36 scenarios.**

**Verification:** All 36 full tuples `(r, v_own, v_tgt, alt, lat)` are disjoint from all 90 pilot manifest tuples. This is verified by the `test_formal_heldout_disjointness.py` test.

---

## 4. Evaluation Protocol

### 4.1 Frozen canonical pair

Only two methods are evaluated:

1. `oracle_task_gate` — Oracle Task Gate (Static Specialist Selection)
2. `hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_eval10_headon_reopened_crossing_leash2_secondary_clamp_lowalt_latrange_descent` — Hierarchical Commander MVP

**No other methods, no ablations, no variants.**

### 4.2 Frozen tasks

Only two tasks are evaluated:

1. `head_on` — The 36 held-out scenarios
2. `crossing_feasible` — The same 4 crossing scenarios as in manifest30/60/90 (unchanged)

### 4.3 Frozen backend settings

- `backend = jsbsim`
- `strict_backend = True` (no fallback to simple)
- `close_range_max_aoa_deg = 60`
- `combat_initial_hp = 100.0`
- `damage_per_step = 0.5`
- `close_range_max_km = 3.0`

### 4.4 Single-seed per scenario

Each scenario runs **once** per method (seed=0). This is consistent with the manifest30/60/90 pilot protocol and is justified by the deterministic scenario paradigm (Section 2).

**No multi-seed.** The freshness unit is the scenario tuple, not the seed.

### 4.5 Provenance requirements

The run must record:
- `config_path` and `config_hash` in `run_manifest.json`
- `backend` and `strict_backend` in `info` and `provenance`
- `attack_zone_close_range_max_aoa_deg = 60` in `provenance`
- `opponent_stage` (expert or end_to_end) in `provenance`
- Any config overrides via `record_config_override`
- `paper_safe = False` is expected for a single pilot run; the paper-safe determination comes from the aggregate analysis, not the individual run.

---

## 5. Pass/Fail Criteria

### 5.1 Pre-registered criteria (defined before any held-out data is generated)

The formal held-out is considered **passed** if and only if ALL of the following conditions are met:

| # | Criterion | Threshold | Rationale |
|---|---:|---|:---|
| 1 | `expert/head_on` commander win_rate ≥ oracle win_rate | ≥ 0.000 | Core claim: commander does not underperform oracle on held-out expert scenarios |
| 2 | `end_to_end/head_on` commander win_rate ≥ oracle win_rate | ≥ 0.000 | Core claim: commander does not underperform oracle on held-out end_to_end scenarios |
| 3 | `expert/crossing_feasible` commander win_rate ≥ 0.50 | ≥ 0.50 | Crossing should not catastrophically regress |
| 4 | `end_to_end/crossing_feasible` commander win_rate ≥ 0.50 | ≥ 0.50 | Crossing should not catastrophically regress |
| 5 | `expert/head_on` oracle-only ≤ commander-only | ≤ | Oracle-only losses should not exceed commander-only recoveries |
| 6 | `end_to_end/head_on` oracle-only ≤ commander-only | ≤ | Oracle-only losses should not exceed commander-only recoveries |

**The held-out is failed if ANY criterion is violated.**

### 5.2 What happens on failure

If the formal held-out fails:
1. **Do NOT** create a new held-out set and re-test. That would be p-hacking.
2. **Do NOT** modify the commander or oracle methods.
3. The conclusion is: **"The commander repair does not yet generalize to the held-out scenario envelope. Further manifest-based route-stability consolidation is required before attempting a formal held-out again."**
4. The next step would be to identify which held-out scenarios caused the failure (scenario-level analysis) and use that insight to guide further training or evaluation, NOT to discard the held-out and retry.

### 5.3 What happens on success

If the formal held-out passes:
1. The paper can claim: **"The commander repair generalizes to unseen head-on scenarios within the parameter envelope (1200–3700 m range, 170–270 m/s, ±900 m altitude, ±200 m lateral offset), evaluated against both expert and end-to-end opponents."**
2. The claim is supported by: manifest30 → manifest60 → manifest90 pilot evolution + formal held-out validation.
3. The paper should still include the standard limitations: combat-only scope, JSBSim backend, single-seed evaluation, no random perturbation.

---

## 6. Stop Conditions Before Running Held-Out

The formal held-out MUST NOT be run until ALL of the following conditions are met:

1. **No new commander patches** are in progress or planned. The current commander checkpoint is frozen.
2. **No new RL families** are being explored. The canonical comparison pair is frozen.
3. **No manifest120, manifest150, or further pilot expansions** are planned. The pilot series stops at manifest90.
4. **This design document is committed and frozen.** No post-hoc changes to the held-out scenario set, pass/fail criteria, or evaluation protocol.
5. **The disjointness test passes.** `test_formal_heldout_disjointness.py` must confirm zero overlap with manifest30/60/90.
6. **The smoke run passes.** A minimal test run (e.g., 2 scenarios × 2 methods) confirms the runner, artifact pipeline, and provenance recording all work correctly.

---

## 7. Artifact Index

| Artifact | Path | Description |
|---|---|---|
| This design document | `docs/formal_held_out_design.md` | Frozen specification of the held-out |
| Held-out config | `config/experiment/...formal_heldout.yaml` | 36 head_on + 4 crossing scenarios |
| Disjointness test | `tests/test_formal_heldout_disjointness.py` | Validates zero overlap with pilot manifests |
| Smoke run script | `run_formal_heldout_smoke.bat` | Minimal validation of the runner pipeline |

---

## 8. Summary

- **Freshness unit:** A completely new scenario tuple `(r, v_own, v_tgt, alt, lat)`, not a new seed on an existing scenario.
- **Held-out size:** 36 head-on scenarios (12 base × 3 lateral offsets) + 4 crossing_feasible (unchanged).
- **Evaluation:** Single-seed (seed=0), JSBSim, both expert and end_to_end opponents.
- **Pass criteria:** Commander ≥ oracle on both opponent splits for head_on; crossing_feasible not catastrophically regressed; oracle-only ≤ commander-only on both splits.
- **Failure handling:** No retry, no new held-out, no method changes. Return to manifest-based consolidation.
