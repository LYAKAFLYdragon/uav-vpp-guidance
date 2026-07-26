# P4 Geometry Gate Failure Attribution Audit — Bugfix Design

## Overview

The authorized run `GEO-20260713-R1` (geometry-pretrain-only, 200,000 steps per skill) completed and its
pre-registered per-skill gate emitted `all_skills_ready_for_combat_finetune = false` with the hard
constraint `next_stage = do_not_start_combat_finetune`. **That verdict is legitimate negative evidence and
is NOT the bug.** The gate did exactly what it was designed to do.

The bug is a defect in the **failure-attribution and status-reporting process** surrounding that verdict:

- There is no traceable, per-skill × per-cell attribution matrix explaining *why* each gate check failed.
- The process cannot distinguish "field absent (unverifiable)" from "field present with value 0 (a
  legitimate 0-step count under current gate semantics)".
- There is no explicit reconciliation between the declared cells and the cells the gate actually
  **emits** across the four skills' `selected_gate.geometry_phase_coverage`. The verified reconciliation
  against the canonical `D:\laiyu_ssh\uav-vpp-guidance` evidence is **26 declared / 26 emitted / diff 0 /
  extra 0** for every skill (`pursuit_conversion` 6/6, `lead_intercept` 6/6, `defensive_extension` 6/6,
  `reentry_recovery` 8/8). The absence of a recorded reconciliation and per-cell attribution of the
  present-value-0 declared cells is the defect, NOT any declared-vs-emitted difference.
- There is no strict provenance boundary separating the currently available `D:\laiyu_ssh\uav-vpp-guidance`
  artifacts from the missing frozen run inputs (run commit / `training_plan.json` /
  `resolved_geometry_config.json`), so a provenance gap risks being read as a data defect.
- There is no controlled label for a contract/provenance inconsistency, so such a mismatch risks being
  misclassified as skill-capability failure or physical unreachability.
- The execution checklist still shows P4 as `in progress` (`[>]`, single skill `pursuit_conversion` still
  training), misrepresenting the true completed negative-evidence state.

> **HISTORICAL PREMISE — NOT VERIFIED BY CURRENTLY AVAILABLE ARTIFACTS.** An earlier version of this
> design asserted a declared-vs-emitted divergence of **26 declared / 22 emitted / 4 diff** (the
> `crossing_entry`-related cells for `lead_intercept` and `reentry_recovery`). That premise has been
> empirically **DISPROVEN** against `src/p4_geometry_pretrain_gate.json`, the four
> `src/skills/<skill>/geometry_pretrain_summary.json`, and
> `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`. The verified reconciliation is
> **26/26/diff 0/extra 0**. A 22-emitted output is **structurally impossible** under the current source:
> `evaluate_skill_gate` (`src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py:511-546`) emits the
> full Cartesian product of declared `geometry_states` × `phases`, filling absent combinations with 0 via
> `.get(..., 0)`, so the emitted count ALWAYS equals the declared count. The 26/22/4 numbers are retained
> in this document ONLY as a labeled historical premise / audit subject for provenance reconciliation,
> never as a fact about the P4 gate.
>
> **The real failure structure (verified):** `geometry_phase_coverage = false` is driven by declared cells
> emitted with value 0 (e.g. `lead_intercept` `advantage:post_merge = 0`, `neutral:post_merge = 0`), NOT by
> any missing/absent cell; `intent_progress` fails for three skills (e.g. `lead_intercept`
> `intent_progress_fraction = 0.498 < 0.55`); and `reentry_recovery` passes `intent_progress` but is still
> blocked by phase-coverage failure. The top-level gate JSON and the four per-skill summaries agree exactly
> on `geometry_phase_coverage` (cross-artifact consistency is CLEAN).

The fix (`F'`) is to produce a correct, evidence-backed, reproducible attribution audit — an audit `.md`
(zh) plus a matrix `.csv` and `.json`, driven by a configurable analysis script in `scripts/` with a
minimal test — and to update the checklist to its final P4 negative-evidence state. `F'` explicitly does
**not** retrain, does **not** re-evaluate, does **not** relax or edit the pre-registered gate, and does
**not** touch any P4 outputs, checkpoints, model, reward, or the canonical two-skill paper mainline. The
final conclusion must remain: **under the pre-registered gate, combat finetune is NOT allowed to start,
and this audit has no authority to change that.**

A structural limitation is carried through the whole design: coverage steps are recorded keyed by the
scenario's **fixed `initial_class`**, not by a per-step re-classified live posture
(`_record_episode_summary` at `src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py:451` writes
`state_phase_steps[f"{initial_state}:{metric['phase']}"]`; call sites pass
`scenario["metadata"]["initial_class"]` at line 806 for training and line 677 for dev30). Therefore this
audit can only prove **initial-class × phase** coverage and CANNOT prove whether the dynamic five-state
posture actually occurred during `post_merge` / `re_entry`.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug — an audit/reporting request against
  `GEO-20260713-R1` where the process cannot produce a traceable per-skill × per-cell attribution, cannot
  reconcile declared vs emitted cells, cannot distinguish absent-vs-zero fields, and lacks a controlled
  label for contract/provenance inconsistency.
- **Property (P)**: The desired behavior for C — an auditable, field-path-traceable attribution matrix +
  audit doc + reconciled contract preflight, with the correct `do_not_start_combat_finetune` conclusion
  preserved.
- **Preservation**: All P4 outputs, gate definition/thresholds/semantics, checkpoints, model, reward,
  canonical two-skill mainline, locked stages (combat/P5/P6/P7), local worktree state, and passing
  sub-checks must remain exactly as they are.
- **initial-class × phase**: The only coverage claim the recorded data supports — steps keyed by the
  scenario's fixed `initial_class` label crossed with the recorded `phase`. NOT a claim about dynamic live
  posture.
- **Declared cell**: A `skill × geometry_state × phase` triple derived from the shared-skill registry
  `geometry_phase_coverage` blocks (`geometry_states` × `phases`), used as a documented
  `provenance_unverified` substitute for the missing frozen `training_plan.json`. Total across four
  skills: **26** (`pursuit_conversion` 3×2=6, `lead_intercept` 3×2=6, `defensive_extension` 2×3=6,
  `reentry_recovery` 4×2=8).
- **Emitted cell**: A `state:phase` key actually present in a skill's
  `selected_gate.geometry_phase_coverage` map. **Emitted cell total on the current artifact: 26** — equal
  to the declared total, because `evaluate_skill_gate` (`thesis_shared_skill_geometry.py:511-546`) emits
  the full Cartesian product of declared states × phases (absent combinations filled with 0).
- **Diff cell**: A declared cell with no corresponding emitted key. **On the current artifact the diff set
  is EMPTY (26 − 26 = 0); there is no diff cell.** (The historical premise of a 4-cell `crossing_entry`
  diff for `lead_intercept` and `reentry_recovery` is DISPROVEN and structurally impossible under
  `evaluate_skill_gate:511-546` — retained only as an unverified historical premise.)
- **selected_gate**: The gate snapshot of the **SELECTED checkpoint** for a skill in
  `p4_geometry_pretrain_gate.json` (`skills.<skill>...gate`, structure per `evaluate_skill_gate` in
  `thesis_shared_skill_geometry.py`, return block at `:567-575`). Training selects the candidate
  checkpoint by **whether it passes the gate, then by intent progress** (selection/sort logic at
  `thesis_shared_skill_geometry.py:816`); `selected_gate` is therefore the gate object of that chosen
  checkpoint — **NOT** the last dev30 evaluation. The audit report SHALL record the corresponding
  evaluation step (the checkpoint's evaluation step) for the selected checkpoint of each skill.
- **`evaluate_skill_gate`**: The function in `thesis_shared_skill_geometry.py` that computes
  `checks`, `geometry_phase_coverage`, `intent_progress_fraction`, `profile_coverage`,
  `safety_by_opponent`, and `status` for a skill.
- **F**: The original (defective) attribution/reporting process. **F'**: The fixed audit process.

## Bug Details

### Bug Condition

The bug manifests when a reviewer or downstream decision process inspects the state of `GEO-20260713-R1`
to decide whether combat finetune is allowed. The attribution/reporting process is unable to produce a
reproducible, field-path-traceable, per-cell explanation of the gate failure; it cannot reconcile
declared vs emitted cells; it conflates "field absent" with "field present value 0"; and it has no
controlled label for a contract/provenance inconsistency.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type AuditRequest
         (references the frozen GEO-20260713-R1 evidence set)
  OUTPUT: boolean

  RETURN input.target_run == "GEO-20260713-R1"
         AND gateResultExists(input)                 // p4_geometry_pretrain_gate.json present
         AND requestsFailureAttribution(input)        // a why/next-step decision is being made
         AND (
              NOT hasPerCellAttributionMatrix(input)              // 1.1
              OR NOT tracesEachCellToFieldPath(input)             // 1.2
              OR conflatesAbsentWithZero(input)                   // 1.3
              OR NOT distinguishesNotObservedFromObservedFail(input) // 1.4
              OR NOT reconcilesDeclaredVsEmittedCells(input)      // 1.8
              OR NOT hasContractProvenanceLabel(input)            // 1.9
              OR overclaimsDynamicPosture(input)                  // 1.10
              OR NOT frozenProvenanceRecorded(input)              // 1.6
             )
END FUNCTION
```

Note: the gate verdict itself (`all_skills_ready_for_combat_finetune == false`) is NOT part of the bug
condition. `isBugCondition` is about the *attribution/reporting* defect, never about the gate's `false`
result being "wrong".

### Examples

- **Missing per-cell attribution** — `p4_geometry_pretrain_gate.json → skills.defensive_extension →
  (selected checkpoint) → gate.geometry_phase_coverage` shows `disadvantage:post_merge = 0`,
  `disadvantage:re_entry = 0`, `neutral:post_merge = 0`, `neutral:re_entry = 0`. *Expected*: each of these
  four cells attributed individually to a field path with a controlled label. *Actual (F)*: no per-cell
  matrix exists; the failure is reported only as "geometry_phase_coverage: false".
- **Present-value-0 attribution (the dominant real case)** — `lead_intercept`'s
  `selected_gate.geometry_phase_coverage` (`src/skills/lead_intercept/geometry_pretrain_summary.json:1685`
  and the matching top-level block in `src/p4_geometry_pretrain_gate.json`) shows `advantage:post_merge = 0`
  and `neutral:post_merge = 0` **present with value 0**, alongside `crossing_entry:post_merge = 182` and
  `crossing_entry:pre_merge = 1092` (both present, non-zero). *Expected*: each present-value-0 cell reported
  as "recorded as 0 steps under current gate semantics" and attributed individually to a field path.
  *Actual (F)*: no per-cell matrix; failure reported only as "geometry_phase_coverage: false". NOTE: on the
  current artifact ALL declared cells are PRESENT, so the dominant case is `present_value_0`, never absent.
- **Absent vs zero distinction (analytically retained, not exercised on this run)** — the absent branch of
  the encoding is kept for reuse, but on the `D:\laiyu_ssh\uav-vpp-guidance` evidence no declared cell is
  absent. Reconciliation is 26 declared / 26 emitted / diff 0 / extra 0 for every skill, so the
  absent-vs-present-value-0 discrimination resolves entirely to `present_value_0` / `present_value_gt_0`
  for this artifact. Any claim of a `crossing_entry` cell being "absent" is the disproven historical
  premise, not a fact.
- **Reconciliation record gap (the genuine defect)** — the four
  `selected_gate.geometry_phase_coverage` maps emit exactly the 26 declared cells (diff 0, extra 0), yet no
  reconciliation record and no per-cell attribution of the present-value-0 cells exists. *Expected*: the
  reconciliation preflight records 26/26/diff-0/extra-0 and the matrix attributes each present-value-0 cell.
  *Actual (F)*: no reconciliation record; the divergence historically alleged (26/22/4) is unverified and
  structurally impossible under `evaluate_skill_gate:511-546`.
- **Checklist misrepresentation** — the checklist line
  `reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md:269` reads `[>] …当前
  pursuit_conversion 正在训练…`. *Expected*: final negative-evidence state (run complete, all four skills
  200k, formal negative evidence). *Actual (F)*: represents an in-progress single-skill run.
- **Edge case (passing sub-checks)** — `reentry_recovery` passes intent progress; all four skills pass
  profile coverage and safety. *Expected*: reported accurately as passes and preserved. This is NOT part
  of the bug and must not be recharacterized as failure (Requirement 3.9).

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- The P3 temporal encoder checkpoint
  (`p3_temporal_encoder_phaseconditional_v2/checkpoints/best.pt`, SHA256
  `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`) and its contract (10 frames × 16-D
  history, 32-D frozen embedding, 66-D skill input, 3-D normalized VPP) remain frozen (Req 3.1).
- All current P4 outputs, checkpoints, model, and reward remain untouched: no retraining, no
  re-evaluation reruns, no parameter changes, no gate relaxation (Req 3.2).
- The pre-registered per-skill gate definition, its thresholds
  (`minimum_geometry_phase_coverage_per_declared_cell: 2`, `minimum_intent_progress_fraction: 0.55`), and
  its coverage semantics remain unmodified (Req 3.3).
- Combat finetune, P5 high-level PPO, P6 ablation, and P7 heldout60 remain locked (Req 3.4).
- The canonical two-skill paper mainline results, docs, and checkpoint registry remain unmodified
  (Req 3.5).
- Any unrelated local working-branch changes are preserved via a separate git worktree; the remote is
  synced first; local state is never reset/overwritten/deleted (Req 3.6).
- Passing sub-checks (`reentry_recovery` intent progress; all four skills' profile coverage and safety)
  are reported accurately and not recharacterized as failures (Req 3.9).

**Scope:**
All inputs that do NOT satisfy `isBugCondition` are completely unaffected by this fix. This includes:
- Any request that is not an attribution/decision request against `GEO-20260713-R1`.
- The gate's own computation and its `false` verdict (which is preserved verbatim as evidence).
- Reads of unrelated runs, the two-skill mainline, or other phases.

The gate's coverage semantics are honored, not "fixed": the audit reports on them; it never edits them
(Req 3.3, 3.8). The raw-train-trajectory prohibition (`raw_train_trajectory_persistence: "prohibited"`,
`thesis_shared_skill_geometry.py:752`) is honored — at most a static reachability analysis from manifest +
phase rules is performed, and the absence of per-step evidence is reported honestly (Req 3.8).

_The expected correct behavior for buggy inputs is defined in the Correctness Properties section
(Property 1). This section captures what must NOT change._

## Hypothesized Root Cause

> **Framing statement (authoritative):** There is **NO declared-versus-emitted divergence** on the current
> `D:\laiyu_ssh\uav-vpp-guidance` artifact — reconciliation is verified 26 declared / 26 emitted / diff 0 /
> extra 0. Under the current `evaluate_skill_gate` implementation
> (`thesis_shared_skill_geometry.py:511-546`), every declared state-phase key IS emitted, with an explicit
> zero when unobserved. The genuine issue this audit exists to attribute is therefore (i) the absence of a
> recorded reconciliation and per-cell attribution of the present-value-0 declared cells, and (ii) a
> **provenance gap**: the frozen run commit, `training_plan.json`, and `resolved_geometry_config.json` are
> NOT available in this repo, and the historical 26/22/4 premise cannot be checked against them. Where
> provenance cannot be established, the only admissible label is `contract_or_provenance_mismatch` with
> `provenance_unverified` — triggered by the missing provenance inputs and the unverifiable historical
> premise, NOT by any declared-vs-emitted diff (which is 0).

Based on the bug analysis, the defect stems from the absence of a dedicated, evidence-driven attribution
layer over the frozen gate output. The most likely contributing causes are:

1. **The historical 26/22/4 divergence is a PROVENANCE gap, not a data divergence (the divergence is NOT
   present on this artifact)**: In the current implementation, `evaluate_skill_gate`
   (`thesis_shared_skill_geometry.py:511`) iterates over **ALL** `required_states × required_phases` and
   explicitly emits a zero-valued cell via `combined["state_phase_steps"].get(..., 0)` (coverage
   construction at `thesis_shared_skill_geometry.py:544`). Consequently the emitted count ALWAYS equals the
   declared count, and the verified reconciliation on the canonical evidence is 26/26/diff-0/extra-0. The
   earlier claim that the plan declared 26 while the gate emitted only 22 (a 4-cell `crossing_entry` diff)
   is **structurally impossible** under this source and is retained only as an **unverified historical
   premise**. The real, attributable issue is a **provenance gap** whose sources include:
   - (a) the frozen run commit that produced the artifacts cannot be located in this repo,
   - (b) the frozen `training_plan.json` is missing (declared cells are substituted from the registry as a
     documented `provenance_unverified` stand-in), and
   - (c) the frozen `resolved_geometry_config.json` is missing, so the run's resolved config/schema_version
     cannot be pinned.

   The audit MUST attempt to locate the run commit that produced the frozen artifacts. When it cannot be
   located, only `provenance_unverified` may be written, and **NO stronger causal explanation is
   permitted**. The 26/22/4 premise is pre-assigned `contract_or_provenance_mismatch` (as an unverifiable
   historical claim) pending provenance reconciliation; it is NOT attributed to any current declared-vs-
   emitted difference, because that difference is 0.

2. **Absent-vs-zero ambiguity in the coverage map (dominant case is present-value-0)**:
   `coverage[f"{state}:{phase}"] = combined["state_phase_steps"].get(..., 0)`
   (`thesis_shared_skill_geometry.py:544`) legitimately emits `0` for a declared cell that never occurred.
   On the current artifact every declared cell is PRESENT, so the dominant case is `present_value_0` (e.g.
   `lead_intercept advantage:post_merge = 0`, `neutral:post_merge = 0`). The absent branch of the encoding
   is analytically retained for reuse but is NOT exercised by this run. A downstream reader that does not
   distinguish "0 emitted" from "key never present" still risks wrongly treating absence as physical
   impossibility — but on this artifact no cell is absent.

3. **Missing controlled label for contract/provenance inconsistency**: The existing five labels describe
   only skill/observability outcomes; a training-plan ↔ gate-output ↔ registry inconsistency has no home
   and gets forced into a skill-capability or unreachability bucket.

4. **initial-class × phase recording keyed on fixed `initial_class`**: Because
   `_record_episode_summary` (`thesis_shared_skill_geometry.py:451`, call sites 677/806) keys steps by the
   scenario's fixed `initial_class`, the coverage map cannot evidence dynamic posture in
   `post_merge` / `re_entry`; any "state × phase" reading over-claims dynamic posture.

5. **No frozen provenance record**: Result JSONs alone are treated as sufficient; the SHA256s, top-level
   field lists, `resolved_geometry_config.json.schema_version`, the registry/dev30 manifest, and the run
   commit SHA are not captured, so the frozen run cannot be distinguished from the working tree.

## Correctness Properties

Property 1: Bug Condition — Traceable Per-Cell Attribution With Reconciled Contract

_For any_ audit input where the bug condition holds (`isBugCondition` returns true), the fixed audit
process SHALL produce an initial-class × phase attribution matrix covering all four skills whose rows
count equals the declared-cell count auto-derived from the registry (a `provenance_unverified` substitute
for the missing `training_plan.json`; this run: 26), SHALL run BOTH preflights before populating the
matrix — a Gate Contract Integrity Preflight (2a) that reconciles declared cells (26) against emitted cells
and records the verified result **26 declared / 26 emitted / diff 0 / extra 0** (the diff set is EMPTY on
this artifact), which is NOT a halting condition (the audit STILL builds the full 26-row matrix and exits
0), and a Cross-Artifact Consistency Preflight (2b) that deep-checks the top-level
`p4_geometry_pretrain_gate.json` `skills.<name>` against the four
`skills/<name>/geometry_pretrain_summary.json` files and, on the current artifact, confirms they agree
exactly (2b CLEAN); (2b halts ONLY on inconsistency, emitting a short integrity report, producing NO
attribution conclusions, and exiting NON-ZERO) — SHALL link every cell to a specific
JSON/config/manifest/source field path with counts, SHALL distinguish field-absent (`unverifiable`) from
field-present-value-0 ("recorded as 0 steps under current gate semantics") across the three status columns
(`train_field_status`, `dev_gate_field_status`, `emission_status`) while reporting that on this artifact
every declared cell is PRESENT (dominant case `present_value_0`, absent branch analytically retained but
not exercised), SHALL classify every failure using only the six controlled labels (including
`contract_or_provenance_mismatch`, which on THIS artifact is triggered by the missing provenance inputs and
the unverifiable historical 26/22/4 premise — NOT by any declared-vs-emitted diff) with a secondary
`reason_code`, SHALL carry the initial-class × phase limitation into column
semantics/conclusions/per-row confidence, SHALL freeze provenance (SHA256s, top-level field lists,
`schema_version` when available, manifest, run commit SHA or `provenance_unverified`), and SHALL conclude
that combat finetune is NOT allowed under the pre-registered gate.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.9, 2.10, 2.11, 2.8**

Property 2: Preservation — Frozen Assets Unchanged (SHA-verified, allowlisted writes only)

_For any_ input where the bug condition does NOT hold (`isBugCondition` returns false), the fixed process
SHALL leave all frozen P4 assets byte-identical. Because the original process `F` is not an executable
program, preservation is verified by computing the P4 artifact SHA256 set **before and after** running the
audit and asserting that: (a) the only files written are the allowlisted three reports plus the execution
checklist; (b) every other P4 artifact's SHA256 is unchanged; and (c) the gate verdict
(`all_skills_ready_for_combat_finetune == false`), the gate thresholds (`min coverage 2`,
`min intent fraction 0.55`), and the P3 encoder checkpoint SHA256 are unchanged. This preserves: the
pre-registered gate definition/thresholds/semantics and its `false` verdict, all P4
outputs/checkpoints/model/reward, the P3 encoder and its contract, the locked stages (combat/P5/P6/P7), the
canonical two-skill mainline, local worktree state, the raw-train-trajectory prohibition, and all passing
sub-checks (`reentry_recovery` intent progress; four-skill profile coverage and safety), which SHALL be
reported accurately and never recharacterized as failures.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9**

## Fix Implementation

### 0. Strict Provenance Boundary

This audit covers **only** the P4 evidence currently available in the in-repo `D:\laiyu_ssh\uav-vpp-guidance`
layout:
- gate JSON at `src/p4_geometry_pretrain_gate.json`,
- the four per-skill summaries at `src/skills/<skill>/geometry_pretrain_summary.json`,
- the registry at `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`.

It is **NOT** a reproduction of the original `GEO-20260713-R1` run that lived under the E: drive results
tree. The following are treated as a hard boundary (Req 2.6, 1.6):
- The frozen **run commit SHA**, **`training_plan.json`**, and **`resolved_geometry_config.json`** are NOT
  present in this repo. Each missing input is recorded explicitly as `provenance_unverified`.
- The current working-tree / HEAD source is **NEVER** substituted for frozen run facts. The frozen
  provenance block and working-tree git info are recorded as **separate** fields so they can never be
  confused.
- **Declared cells are derived from the shared-skill registry**
  (`config/experiment/thesis_five_state_shared_skill_registry_v1.yaml` `geometry_phase_coverage` blocks) as
  a documented substitute for the missing `training_plan.json`. That substitution is itself recorded as
  `provenance_unverified`.
- The `--dev30-manifest-path` is optional; if the dev30 manifest is not present in the repo it is recorded
  as `provenance_unverified`.

No drive letters appear in logic; all defaults are **relative repo paths**.

### 1. Inputs and Provenance Freezing

The audit freezes an explicit evidence set. All paths are **configurable** (no hardcoded drive letters);
defaults point at the in-repo `D:\laiyu_ssh\uav-vpp-guidance` layout (expressed as relative repo paths) but
are overridable via config/CLI (Req 2.6, 3.7).

**Evidence inputs (frozen):**

| Input | Default location (relative repo path, overridable) | Role | Availability |
|---|---|---|---|
| Four per-skill summaries | `src/skills/<skill>/geometry_pretrain_summary.json` | per-skill dev30 gate evidence + `selected_gate` | PRESENT |
| `p4_geometry_pretrain_gate.json` | `src/p4_geometry_pretrain_gate.json` | top-level combined gate + `selected_gate` per skill | PRESENT |
| registry as of run | `--registry-path` (REQUIRED) default `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml` | declared cells source (`provenance_unverified` substitute for `training_plan.json`); declared geometry states/phases/profiles | PRESENT |
| `training_plan.json` | `--training-plan` (OPTIONAL) | frozen declared cells | MISSING → `provenance_unverified` |
| `resolved_geometry_config.json` | `--resolved-config` (OPTIONAL) | frozen resolved config incl. `schema_version` | MISSING → `provenance_unverified` |
| dev30 manifest as of run | `--dev30-manifest-path` (OPTIONAL) | dev30 evaluation manifest | if absent → `provenance_unverified` |
| run commit SHA | git of the frozen run (NOT working tree) | provenance pin | MISSING → `provenance_unverified` |

Default results-root layout is the in-repo `D:\laiyu_ssh\uav-vpp-guidance` tree, referenced only via
relative repo paths (`src/…`, `config/…`); never hardcoded as an absolute drive path inside logic.

**For each input file that IS available the audit records** (Req 2.6, 1.6):
- `sha256` (canonical hash of file bytes),
- `top_level_fields`: sorted list of JSON top-level keys (there is NO formal JSON Schema and NO embedded
  `run_id` per summary — this is recorded as a fact, not treated as a defect),
- for `resolved_geometry_config.json` (when available): `schema_version`.

**Provenance rule** (Req 2.6): for any missing frozen input (run commit SHA, `training_plan.json`,
`resolved_geometry_config.json`, dev30 manifest) record it explicitly as `provenance_unverified`; NEVER
substitute the current working-tree source for frozen run facts. The declared-cell derivation from the
registry is likewise flagged `provenance_unverified`. The frozen provenance block and the working-tree git
info are recorded as **separate** fields so they can never be confused.

**Confirmed gate semantics** (cited, not restated loosely) (Req 2.6):
- coverage unit = **policy steps** (not episodes); gate minimum = **≥ 2 steps per declared cell**
  (`minimum_geometry_phase_coverage_per_declared_cell: 2`, implemented at `thesis_shared_skill_geometry.py:544`/`:561`);
- intent progress = `intent_progress_steps / policy_steps`, a step counts toward the numerator only if its
  `intent_progress > 0` (`thesis_shared_skill_geometry.py:451`), evaluated against threshold **0.55**
  (`minimum_intent_progress_fraction: 0.55`, `:564`).

### 2. Preflights (BOTH run BEFORE the matrix is populated)

Two independent preflights run before any per-cell attribution. They handle **two fundamentally different
kinds of inconsistency** and therefore have **different halting behavior** — do not conflate them:

- **2a. declared-vs-emitted reconciliation (the audit subject for this run).** On the current
  `D:\laiyu_ssh\uav-vpp-guidance` artifact the reconciliation is verified **26 declared / 26 emitted /
  diff 0 / extra 0** — the diff set is **EMPTY**. The historical premise of a 26/22/4 divergence is
  DISPROVEN and structurally impossible under `evaluate_skill_gate:511-546`. Whether the diff is empty (as
  here) or not, 2a is **not** an artifact-integrity failure: the audit **STILL builds the full 26-row
  matrix** and **exits 0 (success)**. On this artifact there are no diff rows to label; any residual
  `contract_or_provenance_mismatch` comes from the provenance gap (§0), not from a declared-vs-emitted
  diff. Continue → full 26-row matrix → exit 0.
- **2b. top-level gate vs the four per-skill `geometry_pretrain_summary.json` disagreement (genuine
  artifact-integrity failure).** If the top-level combined gate disagrees with the per-skill summaries,
  the paired artifacts cannot be trusted at all. The audit emits **only a short integrity report**, exits
  **NON-ZERO**, and produces **NO capability-attribution conclusions** (no matrix). On the current artifact
  2b is **CLEAN**: the top-level gate JSON and the four per-skill summaries agree exactly on
  `geometry_phase_coverage`. Integrity report → non-zero exit → no attribution (only when a disagreement
  exists).

Both preflight results are recorded before the matrix is populated; only a 2b failure halts the audit. On
this artifact 2a diff is EMPTY and 2b is CLEAN.

#### 2a. Gate Contract Integrity Preflight (declared vs emitted reconciliation)

**Data structures:**
```
DeclaredCell   = { skill, geometry_state, phase }              // from registry (provenance_unverified substitute for training_plan.json)
EmittedCell    = { skill, geometry_state, phase, steps:int }   // from selected_gate.geometry_phase_coverage
ReconcileResult = {
  declared: List[DeclaredCell],           // expected 26
  emitted:  List[EmittedCell],            // verified 26 (equals declared)
  diff:     List[DeclaredCell],           // declared - emitted, verified 0 (EMPTY)
  extra:    List[EmittedCell],            // emitted - declared, verified 0
}
```

**Derivation of declared cells** from the registry `geometry_phase_coverage` blocks
(`geometry_states` × `phases`), used as a `provenance_unverified` substitute for the missing
`training_plan.json`:

| Skill | states × phases | declared cells |
|---|---|---|
| `pursuit_conversion` | 3 × 2 | 6 |
| `lead_intercept` | 3 × 2 | 6 |
| `defensive_extension` | 2 × 3 | 6 |
| `reentry_recovery` | 4 × 2 | 8 |
| **Total** | | **26** |

**Comparison algorithm:**
```
FUNCTION reconcile(declared_source, gate_json):     // declared_source = registry (training_plan.json substitute)
  declared = []
  FOR skill IN declared_source.skills:
    FOR state IN skill.geometry_states:             // registry geometry_phase_coverage.geometry_states
      FOR phase IN skill.phases:                    // registry geometry_phase_coverage.phases
        declared.append(DeclaredCell(skill.name, state, phase))
  emitted = []
  FOR skill IN gate_json.skills:
    cov = selected_gate(skill).geometry_phase_coverage        // selected-checkpoint gate map
    FOR "state:phase" -> steps IN cov:
      emitted.append(EmittedCell(skill.name, state, phase, steps))
  declared_keys = { (c.skill, c.state, c.phase) for c in declared }
  emitted_keys  = { (c.skill, c.state, c.phase) for c in emitted }
  diff  = [ c for c in declared if key(c) NOT IN emitted_keys ]   // verified EMPTY (0)
  extra = [ c for c in emitted  if key(c) NOT IN declared_keys ]  // verified 0
  RETURN ReconcileResult(declared, emitted, diff, extra)
```

The preflight result is recorded (verified counts **26 / 26 / diff 0 / extra 0** and the empty diff list)
**before** the matrix is populated so downstream attribution rows can reference it. On this artifact there
are no diff cells to label; the `contract_or_provenance_mismatch` label is instead driven by the
provenance gap (§0) and the unverifiable historical premise (Req 2.9, 2.11, 1.8, 1.9).

**Halting behavior for 2a: continue, build the full matrix, exit 0.** On the current artifact the diff is
EMPTY, so there is nothing to halt on; even a non-empty diff would be the **audit subject**, not a fatal
artifact-integrity error. The audit STILL builds the full 26-row matrix (every row carries
`emission_status = emitted` on this run, since diff is empty), emits all three reports, and **exits 0
(success)**. 2a never triggers a non-zero exit and never suppresses attribution.

#### 2b. Cross-Artifact Consistency Preflight (top-level gate vs per-skill summaries)

A deep consistency check reconciles the **top-level** `p4_geometry_pretrain_gate.json` `skills.<name>`
block against the **four** per-skill `skills/<name>/geometry_pretrain_summary.json` files. The top-level
combined gate and the per-skill summaries must agree on the same run's facts; if they disagree, the paired
artifacts cannot be trusted for fine-grained attribution.

**Deep-consistency algorithm:**
```
FUNCTION crossArtifactConsistency(gate_json, per_skill_summaries):
  problems = []
  FOR skill IN gate_json.skills:
    top    = gate_json.skills[skill.name]
    detail = per_skill_summaries[skill.name]          // skills/<name>/geometry_pretrain_summary.json
    IF detail IS MISSING:
      problems.append((skill.name, "missing_per_skill_summary")); CONTINUE
    # Compare the shared, run-identifying facts field-by-field:
    ASSERT_EQ_OR_RECORD(top.selected_gate.geometry_phase_coverage,
                        detail.selected_gate.geometry_phase_coverage, problems, skill, "geometry_phase_coverage")
    ASSERT_EQ_OR_RECORD(top.selected_gate.intent_progress_fraction,
                        detail.selected_gate.intent_progress_fraction, problems, skill, "intent_progress_fraction")
    ASSERT_EQ_OR_RECORD(top.selected_gate.status,
                        detail.selected_gate.status, problems, skill, "status")
    ASSERT_EQ_OR_RECORD(top.selected_gate.profile_coverage,
                        detail.selected_gate.profile_coverage, problems, skill, "profile_coverage")
    ASSERT_EQ_OR_RECORD(top.selected_gate.safety_by_opponent,
                        detail.selected_gate.safety_by_opponent, problems, skill, "safety_by_opponent")
    ASSERT_EQ_OR_RECORD(top.selected_checkpoint_step,
                        detail.selected_checkpoint_step, problems, skill, "selected_checkpoint_step")
  RETURN problems      // empty == consistent
```

**Halting behavior for 2b (genuine artifact-integrity failure): integrity report, NON-ZERO exit, no
attribution (Req 2.9, 2.11, 1.9).** Unlike 2a, a 2b disagreement means the top-level gate and the per-skill
summaries describe different facts about the same run, so the paired artifacts cannot be trusted for any
capability attribution. If `crossArtifactConsistency` returns any problem, the audit:
1. records the specific inconsistent field path(s) in a short **integrity report**,
2. does **NOT** build the attribution matrix and does **NOT** emit any capability-attribution conclusions,
3. exits **NON-ZERO**,
4. surfaces this as an input to a subsequent P4 design review, never as a skill-capability failure and
   never resolved by editing the gate.

This is the key distinction from 2a: a 2a reconciliation (even a non-empty diff) → continue, full matrix,
exit 0; a 2b top-level-vs-per-skill disagreement → integrity report, non-zero exit, no attribution. On the
current artifact 2b is **CLEAN** (top-level gate and per-skill summaries agree exactly), so the audit
continues to the full matrix and exits 0.

Both preflight results (2a reconciliation = 26/26/diff-0/extra-0; 2b cross-artifact consistency = CLEAN)
are recorded before the matrix is populated; only a 2b failure halts the run.

### 3. Attribution Matrix Schema

Each matrix row corresponds to one declared cell (26 rows expected). Columns (Req 2.1):

| Column | Meaning |
|---|---|
| `skill` | skill name |
| `initial_class` | declared taxonomy initial-class label (the "state" — explicitly the fixed initial_class label, NOT per-step live posture) |
| `phase` | required phase |
| `profile_validity_precondition` | profile / validity-mask precondition for the cell |
| `train_design_requires_coverage` | whether the training-distribution design requires this cell |
| `train_coverage_evidence` | training coverage field path + count |
| `train_coverage_steps` | integer step count (or `absent`) |
| `dev30_coverage_evidence` | dev30 coverage field path + count |
| `dev30_coverage_steps` | integer step count (or `absent`) |
| `train_field_status` | `present_value_0` \| `present_value_gt_0` \| `absent` — derived from `train_aggregate.state_phase_steps` (training coverage) |
| `dev_gate_field_status` | `present_value_0` \| `present_value_gt_0` \| `absent` — derived from `selected_gate.geometry_phase_coverage` (dev gate coverage) |
| `emission_status` | `emitted` \| `declared_not_emitted` — declared-vs-emitted reconciliation outcome for the cell (from preflight 2a). **On this run every row is `emitted`** (diff 0) |
| `gate_min_steps` | 2 (from `minimum_geometry_phase_coverage_per_declared_cell`) |
| `cell_gate_passed` | **per-cell** value computed BY THE AUDIT: `(dev_gate_field_present AND steps >= 2)`. This is the audit's own per-cell derivation, NOT a value the gate emits per cell |
| `skill_geometry_phase_gate_passed` | **skill-level** aggregate verdict read from the gate (`geometry_phase_coverage` true/false is a whole-skill aggregate, NOT per-cell) |
| `skill_intent_progress_passed` | **skill-level** aggregate verdict read from the gate (whole-skill) |
| `intent_progress_fraction` | skill-level `intent_progress_steps/policy_steps` |
| `intent_progress_threshold` | 0.55 |
| `attribution_label` | one of the six controlled labels |
| `reason_code` | secondary, machine-readable reason carried alongside the label (e.g. `dev_coverage_gap`, `initial_class_phase_limitation`, `provenance_unverified`); empty when not applicable |
| `evidence_paths` | JSON/config/manifest/source field paths backing this row |
| `confidence` | conclusion confidence (carries initial-class × phase limitation) |
| `next_step_recommendation` | recommended next action for this cell |

Every column that reads "state" is the fixed `initial_class` label (Req 2.1, 2.10).

**Three separate status columns (Req 2.1, 2.3, 1.3):** a single `field_status` conflates two independent
coverage sources and the reconciliation outcome. The schema therefore carries three distinct columns:
- `train_field_status` — training coverage presence/value from `train_aggregate.state_phase_steps`;
- `dev_gate_field_status` — dev gate coverage presence/value from `selected_gate.geometry_phase_coverage`;
- `emission_status` — the declared-vs-emitted reconciliation outcome (`emitted` when the declared cell has
  a corresponding key in `selected_gate.geometry_phase_coverage`, otherwise `declared_not_emitted`).

These are never collapsed into one column: training coverage, dev gate coverage, and the emission
reconciliation outcome are independent facts.

**CRITICAL — cell-level vs skill-level verdicts (Req 2.1, 2.4):** `geometry_phase_coverage`'s `true`/`false`
is a **whole-skill aggregate verdict**, not a per-cell verdict. The gate does **NOT** emit a per-cell
pass/fail. Readers MUST NOT interpret the skill-level gate verdict as if it were a per-cell verdict. The
matrix therefore splits the concern into three distinct columns:
- `cell_gate_passed` = `(dev_gate_field_present AND steps >= 2)` — **computed per cell by the audit** from
  `dev_gate_field_status` / the dev gate coverage steps;
- `skill_geometry_phase_gate_passed` — **skill-level aggregate** taken directly from the gate;
- `skill_intent_progress_passed` — **skill-level aggregate** taken directly from the gate.

`cell_gate_passed` is an audit-derived helper; the two `skill_*` columns are verbatim gate aggregates and
carry the same value for every row of a given skill.

**Controlled attribution labels** (the ONLY allowed values) (Req 2.4):
`not_observed_cannot_assess`, `train_coverage_gap`, `observed_behavior_failure`,
`telemetry_or_gate_observability_gap`, `mixed_or_inconclusive`, `contract_or_provenance_mismatch`.

**Field-absent vs field-present-value-0 encoding** (Req 2.3, 1.3):
The encoding is applied **independently** to each of the two coverage sources (`train_field_status` from
`train_aggregate.state_phase_steps` and `dev_gate_field_status` from `selected_gate.geometry_phase_coverage`):
- `*_field_status = present_value_0` ⇒ report as "recorded as 0 steps under current gate semantics"
  (legitimate 0 emitted at `thesis_shared_skill_geometry.py:544`), NOT unverifiable.
- `*_field_status = present_value_gt_0` ⇒ the source recorded a positive step count for the cell.
- `*_field_status = absent` ⇒ the key is missing from that source's map; mark `unverifiable` for that
  source. When the declared cell has no key in `selected_gate.geometry_phase_coverage`, this is also
  recorded as `emission_status = declared_not_emitted` (a preflight-2a reconciliation outcome, distinct
  from a coverage count of 0).
- In NEITHER case infer "physically impossible" from mere absence; a structural non-observability claim
  requires manifest initial geometry + phase rules and must still report the absence of raw per-step
  evidence.

**Label decision logic (executable — no undefined predicates).** The previous draft referenced
`observed_behavior_failure(row)` and `coverageNeverProduced(row)` without any executable definition. That
is removed. The honest, executable truth is that **P4's `intent_progress` is ONLY a skill-level
aggregate**, which is insufficient to declare a per-cell "behavior failure". The audit therefore has **no
cell-level reward / behavior-quality signal** and MUST NOT force any cell to `observed_behavior_failure`.

The label logic instead uses the **train vs dev coverage divergence**. Both
`train_aggregate.state_phase_steps` (training coverage, via `train_field_status`) and
`selected_gate.geometry_phase_coverage` (dev gate coverage, via `dev_gate_field_status`) enter row-level
evidence. The function returns BOTH a controlled label and a `reason_code`:

On the current artifact the diff set is EMPTY, so branch 0 below (the per-cell `declared_not_emitted`
route) is NOT exercised by any row; the `contract_or_provenance_mismatch` label is instead attached at the
run level to the provenance gap (§0) and the unverifiable historical premise. The per-cell branch is
retained for reuse on inputs that do exhibit a diff.

```
FUNCTION assignLabel(row, reconcile) -> (label, reason_code):
  # 0. Contract/provenance takes precedence: declared but not emitted.
  #    NOTE: on this run diff is EMPTY, so this branch fires for no row; run-level
  #    provenance_unverified drives contract_or_provenance_mismatch instead.
  IF cellInDiffSet(row, reconcile):                        # emission_status == declared_not_emitted
      RETURN ("contract_or_provenance_mismatch", "")       # Req 2.11 / 1.9

  # Row-level evidence: integer step counts (or the sentinel `absent`).
  train = row.train_coverage_steps                         # from train_aggregate.state_phase_steps
  dev   = row.dev30_coverage_steps                         # from selected_gate.geometry_phase_coverage

  # 1. Absent dev gate field -> unverifiable, no evidence.
  IF row.dev_gate_field_status == "absent":
      RETURN ("not_observed_cannot_assess", "")

  # 2. train=0 AND dev=0 -> never given the chance in either distribution.
  IF train == 0 AND dev == 0:
      IF row.train_design_requires_coverage:
          RETURN ("train_coverage_gap", "")                # declared+required, 0 steps anywhere
      ELSE:
          RETURN ("not_observed_cannot_assess", "")

  # 3. train>0 AND dev=0 -> a DEV/EVALUATION coverage count of zero.
  #    This is NOT strong enough to assert a telemetry/gate observability defect;
  #    we can only say the dev coverage was not observed and cannot assess it here.
  #    MUST NOT be attributed to training (training did produce the posture).
  IF train > 0 AND dev == 0:
      RETURN ("not_observed_cannot_assess", "dev_coverage_gap")   # dev count 0, not a proven observability gap

  # 4. train>0 AND dev>0 but no cell-level quality metric ->
  #    cannot assert behavior failure (only skill-level intent_progress exists).
  IF train > 0 AND dev > 0:
      RETURN ("mixed_or_inconclusive", "")                 # insufficient cell-level evidence

  # 5. Any residual combination (e.g. train=absent while dev present) -> inconclusive.
  RETURN ("mixed_or_inconclusive", "")
```

**Executable rules enforced (Req 2.4, 1.4):**
- A cell with `steps > 0` but **no** cell-level reward / behavior-quality evidence MUST be labeled
  `mixed_or_inconclusive` — it is **never** forced to `observed_behavior_failure`.
- `train == 0 AND dev == 0` → `train_coverage_gap` (if the design requires the cell) or
  `not_observed_cannot_assess`.
- `train > 0 AND dev == 0` → `not_observed_cannot_assess` with `reason_code = dev_coverage_gap`. A dev
  coverage count of zero is **not** strong enough to assert a telemetry/gate observability defect; it is a
  dev/evaluation coverage gap that cannot be assessed from the current evidence, and it MUST NOT be
  attributed to training.
- `train > 0 AND dev > 0` but missing cell-quality metrics → still cannot assert behavior failure
  (`mixed_or_inconclusive`).

**`telemetry_or_gate_observability_gap` is reserved (Req 2.4, 2.10, 1.4, 1.10):** this label is used ONLY
for the semantic limitation that dynamic posture cannot be inferred from the fixed `initial_class` label
(the initial-class × phase limitation), NOT for a dev coverage count of zero. Rows whose claim depends on
dynamic posture in `post_merge` / `re_entry` carry this label (with `reason_code =
initial_class_phase_limitation`) to mark that the gate/telemetry cannot observe the dynamic five-state
posture — see §4. A `dev == 0` count is never sufficient to earn this label.

`observed_behavior_failure` remains in the controlled vocabulary but can only be assigned if and when a
genuine **cell-level** behavior-quality metric becomes available. The current metric is a **skill-level
aggregate only** (`intent_progress` is a whole-skill fraction); with no cell-level quality telemetry, a
cell can only be `mixed_or_inconclusive`, never `observed_behavior_failure`. Under the current P4 evidence
set **no row is eligible** for `observed_behavior_failure`.

**Coverage/intent semantics encoded per row**: coverage unit = policy steps; `gate_min_steps = 2`;
`intent_progress = intent_progress_steps/policy_steps`; `intent_progress_threshold = 0.55` (Req 2.6).

### 4. initial-class × phase Limitation Propagation

The limitation (recording keyed on fixed `initial_class` per
`thesis_shared_skill_geometry.py:451`, call sites 677/806; cannot prove dynamic posture in
`post_merge`/`re_entry`) is propagated into (Req 2.10, 1.10):

- **Column semantics**: the `initial_class` column is explicitly labeled as the fixed initial posture
  label; the matrix header documents that no column asserts dynamic live posture.
- **Per-row confidence**: any row for a `post_merge`/`re_entry` phase whose claim would depend on dynamic
  posture is capped in confidence with an explicit note that dynamic posture is unproven.
- **Reserved label**: this initial-class × phase limitation is the ONLY situation that earns
  `telemetry_or_gate_observability_gap` (with `reason_code = initial_class_phase_limitation`) — it marks
  the semantic inability to infer dynamic posture from the fixed initial-class label, and is never used
  for a mere `dev == 0` coverage count.
- **Conclusions**: the audit conclusion states coverage is proven only for initial-class × phase.

Where a structural-non-observability question arises, the audit performs a **static reachability analysis
from manifest initial geometry + phase rules only** and honors the raw-trajectory prohibition
(`thesis_shared_skill_geometry.py:752`): no per-step trajectory evidence is used or required; the absence
of raw evidence is reported honestly (Req 2.3, 3.8).

### 5. Output Artifacts

Three deliverables, mutually consistent (same rows/data across csv/json) (Req 2.7):

1. `reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md` — narrative audit in
   Chinese: overview, strict provenance boundary, frozen provenance block (with `provenance_unverified`
   markers for the missing run commit / `training_plan.json` / `resolved_geometry_config.json` / dev30
   manifest), gate-semantics citations, preflight reconciliation (verified 26/26/diff-0/extra-0, 2b CLEAN),
   the attribution matrix, the initial-class × phase limitation, and the explicit combat-finetune
   conclusion.
2. `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv` — one row per declared cell
   (26 rows), columns per the schema above.
3. `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json` — the same rows/data as an
   array of objects, plus a `preflight` block (`declared_count=26`, `emitted_count=26`, `diff_count=0`,
   `extra_count=0`, empty `diff` list) and a `provenance` block that records the `provenance_unverified`
   inputs.

The script emits csv and json from a single in-memory matrix so they are byte-consistent in content
(same row order, same values). The `.md` reports the verified 26 declared / 26 emitted / diff 0 / extra 0
reconciliation and explicitly flags the historical 26/22/4 premise as an unverified, structurally
impossible claim.

### 6. Analysis Script Design

**Location & conventions**: an importable module under the package plus a thin CLI in `scripts/`, per the
`src/uav_vpp_guidance` layout and AGENTS.md conventions:

- Module: `src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py`
  (pure functions; no drive letters; no side effects on import).
- CLI: `scripts/analyze_p4_geometry_gate_failure_attribution.py` (argparse; configurable input paths with
  sensible defaults; calls the module).

**Module functions:**
```
load_json(path) -> (obj, sha256, top_level_fields)
freeze_provenance(inputs, run_commit|None) -> ProvenanceBlock   # provenance_unverified for any absent input
derive_declared_cells(declared_source) -> List[DeclaredCell]    # registry (provenance_unverified substitute for training_plan.json); count derived from ANY input
collect_emitted_cells(gate_json) -> List[EmittedCell]           # count derived from ANY input
reconcile(declared, emitted) -> ReconcileResult                 # diff derived, not hardcoded
cross_artifact_consistency(gate_json, per_skill_summaries) -> List[Problem]   # 2b preflight (empty == consistent)
emit_integrity_report(problems, path) -> None    # 2b failure path: short report, then caller exits NON-ZERO
build_matrix(gate_json, declared, reconcile, semantics) -> List[Row]
assign_label(row, reconcile) -> (str, str)        # (label, reason_code)
emit_csv(rows, path); emit_json(rows, reconcile, provenance, path); emit_markdown(...)
self_check(rows, reconcile) -> None   # generic internal-consistency checks (see below); NO hardcoded 26/22/4
```

**Adaptive derivation — NO hardcoded counts in the production module/CLI (Req 2.7, 3.7):** The CLI/module
MUST adaptively derive `declared`, `emitted`, and `diff` counts from **whatever input is provided**. The
verified counts for the current `D:\laiyu_ssh\uav-vpp-guidance` artifact (26 / 26 / diff 0 / extra 0) are
properties of that artifact ONLY and MUST NOT be baked into the script; likewise the disproven historical
26 / 22 / 4 premise MUST NOT be baked in. This keeps the tool a reusable analysis tool with configurable
paths. Only the **integration test** against the frozen artifact asserts the concrete 26 / 26 / 0 / 0.

`self_check(rows, reconcile)` validates **internal consistency generically**, independent of the specific
run:
- `len(rows) == len(reconcile.declared)` (one row per declared cell);
- `len(reconcile.emitted) + len(reconcile.diff) == len(reconcile.declared)` when `extra` is empty
  (declared = emitted-that-are-declared + diff);
- `reconcile.extra == []` (no emitted cell falls outside the declared set — no extra cells);
- every row's `attribution_label` is one of the six controlled labels;
- every diff cell is labeled `contract_or_provenance_mismatch`.

These are relational invariants that hold for any input, not fixed numbers.

**CLI behavior** (Req 2.7):
1. Validate all four skills exist in the gate JSON.
2. Validate each required gate field is readable per skill (`gate.checks`,
   `gate.geometry_phase_coverage`, `gate.intent_progress_fraction`, `gate.profile_coverage`,
   `gate.safety_by_opponent`).
3. Run **both preflights** with their distinct halting behavior:
   - **2b cross-artifact consistency FIRST** (top-level `p4_geometry_pretrain_gate.json` `skills.<name>`
     vs the four `skills/<name>/geometry_pretrain_summary.json`). If 2b finds ANY inconsistency, emit ONLY
     a short integrity report, **exit NON-ZERO**, and do **NOT** build the matrix or emit any
     capability-attribution conclusions.
   - **2a declared-vs-emitted reconciliation** (only reached when 2b passes). A non-empty diff is the known
     audit subject: pre-assign the diff cells `contract_or_provenance_mismatch` / `declared_not_emitted`
     and **continue** — this does NOT trigger a non-zero exit. On the frozen artifact the diff is EMPTY, so
     no per-cell diff rows are produced; the run-level `contract_or_provenance_mismatch` provenance flag is
     recorded separately (§0).
4. Auto-derive the expected cell set from the registry when `training_plan.json` is the missing frozen
   input (declared cells substituted from the registry `geometry_phase_coverage` blocks and flagged
   `provenance_unverified`); the count is derived from whichever input is provided — not hardcoded.
5. Build the full matrix (including any 2a diff rows); emit csv + json + md.
6. Run the generic `self_check` (relational invariants above) and fail loudly (non-zero exit) on any
   internal inconsistency. The script does NOT assert concrete run-specific counts.
7. On success (2b clean, matrix built), **exit 0** even when the 2a diff is non-empty.

**Graceful degradation for missing frozen inputs (Req 2.6, item 8):** the registry (`--registry-path`) is
present and is the source of declared cells, so it is required. `--training-plan`, `--resolved-config`, and
`--dev30-manifest-path` may be the missing frozen inputs; when absent, the CLI marks each
`provenance_unverified` and continues rather than hard-failing. `--gate-json` may be pointed at the actual
in-repo path `src/p4_geometry_pretrain_gate.json` (results-root defaults to the repo).

**CLI surface (defaults overridable, no hardcoded drive letters):**
```
--results-root PATH         # default: repo root; the gate JSON actually lives at src/p4_geometry_pretrain_gate.json
--gate-json PATH            # default: <results-root>/p4_geometry_pretrain_gate.json; here point at src/p4_geometry_pretrain_gate.json
--training-plan PATH        # OPTIONAL; if the missing frozen input -> provenance_unverified (declared cells from registry)
--resolved-config PATH      # OPTIONAL; if the missing frozen input -> provenance_unverified
--registry-path PATH        # REQUIRED: shared-skill registry (declared-cell source; provenance_unverified substitute for training_plan.json)
--dev30-manifest-path PATH  # OPTIONAL (separate from --registry-path); if absent -> provenance_unverified (no hard fail)
--run-commit SHA            # optional; if absent -> provenance_unverified
--out-md / --out-csv / --out-json PATH   # default: reports/…_20260714[_zh].{md,csv,json}
```

**Minimal test**: `tests/test_p4_gate_failure_attribution.py` on a small **synthetic** fixture (NOT the
frozen artifact). It exercises the generic logic without any hardcoded run counts:
- asserts all four skills are present and each gate field is readable;
- asserts the matrix row count equals the derived expected declared count (whatever the fixture declares);
- asserts diff-set detection works (a fixture with a declared-but-not-emitted cell yields a non-empty
  diff, and a fully-emitted fixture yields an empty diff);
- asserts the generic `self_check` relational invariants hold
  (`emitted + diff == declared` when `extra` empty; `extra == []`);
- asserts label assignment maps a diff cell to `contract_or_provenance_mismatch`, an absent field to
  `not_observed_cannot_assess`, and a required cell with `train == 0 AND dev == 0` to `train_coverage_gap`.

**Frozen-artifact integration test (the ONLY place the concrete run counts are asserted)**: a separate
integration test runs the CLI/module against the frozen `GEO-20260713-R1` evidence and asserts the concrete
`declared == 26`, `emitted == 26`, `diff == 0`, `extra == 0` (diff set EMPTY), that the historical 26/22/4
premise is NOT reproducible, that `lead_intercept` `advantage:post_merge` / `neutral:post_merge` are
`present_value_0`, that 2b is CLEAN, that a run-level `contract_or_provenance_mismatch` provenance flag is
raised for the missing frozen inputs, and that the CLI exits 0. These run-specific assertions live only
here, never in the production module.

### 7. Checklist Update

Edit `reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md` to the final P4
negative-evidence state (Req 2.5). Precise edits:

- Line ~269: change `[>]` → `[x]` and rewrite to state the run is **complete**: all four skills reached
  200k steps; the run produced `p4_geometry_pretrain_gate.json` with
  `all_skills_ready_for_combat_finetune = false` / `do_not_start_combat_finetune`.
- Line ~270 (the `明确边界` bullet): change `[ ]` → `[x]` and rewrite to the final state — profile
  coverage and safety **all pass** (4/4); phase coverage **all fail** (4/4); intent progress **fail for
  three** (`defensive_extension`, `lead_intercept`, `pursuit_conversion`) while `reentry_recovery`
  **passes** intent progress but stays blocked by phase-coverage failure; this is **formal negative
  evidence**; combat finetune / P5 / P6 / P7 remain **locked**; the next action is the **gate-failure
  attribution decision, not retraining**.
- Line ~402 (section 8 status bullet): change `[>]` → `[x]` with the same final negative-evidence summary.

The canonical two-skill mainline sections MUST NOT be touched (Req 3.5).

## Testing Strategy

### Validation Approach

Two-phase: first surface counterexamples that demonstrate the attribution defect on the current (unfixed)
process against the frozen evidence, then verify the fixed audit produces a traceable, reconciled matrix
and preserves everything else. **No training or eval reruns** are performed (Req 3.2).

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples demonstrating the attribution/reporting defect BEFORE building the
audit, and confirm or refute the root-cause hypotheses (missing reconciliation; absent-vs-zero conflation;
missing contract label). If refuted, re-hypothesize.

**Test Plan**: Run the reconciliation and matrix-derivation logic on the frozen
`src/p4_geometry_pretrain_gate.json`, the four `src/skills/<skill>/geometry_pretrain_summary.json`, and the
registry, and assert the defect signatures.

**Test Cases**:
1. **Declared/emitted reconciliation (verified clean)** — derive 26 declared and 26 emitted; assert
   `diff == 0` and `extra == 0` for every skill (`pursuit_conversion` 6/6, `lead_intercept` 6/6,
   `defensive_extension` 6/6, `reentry_recovery` 8/8), confirming the emitted set equals the declared set.
   Assert the historical 26/22/4 premise is NOT reproducible against these artifacts (a 22-emitted output
   is structurally impossible under `evaluate_skill_gate:511-546`). The defect surfaced under F is the
   ABSENCE of any recorded reconciliation and per-cell attribution of the present-value-0 cells, not a
   declared-vs-emitted gap.
2. **Present-value-0 attribution** — assert `lead_intercept` `advantage:post_merge` and `neutral:post_merge`
   are `present_value_0` (both present with value 0) while `crossing_entry:post_merge` / `crossing_entry:pre_merge`
   are `present_value_gt_0`; assert F provides no per-cell attribution of the present-value-0 cells. On this
   artifact NO declared cell is `absent`, so the dominant case is `present_value_0`.
3. **Missing contract/provenance label** — assert the missing frozen inputs (run commit,
   `training_plan.json`, `resolved_geometry_config.json`) and the unverifiable historical 26/22/4 premise
   cannot be described by the original five labels, requiring `contract_or_provenance_mismatch` at the run
   level (NOT as any per-cell declared-vs-emitted diff row).
4. **Cross-artifact inconsistency (2b)** — construct a fixture where the top-level
   `p4_geometry_pretrain_gate.json` `skills.<name>` disagrees with a per-skill
   `skills/<name>/geometry_pretrain_summary.json` (e.g. a differing `intent_progress_fraction`); assert
   `cross_artifact_consistency` flags it and that F' emits ONLY the integrity report and exits NON-ZERO
   (no matrix), whereas F would silently proceed to capability attribution. On the frozen artifact 2b is
   verified CLEAN (top-level gate agrees with the four per-skill summaries), so the audit continues to a
   full matrix and exits 0.
5. **Checklist state** — assert the checklist still shows `[>]`/in-progress (misrepresentation).

**Expected Counterexamples**:
- 26 declared cells emitted (diff 0), yet the present-value-0 cells go unattributed and no reconciliation
  record exists under F.
- Missing frozen provenance inputs read as data defects instead of `provenance_unverified`.
- Possible causes: no reconciliation record, no per-cell present-value-0 attribution, missing controlled
  provenance label.

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed audit produces the expected
behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := runAudit_fixed(input)
  ASSERT result.matrix.rowCount == deriveDeclaredCount(input.registry)        // frozen GEO: 26 (2b clean)
  ASSERT result.preflight == { declared:26, emitted:26, diff:0, extra:0 }     // frozen GEO artifact; empty diff, exit 0
  ASSERT result.preflight.diff == []                                          // no declared-vs-emitted gap
  ASSERT NOT reproducible(historical_premise = { emitted:22, diff:4 })        // 22/4 disproven, structurally impossible
  // 2b (cross-artifact) behavior: if clean -> continue + full matrix + exit 0 (the frozen GEO case);
  //                               if inconsistent -> integrity report + NON-ZERO exit + NO matrix.
  ASSERT result.cross_artifact_problems == []  // frozen GEO: top-level gate agrees with per-skill summaries
  ASSERT result.run_level_provenance_flag == "contract_or_provenance_mismatch"  // missing run commit/training_plan/resolved config
  ASSERT everyRowHasFieldPathEvidence(result.matrix)
  ASSERT distinguishesAbsentFromZero(result.matrix)
  ASSERT everyLabel IN CONTROLLED_LABELS
  ASSERT diffCellsLabeled(result.matrix, "contract_or_provenance_mismatch")
  ASSERT carriesInitialClassPhaseLimitation(result)
  ASSERT provenanceFrozen(result) OR result.provenance == "provenance_unverified"
  ASSERT result.conclusion == "combat_finetune_NOT_allowed"
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed process produces the
same result as the original.

**Why the naive equality test is not valid:** `runAudit_original(input) == runAudit_fixed(input)` is NOT
testable, because the original process `F` is a defective *human/reporting* process, not an executable
program that can be invoked and compared. There is no runnable `runAudit_original` to diff against.

**Preservation is therefore verified CONCRETELY (no random-input generation):** in a clean git worktree,
record a SHA256 manifest of the specified P4 artifacts, run the audit once, assert those artifact hashes
are unchanged, and use `git diff --name-only` to assert writes are limited to the allowlist for this task.
```
# Preservation check (concrete, single run in a clean worktree — NOT random-input PBT):
before := sha256_manifest(P4_ARTIFACT_LIST)        # SHA256 of each specified P4 output/checkpoint/gate/config/mainline file
run_audit_fixed(input)                              # F' executes once (read-only except allowlisted outputs)
after  := sha256_manifest(P4_ARTIFACT_LIST)

# 1. Every specified P4 artifact SHA256 is byte-identical before and after.
FOR path IN P4_ARTIFACT_LIST DO
  ASSERT before[path] == after[path]
END FOR

# 2. git diff --name-only shows writes limited to THIS task's allowlist only:
#    the new module, the new CLI script, the new test, the three reports, and the checklist.
ALLOWLIST := {
  "src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py",
  "scripts/analyze_p4_geometry_gate_failure_attribution.py",
  "tests/test_p4_gate_failure_attribution.py",
  "reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md",
  "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv",
  "reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json",
  "reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md",
}
ASSERT set(git_diff_name_only()) SUBSET_OF ALLOWLIST

# 3. Gate verdict, thresholds, and checkpoint SHA are unchanged.
ASSERT gate_verdict(after)   == gate_verdict(before)          # all_skills_ready_for_combat_finetune == false
ASSERT gate_thresholds(after) == gate_thresholds(before)      # min coverage 2 ; min intent fraction 0.55
ASSERT checkpoint_sha(after) == checkpoint_sha(before)        # P3 encoder best.pt SHA256 unchanged
```

**Testing Approach**: Preservation is verified by this concrete before/after SHA256 comparison plus the
`git diff --name-only` allowlist assertion — a single deterministic run in a clean worktree. No
property-based testing and no randomly generated non-audit inputs are used; the audit is a frozen,
read-only-except-allowlisted-outputs process, so a concrete SHA manifest is the correct and sufficient
check.

**Test Plan**: In a clean worktree, compute the SHA256 manifest of the specified P4 artifacts, run the
fixed audit once, assert those hashes are unchanged, and assert `git diff --name-only` is a subset of the
allowlist (module, script, test, three reports, checklist); verify gate verdict, thresholds, and
checkpoint SHA are unchanged.

**Test Cases**:
1. **Frozen artifacts unchanged (SHA manifest)** — in a clean worktree, compute the P4 artifact SHA256
   manifest before and after; assert every specified artifact SHA256 is byte-identical.
2. **Allowlisted writes only (`git diff --name-only`)** — assert the changed-path set is a subset of
   {module, CLI script, test, three reports, checklist}.
3. **Gate verdict / thresholds / checkpoint SHA preserved** — assert the `false` verdict is reported
   verbatim, the thresholds equal `2` and `0.55`, and the P3 encoder `best.pt` SHA256 is unchanged.
4. **Passing sub-checks preserved** — assert `reentry_recovery` intent-progress pass and all four skills'
   profile/safety passes are reported as passes, not failures (Req 3.9).

### Unit Tests

- Reconciliation on a **synthetic** fixture: declared-cell derivation, emitted-cell collection, diff —
  counts derived from the fixture, NOT hardcoded 26/22/4.
- Generic `self_check` relational invariants (`emitted + diff == declared` when `extra` empty;
  `extra == []`).
- Cross-artifact consistency: a fixture whose top-level gate disagrees with a per-skill summary yields a
  non-empty problem list and forces `contract_or_provenance_mismatch`.
- Absent-vs-zero classification and label assignment for each reachable controlled label under the
  executable rules (no `observed_behavior_failure` forced without cell-level quality evidence).
- Provenance freezing including `provenance_unverified` fallback.
- csv/json content consistency (identical rows/values).

### Parametrized Tests (over pure functions — no PBT, no `hypothesis`)

The repo dev dependencies include `pytest` but **not** `hypothesis`; this frozen audit does NOT introduce a
property-based-testing dependency. The specification-level correctness Properties 1 and 2 remain the
authoritative properties, but they are verified with a small number of **parametrized `pytest`** cases over
the pure functions, using concrete/enumerated inputs rather than generated ones:

- **Reconciliation** (`@pytest.mark.parametrize`): fixtures with fully-emitted, one-missing, and
  several-missing declared/emitted cell sets; assert `reconcile` produces the correct `diff`/`extra` sets
  and that the `self_check` relational invariants hold (`emitted + diff == declared` when `extra` empty;
  `extra == []`).
- **Field status (the three columns)**: enumerate `absent` / `present_value_0` / `present_value_gt_0` for
  both `train_field_status` and `dev_gate_field_status`, plus `emitted` / `declared_not_emitted` for
  `emission_status`; assert the encoding never infers "physically impossible" from absence.
- **Label safety**: enumerate the `(train, dev, requires_coverage, in_diff)` combinations and assert the
  `(label, reason_code)` pairs match the rules — including `train>0 & dev=0 →
  (not_observed_cannot_assess, dev_coverage_gap)` and that `observed_behavior_failure` is never produced
  under the current cell-quality-free evidence set.
- **Cross-artifact consistency (2b)**: a consistent fixture yields an empty problem list (exit 0 path); an
  inconsistent fixture yields a non-empty list that drives the integrity-report / non-zero-exit path with
  no matrix.

### Integration Tests

- End-to-end script run on a small **synthetic** fixture with a 2a-style declared-vs-emitted diff and a
  clean 2b: emits md+csv+json, generic `self_check` passes, all four skills validated, the diff rows are
  labeled `contract_or_provenance_mismatch`, and **exit code 0** — no run-specific counts asserted here.
- **2b integrity-failure integration test**: run on a fixture where the top-level gate disagrees with a
  per-skill `geometry_pretrain_summary.json`; assert the audit emits ONLY the integrity report, produces
  NO matrix / no attribution conclusions, and exits **NON-ZERO**.
- **Frozen-artifact integration test (the ONLY place the concrete run counts are asserted)**: run on the
  frozen `GEO-20260713-R1` evidence under `D:\laiyu_ssh\uav-vpp-guidance` (read-only) and assert
  `declared == 26`, `emitted == 26`, `diff == 0`, `extra == 0` (the diff set is EMPTY), and that the
  historical 26/22/4 premise is NOT reproducible; assert `lead_intercept` `advantage:post_merge` and
  `neutral:post_merge` are `present_value_0` (present, not absent); assert both preflights run (2b verified
  CLEAN → continue), the full 26-row matrix is produced with `emission_status == emitted` on all 26 rows,
  the conclusion is `combat_finetune_NOT_allowed`, a **run-level** `contract_or_provenance_mismatch`
  provenance flag is surfaced for the missing frozen inputs (run commit / `training_plan.json` /
  `resolved_geometry_config.json`), and the script **exits 0**.
- Checklist edit integration: assert P4 lines reach the final negative-evidence state and the two-skill
  mainline sections are unchanged.

## Verification Plan

No training or eval reruns. Exact commands:

```
# 1. Run on the frozen evidence (read-only); emits the three artifacts and runs the generic self-check.
#    (The concrete run counts 26/26/0/0 are asserted only by the frozen-artifact integration test, not the CLI.)
#    results-root defaults to the repo, so the gate JSON actually lives at src/p4_geometry_pretrain_gate.json;
#    --gate-json is pointed there explicitly. The dev30 manifest / resolved config may be missing frozen
#    inputs; the CLI degrades gracefully (marks them provenance_unverified) rather than hard-failing.
python scripts/analyze_p4_geometry_gate_failure_attribution.py \
    --results-root <results-root> \
    --gate-json src/p4_geometry_pretrain_gate.json \
    --training-plan <results-root>/training_plan.json \
    --resolved-config <results-root>/resolved_geometry_config.json \
    --registry-path config/experiment/thesis_five_state_shared_skill_registry_v1.yaml \
    --dev30-manifest-path <dev30-manifest-path> \
    --out-md reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md \
    --out-csv reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv \
    --out-json reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json

# 2. Minimal test for the new module
python -m pytest tests/test_p4_gate_failure_attribution.py -v
```

**A passing result looks like:**
- The script exits 0 and prints its generic self-check summary (`rows == declared`,
  `emitted + diff == declared`, `extra == 0`). On the frozen GEO artifact these resolve to
  `declared=26, emitted=26, diff=0, extra=0` with an EMPTY diff — but the script derives these numbers from
  the input rather than hardcoding them, and the historical 26/22/4 premise is not reproducible.
- The three artifacts exist and are mutually consistent (csv and json contain the same rows/values).
- `pytest` reports all tests passed: four skills present, gate fields readable, matrix row count == derived
  declared count, diff-set detection works on the synthetic fixture, cross-artifact consistency is
  enforced, and the label logic maps a diff cell to `contract_or_provenance_mismatch`. The separate
  frozen-artifact integration test asserts the concrete 26/26/diff-0/extra-0, a clean 2b, a run-level
  `contract_or_provenance_mismatch` provenance flag, and exit 0.
- The audit's stated conclusion is: **under the pre-registered gate, combat finetune is NOT allowed to
  start; this audit has no authority to change that** (Req 2.8, 2.11).
