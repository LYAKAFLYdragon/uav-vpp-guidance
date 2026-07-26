# Implementation Plan

This is a strictly **non-training, non-tuning** attribution audit against the frozen run
`GEO-20260713-R1`, driven entirely by the P4 evidence currently available in the in-repo
`D:\laiyu_ssh\uav-vpp-guidance` layout (`src/p4_geometry_pretrain_gate.json`, the four
`src/skills/<skill>/geometry_pretrain_summary.json`, and
`config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`). No task in this plan retrains,
re-evaluates, edits the pre-registered gate, or touches checkpoints/model/reward or the canonical two-skill
mainline. All input paths are configurable (no hardcoded drive letters).

**Verified reconciliation (authoritative):** the declared-vs-emitted reconciliation on the current artifact
is **26 declared / 26 emitted / diff 0 / extra 0** (`pursuit_conversion` 6/6, `lead_intercept` 6/6,
`defensive_extension` 6/6, `reentry_recovery` 8/8). The historical **26 declared / 22 emitted / 4 diff**
premise is **DISPROVEN** and structurally impossible under `evaluate_skill_gate`
(`src/uav_vpp_guidance/training/thesis_shared_skill_geometry.py:511-546`), which emits the full Cartesian
product of declared states × phases (absent combinations filled with 0 via `.get(..., 0)`), so emitted
ALWAYS equals declared. The genuine defect is (i) the absence of a recorded reconciliation and per-cell
attribution of the present-value-0 declared cells, and (ii) a **provenance gap** (missing run commit /
`training_plan.json` / `resolved_geometry_config.json` / dev30 manifest), which is the ONLY source of any
`contract_or_provenance_mismatch` on this artifact — NOT any declared-vs-emitted diff (which is 0).

No task introduces a property-based-testing dependency: the repo dev deps include `pytest` but **not**
`hypothesis`, so every test below is concrete or `@pytest.mark.parametrize`d over pure functions. The
durable evidence test and the (xfail) audit-output test come FIRST, then implementation, then parametrized
+ concrete-preservation tests, then un-xfail the audit test, then the optional frozen-artifact integration
test, then verification & delivery.

Every pytest invocation is exactly `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
(NO `--run` flag — the repo does not register it). The audit tool uses Python stdlib only (`json`, `csv`,
`hashlib`, `argparse`, `subprocess`) and runs on **Python 3.10.9** (3.11 is unavailable on this host).

Tasks marked with an asterisk (`*`) are optional.

---

## Phase 0: Safe Working Environment (before any code change)

- [x] 0. Prepare a safe, non-destructive working environment
  - Sync the remote for branch `research/thesis-five-state-shared-intent-v1` first (fetch, do NOT reset/overwrite/delete local state)
  - If unrelated local working-branch changes exist, create a **separate git worktree** for this task rather than switching/resetting in place
  - Confirm no operation in this plan will retrain, re-evaluate, edit the gate, or modify checkpoints/model/reward or the canonical two-skill mainline
  - Confirm all input paths will be passed via CLI flags/config (no hardcoded drive letters in logic)
  - _Requirements: 3.6, 3.2, 3.3, 3.4, 3.5, 3.7_

---

## Phase 1: Durable Evidence Test + xfail Audit-Output Test (BEFORE the fix)

- [x] 1. Write the durable frozen-evidence fact test `test_frozen_evidence_reconciliation_and_disproven_premise`
  - **Property 1: Bug Condition** - Verified Reconciliation + Disproven 26/22/4 Premise (evidence half)
  - **GOAL**: Establish the durable, tool-independent evidence of the REAL reconciliation and the disproven
    historical premise. This is a STATIC FACT test on the ORIGINAL frozen artifact — it MUST PASS both
    BEFORE and AFTER implementation (it does not depend on the audit tool existing)
  - Read the raw declared-vs-emitted facts DIRECTLY (no import of the audit module):
    - derive declared cells from the registry
      `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`
      `geometry_phase_coverage` blocks (`geometry_states` × `phases`) — the documented
      `provenance_unverified` substitute for the missing `training_plan.json`
    - collect emitted keys from each skill's `selected_gate.geometry_phase_coverage` in
      `src/p4_geometry_pretrain_gate.json` and the four
      `src/skills/<skill>/geometry_pretrain_summary.json` (the SELECTED checkpoint's gate snapshot, not the
      last dev30 eval)
  - Assert the VERIFIED frozen facts hold: `declared == 26`, `emitted == 26`, `diff == 0`, `extra == 0`,
    and per skill `pursuit_conversion` 6/6, `lead_intercept` 6/6, `defensive_extension` 6/6,
    `reentry_recovery` 8/8
  - Assert the historical **26/22/4** premise is NOT reproducible from the artifact (the diff set is EMPTY);
    it is structurally impossible under `evaluate_skill_gate` (`thesis_shared_skill_geometry.py:511-546`)
    which emits the full Cartesian product filled with 0
  - Assert the present-value-0 facts DIRECTLY on the artifact: `lead_intercept.advantage:post_merge` is
    **present with value 0** (NOT absent), `lead_intercept.neutral:post_merge` is **present with value 0**
    (NOT absent), and the `crossing_entry` cells are present and non-zero (e.g.
    `crossing_entry:post_merge = 182`, `crossing_entry:pre_merge = 1092`)
  - **EXPECTED OUTCOME**: Test PASSES on the UNFIXED artifact (the facts are already true in the frozen
    evidence) and continues to PASS after implementation — it is the durable evidence anchor documenting the
    real reconciliation and the disproven premise
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - Mark complete when the test is written, run, and passing against the frozen artifact
  - _Requirements: 1.1, 1.2, 1.3, 1.8, 1.9, 1.6_

- [x] 2. Write the audit-output test `test_audit_emits_reconciled_attribution` (marked xfail BEFORE the fix)
  - **Property 1: Bug Condition** - Traceable Per-Cell Attribution With Reconciled Contract (fixed-output half)
  - **IMPORTANT**: This test exercises the NEW audit output (module/CLI). Because the audit tool does NOT
    exist yet, mark it `@pytest.mark.xfail(reason="audit tool not implemented yet", strict=True)` for now
  - Assert F' (the audit) emits the full **26-row reconciled matrix** where **every row** has
    `emission_status == emitted` (the diff set is EMPTY on this artifact — there are NO per-cell
    `declared_not_emitted` rows)
  - Assert the three status columns are present and populated per row: `train_field_status`,
    `dev_gate_field_status`, `emission_status`, plus the `reason_code` column
  - Assert a **run-level** `contract_or_provenance_mismatch` provenance flag is raised for the MISSING
    frozen inputs (run commit / `training_plan.json` / `resolved_geometry_config.json`) — NOT for any
    per-cell declared-vs-emitted diff (which is 0)
  - Assert every row links to JSON/config/manifest/source field paths (`evidence_paths`) and distinguishes
    field-absent (`unverifiable`) from field-present-value-0 ("recorded as 0 steps under current gate
    semantics"); on this artifact the dominant case is `present_value_0`
  - Assert the frozen provenance block is emitted with `provenance_unverified` markers (SHA256s, top-level
    field lists, `schema_version` when available, manifest, run commit SHA or `provenance_unverified`),
    kept separate from working-tree git info
  - Assert the conclusion is `combat_finetune_NOT_allowed`
  - **EXPECTED OUTCOME (now)**: `xfail` (the audit tool is not implemented). The xfail marker is removed in
    Phase 4 after implementation, at which point the test must PASS
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v` and confirm it reports `xfail`
  - Mark complete when the test is written, marked xfail, run, and reported as xfail
  - _Requirements: 1.4, 1.10, 2.1, 2.2, 2.3, 2.4, 2.6, 2.9, 2.10, 2.11, 2.8_

---

## Phase 2: Preservation Baseline Observation (BEFORE the fix)

- [x] 3. Record the concrete preservation baseline (observation-first, NO random-input PBT)
  - **Property 2: Preservation** - Frozen Assets Unchanged (SHA-verified, allowlisted writes only)
  - **IMPORTANT**: This is a CONCRETE baseline, not a property-based test. Do NOT generate random non-audit
    inputs; the audit is a frozen, read-only-except-allowlisted-outputs process
  - In a clean git worktree, compute and record a **SHA256 manifest** of the specified P4 artifact list.
    The P4 artifact list is the in-repo evidence (`src/p4_geometry_pretrain_gate.json`, the four
    `src/skills/<skill>/geometry_pretrain_summary.json` under `src/skills/**`, and the registry
    `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`) plus the mainline files (every P4
    output/checkpoint/gate/config/mainline file to be preserved)
  - Record the gate verdict (`all_skills_ready_for_combat_finetune == false`), the gate thresholds
    (min coverage `2`, min intent fraction `0.55`), and the P3 encoder `best.pt` SHA256
    (`385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`)
  - Record the passing sub-checks on the frozen evidence: `reentry_recovery` intent-progress pass; all four
    skills' profile coverage and safety pass
  - Persist this baseline manifest so the Phase 4 preservation test can assert byte-identical hashes after
    the audit runs
  - **EXPECTED OUTCOME**: Baseline recorded on the UNFIXED state; nothing is modified in this step
  - Mark complete when the SHA256 manifest and recorded invariants are captured
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9_

---

## Phase 3: Fix Implementation

- [x] 4. Fix for the P4 gate failure-attribution and status-reporting defect

  - [x] 4.1 Implement the attribution module `src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py`
    - Python **stdlib only** (`json`, `csv`, `hashlib`, `argparse`, `subprocess`); runs on Python **3.10.9**
      (note: 3.11 is unavailable on this host). Pure functions, no side effects on import, no hardcoded
      drive letters, no hardcoded 26/26/0/0 and no hardcoded 26/22/4
    - `load_json(path) -> (obj, sha256, top_level_fields)`
    - `freeze_provenance(inputs, run_commit|None) -> ProvenanceBlock`: mark the MISSING frozen inputs (run
      commit / `training_plan.json` / `resolved_geometry_config.json` / dev30 manifest) as
      `provenance_unverified`, and add a **run-level `contract_or_provenance_mismatch` flag** for the
      provenance gap; frozen provenance and working-tree git info kept as separate fields
    - `derive_declared_cells(declared_source)`: derive declared cells from the **registry** as a documented
      `provenance_unverified` substitute for the missing `training_plan.json` (count adaptive from input);
      `collect_emitted_cells(gate_json)` from `selected_gate.geometry_phase_coverage` — the SELECTED
      checkpoint's gate snapshot (NOT the last dev30 eval); also record that selected checkpoint's
      evaluation step per skill
    - `reconcile(declared, emitted) -> {declared, emitted, diff, extra}` with adaptively derived counts (NO
      hardcoded counts); on this artifact the result is 26/26/diff-0/extra-0
    - `cross_artifact_consistency(gate_json, per_skill_summaries) -> List[Problem]` (2b) deep-checking top-level `skills.<name>` vs the four `skills/<name>/geometry_pretrain_summary.json` (empty == consistent; CLEAN on this artifact)
    - `emit_integrity_report(problems, path)` for the 2b failure path (short report; caller then exits NON-ZERO)
    - `build_matrix(...)` producing one row per declared cell with the full column schema, including the THREE status columns `train_field_status` (from `train_aggregate.state_phase_steps`), `dev_gate_field_status` (from `selected_gate.geometry_phase_coverage`), and `emission_status` (`emitted` | `declared_not_emitted` from the 2a reconciliation — **`emitted` for all rows on this artifact**), a `reason_code` column, plus `cell_gate_passed` (audit-computed `dev_gate_field_present AND steps >= 2`) vs the verbatim skill-level aggregates `skill_geometry_phase_gate_passed` / `skill_intent_progress_passed`
    - `assign_label(row, reconcile) -> (label, reason_code)` implementing the executable rules from design §3: diff cell -> `(contract_or_provenance_mismatch, "")` (NOT exercised per-row on this artifact — diff is empty); `dev_gate_field_status == absent` -> `(not_observed_cannot_assess, "")`; train=0 & dev=0 & requires -> `(train_coverage_gap, "")`, else `(not_observed_cannot_assess, "")`; **train>0 & dev=0 -> `(not_observed_cannot_assess, dev_coverage_gap)`**; train>0 & dev>0 -> `(mixed_or_inconclusive, "")`; residual -> `(mixed_or_inconclusive, "")`
    - `observed_behavior_failure` is NEVER assigned under the current aggregate-only evidence (no cell-level quality metric exists); `telemetry_or_gate_observability_gap` is reserved ONLY for the initial-class × phase limitation (with `reason_code = initial_class_phase_limitation`), never for a `dev == 0` count
    - Three-column `field_status` encoding: `present_value_0` ("recorded as 0 steps under current gate semantics") vs `present_value_gt_0` vs `absent` (`unverifiable`); never infer "physically impossible" from absence. On this artifact every declared cell is PRESENT, so the dominant case is `present_value_0`
    - Propagate the initial-class × phase limitation into column semantics and per-row confidence
    - `emit_csv`, `emit_json` (rows + `preflight` block + `provenance` block), `emit_markdown` from a single in-memory matrix (byte-consistent content)
    - `self_check(rows, reconcile)` with generic relational invariants only (`len(rows) == len(declared)`, `emitted + diff == declared` when `extra` empty, `extra == []`, every label in the six controlled labels, every diff cell labeled `contract_or_provenance_mismatch`) — NO hardcoded run counts
    - _Bug_Condition: isBugCondition(input) for GEO-20260713-R1 attribution requests (design Glossary/Bug Condition)_
    - _Expected_Behavior: expectedBehavior = design Property 1 (traceable per-cell 26-row matrix, all rows emitted + reconciliation 26/26/diff-0/extra-0 + three status columns + reason_code + six controlled labels + frozen provenance with provenance_unverified + run-level contract_or_provenance_mismatch + combat_finetune_NOT_allowed)_
    - _Preservation: design Preservation Requirements (frozen assets, gate semantics, raw-trajectory prohibition)_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 2.9, 2.10, 2.11, 3.3, 3.7, 3.8, 1.3, 1.4, 1.8, 1.9, 1.10_

  - [x] 4.2 Implement the CLI `scripts/analyze_p4_geometry_gate_failure_attribution.py` with 2b-first / 2a-continue behavior
    - argparse with configurable paths: `--results-root`, `--gate-json` (points at `src/p4_geometry_pretrain_gate.json` on this repo), `--registry-path` **REQUIRED** (declared-cell source; `provenance_unverified` substitute for `training_plan.json`), `--training-plan` **OPTIONAL**, `--resolved-config` **OPTIONAL**, `--dev30-manifest-path` **OPTIONAL** (separate from `--registry-path`), `--run-commit` (optional), `--out-md`/`--out-csv`/`--out-json`
    - **Graceful degradation**: when `--training-plan`, `--resolved-config`, or `--dev30-manifest-path` are the missing frozen inputs, mark each `provenance_unverified` and **continue** — do NOT hard-fail; declared cells are derived from the required registry
    - Validate all four skills exist and each gate field is readable (`gate.checks`, `geometry_phase_coverage`, `intent_progress_fraction`, `profile_coverage`, `safety_by_opponent`)
    - Run **2b Cross-Artifact Consistency FIRST**: top-level `src/p4_geometry_pretrain_gate.json` `skills.<name>` vs the four `skills/<name>/geometry_pretrain_summary.json`. On ANY disagreement (genuine artifact-integrity failure), emit ONLY a short integrity report, produce NO attribution / NO matrix, and **exit NON-ZERO**. On this artifact 2b is CLEAN → continue
    - Only if 2b is clean, run **2a declared-vs-emitted reconciliation**: on this artifact the result is
      **26 declared / 26 emitted / diff 0 / extra 0** (empty diff). A non-empty diff would be the audit
      subject (labeled `contract_or_provenance_mismatch` / `declared_not_emitted`) and does NOT halt the
      run; on this artifact there are no diff rows. **Continue** → build the full matrix
    - Auto-derive the expected cell set from the registry (adaptive count); build the full 26-row matrix (all rows `emission_status = emitted`); emit md+csv+json; run generic `self_check`; record a **run-level `contract_or_provenance_mismatch`** for the missing provenance inputs; **exit 0**
    - Do NOT assert concrete run-specific counts in the CLI (26/26/0/0 lives only in the frozen-artifact integration test)
    - _Bug_Condition: isBugCondition(input) for GEO-20260713-R1 attribution requests_
    - _Expected_Behavior: design Property 1 + design §2 (2b runs first and halts non-zero on integrity failure; 2a reconciliation 26/26/diff-0 continues to full matrix and exit 0; run-level provenance mismatch recorded for missing provenance)_
    - _Preservation: read-only against evidence except allowlisted outputs; no gate edits_
    - _Requirements: 2.7, 2.9, 2.11, 3.7, 3.2, 3.3_

  - [x] 4.3 Produce the three output artifacts by running the audit (read-only against evidence)
    - `reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md` (Chinese narrative: overview, strict provenance boundary, frozen provenance block with `provenance_unverified` markers + host/interpreter note, gate-semantics citations, preflight reconciliation **26/26/diff-0/extra-0** (2b CLEAN), attribution matrix with the three status columns + reason_code, the selected checkpoint's evaluation step per skill, initial-class × phase limitation, explicit combat-finetune conclusion). The md MUST explicitly flag the historical **26/22/4** premise as **unverified / disproven / structurally impossible** and state the REAL failure attribution: `geometry_phase_coverage = false` is driven by present-value-0 `post_merge`/`re_entry` declared cells (e.g. `lead_intercept advantage:post_merge = 0`, `neutral:post_merge = 0`) plus `intent_progress < 0.55` for three skills; `reentry_recovery` passes intent progress but stays blocked by phase-coverage failure. Include the verified per-skill intent fractions (`defensive_extension 0.5325`, `lead_intercept 0.4977`, `pursuit_conversion 0.5193`, `reentry_recovery 0.5820`) and the zero cells
    - `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv` (**26 rows, all `emission_status = emitted`**, full schema columns)
    - `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json` (same rows + `preflight` block `declared_count 26 / emitted_count 26 / diff_count 0 / extra_count 0` with empty `diff` list + `provenance` block with `provenance_unverified` markers + host/interpreter note)
    - Conclusion must state: under the pre-registered gate, combat finetune is NOT allowed to start, and this audit has no authority to change that
    - _Expected_Behavior: design Property 1 + design §5 Output Artifacts (mutually consistent md/csv/json; verified 26/26/diff-0; disproven 26/22/4 premise flagged)_
    - _Preservation: writes limited to the three allowlisted reports_
    - _Requirements: 2.7, 2.8, 2.10, 2.11_

  - [x] 4.4 Update the execution checklist to the final P4 negative-evidence state
    - Edit `reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md` at lines ~269/270/402
    - P4 run complete (`[>]`/`[ ]` -> `[x]`), all four skills reached 200k steps, profile/safety all pass (4/4), phase coverage all fail (4/4), three intent progress fail (`defensive_extension`, `lead_intercept`, `pursuit_conversion`) while `reentry_recovery` passes intent progress but stays blocked; formal negative evidence; combat/P5/P6/P7 remain locked; next action = gate-failure attribution decision, NOT retraining
    - Do NOT touch the canonical two-skill mainline sections
    - _Preservation: two-skill mainline unchanged_
    - _Requirements: 2.5, 3.5_

  - [x] 4.5 Write the parametrized `pytest` suite over pure functions in `tests/test_p4_gate_failure_attribution.py`
    - Synthetic fixtures only (NOT the frozen artifact); no hardcoded run counts; NO `hypothesis`, NO PBT — use `@pytest.mark.parametrize` over concrete/enumerated inputs
    - **Reconciliation** (parametrized): fully-emitted, one-missing, and several-missing declared/emitted cell sets; assert `reconcile` produces the correct `diff`/`extra` sets and that `self_check` relational invariants hold (`emitted + diff == declared` when `extra` empty; `extra == []`). NOTE: the frozen artifact has diff 0, so the diff-mapping is exercised on a **synthetic fixture only**
    - **Three field-status columns** (parametrized): enumerate `absent`/`present_value_0`/`present_value_gt_0` for both `train_field_status` and `dev_gate_field_status`, plus `emitted`/`declared_not_emitted` for `emission_status`; assert the encoding never infers "physically impossible" from absence
    - **Label safety** (parametrized): enumerate the `(train, dev, requires_coverage, in_diff)` combinations and assert the `(label, reason_code)` pairs match design §3 — including `train>0 & dev=0 -> (not_observed_cannot_assess, dev_coverage_gap)`, a (synthetic, non-frozen) diff cell -> `(contract_or_provenance_mismatch, "")`, absent -> `not_observed_cannot_assess`, train=0 & dev=0 & requires -> `train_coverage_gap`; assert `observed_behavior_failure` is NEVER produced under the cell-quality-free evidence set
    - **Cross-artifact consistency (2b)** (parametrized): a consistent fixture yields an empty problem list (exit-0 path); an inconsistent fixture yields a non-empty list that drives the integrity-report / non-zero-exit path with NO matrix
    - Provenance freezing including `provenance_unverified` fallback and the run-level `contract_or_provenance_mismatch` flag; csv/json content consistency (identical rows/values)
    - These parametrized cases VERIFY spec-level Property 1 and Property 2 with concrete inputs (the Properties remain the authoritative spec-level correctness statements)
    - _Expected_Behavior: design Testing Strategy (Unit Tests + Parametrized Tests over pure functions; no PBT/hypothesis)_
    - _Requirements: 2.7, 2.2, 2.3, 2.4, 2.9, 2.11, 1.3, 1.4_

---

## Phase 4: Fix Verification (un-xfail the audit test + concrete preservation)

- [x] 5. Remove the xfail marker from `test_audit_emits_reconciled_attribution` and assert it passes
  - **Property 1: Expected Behavior** - Traceable Per-Cell Attribution With Reconciled Contract
  - **IMPORTANT**: This is the SAME test written in Phase 1 task 2 — do NOT write a new test. Simply remove
    the `@pytest.mark.xfail(...)` marker now that the audit tool exists
  - The test asserts F' emits the full **26-row reconciled matrix** where every row has
    `emission_status == emitted` (diff empty), the three status columns, the `reason_code` column, the
    frozen provenance block with `provenance_unverified` markers, the run-level
    `contract_or_provenance_mismatch` flag for the missing frozen inputs, and the
    `combat_finetune_NOT_allowed` conclusion
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - **EXPECTED OUTCOME**: The (now un-xfailed) test PASSES, confirming the attribution defect is resolved.
    `test_frozen_evidence_reconciliation_and_disproven_premise` (task 1) also still PASSES (durable evidence
    unchanged)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.9, 2.10, 2.11, 2.8_

- [x] 6. Verify preservation concretely (SHA256 manifest + `git diff --name-only` allowlist)
  - **Property 2: Preservation** - Frozen Assets Unchanged (SHA-verified, allowlisted writes only)
  - **IMPORTANT**: Concrete verification, NO random-input PBT. Reuse the baseline manifest recorded in Phase 2 task 3
  - In the clean worktree, run the audit once (read-only except allowlisted outputs), then recompute the SHA256 manifest of the specified P4 artifacts and assert every specified artifact hash is byte-identical to the baseline
  - Use `git diff --name-only` to assert the changed-path set is a SUBSET of the allowlist: the new module (`src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py`), the CLI script (`scripts/analyze_p4_geometry_gate_failure_attribution.py`), the test (`tests/test_p4_gate_failure_attribution.py`), the three reports, and the execution checklist
  - Assert the gate verdict (`all_skills_ready_for_combat_finetune == false`), thresholds (min coverage `2`, min intent fraction `0.55`), and the P3 encoder `best.pt` SHA256 are unchanged
  - Assert passing sub-checks are reported as passes, never recharacterized as failures (`reentry_recovery` intent progress; all four skills' profile coverage and safety)
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - **EXPECTED OUTCOME**: All preservation assertions PASS (no regressions; writes limited to the allowlist)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9_

- [ ]* 7. Write the frozen-artifact integration test (the ONLY place 26/26/diff-0 is asserted)
  - Separate integration test running the CLI/module against the frozen `GEO-20260713-R1` evidence (read-only)
  - Assert `declared == 26`, `emitted == 26`, `diff == 0`, `extra == 0` (diff set EMPTY), and that the
    historical **26/22/4** premise is NOT reproducible
  - Assert `lead_intercept` `advantage:post_merge` and `neutral:post_merge` are `present_value_0` (present with value 0, not absent)
  - Assert 2b is CLEAN (top-level gate agrees with the four per-skill summaries) so the run continues, the full 26-row matrix is produced (all rows `emission_status = emitted`), a **run-level `contract_or_provenance_mismatch` provenance flag** is raised for the missing frozen inputs, and the script **exits 0**
  - Assert the conclusion is `combat_finetune_NOT_allowed`
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - _Requirements: 2.7, 2.9, 2.11, 2.8_

- [ ]* 8. Write the 2b integrity-failure integration test (non-zero-exit path)
  - Separate integration test on a **synthetic** fixture where the top-level gate disagrees with a per-skill `geometry_pretrain_summary.json` (e.g. a differing `intent_progress_fraction`)
  - Assert the audit emits ONLY the short integrity report, produces NO matrix / NO attribution conclusions, and exits **NON-ZERO**
  - This covers the 2b halting behavior distinct from the 2a continue-and-exit-0 path of task 7
  - Run `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - _Requirements: 2.9, 2.11_

---

## Phase 5: Verification & Delivery

- [x] 9. Checkpoint - Ensure all tests pass and deliver
  - Run the CLI on the frozen evidence (read-only; a clean 2b → reconciliation 26/26/diff-0 → full matrix → exit 0); optional flags may be omitted (→ `provenance_unverified`):
    `python scripts/analyze_p4_geometry_gate_failure_attribution.py --gate-json src/p4_geometry_pretrain_gate.json --registry-path config/experiment/thesis_five_state_shared_skill_registry_v1.yaml --out-md reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md --out-csv reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv --out-json reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json`
  - Confirm `--gate-json` points at `src/p4_geometry_pretrain_gate.json` and `--registry-path` at `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml`; the optional `--training-plan` / `--resolved-config` / `--dev30-manifest-path` flags may be omitted (each recorded as `provenance_unverified`)
  - Run the pytest suite (NO `--run` flag): `python -m pytest tests/test_p4_gate_failure_attribution.py -v`
  - Report the exact commands and their results
  - Commit only this task's artifacts (module, CLI, test, three reports, checklist edit) with a clear message, e.g. `docs(thesis): attribute P4 geometry gate failures`
  - Push to `origin/research/thesis-five-state-shared-intent-v1`
  - Report must include: the verified **26 declared / 26 emitted / diff 0 / extra 0** reconciliation, the `provenance_unverified` list (run commit / `training_plan.json` / `resolved_geometry_config.json` / dev30 manifest), the changed files (`git diff --name-only` allowlist), the test results, and the one-line conclusion (under the pre-registered gate, combat finetune is NOT allowed to start; this audit has no authority to change that)
  - Ensure all tests pass; ask the user if questions arise
  - _Requirements: 2.7, 2.8, 3.2, 3.5, 3.6_
