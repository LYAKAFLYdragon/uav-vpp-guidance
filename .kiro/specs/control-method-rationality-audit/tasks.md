# Implementation Plan

This is a strictly **read-only, non-training, non-tuning** investigation of the 18
control-method findings (`F01`–`F18`) from the rationality audit. No task in this plan
modifies guidance, flight-control, VPP, reward, observation, optimizer source, or any config.
No task retrains, re-optimizes gains, or touches checkpoints.

python environment: py3.11

**Branch**: `research/control-method-rationality-audit` (already created).

**Verified facts driving this plan** (established by reading source during the audit — the
harness must re-derive them, not hardcode them):

- `los_rate_guidance.py::_compute_nz_cmd` computes
  `base_nz + k_los·arctan2(rel_z, horiz) + k_pos·(distance/distance_scale_m)` — **no λ̇ term**.
- The roll channel damps on `own_state["roll_rad"]` (**angle**, not rate); no `ψ̇ = g·tan(φ)/V`.
- `config/guidance.yaml` mode-switch `aspect_threshold_deg: 25.0`, but
  `_evaluate_mode_switch_gate` falls back to `15.0`; `crossing_aspect_threshold_deg` unset.
- `_mode_switch_latched` is cleared only in `reset()` → latches for the whole episode.
- `config/guidance.yaml` VPP `action_dim: 3`, but `VirtualPointGenerator.__init__` defaults to `5`.
- `cem.py`: `candidates=12`, `elite_ratio=0.25` ⇒ `n_elite = max(1, 3) = 3`; diagonal
  `np.std(elite, axis=0)`; `convergence_tol=0.001`.
- `bilevel_trainer.py::_compute_regret` returns `max(0, 1 − best_known_SR)` (monotone
  non-increasing — not true regret); `_save_policy_snapshot` is defined but never used for rollback.
- `observation.py::build_observation` base segment is **16** features in the `AGENTS.md` order.
- **F15's premise is false**: `(sin θ, cos θ)` is injective — 30° and 330° share `cos` but
  differ in `sin`. This must be proven by an executable witness, not asserted.

**Testing constraints**: every pytest invocation is exactly
`python -m pytest tests/test_control_method_audit.py -v` (NO `--run` flag — the repo does not
register one). **No `hypothesis` / no property-based testing** — the repo dev deps do not
include `hypothesis`; use concrete or `@pytest.mark.parametrize`d tests over pure functions.
Numeric diagnostics are closed-form pure functions, **not** episode rollouts.

Tasks marked with an asterisk (`*`) are optional.

---

## Phase 0: Safe Working Environment (before any code change)

- [x] 0. Confirm the non-destructive working environment
  - Confirm the active branch is `research/control-method-rationality-audit`
  - Confirm no task in this plan will modify `src/uav_vpp_guidance/guidance/**`,
    `flight_control/**`, `virtual_point/**`, `envs/reward.py`, `envs/observation.py`,
    `envs/tracking_env.py`, `gain_optimizer/**`, or `config/**`
  - Confirm no task retrains, re-optimizes gains, or modifies checkpoints
  - Confirm all input paths will be passed via CLI flags (no hardcoded drive letters)
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

---

## Phase 1: Preservation Baseline (BEFORE writing the harness)

- [x] 1. Record the SHA256 preservation baseline of the control path
  - **Property 2: Preservation** — Zero runtime behavior change
  - **IMPORTANT**: concrete baseline, NOT property-based testing. Do not generate random inputs
  - Compute and persist a SHA256 manifest of the full preserved set: every file under
    `src/uav_vpp_guidance/guidance/`, `src/uav_vpp_guidance/flight_control/`,
    `src/uav_vpp_guidance/virtual_point/`, `src/uav_vpp_guidance/gain_optimizer/`, plus
    `src/uav_vpp_guidance/envs/reward.py`, `envs/observation.py`, `envs/tracking_env.py`, and
    every file under `config/`
  - Record the baseline in a form the Phase 4 preservation test can load and compare against
  - **EXPECTED OUTCOME**: baseline captured; nothing modified in this step
  - _Requirements: 10.1, 10.2, 10.3_

---

## Phase 2: Diagnostics + xfail Report Test (BEFORE the harness exists)

- [x] 2. Write the numeric-diagnostic tests (these PASS immediately — pure math)
  - **Property 1: Expected Behavior** — source-backed verdicts (numeric half)
  - **GOAL**: establish the durable, harness-independent numeric evidence. These are closed-form
    facts, so they must PASS both before and after the harness exists
  - Write `tests/test_control_method_audit.py` with parametrized tests over pure functions:
    - `filter_time_constant(alpha, dt) = -dt/log(1-alpha)`: assert `(0.3, 0.2) ≈ 0.5601 s`
      (the F03/F06/F11 phase-lag claim); assert τ decreases monotonically as α increases
    - `offset_angular_deviation(offset_m, range_m) = degrees(atan2(offset_m, range_m))`: assert
      `(500, 2500) ≈ 11.31°`, and assert the SAME metric offset yields a strictly LARGER angle at
      shorter range — this is the executable core of the F05 semantics-drift finding
    - `cem_elite_count(candidates, elite_ratio) = max(1, int(n*ratio))`: assert `(12, 0.25) -> 3`
      and `(70, 0.25) -> 17`; assert the `max(1, ...)` floor holds for tiny populations
    - `sincos_injective_witness()`: assert 30° and 330° share `cos` (≈0.866) but differ in `sin`
      (+0.5 vs −0.5) — the **executable disproof of F15**
    - `dwell_seconds(steps, dt)`: assert `(3, 0.2) -> 0.6 s`; `range_travel(closing_mps, s)` at a
      stated closing speed for the F17 dwell adequacy claim
    - `roll_rate_limit_deg_s(1.5) ≈ 85.94°/s` for the F04 comparison against the documented
      F-16-class reference
  - **EXPECTED OUTCOME**: all diagnostic tests PASS (they are closed-form math, independent of the harness)
  - Run `python -m pytest tests/test_control_method_audit.py -v`
  - _Requirements: 1.3, 1.4, 2.1, 2.2, 3.1, 6.2, 7.2, 9.4_

- [x] 3. Write the report-output test `test_audit_emits_triaged_findings` (marked xfail BEFORE the harness)
  - **Property 1: Expected Behavior** — source-backed verdicts (report half)
  - **IMPORTANT**: this exercises the NEW harness output. Because the harness does not exist yet,
    mark it `@pytest.mark.xfail(reason="audit harness not implemented yet", strict=True)`
  - Assert the harness emits exactly **18 findings** with ids exactly `F01`..`F18`, no duplicates
  - Assert every finding has a `verdict` from the four controlled labels
    (`confirmed_defect`, `confirmed_risk`, `by_design_ok`, `premise_incorrect`), ≥1
    `evidence_paths` entry, a `failure_mode`, a `recommendation`, a `priority`, and a `confidence`
  - Assert **F15 is `premise_incorrect` with `priority == 0`** (the rubber-stamp guard)
  - Assert every `premise_incorrect` record has `priority == 0` and every `confirmed_*` has
    `priority >= 1`
  - Assert the json carries `source_facts`, `diagnostics`, `summary` (with the four verdict
    buckets + `remediation_backlog`), and a `provenance` block (git commit/branch, python
    version, platform)
  - Assert `F18` is listed in `audit_scope.analytic_only_findings` (no run-based quantification)
  - **EXPECTED OUTCOME (now)**: `xfail`. The marker is removed in Phase 4, where it must PASS
  - Run `python -m pytest tests/test_control_method_audit.py -v` and confirm it reports `xfail`
  - _Requirements: 9.1, 9.2, 9.3, 1.5, 2.4, 3.3, 4.3, 5.3, 6.3, 7.3, 8.2_

---

## Phase 3: Harness Implementation

- [ ] 4. Implement the control-method audit harness

  - [x] 4.1 Implement `src/uav_vpp_guidance/evaluation/control_method_audit.py`
    - Pure functions, no side effects on import, no hardcoded drive letters, **no hardcoded
      per-finding verdicts** (verdicts come from `triage`)
    - `filter_time_constant`, `offset_angular_deviation`, `cem_elite_count`,
      `sincos_injective_witness`, `dwell_seconds`, `range_travel`, `roll_rate_limit_deg_s` — the
      pure numeric diagnostics already tested in task 2
    - `load_source_facts(paths) -> SourceFacts`: read **effective** values by constructing the
      real objects with the real config (`LOSRateGuidance`, `ProportionalNavigationGuidance`,
      `HybridGuidance`, `EnhancedLowLevelController`, `JSBSimActuatorInterface`,
      `RewardCalculator`, `VirtualPointGenerator`, `CEMGainOptimizer`) and reading their
      attributes — never substituting literals. Also read `config/guidance.yaml` directly for the
      config-vs-code mismatch facts (mode-switch `25.0` vs code `15.0`; VPP `action_dim` 3 vs 5)
      and call `build_observation` to obtain the real base dim (16) and feature-name order
    - `run_diagnostics(facts) -> DiagnosticResults` composing the pure functions
    - `FINDING_SPECS`: the single canonical table declaring, for each of `F01`..`F18`, its id,
      dimension, Chinese title, evidence paths, and which diagnostics it depends on — identity
      only, **no verdicts**
    - `triage(facts, diagnostics) -> List[FindingRecord]` implementing the ordered rules from
      design §Triage: premise-disproof → `premise_incorrect` (priority 0); wrong
      physical/mathematical relation → `confirmed_defect`; defensible but quantified failure mode
      → `confirmed_risk`; otherwise `by_design_ok`. Priority from the design's declared impact
      ordering (F01, F06, F05, F02, F16, F08/F09 first), deterministic, not ad-hoc
    - `self_check(records)`: generic relational invariants only — 18 records, ids exactly
      `F01..F18` with no duplicates, every verdict in the controlled set, every record has ≥1
      evidence path, numeric claims have a `diagnostic_ref`, `premise_incorrect ⇒ priority == 0`,
      `confirmed_* ⇒ priority >= 1`, and ≥1 record is `premise_incorrect`. **No hardcoded
      per-finding verdict list**
    - `emit_json(records, path)` and `emit_markdown(records, path)` generated from a single
      in-memory structure so md and json cannot disagree
    - Graceful degradation: if a module/config cannot be read or imported, fall back to
      source-text evidence, set `confidence = low`, annotate `notes`, and continue — never
      fabricate a fact, never crash the whole run
    - _Bug_Condition: isBugCondition(finding) — the finding premise is verifiable against
      committed source or closed-form math (design Glossary/Bug Condition)_
    - _Expected_Behavior: design Property 1 (every finding gets a source-backed verdict with
      evidence paths, diagnostic refs, failure mode, recommendation, priority, confidence)_
    - _Preservation: design Property 2 (read-only against all control source and config)_
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 8.1, 8.2, 10.1, 10.4_

  - [x] 4.2 Implement the CLI `scripts/audit_control_method_rationality.py`
    - argparse with configurable paths: `--guidance-config` (default `config/guidance.yaml`),
      `--source-root` (default `src/uav_vpp_guidance`), `--out-md`, `--out-json`
    - Order of operations: `load_source_facts` → `run_diagnostics` → `triage` → `self_check` →
      write artifacts. **Write only after `self_check` passes** so a malformed report can never be
      committed; on `self_check` failure, exit NON-ZERO having written nothing
    - Abort before writing if any output path falls outside the allowlist
    - Record a `provenance` block (git commit/branch, python version, platform, timestamp)
    - Exit 0 on success
    - Do NOT assert concrete per-finding verdicts in the CLI (those live in the integration test)
    - _Expected_Behavior: design Property 1 + design §Error Handling (atomic write after self_check;
      non-zero exit on invariant violation)_
    - _Preservation: writes limited to the two allowlisted report artifacts_
    - _Requirements: 9.1, 9.3, 10.1, 10.2_

  - [x] 4.3 Write the source-fact tests (extend `tests/test_control_method_audit.py`)
    - Assert `LOSRateGuidance` built from `config/guidance.yaml` reports the documented limits
      (nz `[-2, 7]`, roll_rate `[-1.5, 1.5]`, throttle `[0.4, 0.9]`) and `alpha_filter = 0.3`, and
      that the audit reads them **from the instance**, not from a literal
    - Assert `CEMGainOptimizer` over the 7-D gain space reports `candidates = 12`,
      `elite_ratio = 0.25`, and that `cem_elite_count` on those values is `3`
    - Assert `build_observation` on a minimal state pair returns a **16**-length base vector whose
      feature-name order matches the `AGENTS.md` base ordering (guards the schema contract and
      proves F14's "16-D base" premise)
    - Assert the config-vs-code mismatches: mode-switch `aspect_threshold_deg == 25.0` in config
      vs `15.0` code fallback; VPP `action_dim == 3` in config vs generator default `5`
    - Assert `HybridGuidance` reports `hysteresis_m = 500.0` and `min_dwell_steps = 3`
    - Assert `RewardCalculator` defaults: `w_safety = 2.0`, `w_angle = 0.8`,
      `terminal_success = 200`, `terminal_crash = -300`
    - _Expected_Behavior: design §Components (`SourceFacts` read from real modules, never hardcoded)_
    - _Requirements: 1.1, 1.2, 1.4, 2.3, 3.1, 5.1, 5.2, 6.1, 7.1, 7.2_

  - [x] 4.4 Write the parametrized triage-rule tests
    - Synthetic finding inputs only (not the real 18) — assert the rule ordering from design §Triage
    - Parametrize `(premise_disproved, wrong_relation, quantified_failure_mode)` and assert the
      resulting verdict: disproof → `premise_incorrect` (priority 0); wrong relation →
      `confirmed_defect`; quantified failure mode → `confirmed_risk`; none → `by_design_ok`
    - Assert `triage` is pure: identical `(facts, diagnostics)` input yields identical output
    - Assert `self_check` raises on each invariant violation: wrong record count, duplicate id,
      unknown verdict label, missing evidence path, `premise_incorrect` with non-zero priority,
      `confirmed_*` with priority 0, and zero `premise_incorrect` records
    - _Expected_Behavior: design §Triage rules + §self_check (generic invariants, no hardcoded verdicts)_
    - _Requirements: 1.5, 2.4, 3.3, 4.3, 5.3, 6.3, 7.3, 8.2, 9.2_

  - [x] 4.5 Run the harness to produce the two report artifacts
    - `reports/control_method_rationality_audit_findings_20260725_zh.md` — Chinese report with the
      design §Data Models section order: 审查范围与方法（read-only / analytic-only 边界 / 参考场景）;
      结论摘要（四类判定分布 + 优先级 backlog）; 逐项发现 F01–F18（判定、源码证据 file:line、诊断复现、
      失效模式、改进建议、置信度）; 被否证的前提（至少 F15，说明无需改动）; 未量化项说明（F18 仅结构性
      分析，未做 run-based 量化）; 后续修复任务边界（每项修复须走 `AGENTS.md` 契约）
    - `reports/control_method_rationality_audit_findings_20260725.json` — `schema_version`,
      `audit_scope` (with `analytic_only_findings: ["F18"]`), `source_facts`, `diagnostics`,
      `findings` (18 records), `summary` (four verdict buckets + `remediation_backlog`), `provenance`
    - The md and json MUST be mutually consistent (generated from one in-memory structure)
    - The report MUST explicitly state that every recommendation is **advisory only** and that no
      control behavior, config, or checkpoint was changed by this audit
    - _Expected_Behavior: design §Data Models (mutually consistent md/json; explicit advisory boundary)_
    - _Preservation: writes limited to the two allowlisted reports_
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 10.5_

---

## Phase 4: Verification (un-xfail the report test + concrete preservation)

- [x] 5. Remove the xfail marker from `test_audit_emits_triaged_findings` and assert it passes
  - **Property 1: Expected Behavior** — source-backed verdicts
  - **IMPORTANT**: this is the SAME test written in Phase 2 task 3 — do NOT write a new test.
    Simply remove the `@pytest.mark.xfail(...)` marker now that the harness exists
  - The test asserts 18 findings with ids `F01..F18`, controlled verdict labels, evidence paths,
    the `premise_incorrect ⇒ priority 0` rule, F15 specifically `premise_incorrect`, the
    `summary`/`provenance` blocks, and `F18` in `analytic_only_findings`
  - Run `python -m pytest tests/test_control_method_audit.py -v`
  - **EXPECTED OUTCOME**: the un-xfailed test PASSES; all Phase 2 diagnostic tests still PASS
  - _Requirements: 9.1, 9.2, 9.3, 1.5, 6.2_

- [x] 6. Verify preservation concretely (SHA256 manifest + `git diff --name-only` allowlist)
  - **Property 2: Preservation** — Zero runtime behavior change
  - **IMPORTANT**: concrete verification, NO random-input PBT. Reuse the Phase 1 task 1 baseline
  - Recompute the SHA256 manifest of the preserved set after running the audit and assert every
    hash is byte-identical to the baseline (all of `guidance/**`, `flight_control/**`,
    `virtual_point/**`, `gain_optimizer/**`, `envs/reward.py`, `envs/observation.py`,
    `envs/tracking_env.py`, `config/**`)
  - Assert `git diff --name-only` (plus untracked additions) is a SUBSET of the allowlist: the
    spec folder, `scripts/audit_control_method_rationality.py`,
    `src/uav_vpp_guidance/evaluation/control_method_audit.py`,
    `tests/test_control_method_audit.py`, and the two report artifacts
  - Assert `config/guidance.yaml` is unchanged (no gain/limit/threshold was "fixed" in passing)
  - Run `python -m pytest tests/test_control_method_audit.py -v`
  - **EXPECTED OUTCOME**: all preservation assertions PASS
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 7. Write the end-to-end integration test against the real repo
  - Run the CLI against the real source tree (read-only) and assert exit 0, both artifacts written
  - Assert 18 findings, ids exactly `F01..F18`, `self_check` invariants hold
  - Assert F15 is `premise_incorrect`; assert the `remediation_backlog` is priority-ordered and
    non-empty; assert md and json agree on every verdict
  - Run `python -m pytest tests/test_control_method_audit.py -v`
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

---

## Phase 5: Delivery

- [-] 8. Checkpoint — ensure all tests pass and deliver
  - Run the CLI:
    `python scripts/audit_control_method_rationality.py --guidance-config config/guidance.yaml --out-md reports/control_method_rationality_audit_findings_20260725_zh.md --out-json reports/control_method_rationality_audit_findings_20260725.json`
  - Run the suite (NO `--run` flag): `python -m pytest tests/test_control_method_audit.py -v`
  - Report the exact commands and their results
  - Commit only this spec's artifacts (spec folder, harness module, CLI, test, two reports) with a
    clear message, e.g. `docs(audit): triage control-method rationality findings`
  - Push with `-u` to `origin/research/control-method-rationality-audit` (new branch, never main)
  - Report must include: the verdict distribution across the four labels, the priority-ordered
    remediation backlog, the findings marked `premise_incorrect` (at least F15), the changed-file
    allowlist proof, the test results, and the one-line conclusion (this audit changed no control
    behavior; every remediation is advisory and requires a separate gated task)
  - Ensure all tests pass; ask the user if questions arise
  - _Requirements: 9.1, 9.2, 9.3, 10.1, 10.5_
