# Implementation Plan

This is a phased, gated remediation plan for the remaining findings from the control-method rationality audit. It is **not authorization to change runtime behavior immediately**. Until an individual phase passes its entry criteria, its work is limited to specifications, baselines, deterministic test harnesses, offline analysis, and reproducible experiments. Do not modify protected runtime/control/configuration paths during the planning and evidence phases:

- `src/uav_vpp_guidance/guidance/**`
- `src/uav_vpp_guidance/flight_control/**`
- `src/uav_vpp_guidance/virtual_point/**`
- `src/uav_vpp_guidance/envs/reward.py`
- `src/uav_vpp_guidance/envs/observation.py`
- `src/uav_vpp_guidance/envs/tracking_env.py`
- `src/uav_vpp_guidance/gain_optimizer/**`
- `config/**`

**Audit disposition:** verified defects — F01, F02, F06, F07, F16; experiments required — F03, F05, F08, F10–F14, F17, F18; retained by design — F04 (conservative safety envelope); disproven/no change — F15 (sine/cosine encoding remains); F09 is a measurement/terminology follow-up with no training-behavior change in this plan.

**Change-safety contract:** Each implementation phase requires an explicit approval gate after its baseline and experiment evidence are reviewed. Before a protected runtime/config change, create a dedicated implementation branch/worktree, record a SHA256 preservation manifest, record every loaded-config mutation through `record_config_override`, produce portable pipeline manifests/artifact contracts, and validate no unrelated protected path changed. Do not commit or push as part of this plan.

**Experiment contract:** Every comparison uses the same committed scenario matrix, seed list, policy/checkpoint or gain artifact, evaluation horizon, success/safety metrics, and backend selection. Report confidence intervals or per-seed distributions, command line, resolved config, backend/fallback provenance, input hashes, and raw aggregate metrics. A candidate is rejected if it violates an existing safety limit, changes an observation schema without its migration contract, silently changes backend/config semantics, or fails to meet its phase acceptance threshold.

**Required test modes:** Prefer deterministic unit and integration tests for mathematical and state-machine behavior. Run only non-watch tests, e.g. `python -m pytest <target> -v`. Where JSBSim is unavailable, run strict/simple checks only for harness validation and mark transfer evidence unavailable; do not claim F18 is resolved.

---

## Phase 0: Evidence Freeze and Independent Test Infrastructure

- [x] 0. Freeze the audited baseline and remediation decision record
  - Re-run or validate the current rationality-audit report and record the exact source revision, report hashes, config hashes, protected-path SHA256 manifest, baseline scenario matrix, seed list, checkpoint/gain identifiers, and active backend/fallback provenance.
  - Create a machine-readable remediation ledger mapping F01–F18 to disposition, owner phase, required evidence, decision gate, and final status (`implemented_and_verified`, `experiment_rejects_change`, `needs_more_evidence`, `by_design_no_change`, or `disproven_no_change`).
  - Set F04 to `by_design_no_change`, F15 to `disproven_no_change`, and F09 to `needs_more_evidence`/documentation-only; do not leave them in an implementation queue.
  - Add a preservation test that fails if any protected path changes before an approved implementation phase.
  - _Requirements: 2.13, 3.1, 3.2, 3.7, 3.8, 3.9_

- [x] 1. Build reusable deterministic control-analysis fixtures
  - Create pure, non-runtime test fixtures for relative geometry, LOS angle/rate, roll/roll-rate, speed, range, filter cascades, VPP actions, mode-switch events, and actuator command traces.
  - Make all fixtures parameterized and deterministic; cover nominal, near-range, long-range, high closing-rate, reversal, saturation, absent-config, and reset cases.
  - Add reusable metric functions for boundedness, sign consistency, rise/settling/overshoot, delay/phase estimate, chatter count, command slew, return decomposition, and per-seed comparison.
  - _Requirements: 2.1–2.12, 3.9_

---

## Phase 1: Guidance-Physics Defects (F01 and F02)

- [x] 2. Specify and test the F01 guidance-law replacement before editing control code
  - **Bug condition:** a guidance calculation claims LOS-rate semantics while `lambda_dot` is absent and its geometric contribution grows solely with distance.
  - Write a law-specification note defining coordinate frame, LOS-rate calculation, units, signs, load-factor composition, range/closing validity limits, fallback behavior, and whether the replacement is a renamed geometric controller or a true LOS-rate/PN-family law.
  - Add pure geometry tests that prove: zero LOS rate yields no LOS-rate contribution; mirrored geometry produces the expected sign reversal; bounded inputs yield bounded commands before saturation; increasing range with unchanged angular/LOS-rate error does not increase the geometric correction unless explicitly justified by the selected law; singular/near-zero range is safe.
  - Establish a candidate-versus-frozen-baseline offline trace evaluation over the reference envelope; do not change runtime code until the review accepts physical semantics and no safety invariant regresses.
  - **Gate:** approve one explicit law and compatibility/migration strategy; otherwise log `needs_more_evidence` and retain baseline.
  - _Requirements: 1.1, 2.1, 3.1, 3.2, 3.9_

- [x] 3. Specify and test the F02 roll two-loop/damping remedy before editing control code
  - **Bug condition:** damping is proportional to roll angle rather than an explicitly defined roll-rate signal, with no tested heading-demand-to-bank/roll-rate relationship.
  - Define the outer heading/turn-demand loop, inner roll-rate/roll-attitude loop, signal sources, airspeed handling, gain units, saturation/anti-windup behavior, and the low-level controller responsibility boundary.
  - Add deterministic step, opposite-step, bank-release, speed-sweep, and saturation tests. Assert damping reduces rate-dependent oscillation, respects F04 limits, and preserves command sign and reset semantics.
  - Compare the candidate against baseline on the frozen scenario/seed matrix, including JSBSim when available; report tracking, safety, oscillation, and control-effort metrics rather than only success rate.
  - **Gate:** approve only if physical tests pass and no safety/stability metric regresses; otherwise retain baseline and record the failed hypothesis.
  - _Requirements: 1.2, 2.2, 3.1, 3.2, 3.9_

- [x] 4. Implement only the approved F01/F02 design in a dedicated gated change
  - Start only after tasks 2 and 3 approve their designs. Modify the smallest approved runtime surface; do not bundle F01/F02 with timing, VPP, reward, or observation changes.
  - Add unit tests for the approved law and controller interfaces plus integration tests proving current safety saturation behavior is retained.
  - Recompute the protected-path preservation manifest and compare baseline behavior on the frozen matrix; record config overrides, resolved config, and artifact provenance.
  - _Requirements: 2.1, 2.2, 3.1–3.6, 3.9_

---

## Phase 2: Time Scale, PN, Filter, and Actuator Evidence (F06, F03, F11)

- [x] 5. Define a multi-rate architecture and end-to-end delay budget for F06
  - **Bug condition:** high-level action, guidance, filtering, and low-level control all execute at 5 Hz without declared scheduling, hold, or latency semantics.
  - Document selectable high-level/guidance/low-level rates; require integer ratios or explicit resampling; define zero-order hold/interpolation, order of updates, reset, telemetry timestamps, and how filters advance at each rate.
  - Add deterministic scheduler tests for exact invocation counts, command-hold/interpolation behavior, reset, non-integer-ratio rejection, and legacy single-rate equivalence.
  - Measure baseline and candidate end-to-end delay/phase across the command spectrum, reporting the existing approximately 0.5607 s filter time constant as one component rather than total latency.
  - **Gate:** adopt a changed cadence only after it meets declared latency/stability/safety targets and reproduces legacy behavior when legacy cadence is selected.
  - _Requirements: 1.3, 2.3, 3.1, 3.5, 3.6, 3.9_

- [ ] 6. Run controlled F03/F11 dynamics experiments and decide PN/filter/actuator changes
  - **Risk condition:** enabled filter stages and actuator behavior lack a validated cumulative delay/phase and rate-limit budget; PN bank compensation is unvalidated.
  - Enumerate every enabled filter and actuator dynamic by configuration; measure per-stage and total delay, phase, peak command, and saturation under deterministic sweeps and representative engagement traces.
  - Evaluate one change at a time: PN filter tuning, bank-compensation candidate, explicit actuator slew limit, and filter consolidation. Keep F04 limits unchanged in every candidate.
  - Require simple and JSBSim evidence where applicable; mark any unavailable JSBSim run as incomplete rather than passing.
  - **Gate:** select only a candidate with improved/equal response and no safety, command-rate, or tracking regression; otherwise classify F03/F11 as `experiment_rejects_change` or `needs_more_evidence`.
  - _Requirements: 1.7, 2.7, 3.1–3.6, 3.9_

- [ ] 7. Implement approved timing/dynamics changes incrementally
  - Implement F06 scheduling first, then at most one approved F03/F11 candidate per change set; do not combine with F01/F02/VPP changes.
  - Add deterministic scheduler/dynamics tests, regression tests for legacy timing selection, and artifact-backed scenario evaluations.
  - Verify protected-path diff allowlist, config-override provenance, backend provenance, and F04 safety-limit preservation.
  - _Requirements: 2.3, 2.7, 3.1–3.6, 3.9_

---

## Phase 3: VPP Semantics and Action-Contract Defects (F05 and F07)

- [ ] 8. Compare fixed-metre, normalized, and distance-scaled VPP semantics for F05
  - **Risk condition:** a fixed offset changes tactical angular meaning as range changes; the configured lateral bound is 800 m and must not be confused with the audit's hypothetical 500 m example.
  - Define candidate representations, action bounds, coordinate frames, near-zero-range behavior, clipping, smoothing, and reversible conversion/telemetry fields.
  - Test action-to-offset monotonicity, bounds, symmetry, distance scaling, near/far angular consistency, and current fixed-metre behavior as the baseline.
  - Evaluate candidate representations with the frozen scenario/seed matrix and explicitly compare tactical, safety, tracking, command-effort, and policy stability outcomes.
  - **Gate:** change semantics only if the candidate meets predeclared metrics and a checkpoint/action-space migration plan is accepted; otherwise keep fixed-metre semantics and record the risk result.
  - _Requirements: 1.6, 2.4, 3.1, 3.3, 3.5, 3.6, 3.9_

- [x] 9. Repair the F07 VPP action-dimension contract
  - **Bug condition:** configuration declares 3 action dimensions while constructor fallback selects 5 when configuration is omitted.
  - Define one canonical action-dimension source and validation boundary spanning config schema, generator construction, environment/action space, policy/checkpoint metadata, loaders, and evaluation scripts.
  - Add tests for explicit compatible dimension, missing dimension, mismatched dimension, stale checkpoint metadata, and legacy artifact migration/rejection. Missing or mismatched values must fail descriptively; they must not silently choose 5.
  - If a config default or loader is changed, record the mutation with `record_config_override`; do not update a checkpoint in place.
  - **Gate:** approve runtime implementation once all construction paths agree and legacy handling is explicit.
  - _Requirements: 1.4, 2.5, 3.1, 3.5, 3.6, 3.9_

- [x] 10. Implement approved VPP changes with checkpoint/action migration safety
  - Implement F07 independently first; implement F05 only after task 8 approves an action-semantic migration.
  - Add action-space/schema compatibility checks before policy load, version any changed metadata, and evaluate legacy artifacts only through an explicit compatibility mode with provenance.
  - Confirm no observation base order, backend semantics, or unrelated config behavior changes.
  - _Requirements: 2.4, 2.5, 3.1, 3.3–3.6, 3.9_

---

## Phase 4: Mode Switching and Flight-Control Experiment Gates (F16, F17, F10)

- [x] 11. Replace the F16 implicit latch with a specified, tested mode-state machine
  - **Bug condition:** once triggered, the mode remains latched until reset and missing configuration changes the threshold from configured 25 degrees to code-default 15 degrees.
  - Define states, entry guard, hold condition, release guard, hysteresis/dwell interaction, reset, timeout/fail-safe behavior, telemetry, and canonical threshold/default ownership.
  - Add deterministic transition-table tests for non-trigger, entry, continued hold, release, re-entry, reset, absent config, explicit override, and crossing/non-crossing scenarios. Assert explicit configuration is never silently overridden and every state transition is observable.
  - Compare permanent-latch, releaseable-latch, and state-machine candidates using frozen scenarios/seeds before implementation.
  - **Gate:** approve only a state model that improves the identified episode behavior without introducing mode chatter, safety loss, or backend/config provenance violations.
  - _Requirements: 1.5, 2.6, 3.1, 3.4–3.6, 3.9_

- [x] 12. Quantify F17 hysteresis/dwell and F10 scheduling/arbitration before changing either
  - Build a representative closing-speed and engagement-state sweep. Measure F17 switch delay, range travelled, chatter, missed switch opportunities, and task/safety outcomes for dwell/hysteresis candidates.
  - Build qbar, altitude, AoA, and command-source sweeps for F10. Define a single arbitration sequence for guidance, altitude-hold, energy, and protection contributions; test saturation precedence and telemetry attribution.
  - Compare baseline fixed mapping to an explicitly named scheduled-controller candidate without changing the default. Treat lack of sufficient JSBSim evidence as `needs_more_evidence`.
  - **Gate:** accept a dwell/hysteresis or qbar/AoA scheduling change only when it passes stated stability, safety, and performance margins; otherwise preserve baseline.
  - _Requirements: 2.8, 2.11, 3.1–3.6, 3.9_

- [ ] 13. Implement only approved F16/F17/F10 changes as isolated change sets
  - Implement F16 first because it is a confirmed defect; retain F17/F10 as experiments until their gates approve a candidate.
  - Add transition, arbitration, qbar-envelope, and no-chatter tests; preserve F04 saturation limits and report all config overrides/provenance.
  - Require targeted tests plus scenario evaluation before marking the ledger `implemented_and_verified`.
  - _Requirements: 2.6, 2.8, 2.11, 3.1–3.6, 3.9_

---

## Phase 5: Optimization, Reward, Observation, and Transfer Validation (F08, F12–F14, F18; F09 documentation)

- [x] 14. Validate CEM credibility and clarify F09 without changing training behavior
  - **Risk condition:** active 5-D CEM uses 12 candidates and 3 elites with diagonal spread; its adequacy is unproven. F09's current quantity is not paired regret but changing rollback behavior is out of scope.
  - Establish seed-controlled CEM experiments comparing population/elite/covariance/early-stop candidates under a fixed compute budget; report coverage, convergence variation, best/median/worst result, and out-of-sample JSBSim validation where available.
  - Add truthful metric labels/documentation for F09, including its relation to best-known success rate and lack of rollback; do not alter bilevel training/rollback logic.
  - **Gate:** adopt CEM changes only with predeclared statistical and compute-budget benefit; otherwise keep baseline. Mark F09 `by_design_no_change` unless a separately approved training-method spec is created.
  - _Requirements: 2.9, 3.1, 3.5, 3.6, 3.8, 3.9_

- [x] 15. Validate reward risk F12/F13 through decomposition and anti-exploitation tests
  - Produce return decomposition across step and terminal components, safety-trigger occupancy, and counterfactual toggles for disabled `w_alive`/`w_closing` terms without changing default reward behavior.
  - Evaluate reward candidates only in isolated ablations with a fixed scenario/seed protocol and report success, crash/safety, return composition, exploit indicators, and transfer results.
  - **Gate:** retain default weights when a candidate lacks statistically credible task/safety benefit; any approved config mutation must be recorded with `record_config_override` and must not be hidden in a script.
  - _Requirements: 2.9, 3.1, 3.5, 3.6, 3.9_

- [x] 16. Establish the F14 observation-information experiment and migration gate
  - Audit which proposed signals are already available through existing optional observation segments and which are genuinely absent. Define feature sources, units, update time, normalization, availability/fallback, and leakage policy for every candidate.
  - Compare baseline against minimal additions and history candidates using fixed training/evaluation protocols. Measure task/safety benefit, policy sensitivity, feature availability, and backend parity.
  - If and only if an extension passes the experiment gate, create a separate observation-schema implementation task requiring: schema-version bump, unchanged base-feature order, deterministic names, `include_*` flag, `observation_schema`/provenance update, dimensional/flag tests, README and `AGENTS.md` documentation, and checkpoint compatibility/migration handling.
  - Explicitly exclude F15: do not add an angle-magnitude feature because the sine/cosine premise is disproven.
  - _Requirements: 2.10, 3.1, 3.3–3.7, 3.9_

- [x] 17. Quantify simple-to-JSBSim transfer for F18
  - Lock a common evaluation matrix with identical scenarios, seeds, policy/gain versions, control frequency, action contract, and metric definitions; verify backend selection is requested transparently and final backend/fallback fields appear in every result.
  - Evaluate baseline and each approved candidate on simple and strict JSBSim. Report per-seed performance deltas, safety/termination categories, actuator/energy/command traces, and fallback occurrences. A fallback run is not JSBSim evidence.
  - Package each run through the portable pipeline, including resolved config, `config_overrides`, input/output hashes, run manifest, artifact contract, and paper-safety status.
  - **Gate:** do not claim sim-to-sim transfer or promote final CEM/bilevel gains until strict JSBSim passes predefined safety and performance acceptance criteria.
  - _Requirements: 2.12, 3.4–3.6, 3.9_

---

## Phase 6: Decision, Regression, and Delivery Evidence

- [x] 18. Complete the remediation ledger and publish independent decisions
  - For each F01–F18, link the baseline, test IDs, experiment artifacts, acceptance thresholds, result, and final ledger status. Do not collapse verified defects and experimental risks into one status.
  - Required final no-change records: F04 `by_design_no_change`; F15 `disproven_no_change`; F09 training behavior unchanged unless a new separately approved spec exists.
  - Required final verified-defect records: F01, F02, F06, F07, and F16 either `implemented_and_verified` or `needs_more_evidence` with an explicit blocker; they must not be silently abandoned.
  - _Requirements: 2.13, 3.1–3.9_

- [ ] 19. Run regression, compatibility, and preservation verification for every approved change
  - Run targeted unit/integration tests for changed behavior; the existing observation/provenance tests when applicable; scheduler/dynamics/action-contract/mode-state tests; and strict-backend smoke/evaluation checks when JSBSim is available.
  - Compare the protected-path SHA256 manifest, scenario-matrix results, observation schema/version, backend/fallback provenance, config overrides, and pipeline artifact contract to their pre-change baselines.
  - Confirm F04 limits, F15 encoding, base observation order, explicit config precedence, and absence of silent backend overrides are preserved.
  - Report validation commands and results without committing or pushing.
  - _Requirements: 3.1–3.9_
