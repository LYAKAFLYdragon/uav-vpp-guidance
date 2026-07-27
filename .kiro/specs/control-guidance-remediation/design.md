# Control Guidance Remediation Bugfix Design

## Overview

This design remediates the verified control-method defects F01, F02, F06, F07, and F16 while deciding F03, F05, F08, F10–F14, F17, and F18 through controlled evidence. It deliberately separates evidence production from behavioral implementation: planning, baseline, fixture, and experiment work may add reports, tests, analysis utilities, and portable pipeline stages, but SHALL NOT modify protected runtime/control/configuration paths. A protected change is permitted only after the immediately preceding finding-specific evidence gate records an explicit approval.

The implementation strategy is incremental and reversible. Each phase freezes its inputs, evaluates one candidate against the same scenario/seed/backend matrix, records a machine-readable decision, and—only when approved—changes the smallest allowlisted runtime surface. F01/F02, timing/dynamics, VPP, mode switching/control scheduling, and optimization/reward/observation work remain isolated change sets. F04 safety limits and F15 sine/cosine encoding are preserved; F09 permits terminology or measurement clarification only.

The user requested execution of all tasks after prerequisites are complete. Accordingly, task execution may proceed sequentially after this design phase is approved, but that request does not waive any evidence gate: a downstream implementation task is runnable only when every listed prerequisite is complete and its gate status is `approved`. A rejected or inconclusive gate records the required no-change disposition and skips the corresponding runtime mutation while allowing independent later phases to continue.

## Glossary

- **Bug_Condition (C)**: An input/configuration/execution state exposing one of the verified defects F01, F02, F06, F07, or F16.
- **Property (P)**: The required correct behavior for an input satisfying `C`, including physical semantics, canonical contracts, deterministic scheduling, or explicit state transitions.
- **Preservation**: Equivalence of behavior and artifacts outside an approved finding-specific change, including byte-identical protected paths during evidence work.
- **Protected_Path**: Any runtime/control/configuration path listed in `tasks.md`, including guidance, flight control, VPP, reward, observation, tracking environment, optimizer, and `config/**`.
- **Evidence_Gate**: A machine-readable decision with finding ID, candidate, evidence hashes, thresholds, results, reviewer decision, timestamp, and status `approved`, `rejected`, or `needs_more_evidence`.
- **Frozen_Baseline**: The source revision, protected-path SHA256 manifest, resolved configs, scenario matrix, seeds, policy/checkpoint or gain identifiers, metrics, and backend provenance captured before candidate evaluation.
- **Canonical_Action_Dimension**: The single action dimension agreed across config, generator, environment/action space, policy/checkpoint metadata, loaders, and evaluation scripts.
- **Legacy_Single_Rate_Mode**: The selectable configuration reproducing the current 0.2 s high-level/guidance/low-level cadence.
- **Observation_Schema**: The versioned schema containing flags, dimension, and deterministic `feature_names`; its 16-feature base order is immutable in this remediation unless a separately gated migration is approved.
- **Backend_Provenance**: Requested and final backend plus `backend_fallback_occurred` and `backend_fallback_reason`, recorded in result info and provenance.
- **Config_Override_Record**: A `record_config_override` entry containing key, old value, new value, and source whenever loaded YAML is mutated.
- **Portable_Artifact**: A bundle containing resolved config, run manifest, artifact contract, input/output SHA256 hashes, command provenance, outputs, bundle hashes, and paper-safety status.
- **F**: The frozen original implementation.
- **F′**: A finding-specific candidate or approved fixed implementation.
## Bug Details

### Bug Condition

The verified bug condition is a disjunction of five independently gated defects. Risk findings are not members of `C`; they remain hypotheses until an experiment gate approves a change.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type RemediationInput
  OUTPUT: boolean

  f01 := input.path = "los_normal_load"
         AND input.law_claims_los_rate_semantics
         AND (NOT input.explicit_los_angular_rate_used
              OR input.geometric_correction_grows_only_with_range)

  f02 := input.path = "roll_command"
         AND input.signal_labeled_as_damping
         AND input.feedback_signal = roll_angle
         AND input.explicit_roll_rate_feedback_is_absent

  f06 := input.path = "control_schedule"
         AND input.high_level_period = 0.2 seconds
         AND input.guidance_period = 0.2 seconds
         AND input.low_level_period = 0.2 seconds
         AND input.explicit_schedule_contract_is_absent

  f07 := input.path = "vpp_action_construction"
         AND (input.action_dimension_is_missing
              OR dimensionsDisagree(input.config, input.generator,
                                     input.action_space, input.checkpoint))
         AND input.construction_silently_selects_semantics

  f16 := input.path = "mode_switch"
         AND (input.gate_was_previously_triggered AND input.release_guard_holds
              AND input.mode_remains_latched_until_reset
              OR input.aspect_threshold_is_missing
                 AND input.code_default != input.canonical_default)

  RETURN f01 OR f02 OR f06 OR f07 OR f16
END FUNCTION
```

The expected predicate used by fix checking is:
```
FUNCTION expectedBehavior(result)
  INPUT: result of type RemediationResult
  OUTPUT: boolean

  RETURN result.is_finite
         AND result.physical_units_and_signs_match_specification
         AND result.satisfies_finding_specific_acceptance_thresholds
         AND result.f04_limits_are_unchanged
         AND result.protected_diff_is_within_approved_allowlist
         AND result.backend_provenance_is_complete
         AND result.config_overrides_are_recorded
         AND result.artifact_contract_is_valid
END FUNCTION
```

### Examples

- **F01 range growth:** With unchanged LOS elevation/LOS-rate error, increasing range from 800 m to 2500 m currently increases the `k_pos * distance / distance_scale_m` contribution. A true LOS-rate or bounded geometric candidate must not increase correction solely due to range.
- **F01 zero rate:** Parallel relative motion with zero LOS angular rate must produce zero LOS-rate contribution. Near-zero range must remain finite and use documented fallback/capture behavior.
- **F02 independent signals:** Two states with equal roll angle but opposite roll rates currently receive equal damping contribution; the corrected rate-feedback term must reverse sign, subject to the documented inner-loop convention.
- **F06 cadence:** One second currently gives five high-level decisions and no independently declared guidance/low-level invocation contract. Legacy mode must reproduce that trace; an approved multi-rate mode must produce exact configured counts and holds.
- **F07 omitted dimension:** `VirtualPointGenerator({})` currently selects 5 while the audited active configuration declares 3. After repair, omission or disagreement must fail descriptively or resolve through one explicitly documented canonical source—never silently choose 5.
- **F16 release:** After a gate activates, a later state satisfying the approved release guard currently remains effective with reason `latched`; an approved releaseable state model must transition and expose the transition in telemetry.
- **F16 missing threshold:** Missing `aspect_threshold_deg` currently uses 15 degrees while the audited config uses 25 degrees. Missing configuration must be rejected or resolved from one canonical default source, while an explicit user value always wins.
- **F05 risk example:** An 800 m lateral offset has approximately 17.74° angular meaning at 2500 m and 45° at 800 m. This quantifies an experiment, but does not authorize a semantic change.
- **F17 edge case:** Three dwell steps at 5 Hz equal 0.6 s; at 300 m/s closing speed the engagement travels about 180 m. Candidate dwell/hysteresis changes require measured benefit and no chatter/safety regression.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Before a gate approval, every Protected_Path remains byte-identical to the Frozen_Baseline SHA256 manifest.
- Each approved implementation changes only its finding-specific allowlist; unrelated protected files remain hash-identical.
- F04 limits remain `nz [-2, 7]`, roll rate `[-1.5, 1.5] rad/s`, and throttle `[0.4, 0.9]` unless a separate JSBSim-backed safety decision exists.
- Capture-radius, finite-value fallback, reset, clipping, filtering, and command-postprocessing semantics remain unchanged except where the approved design explicitly names them.
- The 16 base observation features retain their exact names and order; existing optional segments retain flag and dimensional semantics. F15 adds no angle-magnitude feature.
- Backend selection remains transparent: requested/final backend, fallback occurrence, and fallback reason are recorded; fallback evidence is never labeled strict JSBSim evidence.
- Explicit user configuration retains precedence. Every approved script mutation of loaded YAML calls `record_config_override` and appears in provenance.
- Existing fixed-metre VPP semantics, reward weights, optimizer behavior, hybrid dwell/hysteresis, flight-control mapping, and observation schema remain defaults unless their own evidence gate approves a candidate.
- F09 bilevel training and rollback behavior remain unchanged; only truthful terminology/measurement documentation is permitted.
- Every experiment preserves the same scenario definitions, seed list, horizon, policy/checkpoint or gain artifacts, metric definitions, and backend request across baseline/candidate comparisons.

**Scope:**
All inputs outside the approved finding-specific Bug_Condition are preserved by differential tests against `F`. Planning and evidence tasks may write only non-protected tests, analysis support, reports, manifests, and artifacts. Runtime work is blocked unless the preceding gate is explicitly `approved`; `rejected` records `experiment_rejects_change`, and inconclusive evidence records `needs_more_evidence` without mutation.

## Hypothesized Root Cause

1. **F01 semantic/physical mismatch:** `LOSRateGuidance.compute_command` derives elevation from relative position and `_compute_nz_cmd` combines it with a range-growing position term, but no relative velocity/LOS angular-rate term is present. The class name and gain names therefore imply physics the implementation does not provide.
2. **F02 state-variable mismatch:** The roll command is `k_roll * heading_error - k_damp * current_roll`; `current_roll` is angle, not angular rate. The outer heading demand and inner rate/attitude responsibilities are not separated, so gain units and coordinated-turn behavior are ambiguous.
3. **F06 orchestration coupling:** `CloseRangeTrackingEnv.step` advances one `high_level_dt` and performs VPP, guidance, command filtering, and environment stepping in the same call. PN also defaults to `dt=0.2`; no scheduler owns independent clocks, hold/interpolation rules, or timestamped latency accounting.
4. **F07 distributed defaults:** `VirtualPointGenerator.__init__` uses `config.get("action_dim", 5)`, while active policy/config paths commonly use 3 and loaders validate dimensions inconsistently. Action semantics lack one mandatory validation boundary.
5. **F16 implicit latch/default duplication:** `tracking_env` sets `_mode_switch_latched=True` after gate activation and clears it only in reset. `_evaluate_mode_switch_gate` owns fallback values (15°, 3000 m, 100 m/s) separately from configuration, creating two default authorities and no explicit release transition.
6. **Risk findings lack causal evidence:** Fixed-metre VPP scaling, filter/actuator cascades, controller scheduling, CEM sampling, rewards, observations, dwell/hysteresis, and simulator transfer are plausible risks but not proven defects. Changing them together would confound attribution and threaten regression control.
7. **Cross-cutting reproducibility risk:** Without immutable hashes, resolved config, backend/fallback records, override records, and artifact contracts, a measured difference could be caused by environment or configuration drift rather than the candidate.
## Correctness Properties

Property 1: Bug Condition - F01 Bounded Physical Guidance

_For any_ valid relative geometry where F01's bug condition holds, an approved F01 implementation SHALL use the explicitly specified physical signals and units, SHALL produce zero LOS-rate contribution for zero LOS angular rate, SHALL reverse the appropriate contribution under mirrored geometry, SHALL remain finite near zero range, and SHALL not increase a geometric correction solely because range increases while angular/LOS-rate error is unchanged.

**Validates: Requirements 2.1**

Property 2: Preservation - Protected Paths and Gate Enforcement

_For any_ planning or evidence task, and for any implementation task whose preceding Evidence_Gate is not `approved`, all Protected_Path hashes SHALL equal the Frozen_Baseline; for an approved implementation, changed protected paths SHALL be a subset of that gate's explicit allowlist.

**Validates: Requirements 3.1, 3.9**

Property 3: Bug Condition - F02 Roll-Rate Damping

_For any_ pair of states that isolates roll rate from roll angle, an approved F02 implementation SHALL apply damping according to the documented roll-rate signal or equivalent inner-loop feedback, produce the specified sign reversal for opposite rates, preserve reset/sign conventions, and keep final commands inside F04 limits.

**Validates: Requirements 2.2**

Property 4: Bug Condition - F06 Deterministic Multi-Rate Scheduling

_For any_ valid configured integer-related schedule, the scheduler SHALL invoke high-level, guidance, and low-level stages at exact configured timestamps with specified hold/interpolation and reset semantics; for Legacy_Single_Rate_Mode it SHALL reproduce the frozen invocation and command trace, and it SHALL reject unsupported ratios rather than silently drift.

**Validates: Requirements 2.3**

Property 5: Bug Condition - F07 Canonical Action Contract

_For any_ construction, policy load, or evaluation input where action dimensions are missing, incompatible, or stale, the repaired contract SHALL fail descriptively before action interpretation; for compatible inputs, every boundary SHALL report the same Canonical_Action_Dimension and preserve the approved mapping.

**Validates: Requirements 2.5**

Property 6: Bug Condition - F16 Explicit Mode-State Transitions

_For any_ sequence of non-crossing, entry, hold, release, re-entry, and reset events, an approved F16 state machine SHALL follow its transition table deterministically, expose every transition/reason in telemetry, use one canonical missing-config policy, and never override an explicit threshold.

**Validates: Requirements 2.6**

Property 7: Preservation - Safety Envelope and Existing Control Defenses

_For any_ candidate command trace, F′ SHALL retain F04 command limits and shall preserve finite fallback, capture handling, clipping, and post-processing behavior outside the approved change; any safety-limit violation SHALL reject the candidate.

**Validates: Requirements 3.2, 3.9**

Property 8: Preservation - Observation Schema and F15 Encoding

_For any_ existing observation flag combination, F′ SHALL preserve the ordered 16-feature base and current optional-segment names/dimensions; an approved F14 extension SHALL bump schema version, append deterministic names behind an `include_*` flag, update provenance/docs/tests and enforce checkpoint migration, while no candidate SHALL add an F15 angle-magnitude feature.

**Validates: Requirements 2.10, 3.3, 3.7**

Property 9: Preservation - Backend and Config Provenance

_For any_ remediation run, outputs SHALL record final backend, fallback occurrence/reason and all loaded-config mutations through `record_config_override`; no fallback run SHALL satisfy a strict-JSBSim gate and no unrecorded override SHALL be paper-safe.

**Validates: Requirements 3.4, 3.5**

Property 10: Preservation - Portable Reproducible Evidence

_For any_ baseline or candidate run, the artifact bundle SHALL contain a valid run manifest, artifact contract, resolved config, command line, git/platform metadata, config overrides, input/output hashes, declared outputs, and paper-safety result, and bundle verification SHALL reproduce every stored hash on another path.

**Validates: Requirements 3.6, 3.9**

Property 11: Experiment Gate - F03/F11 Dynamics

_For any_ PN/filter/bank-compensation/actuator candidate, the decision SHALL be based on per-stage and aggregate delay/phase, peak, saturation, slew, tracking, stability, and safety evidence under identical inputs; a candidate that regresses any declared safety/stability gate or lacks required strict-JSBSim evidence SHALL not modify runtime defaults.

**Validates: Requirements 2.7**

Property 12: Experiment Gate - F05 VPP Semantics

_For any_ range and bounded action in the declared envelope, each F05 candidate mapping SHALL be finite, bounded, monotonic, symmetric as specified, and safe near zero range; adoption SHALL occur only when paired evidence meets tactical, safety, tracking, command-effort, policy-stability, and migration thresholds.

**Validates: Requirements 2.4**

Property 13: Experiment Gate - F10/F17 Control and Switching

_For any_ representative envelope point or switching trace, candidates SHALL use the specified command-arbitration order and report dwell delay, travelled range, chatter, saturation attribution, tracking, stability, and safety; no scheduling or hysteresis default SHALL change unless all predeclared margins pass.

**Validates: Requirements 2.8, 2.11**

Property 14: Experiment Gate - F08/F12/F13 Optimization and Reward

_For any_ candidate evaluated under the frozen compute budget, seeds, and scenario matrix, reports SHALL include CEM coverage/convergence distributions or reward/terminal decomposition and exploit indicators as applicable; a candidate SHALL be rejected unless its statistical, compute, task, and safety gates pass without hidden config mutation.

**Validates: Requirements 2.9**

Property 15: Experiment Gate - F18 Strict Transfer Evidence

_For any_ claimed simple-to-JSBSim transfer result, paired runs SHALL share scenarios, seeds, policy/gain artifacts, control schedule, action contract, and metrics, and the JSBSim member SHALL record final backend `jsbsim` with no fallback; otherwise transfer status SHALL remain `needs_more_evidence`.

**Validates: Requirements 2.12**

Property 16: Preservation - F09 Training Semantics

_For any_ F09 terminology or metric clarification, bilevel update, snapshot, and rollback behavior in F′ SHALL produce the same trace as F for identical inputs; a behavior change requires a separate approved specification.

**Validates: Requirements 3.8**

Property 17: Decision Ledger Completeness

_For any_ completed remediation run, every F01–F18 ledger entry SHALL link evidence and use an allowed final status; F04 SHALL be `by_design_no_change`, F15 `disproven_no_change`, and F01/F02/F06/F07/F16 SHALL be either `implemented_and_verified` or `needs_more_evidence` with an explicit blocker.

**Validates: Requirements 2.13**
## Fix Implementation

### Changes Required

Implementation is split into two classes of work. **Evidence-only work** may add deterministic fixtures, metrics, reports, ledgers, pipeline stages, and tests outside Protected_Paths. **Runtime work** is conditional and may begin only when its finding-specific gate artifact says `approved`. No task commits or pushes.

1. **Evidence freeze and gate controller**
   - Create a protected-path SHA256 manifest and a guard test that runs before and after every evidence task.
   - Create a remediation-ledger schema for F01–F18 and an Evidence_Gate schema containing required evidence hashes, acceptance values, decision, and protected-path allowlist.
   - Make task orchestration interpret `rejected` as an explicit no-change decision and `needs_more_evidence` as a blocked implementation, not as permission to continue mutation.

2. **Reusable deterministic analysis layer**
   - Add pure fixtures for geometry, LOS rate, roll/rate, scheduler clocks, cascaded filters, actuator traces, VPP mappings, mode events, reward decomposition, and paired backend runs.
   - Add metrics for sign/boundedness, response delay, phase, rise/settling/overshoot, slew, chatter, switching travel, safety occupancy, and paired seed distributions.
   - Keep this layer independent from runtime classes so exploratory tests can demonstrate counterexamples against F before any protected edit.

3. **F01/F02 approved guidance change**
   - **Conditional files:** `src/uav_vpp_guidance/guidance/los_rate_guidance.py` and only separately approved interface/support tests.
   - Separate named physical contributions: LOS angular-rate/geometric normal-load computation, outer heading/turn demand, and inner roll-rate damping/tracking.
   - Require explicit units/signs, state-source validation, near-zero range handling, and compatibility behavior. Do not alter `CommandPostProcessor` limits.
   - Implement F01 and F02 only when both task-specific designs are approved; otherwise preserve each rejected component independently.

4. **F06 then F03/F11 approved timing/dynamics changes**
   - **Conditional files:** the minimal scheduler/orchestration surface in `tracking_env.py`, affected guidance/flight-control modules, and config only as explicitly allowlisted.
   - Introduce independent periods with integer-ratio validation or explicitly designed resampling; define update order, zero-order hold/interpolation, filter advancement, reset, and timestamps.
   - Preserve Legacy_Single_Rate_Mode exactly. Implement F06 first; apply no more than one approved filter/bank-compensation/slew candidate per later isolated change.

5. **F07 then optionally F05 VPP changes**
   - **Conditional files:** `virtual_point/generator.py`, action validation/load boundaries, checkpoint metadata, and explicitly approved config schema.
   - Remove the silent constructor fallback as an independent F07 fix; validate one canonical dimension before interpreting actions or loading a policy.
   - Preserve legacy artifacts only through explicit compatibility/migration mode with provenance; never rewrite checkpoints in place.
   - Keep fixed-metre mapping unless the F05 experiment and checkpoint/action migration gate are approved.

6. **F16 then optionally F17/F10 changes**
   - **Conditional files:** the smallest mode-switch surface in `tracking_env.py`, then separately approved hybrid/flight-control modules.
   - Replace the implicit boolean latch with the approved state model and one canonical threshold owner; define entry, hold, release, re-entry, reset, timeout/fail-safe, and transition telemetry.
   - F17 dwell/hysteresis and F10 arbitration/scheduling remain unchanged unless their independent experiments pass.

7. **F08/F12–F14/F18 controlled work**
   - Keep optimizer, reward, observation, and default configs unchanged while gathering evidence.
   - If F14 passes, create a separate schema implementation gate before touching `observation.py`: bump schema version, preserve base order, append deterministic names behind an `include_*` flag, update `observation_schema` and provenance, define checkpoint migration, and update `README.md`, `AGENTS.md`, and dimensional/flag tests.
   - F18 changes no simulator-selection semantics; it validates paired strict backend evidence.

8. **AGENTS.md contract enforcement in every phase**
   - Backend selection uses config first, then legacy `env.use_jsbsim`; strict failure raises and non-strict fallback records final backend, occurrence, and reason.
   - Any script mutation after YAML load calls `record_config_override`, including timing, backend, mode-switch, reward, optimizer, or CLI-derived changes.
   - New result-producing work uses `PipelineStage`/equivalent manifest and artifact contract, snapshots resolved config, records input/output hashes, finalizes paper-safety, and creates a verifiable portable bundle.

9. **Finalization**
   - Re-run targeted tests and the frozen scenario matrix after each approved change; compare protected hashes and all compatibility/provenance contracts.
   - Complete each ledger entry independently. Do not collapse a rejected risk candidate into an unresolved defect or claim JSBSim transfer from fallback data.

## Testing Strategy

### Validation Approach

Validation has three stages per finding: (1) demonstrate the counterexample against F without protected edits, (2) check P for every generated buggy input against a candidate, and (3) compare F and F′ over non-buggy inputs and cross-cutting contracts. Tests are deterministic and non-watch. Commands use forms such as `python -m pytest <target> -v`; watch modes and development servers are prohibited.

Every run records the exact command, source revision, resolved config, seed, scenario, input artifacts, backend request/final selection, fallback fields, overrides, outputs, hashes, and acceptance decision. JSBSim-unavailable runs may validate harness behavior but cannot pass an F18 or JSBSim-required gate.

### Exploratory Bug Condition Checking

**Goal:** Surface concrete counterexamples on unmodified code and confirm or refute each root-cause hypothesis before implementation.

**Test Plan:** Use pure fixtures or isolated calls against the frozen implementation. Capture failures as evidence artifacts and immediately verify the protected-path SHA256 manifest remains unchanged.

**Test Cases:**
1. **F01 range/zero-rate test:** Hold angular geometry and LOS rate fixed while sweeping range; show the range-growing contribution and absent explicit LOS-rate response.
2. **F02 state isolation test:** Hold roll angle constant and reverse roll rate; show unchanged damping contribution, then hold rate constant and vary angle.
3. **F06 schedule trace:** Record one second of high-level, guidance, filter, low-level, and backend invocation timestamps and quantify cascade delay.
4. **F07 construction matrix:** Exercise explicit 3, missing, mismatched, stale-checkpoint, and legacy metadata inputs; capture silent fallback to 5.
5. **F16 transition trace:** Trigger the gate, move to a release-worthy state, and show `latched` persistence; construct missing-threshold cases showing 15° fallback versus audited 25° config.
6. **Risk probes:** Quantify F05 angular mapping, F03/F11 cascade delay, F17 0.6 s dwell/travel, F08 sampling distributions, reward decomposition, observation availability, and paired backend provenance without changing defaults.

**Expected Counterexamples:**
- Normal-load geometric contribution rises with range despite unchanged angular/rate error.
- Roll damping responds to angle but not isolated angular rate.
- All control stages share a 0.2 s orchestration step and no explicit independent scheduler contract.
- Missing VPP dimension silently chooses 5.
- Mode switch remains effective after the entry gate clears and missing threshold resolves to a duplicate code default.

If a counterexample does not occur, the corresponding root cause is refuted, the design is re-hypothesized, and runtime implementation remains blocked.

### Fix Checking

**Goal:** Verify that all inputs satisfying the finding-specific bug condition satisfy the expected predicate after an approved fix.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  REQUIRE evidenceGate(input.finding).status = approved
  result := fixedFunction(input)
  ASSERT expectedBehavior(result)
END FOR
```

F01/F02 use generated geometry and independent attitude/rate states; F06 uses generated valid clock ratios and reset points; F07 uses generated metadata/config combinations; F16 uses generated event sequences checked against a transition model.

### Preservation Checking

**Goal:** Prove non-buggy behavior and all non-approved protected surfaces are unchanged.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT observableBehavior(originalFunction(input))
         = observableBehavior(fixedFunction(input))
END FOR

FOR EACH protectedPath NOT IN approvedGate.allowlist DO
  ASSERT sha256_after(protectedPath) = sha256_frozen(protectedPath)
END FOR
```

**Testing Approach:** Differential property-based tests compare normalized observable outputs, state transitions, reset state, schema, provenance, and artifacts. Hash tests provide stricter byte preservation for unapproved files. Floating-point comparisons use documented tolerances only where an approved numerical change makes bit identity inappropriate.

**Test Cases:**
1. Legacy single-rate trace equivalence and unchanged command hold/filter behavior.
2. Existing 3D/4D/6D action paths and explicitly supported legacy artifact behavior.
3. Non-trigger, explicit-config, command-override, and reset mode-switch paths.
4. F04 saturation, capture radius, finite fallback, command postprocessing, and F15 encoding.
5. All observation flag combinations, deterministic feature names, base order, and checkpoint rejection/migration policy.
6. Backend strict failure/non-strict fallback telemetry and complete config override records.
7. Portable bundle copy/verification and rejection of missing/tampered artifacts.
8. F09 training trace equivalence.

### Unit Tests

- Pure F01 geometry/LOS-rate units, sign, symmetry, boundedness, range invariance, singular range, and saturation interaction.
- F02 roll-angle/rate isolation, step, reversal, speed sweep, saturation, and reset.
- Scheduler invocation count/timestamp, hold/interpolation, ratio rejection, filter advancement, and reset tests.
- VPP canonical dimension, missing/mismatch/stale metadata, bounds, symmetry, monotonicity, and migration errors.
- Mode-state transition-table, explicit threshold precedence, absent config, telemetry, timeout/fail-safe, and reset tests.
- Hash guard, gate schema, ledger status, metric, provenance, manifest, artifact contract, and bundle verification tests.
- Existing `tests/test_tracking_env_no_prediction.py` and `tests/test_common_provenance.py` whenever observation/provenance behavior is touched.

### Property-Based Tests

- Generate finite relative positions/velocities and mirrored/range-scaled geometries for Properties 1 and 3.
- Generate valid/invalid multi-rate ratios, horizons, and reset points for Property 4.
- Generate action dimensions and config/checkpoint/action-space metadata combinations for Property 5.
- Generate mode-event sequences around thresholds, hysteresis, dwell, and reset for Properties 6 and 13.
- Generate protected allowlists/gate states and tampered manifests/bundles for Properties 2 and 10.
- Generate observation flag combinations and candidate appended segments for Property 8.
- Generate paired seeds/scenarios/backend outcomes and config mutations for Properties 9, 11–15.

Each PBT is run with a fixed/reported seed and non-watch command. A failing shrunk counterexample is stored in the finding's evidence artifact.

### Integration Tests

- Full policy/VPP/guidance/postprocessor/backend flow for each approved isolated change, including safety telemetry.
- Legacy and approved multi-rate end-to-end traces with exact scheduler timestamps and latency budget.
- Policy/checkpoint load through environment/action space and generator validation.
- Full F16 entry/hold/release/re-entry/reset episodes and F17 crossing-speed sweeps.
- Baseline/candidate portable pipeline runs with resolved config, provenance, artifact contract, and cross-directory bundle verification.
- Paired simple and strict-JSBSim evaluation; fallback runs are asserted invalid as transfer evidence.
- Final regression matrix verifying protected hashes, F04 limits, F15 encoding, observation schema, backend fields, config overrides, and ledger completion.
