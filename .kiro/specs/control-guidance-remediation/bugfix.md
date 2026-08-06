# Bugfix Requirements Document

## Introduction

The completed control-method rationality audit, represented in the current workspace by `reports/control_method_rationality_audit_findings_20260725.{json,md}` and the historical audit branch, established that the remaining issues are not a single interchangeable defect. It verified five implementation defects: F01 (a guidance law named for LOS rate contains no LOS-rate term and includes a range-growing distance term), F02 (roll damping acts on roll angle rather than roll rate), F06 (high-level action, guidance, and low-level control share a 5 Hz time scale), F07 (VPP action-dimension config and constructor default disagree), and F16 (mode switching latches until reset and has configuration/default threshold ambiguity). It also established risks that require controlled experiments before any behavioral change: F03, F05, F08, F10–F14, F17, and F18.

This bugfix specification defines the evidence, decision gates, and testable behavior needed to remediate the verified defects one by one and to decide the risk findings from experiment rather than assumption. It must preserve the current reference behavior until each individual change is approved by its corresponding gate. F04 is an intentionally conservative, by-design safety envelope and is not a remediation target. F15 is disproven because the `(sin(theta), cos(theta))` pair preserves angle information over `[0, 2π)` and requires no observation-encoding change. F09 remains a terminology/measurement follow-up only; it does not authorize training or rollback behavior changes in this remediation.

No task in this specification changes protected runtime control or configuration files. Future implementation work is explicitly gated and must follow `AGENTS.md`: observation extensions require schema-versioning, deterministic feature names, downstream compatibility, and tests; backend selection must remain transparent and recorded; scripts that mutate loaded configuration must call `record_config_override`; pipeline artifacts must carry manifests and contracts.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the current LOS-rate-named guidance law computes normal-load command in the audited control path THEN the system omits an explicit LOS angular-rate term and adds `k_pos * distance / distance_scale_m`, so the position contribution grows with range rather than expressing a clearly convergent geometric error.

1.2 WHEN the current guidance roll channel computes its roll-rate command THEN the system applies the damping gain to `roll_rad` rather than roll angular rate and does not define a tested coordinated-turn relationship between heading-rate demand, bank, airspeed, and low-level tracking.

1.3 WHEN high-level VPP/RL actions, guidance updates, PN filtering, and low-level control execute in the default path THEN the system runs them at the shared 5 Hz (`dt = 0.2 s`) cadence, so it has no explicit multi-rate schedule or end-to-end latency budget; the audited first-order filter alone has approximately 0.5607 s time constant at `alpha = 0.3`.

1.4 WHEN an action dimension is omitted while constructing a VPP generator rather than supplied from the active configuration THEN the system uses a constructor default of 5 while the audited guidance configuration declares 3, allowing silent action-semantic divergence.

1.5 WHEN a mode-switch gate triggers during an episode THEN the system latches the mode until `reset()`, and WHEN the configured aspect threshold is absent THEN the system can use a code fallback of 15 degrees rather than the audited configured value of 25 degrees.

1.6 WHEN a fixed-metre VPP offset is interpreted as a tactical command across engagement ranges THEN the same offset has materially different angular meaning; the configured 800 m lateral bound corresponds to about 17.74 degrees at 2500 m and 45 degrees at 800 m, but the current behavior provides no validated distance-scaled semantic alternative.

1.7 WHEN PN filtering, guidance filtering, low-level filtering, and actuator behavior are assessed together THEN the system has no validated aggregate delay, phase-margin, actuator-rate, or bank-compensation evidence, so F03 and F11 remain quantified risks rather than confirmed implementation defects.

1.8 WHEN CEM, reward shaping, observation state/history, flight-control scheduling, hybrid dwell/hysteresis, or simple-to-JSBSim transfer is changed based only on the audit narrative THEN the system lacks the required controlled experiment, baseline comparison, and acceptance gate to distinguish a real improvement from a regression.

### Expected Behavior (Correct)

2.1 WHEN a replacement for the F01 guidance computation is proposed THEN the system SHALL define its physical inputs, units, sign conventions, saturation interaction, and naming consistently, SHALL include an explicit LOS-rate term if it continues to claim LOS-rate guidance, and SHALL demonstrate by deterministic geometry tests that the geometric-error contribution is bounded and converges rather than increasing solely because range increases.

2.2 WHEN a replacement for the F02 roll channel is proposed THEN the system SHALL use an explicitly defined roll-rate feedback signal or documented equivalent, SHALL define the handoff between heading/turn demand and low-level roll tracking, and SHALL demonstrate with deterministic step, reversal, and representative-speed tests that damping responds to roll rate without violating the retained safety envelope.

2.3 WHEN a multi-rate remedy for F06 is proposed THEN the system SHALL define independently configurable high-level, guidance, and low-level update periods, their integer or explicitly interpolated scheduling relation, command hold/interpolation semantics, reset behavior, and a cumulative latency budget; it SHALL demonstrate that the existing single-rate configuration remains reproducible when selected and that any new default is supported by measured closed-loop evidence.

2.4 WHEN a VPP semantic remedy for F05 is proposed THEN the system SHALL compare the current fixed-metre representation with distance-scaled and/or angular alternatives over the declared close-range operating envelope, SHALL retain a bounded and documented action-to-offset mapping, and SHALL adopt a new representation only if its controlled evaluation improves or preserves safety, tracking, and tactical metrics against the frozen baseline.

2.5 WHEN VPP action construction occurs after remediation of F07 THEN the system SHALL require or validate one canonical action dimension across configuration, generator, action space, policy/checkpoint metadata, and runtime validation; absent, incompatible, or stale dimensions SHALL fail clearly rather than silently selecting a different action meaning.

2.6 WHEN a mode-switch policy is remediated for F16 THEN the system SHALL define whether its latch is permanent, releaseable, or state-machine-based; SHALL define entry, hold, release, reset, and missing-config behavior; SHALL use one canonical default source for thresholds; and SHALL demonstrate deterministic state transitions for crossing and non-crossing cases without silently changing explicit user configuration.

2.7 WHEN PN/filter/actuator dynamics are considered for F03 and F11 THEN the system SHALL measure and report the complete command-to-actuator delay/phase budget, including all enabled filters and actuator dynamics, SHALL evaluate bank-compensation and explicit rate-limit candidates only through controlled experiments, and SHALL reject changes that worsen safety, stability, or baseline task performance.

2.8 WHEN flight-control scheduling and command arbitration are considered for F10 THEN the system SHALL establish a testable single ordering for guidance, altitude-hold, energy, and protection contributions and SHALL compare fixed mapping with qbar/AoA-scheduled control under representative envelope points before adopting any runtime change.

2.9 WHEN CEM or reward changes are considered for F08, F12, and F13 THEN the system SHALL run deterministic seed-controlled baselines and ablations, SHALL report CEM coverage/convergence and reward-component/terminal-return decomposition, and SHALL approve a changed population, covariance model, early-stop rule, or reward weight only when predefined statistical and safety gates are met.

2.10 WHEN observation remediation is considered for F14 THEN the system SHALL first establish whether AoA, beta, roll, angular-rate, command-tracking error, target turn-rate, or history adds measurable information beyond existing optional segments; any approved observation extension SHALL follow the `AGENTS.md` schema contract, including a schema-version change, deterministic `feature_names`, dimension/flag tests, checkpoint compatibility policy, README and contract documentation.

2.11 WHEN hybrid dwell/hysteresis is considered for F17 THEN the system SHALL quantify crossing distance, switching delay, chatter rate, and task metrics at representative closing speeds, and SHALL alter dwell or hysteresis only when the experiment gate demonstrates reduced harmful delay or chatter without safety regression.

2.12 WHEN simple-to-JSBSim transfer is considered for F18 THEN the system SHALL quantify, using identical scenario definitions, seeds, policy/gain artifacts, and recorded backend provenance, the difference between simple and JSBSim outcomes; final CEM or bilevel gains SHALL not be claimed transferable until they have passed the predefined JSBSim revalidation gate.

2.13 WHEN the remediation evidence is reviewed THEN the system SHALL classify each item as `implemented_and_verified`, `experiment_rejects_change`, `needs_more_evidence`, `by_design_no_change`, or `disproven_no_change`, with F04 recorded as `by_design_no_change` and F15 recorded as `disproven_no_change`.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN no individual remediation gate has approved a runtime change THEN the system SHALL CONTINUE TO preserve byte-identical protected runtime/control and configuration files, including guidance, flight-control, VPP, reward, observation, tracking environment, optimizer, and configuration paths.

3.2 WHEN F04 safety limits are evaluated THEN the system SHALL CONTINUE TO retain the audited conservative normal-load, roll-rate, and throttle limits unless a separate JSBSim-backed safety decision explicitly approves a change; this specification does not treat F04 as a defect.

3.3 WHEN observations are consumed by existing policies and checkpoints THEN the system SHALL CONTINUE TO preserve the 16-feature base ordering and all existing optional-segment semantics unless a versioned migration is approved under the `AGENTS.md` observation-schema contract.

3.4 WHEN training or evaluation selects a backend THEN the system SHALL CONTINUE TO resolve and record the final active backend, fallback occurrence, and fallback reason, and scripts SHALL CONTINUE TO avoid silent backend overrides.

3.5 WHEN a script changes a loaded configuration in an approved future experiment or rollout THEN the system SHALL CONTINUE TO record the mutation through `record_config_override` in provenance, including backend, timing, guidance, optimizer, reward, or mode-switch overrides.

3.6 WHEN a remediation experiment or evaluation emits an artifact THEN the system SHALL CONTINUE TO use reproducible configs, input/output hashes, command provenance, and artifact contracts consistent with the portable pipeline requirements in `AGENTS.md`.

3.7 WHEN F15 is reviewed THEN the system SHALL CONTINUE TO use the existing sine/cosine angle encoding and SHALL NOT add an angle-magnitude feature to remedy the disproven premise.

3.8 WHEN F09 is reviewed THEN the system SHALL CONTINUE TO preserve existing bilevel training behavior; a separately approved documentation/metric task may clarify its name and meaning, but this remediation shall not introduce rollback or change optimization semantics without an independent experiment gate.

3.9 WHEN experiments are run for risk findings THEN the system SHALL CONTINUE TO preserve frozen baselines, use controlled seeds and scenario matrices, retain all pass/fail evidence, and reject any candidate that regresses declared safety, stability, schema compatibility, or backend-provenance requirements.
