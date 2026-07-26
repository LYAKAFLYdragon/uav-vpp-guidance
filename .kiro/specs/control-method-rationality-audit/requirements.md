# Requirements Document

## Introduction

This spec drives a **structured investigation and triage** of the 18 control-method
findings produced by the control-method rationality audit
(`reports/control_method_rationality_audit_prompt.md`). The findings span guidance-law
physics, hierarchical structure, gain optimization, flight control, reward design,
observation completeness, mode switching, and simulation fidelity.

The goal of this spec is **NOT** to blindly implement fixes. It is to:

1. **Verify** each finding against the actual source (some prompt premises were already
   shown to be inaccurate — e.g. finding #15 sin/cos encoding is correct, finding #1's
   law contains no LOS-rate term despite its name).
2. **Reproduce** each risk with a minimal, deterministic diagnostic (unit test, probe
   script, or numeric analysis) so the finding is backed by executable evidence, not
   assertion.
3. **Triage** each finding into: `confirmed_defect`, `confirmed_risk`, `by_design_ok`,
   or `premise_incorrect`, with an evidence trail.
4. **Recommend** a concrete, scoped remediation for each `confirmed_*` finding, ordered
   by impact, without changing frozen behavior in this spec.

Every code fix that changes runtime control behavior is deferred to a **separate,
explicitly gated task** so that the investigation itself never silently alters guidance,
flight-control, reward, or observation semantics. Any change that touches the observation
schema, backend selection, or config overrides MUST follow the contracts in `AGENTS.md`
(schema version bump, `record_config_override`, corresponding tests).

Work is performed on branch `research/control-method-rationality-audit`. The investigation
is read-only against existing checkpoints, configs, and results except for allowlisted new
outputs (diagnostic scripts, tests, and this spec's report artifacts).

### Reference source paths (authoritative)

| Concern | Path |
|---|---|
| LOS-rate ("pursuit") guidance | `src/uav_vpp_guidance/guidance/los_rate_guidance.py` |
| Proportional navigation | `src/uav_vpp_guidance/guidance/proportional_navigation.py` |
| Hybrid guidance | `src/uav_vpp_guidance/guidance/hybrid_guidance.py` |
| VPP generator | `src/uav_vpp_guidance/virtual_point/generator.py` |
| Gain dataclass | `src/uav_vpp_guidance/guidance/gain_config.py` |
| Enhanced / gain-scheduled controller | `src/uav_vpp_guidance/flight_control/enhanced_low_level_controller.py` |
| PID controller family | `src/uav_vpp_guidance/flight_control/pid_controllers.py` |
| Low-level controller | `src/uav_vpp_guidance/flight_control/low_level_controller.py` |
| Actuator interface | `src/uav_vpp_guidance/flight_control/actuator_interface.py` |
| Reward | `src/uav_vpp_guidance/envs/reward.py` |
| Observation | `src/uav_vpp_guidance/envs/observation.py` |
| CEM optimizer | `src/uav_vpp_guidance/gain_optimizer/cem.py` |
| Bilevel trainer | `src/uav_vpp_guidance/gain_optimizer/bilevel_trainer.py` |
| Mode-switch gate | `src/uav_vpp_guidance/envs/tracking_env.py::_evaluate_mode_switch_gate` |
| Guidance config | `config/guidance.yaml` |

### Reference scenario constraints

F-16-class (subsonic) fighter, 3000–5000 m altitude, close-range engagement, default
control frequency 5 Hz (dt = 0.2 s). Findings are scored against these conditions.

---

## Requirements

### Requirement 1 — Guidance-law physics verification (findings #1–#4)

**User Story:** As a control engineer, I want each guidance-law finding verified against
the source and reproduced with a numeric diagnostic, so that I can trust the triage verdict.

#### Acceptance Criteria

1. WHEN finding #1 (LOS-rate law lacks a λ̇ term; `k_pos` scales with raw distance) is
   investigated THEN the system SHALL confirm from `_compute_nz_cmd` that
   `nz = base_nz + k_los·arctan2(rel_z, horiz) + k_pos·(distance/distance_scale_m)`, SHALL
   confirm no LOS-rate (dλ/dt) term is present, AND SHALL emit a numeric probe showing the
   `k_pos` term grows with range (opposite of a well-posed geometric error term).
2. WHEN finding #2 (roll channel: `k_damp` multiplies roll ANGLE not rate; no coordinated
   turn kinematics) is investigated THEN the system SHALL confirm `roll_rate_cmd =
   k_roll·heading_error − k_damp·current_roll` uses `own_state["roll_rad"]`, AND SHALL
   confirm the `ψ̇ = g·tan(φ)/V` coordinated-turn relation is absent from the guidance layer.
3. WHEN finding #3 (PN λ̇ first-order filter α=0.3 phase lag) is investigated THEN the
   system SHALL compute the effective filter time constant τ = −dt/ln(1−α) and confirm it
   is ≈0.56 s at dt=0.2 s, AND SHALL confirm the `nz = 1 + a_z/g + |a_horiz|/g` composition
   adds horizontal accel magnitude unconditionally (no `1/cos φ` bank compensation).
4. WHEN finding #4 (saturation limits) is investigated THEN the system SHALL confirm the
   configured limits (nz∈[-2,7]g, roll_rate∈[-1.5,1.5] rad/s, throttle∈[0.4,0.9]) from
   `config/guidance.yaml`, AND SHALL document each against F-16-class references, flagging
   roll_rate (≈86°/s vs ~270°/s capability) and −2 g as conservative.
5. WHEN each of findings #1–#4 is triaged THEN the system SHALL assign exactly one verdict
   from {`confirmed_defect`, `confirmed_risk`, `by_design_ok`, `premise_incorrect`} with a
   linked evidence artifact (test id or probe output).

### Requirement 2 — Hierarchical structure verification (findings #5–#7)

**User Story:** As a systems designer, I want the layering and timescale findings verified,
so that structural remediation is justified.

#### Acceptance Criteria

1. WHEN finding #5 (VPP metric offset semantics drift with range) is investigated THEN the
   system SHALL confirm from the generator config that `d_long_range/d_lat_range/d_vert_range`
   are fixed metric ranges (±1500/±800/±500 m), AND SHALL emit a probe computing the angular
   deviation a 500 m lateral offset induces at 2500 m range (≈11.3°) and at a near range.
2. WHEN finding #6 (single 5 Hz timescale for RL/guidance/PID; no timescale separation) is
   investigated THEN the system SHALL confirm the shared dt=0.2 s across layers AND compute
   the first-order filter time constant (≈0.56 s) to quantify the phase lag.
3. WHEN finding #7 (action_dim inconsistency: config=3 vs generator default=5) is
   investigated THEN the system SHALL confirm `config/guidance.yaml` sets `action_dim: 3`
   while `VirtualPointGenerator.__init__` defaults to 5, AND SHALL flag the inconsistency.
4. WHEN each of findings #5–#7 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 3 — Gain-optimization verification (findings #8–#9)

**User Story:** As an optimization owner, I want the CEM/bilevel findings verified, so that
optimizer-credibility issues are documented with evidence.

#### Acceptance Criteria

1. WHEN finding #8 (CEM undersampling: 7-D space, 12 candidates, 3 elites, diagonal std) is
   investigated THEN the system SHALL confirm `candidates=12`, `elite_ratio=0.25`
   (⇒ `n_elite=max(1,3)=3`), diagonal `np.std(elite, axis=0)` covariance, and
   `convergence_tol=0.001` early-stop from `cem.py`, AND SHALL document the ≥10·dim
   population heuristic.
2. WHEN finding #9 (bilevel: no convergence guarantee; regret is monotone-nonincreasing and
   not true regret; no rollback) is investigated THEN the system SHALL confirm `_compute_regret`
   returns `max(0, 1 − best_known_SR)` with monotone `best_known`, AND SHALL confirm
   `_save_policy_snapshot` exists but is never used for rollback in `train()`.
3. WHEN each of findings #8–#9 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 4 — Flight-control verification (findings #10–#11)

**User Story:** As a flight-control engineer, I want the actuator-mapping and filtering
findings verified, so that fidelity gaps are documented.

#### Acceptance Criteria

1. WHEN finding #10 (base controller uses fixed-gain feedforward with no dynamic-pressure
   scheduling) is investigated THEN the system SHALL confirm `EnhancedLowLevelController`
   uses constant `nz_to_elevator_gain`/`roll_rate_to_aileron_gain`, AND SHALL confirm
   `GainScheduledEnhancedController`/`RobustPIDController` provide q̄/altitude/AoA scheduling,
   AND SHALL confirm multiple independent nz biases (guidance `k_pos`, `altitude_hold`,
   `energy_boost`) are summed without a single arbitrator.
2. WHEN finding #11 (first-order LPF is the only smoothing; no slew-rate limiter) is
   investigated THEN the system SHALL confirm the command LPF `y=α·u+(1−α)·y_prev` is the
   sole smoothing stage and that no explicit actuator slew-rate limit exists, AND SHALL
   confirm series-filtering stages (guidance optional filter + low-level input filter + PN
   λ̇ filter) are not jointly budgeted.
3. WHEN each of findings #10–#11 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 5 — Reward-design verification (findings #12–#13)

**User Story:** As a reward designer, I want the weight and terminal-magnitude findings
verified, so that reward-shaping risk is quantified.

#### Acceptance Criteria

1. WHEN finding #12 (weight balance; `w_safety=2.0` highest) is investigated THEN the system
   SHALL confirm the default weights from `reward.py` AND confirm the safety penalty only
   activates when `altitude < min_alt + 1000 m`.
2. WHEN finding #13 (terminal magnitude ±200/−300 vs step |r|<~5) is investigated THEN the
   system SHALL confirm the terminal constants AND enumerate reward-hacking surfaces
   (`w_alive`, `w_closing`) that are 0 by default but risky if enabled.
3. WHEN each of findings #12–#13 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 6 — Observation-completeness verification (findings #14–#15)

**User Story:** As an RL engineer, I want observation findings verified against the schema
contract, so that any extension follows `AGENTS.md`.

#### Acceptance Criteria

1. WHEN finding #14 (missing own-body/history/target-maneuver states) is investigated THEN
   the system SHALL confirm the 16-D base schema from `build_observation`, confirm the
   optional segments (gains, guidance_state, saturation, prediction, opponent_stage,
   task_type), AND identify which missing signals (AoA, β, φ, p, command-tracking error,
   target turn-rate) are covered by an existing optional segment vs genuinely absent.
2. WHEN finding #15 (sin/cos encoding "loses angle magnitude") is investigated THEN the
   system SHALL PROVE the premise incorrect by showing `(sin θ, cos θ)` is injective over
   `[0,2π)` (e.g. 30° vs 330° differ in sin), AND SHALL mark finding #15 `premise_incorrect`.
3. WHEN each of findings #14–#15 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 7 — Mode-switch and robustness verification (findings #16–#17)

**User Story:** As a guidance engineer, I want the mode-switch latch and hybrid hysteresis
findings verified, so that dynamic-engagement misbehavior is documented.

#### Acceptance Criteria

1. WHEN finding #16 (mode-switch latches for the whole episode; config/default threshold
   mismatch; narrow aspect coverage) is investigated THEN the system SHALL confirm
   `_mode_switch_latched` is only cleared on `reset()`, confirm `config` sets
   `aspect_threshold_deg: 25` while code default is `15`, AND confirm `crossing_aspect_threshold_deg`
   is unset by default.
2. WHEN finding #17 (hybrid hysteresis: 500 m band, min_dwell=3 steps=0.6 s) is investigated
   THEN the system SHALL confirm the hysteresis/dwell values from `hybrid_guidance.py` AND
   evaluate the 0.6 s dwell against a representative closing speed.
3. WHEN each of findings #16–#17 is triaged THEN the system SHALL assign exactly one verdict
   with a linked evidence artifact.

### Requirement 8 — Simulation-fidelity verification (finding #18)

**User Story:** As an evaluation owner, I want the simple-vs-JSBSim transfer risk documented,
so that gain-transfer degradation is anticipated.

#### Acceptance Criteria

1. WHEN finding #18 (simple backend omits aero coupling, engine lag, q̄-varying limits;
   sim-to-sim degradation) is investigated THEN the system SHALL enumerate the dynamic
   effects omitted by the simple backend AND confirm which evaluation controller is used for
   JSBSim.
2. WHEN finding #18 is triaged THEN the system SHALL assign exactly one verdict AND SHALL
   recommend that final CEM/bilevel gains be re-validated on JSBSim.

### Requirement 9 — Consolidated triage report and remediation backlog

**User Story:** As the project lead, I want one consolidated, evidence-linked report and a
priority-ordered remediation backlog, so that fixes can be scheduled without re-deriving
the analysis.

#### Acceptance Criteria

1. WHEN the investigation completes THEN the system SHALL emit a single Chinese-language
   report `reports/control_method_rationality_audit_findings_20260725_zh.md` containing, for
   each of the 18 findings: the verdict, the source evidence (file:line), the diagnostic
   artifact reference, the failure mode, and the recommended remediation.
2. WHEN the report is emitted THEN it SHALL include a priority-ordered remediation backlog
   (impact-ranked) AND SHALL explicitly list which findings are `premise_incorrect`
   (at least #15) and require no change.
3. WHEN a machine-readable summary is needed THEN the system SHALL emit
   `reports/control_method_rationality_audit_findings_20260725.json` with one record per
   finding: `id`, `dimension`, `verdict`, `evidence_paths`, `diagnostic_ref`,
   `failure_mode`, `recommendation`, `priority`.
4. WHEN the report references a numeric claim (τ≈0.56 s, ≈11.3° offset, n_elite=3) THEN each
   such claim SHALL be reproducible from a committed diagnostic script or test.

### Requirement 10 — Non-destructive investigation guarantees (regression prevention)

**User Story:** As a maintainer, I want the investigation to change no runtime behavior, so
that the audit itself introduces no regressions.

#### Acceptance Criteria

1. WHEN the investigation runs THEN it SHALL NOT modify any guidance, flight-control, reward,
   observation, VPP, or optimizer source module (read-only), writing only diagnostic scripts,
   tests, and the two report artifacts.
2. WHEN the investigation runs THEN it SHALL NOT modify `config/guidance.yaml` or any other
   config; any proposed config change SHALL appear only as a recommendation in the report.
3. WHEN the investigation runs THEN it SHALL NOT retrain, re-optimize gains, or modify
   checkpoints.
4. WHEN a diagnostic needs to instantiate a controller/guidance object THEN it SHALL use the
   public constructors with in-test config dicts and SHALL NOT mutate shared config files.
5. IF any finding's remediation is later implemented THEN it SHALL be a separate, gated task
   that follows the `AGENTS.md` observation-schema / backend / config-override / testing
   contracts.
