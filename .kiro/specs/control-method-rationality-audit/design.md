# Design Document

## Overview

This design specifies a **read-only, evidence-first investigation harness** that verifies,
reproduces, and triages the 18 control-method findings from the rationality audit.

The core design principle: **every verdict must be backed by a committed, re-runnable
artifact**. No finding is accepted or rejected on narrative reasoning alone. Two prompt
premises were already found inaccurate during the initial review, which is exactly why the
harness must re-derive facts from source rather than trust the finding text.

The harness has three layers:

| Layer | Module | Responsibility |
|---|---|---|
| Static source facts | `ControlAuditProbe` (source introspection) | Read the real defaults/limits/structure from the modules and config; no behavior change |
| Numeric diagnostics | `probes/*.py` pure functions | Reproduce each numeric claim (filter τ, offset angle, elite count, injectivity) deterministically |
| Triage + reporting | `ControlAuditReport` | Combine static facts + diagnostics into per-finding verdicts, emit md + json |

### Non-goals

- Not fixing anything. No guidance/flight-control/reward/observation/VPP/optimizer source
  module is modified by this spec.
- Not retraining, not re-optimizing gains, not touching checkpoints or configs.
- Not proving JSBSim-vs-simple degradation empirically (that would require runs). Finding
  #18 is triaged as an analytic/structural risk with an explicit recommendation, and the
  report must state that no run-based quantification was performed.

---

## Glossary

- **Finding**: one of the 18 numbered audit items (`F01`..`F18`).
- **Verdict**: exactly one of
  - `confirmed_defect` — source contradicts sound control practice AND causes wrong behavior.
  - `confirmed_risk` — source is defensible but carries a concrete failure mode under the
    reference scenario.
  - `by_design_ok` — behavior is correct/intentional; no change needed.
  - `premise_incorrect` — the finding's factual premise is false against source/math.
- **Evidence path**: `file:line` (or `file::symbol`) locating the code that establishes a fact.
- **Diagnostic ref**: the test id or probe function that reproduces a numeric claim.
- **Reference scenario**: F-16-class subsonic, 3000–5000 m, close range, dt = 0.2 s (5 Hz).

### Bug Condition

`isBugCondition(finding)` is true when the finding's stated premise is **verifiable against
committed source or closed-form math**. All 18 findings satisfy this except the run-based
portion of F18 (simple→JSBSim degradation magnitude), which is explicitly scoped as
analytic-only.

---

## Properties

### Property 1: Expected Behavior — Every finding gets a source-backed verdict

For every finding `F01..F18`, the harness emits a record containing a verdict from the four
controlled labels, at least one evidence path resolving to real committed source, a
diagnostic ref for every numeric claim, a failure mode, and a recommendation. No finding is
left unlabeled, and no verdict is asserted without evidence.

Specifically, the harness must reproduce these verified facts:

| Fact | Value | Source |
|---|---|---|
| LOS law nz composition | `base_nz + k_los·elev + k_pos·(d/scale)`, no λ̇ | `los_rate_guidance.py::_compute_nz_cmd` |
| Roll damping operand | `own_state["roll_rad"]` (angle, not rate) | `los_rate_guidance.py::compute_command` |
| Filter time constant | τ = −dt/ln(1−α) ≈ 0.56 s at α=0.3, dt=0.2 | closed form + `_apply_internal_filter` |
| PN nz composition | `1 + a_z/g + |a_horiz|/g` (no `1/cos φ`) | `proportional_navigation.py` |
| Limits | nz [-2,7], roll_rate [-1.5,1.5], throttle [0.4,0.9] | `config/guidance.yaml` |
| VPP offset ranges | ±1500 / ±800 / ±500 m fixed metric | `config/guidance.yaml` |
| action_dim mismatch | config `3` vs generator default `5` | `guidance.yaml` + `generator.py` |
| CEM elites | `max(1, int(12·0.25)) = 3` of 12, diagonal std | `cem.py` |
| Regret formula | `max(0, 1 − best_known_SR)`, monotone | `bilevel_trainer.py::_compute_regret` |
| Snapshot unused | `_save_policy_snapshot` never called in `train()` | `bilevel_trainer.py` |
| Base obs dim | 16, fixed order per `AGENTS.md` | `observation.py::build_observation` |
| sin/cos injective | `(sin,cos)` pair uniquely determines θ | closed form |
| Mode-switch latch | cleared only in `reset()` | `tracking_env.py` |
| Threshold mismatch | config `25.0` vs code default `15.0` | `guidance.yaml` + `_evaluate_mode_switch_gate` |
| Hybrid dwell | `min_dwell_steps=3` ⇒ 0.6 s at 5 Hz | `hybrid_guidance.py` |

### Property 2: Preservation — Zero runtime behavior change

After the investigation completes, every control-path source module and every config file is
**byte-identical** to its pre-investigation state. The changed-path set is a subset of the
allowlist:

- `.kiro/specs/control-method-rationality-audit/**`
- `scripts/audit_control_method_rationality.py`
- `src/uav_vpp_guidance/evaluation/control_method_audit.py`
- `tests/test_control_method_audit.py`
- `reports/control_method_rationality_audit_findings_20260725_zh.md`
- `reports/control_method_rationality_audit_findings_20260725.json`

Explicitly preserved (must not change): all of `src/uav_vpp_guidance/guidance/**`,
`flight_control/**`, `virtual_point/**`, `envs/reward.py`, `envs/observation.py`,
`envs/tracking_env.py`, `gain_optimizer/**`, and `config/**`.

---

## Architecture

```
scripts/audit_control_method_rationality.py   (CLI, argparse, configurable paths)
        │
        ├── src/uav_vpp_guidance/evaluation/control_method_audit.py
        │       ├── load_source_facts(paths)        -> SourceFacts        (static, read-only)
        │       ├── run_diagnostics()               -> DiagnosticResults  (pure numeric)
        │       ├── triage(facts, diagnostics)      -> List[FindingRecord]
        │       ├── self_check(records)             -> None | raises
        │       ├── emit_json(records, path)
        │       └── emit_markdown(records, path)
        │
        └── tests/test_control_method_audit.py      (verifies diagnostics + triage rules)
```

### Design decision: diagnostics are pure functions, not simulations

Each numeric claim is reproduced by a **closed-form or single-call** computation rather than
an episode rollout. Rationale: rollouts are slow, seed-sensitive, and would tempt the harness
into becoming a tuning loop. Pure functions make the report's numbers deterministic and
reviewable.

Examples:

- `filter_time_constant(alpha, dt) -> float` — returns `-dt/log(1-alpha)`; used for F03, F06, F11.
- `offset_angular_deviation(offset_m, range_m) -> float` — returns `atan2(offset_m, range_m)`;
  used for F05.
- `cem_elite_count(candidates, elite_ratio) -> int` — mirrors `max(1, int(n·ratio))`; used for F08.
- `sincos_injective_witness(a_deg, b_deg) -> dict` — returns the sin/cos pair for two angles
  sharing a cos value, proving separability; used for F15.
- `dwell_seconds(steps, dt)` and `range_travel(closing_mps, seconds)` — used for F16/F17.

### Design decision: static facts read from real modules, not hardcoded

`load_source_facts` instantiates the real classes with the real config to read effective
values, e.g. constructing `LOSRateGuidance(config)` and reading `nz_min/nz_max/alpha_filter`,
and constructing `CEMGainOptimizer` with the real gain space to read `candidates`/`elite_ratio`.
This ensures the report cannot drift from source. Instantiation uses in-memory config dicts
and never writes back.

For structural facts that cannot be read from an instance (e.g. "λ̇ term absent",
"`_save_policy_snapshot` never called in `train()`"), the harness records an explicit
`evidence_path` plus a short source excerpt, and the corresponding assertion lives in the
test suite as a source-text check rather than a behavioral check.

### Design decision: one canonical finding table

The 18 findings, their dimensions, and their expected verdict category are declared once in a
module-level table (`FINDING_SPECS`) so the md and json outputs are generated from a single
in-memory structure and cannot disagree. The verdict itself is computed by `triage`, not
hardcoded in the table — the table only declares identity, dimension, and which diagnostics
each finding depends on.

---

## Components and Interfaces

### `SourceFacts` (dataclass)

Read-only snapshot of effective values:

```
guidance_limits: dict          # nz_min/max, roll_rate_min/max, throttle_min/max
los_gains: dict                # k_los, k_pos, k_damp, k_roll, k_speed, alpha_filter
los_params: dict               # distance_scale_m, base_nz, capture_radius_m, ...
pn_params: dict                # navigation_constant, los_rate_filter_alpha, dt, max_accel
hybrid_params: dict            # hybrid_mode, range_threshold_m, hysteresis_m, min_dwell_steps
mode_switch_config: dict       # enabled, aspect_threshold_deg, crossing_*, range/closing
mode_switch_code_default: float  # the 15.0 fallback in _evaluate_mode_switch_gate
vpp_config: dict               # action_dim, d_long/lat/vert_range, smoothing_alpha
vpp_generator_default_action_dim: int
reward_weights: dict           # w_range/angle/energy/safety/saturation/smooth, terminals
controller_gains: dict         # nz_to_elevator_gain, roll_rate_to_aileron_gain, PID gains
observation_base_dim: int      # 16
observation_feature_names: list
cem_config: dict               # candidates, elite_ratio, convergence_tol, noise_floor
control_dt: float              # 0.2
evidence_paths: dict           # finding_id -> [file:line, ...]
```

### `DiagnosticResults` (dataclass)

```
filter_tau_s: float                  # ≈0.5601 at alpha=0.3, dt=0.2
pn_filter_tau_s: float
vpp_offset_angle_deg_at_2500m: float # ≈11.31 for 500 m lateral
vpp_offset_angle_deg_at_800m: float  # near-range comparison
cem_elite_count: int                 # 3
cem_recommended_population: int      # 10 * dim = 70
sincos_witness: dict                 # {"30deg": (0.5, 0.866), "330deg": (-0.5, 0.866)}
dwell_seconds: float                 # 0.6
dwell_range_travel_m: float          # closing_mps * dwell_seconds
roll_rate_limit_deg_s: float         # 85.94
f16_roll_rate_reference_deg_s: float # ~270 (documented reference, cited)
terminal_to_step_ratio: float        # |terminal| / typical step magnitude
```

### `FindingRecord` (dataclass -> one json object / one md section)

```
id: str                 # "F01".."F18"
dimension: str          # e.g. "guidance_physics"
title_zh: str
verdict: str            # confirmed_defect | confirmed_risk | by_design_ok | premise_incorrect
evidence_paths: List[str]
diagnostic_ref: List[str]     # test ids / probe function names
observed_fact: str            # what the source actually does
failure_mode: str             # what breaks under the reference scenario
recommendation: str
priority: int                 # 1 = highest; premise_incorrect -> 0 (no action)
confidence: str               # high | medium | low
notes: str                    # e.g. analytic-only scope for F18
```

### Triage rules (executable, in `triage`)

Applied in order per finding:

1. IF a diagnostic **disproves** the finding's premise → `premise_incorrect`, `priority = 0`.
   (F15 is the known case: the sin/cos witness shows separability.)
2. ELSE IF the source fact shows a **wrong physical/mathematical relation** (missing required
   term, wrong operand, inverted trend) → `confirmed_defect`.
   (Expected: F01 missing λ̇ + range-growing `k_pos`; F06 no timescale separation;
   F16 permanent latch.)
3. ELSE IF the source fact is defensible but a diagnostic quantifies a concrete failure mode
   under the reference scenario → `confirmed_risk`.
4. ELSE → `by_design_ok`.

Priority is assigned from an impact ordering declared in the design (see below) and must be a
stable, deterministic function of the verdict set — not ad-hoc.

### Impact ordering (drives `priority`)

1. F01 guidance physics (pursuit mislabeled as LOS-rate; meaningless `k_pos` range term)
2. F06 no timescale separation (amplifies every inner-loop issue)
3. F05 VPP metric-offset semantics drift with range
4. F02 roll channel: no coordinated-turn kinematics, damping on angle not rate
5. F16 mode-switch permanent latch + threshold mismatch
6. F08 / F09 optimizer credibility (CEM undersampling; regret not true regret, no rollback)

Remaining findings (F03, F04, F07, F10, F11, F12, F13, F14, F17, F18) follow, ordered by
dimension then id. F15 gets `priority = 0`.

### `self_check(records)` — generic invariants only

- `len(records) == 18` and ids are exactly `F01..F18` with no duplicates.
- Every `verdict` is one of the four controlled labels.
- Every record has ≥1 `evidence_paths` entry.
- Every record whose text contains a numeric claim has ≥1 `diagnostic_ref`.
- Every `premise_incorrect` record has `priority == 0`; every `confirmed_*` record has
  `priority >= 1`.
- At least one record is `premise_incorrect` (F15), guarding against a rubber-stamp report.
- No hardcoded per-finding verdict list is asserted here — verdicts come from `triage`.

---

## Data Models

### JSON output (`reports/control_method_rationality_audit_findings_20260725.json`)

```json
{
  "schema_version": "1.0.0",
  "audit_scope": {
    "reference_aircraft": "F-16-class subsonic",
    "altitude_band_m": [3000, 5000],
    "control_dt_s": 0.2,
    "analytic_only_findings": ["F18"]
  },
  "source_facts": { "...": "effective values read from modules/config" },
  "diagnostics": { "...": "numeric results" },
  "findings": [ { "id": "F01", "...": "FindingRecord fields" } ],
  "summary": {
    "confirmed_defect": ["F01", "..."],
    "confirmed_risk": ["..."],
    "by_design_ok": ["..."],
    "premise_incorrect": ["F15"],
    "remediation_backlog": ["F01", "F06", "F05", "..."]
  },
  "provenance": {
    "git_commit": "...",
    "git_branch": "research/control-method-rationality-audit",
    "python_version": "...",
    "platform": "...",
    "generated_at": "..."
  }
}
```

### Markdown output (`reports/..._zh.md`)

Chinese-language report, sections in order:

1. 审查范围与方法（read-only、analytic-only 边界、参考场景）
2. 结论摘要（四类判定的分布 + 优先级 backlog 表）
3. 逐项发现（F01–F18）：判定、源码证据、诊断复现、失效模式、改进建议、置信度
4. 被否证的前提（至少 F15），说明无需改动
5. 未量化项说明（F18 仅结构性分析，未做 run-based 量化）
6. 后续修复任务边界（每项修复须走 `AGENTS.md` 契约）

---

## Error Handling

- **Missing module/config**: if a referenced path cannot be read, the harness records the
  finding's `confidence = low` with an explicit `notes` marker and continues; it does not
  fabricate a fact and does not crash the whole run.
- **Import failure of a control module**: the harness falls back to source-text evidence for
  that finding and marks `notes` accordingly. It never substitutes hardcoded defaults for
  effective values silently.
- **`self_check` violation**: raises, and the CLI exits non-zero without writing partial
  reports (write happens only after `self_check` passes) so a malformed report can never be
  committed.
- **Preservation violation**: if the harness detects it would write outside the allowlist, it
  aborts before writing.

---

## Testing Strategy

All tests live in `tests/test_control_method_audit.py`, run with
`python -m pytest tests/test_control_method_audit.py -v` (no `--run` flag; the repo does not
register one). No `hypothesis` / property-based testing — the repo dev deps do not include
`hypothesis`, so every test is concrete or `@pytest.mark.parametrize`d over pure functions.

### Unit tests — numeric diagnostics

- `filter_time_constant`: parametrized over `(alpha, dt)` including `(0.3, 0.2) -> ≈0.5601`;
  assert monotonic decrease in τ as α increases; assert `alpha=1.0` gives τ→0.
- `offset_angular_deviation`: parametrized; assert `(500, 2500) -> ≈11.31°` and that the same
  metric offset yields a strictly larger angle at shorter range (the F05 semantics-drift core).
- `cem_elite_count`: parametrized; assert `(12, 0.25) -> 3` and `(70, 0.25) -> 17`; assert the
  `max(1, ...)` floor.
- `sincos_injective_witness`: assert 30° and 330° share `cos` but differ in `sin`, proving the
  pair is injective — the executable disproof of F15.
- `dwell_seconds` / `range_travel`: assert `(3, 0.2) -> 0.6 s` and travel at a stated closing
  speed.

### Unit tests — source facts

- Constructing `LOSRateGuidance` with `config/guidance.yaml` yields the documented limits and
  `alpha_filter`; assert the audit reads them from the instance, not from a literal.
- `CEMGainOptimizer` with the 7-D gain space reports `candidates=12`, `elite_ratio=0.25`.
- `build_observation` with a minimal state pair returns a 16-length base vector whose feature
  name order matches the `AGENTS.md` base ordering (guards the schema contract while proving
  F14's "16-D base" premise).
- Config-vs-code mismatch facts: assert `config/guidance.yaml` mode-switch
  `aspect_threshold_deg == 25.0` while the code fallback is `15.0`, and that VPP `action_dim`
  is `3` in config while the generator default is `5`.

### Triage tests — parametrized over synthetic finding inputs

- A finding whose premise-disproof diagnostic fires → `premise_incorrect` with `priority == 0`.
- A finding with a wrong-relation source fact → `confirmed_defect`.
- A finding that is defensible but has a quantified failure mode → `confirmed_risk`.
- A finding with neither → `by_design_ok`.
- Assert `triage` is a pure function of `(facts, diagnostics)` — same input, same output.

### Preservation test

- Compute a SHA256 manifest of the preserved set (all of `guidance/**`, `flight_control/**`,
  `virtual_point/**`, `envs/reward.py`, `envs/observation.py`, `envs/tracking_env.py`,
  `gain_optimizer/**`, `config/**`) before and after running the audit; assert byte-identical.
- Assert `git diff --name-only` is a subset of the allowlist.

### Integration test

- Run the CLI end-to-end against the real repo; assert exit 0, both artifacts written, 18
  findings present, ids exactly `F01..F18`, `self_check` invariants hold, and that F15 is
  reported `premise_incorrect`.

---

## Deferred remediation (explicitly out of scope here)

Each `confirmed_*` finding produces a recommendation only. Implementing any of them is a
separate gated task that must:

1. Follow the `AGENTS.md` observation-schema contract (bump schema version, update
   `feature_names`, add a dim-change test) if observations change — relevant to F14.
2. Record every config mutation via `record_config_override` — relevant to F04, F07, F08,
   F12, F13, F16, F17.
3. Respect the backend contract (no silent backend forcing) — relevant to F18.
4. Add tests asserting the new control behavior before changing defaults.

The report must state this boundary so a reader cannot mistake a recommendation for an
applied change.
