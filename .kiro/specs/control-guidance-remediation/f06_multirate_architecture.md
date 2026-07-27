# F06 proposed multi-rate architecture and delay budget — offline evidence only

**Status:** proposed and evidence-only. The frozen runtime remains the selected behavior: high-level action, VPP, guidance, clipping, command filtering, optional communication delay, and backend advance all occur once per `high_level_dt = 0.2 s`. This document introduces no runtime/configuration selection and authorizes no protected-path modification.

## Selectable schedule contract

A future F06 implementation must expose these explicit periods in seconds: `high_level_period_s` (policy/VPP action), `guidance_period_s` (guidance, clipping, and command filter), and `low_level_period_s` (low-level controller/backend application). Each must be positive and finite. The legacy selectable mode is exactly `(0.2, 0.2, 0.2)` seconds (5 Hz each).

The standard candidate studied offline is `(0.1, 0.05, 0.01)` seconds: high-level 10 Hz, guidance/filter 20 Hz, low-level 100 Hz. It is **not a default or approved runtime cadence**. Every slower period must be an integer multiple of the low-level period, and the high-level/guidance ratio must be integral in one direction. A non-integer relationship is rejected unless a separately specified asynchronous resampler defines clock domain, timestamp mapping, drift bound, and interpolation source.

## Update order, holds, filters, reset, and telemetry

All clocks share `t=0`. At a coincident timestamp, execute in this exact order: (1) high-level policy/VPP action, (2) guidance from the latest action, clip, then advance each command-channel filter once, and (3) low-level/controller/backend application. This ordering means a synchronous legacy tick preserves the frozen same-step trace.

A newly generated action is held by zero-order hold until its next high-level update. A guidance command is also held by zero-order hold across lower-level ticks. Linear interpolation is forbidden for a realtime policy action unless an explicit next sample is supplied by an approved predictive source; otherwise it is rejected rather than inferred. If explicit interpolation is approved, telemetry must identify both samples, their timestamps, interpolation fraction, and source.

Each first-order command filter advances **only** at `guidance_period_s`, with `y[k] = alpha*x[k] + (1-alpha)*y[k-1]`; it does not advance on low-level hold ticks. On episode reset, clear action/guidance holds, filter state, sequence counters, and latency accumulators before the first `t=0` tick. The first post-reset filter update is pass-through, matching `FirstOrderCommandFilter` behavior. Any supervisor-requested filter reset must be timestamped and use the same pass-through rule.

Every applied low-level command telemetry record must include `timestamp_s`, `high_level_action_timestamp_s`, `guidance_timestamp_s`, action age, guidance age, filter sequence number/reset flag, requested/effective schedule ID, command resampling mode, and backend timestamp. Artifact provenance must include requested/final backend and fallback fields. A script that changes a loaded schedule config must call `record_config_override`; evidence scripts here do not mutate YAML and record an empty override list.

## Baseline and candidate latency budget

The frozen source order is `action/VPP → guidance → clip → command filter → optional communication delay → backend step`, all at 0.2 s. The audited `alpha=0.3` command filter has a discrete-equivalent time constant

```text
tau_filter = -T_guidance / ln(1-alpha) = -0.2 / ln(0.7) = 0.560735 s.
```

This is a **filter component**, not the end-to-end latency. The budget must separately report action/observation/inference age, action hold, guidance hold, command-filter delay/phase, configured communication delay, low-level/backend scheduling age, actuator dynamics, and sensor/telemetry delay. The task-5 offline sweep measures only action scheduling/hold and command filtering; task 6 must measure the remaining plant/actuator/communication components. No sum may be called total end-to-end delay until all enabled components are measured under the actual resolved configuration.

The analysis uses sinusoidal commands at 0.1, 0.25, 0.5, 1.0, and 2.0 Hz, reports amplitude ratio, phase lag, and apparent delay for the frozen legacy schedule and the proposed offline candidate. Frequencies at or above the guarded guidance Nyquist range are excluded rather than aliased.

## Acceptance gate

A cadence change is blocked unless all deterministic scheduler tests pass, legacy mode reproduces the frozen invocation/command trace, the complete resolved-config delay/phase budget meets the predeclared latency target (provisionally no worse than 0.2 s apparent scheduler/filter delay at the declared command-band comparison points), and paired frozen-matrix baseline/candidate evidence demonstrates no stability, F04 safety, tracking, backend-provenance, or schema regression. The target is provisional until an engineering review specifies the command spectrum and actuator-inclusive margin. Strict-JSBSim evidence is required wherever the approved evaluation plan requires it; a fallback run cannot satisfy the gate.

The present gate is `needs_more_evidence`, has no protected-path allowlist, and retains legacy single-rate behavior. Subsequent protected F06 implementation is **not authorized**.
