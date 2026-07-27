# F16 Mode-Switch State-Machine Evidence Specification

## Scope and authority
This is an **offline evidence model**, not a runtime selection. The current runtime latch and `config/guidance.yaml` remain unchanged until the F16 gate is approved. The candidate has one threshold authority: an explicit `mode_switch.aspect_threshold_deg` wins; when absent, the candidate resolves the canonical default to **25°**. It never silently uses the runtime code fallback of 15°.

## States and transition rules
- `vpp`: normal VPP-policy operation. Enter `direct_track_pn` only after the entry guard succeeds.
- `direct_track_pn`: direct target tracking with proportional navigation. Entry requires enabled switch, finite geometry, range at or below entry range, non-opening range rate, sufficient closing speed, and low-aspect or explicitly configured crossing geometry.
- `failsafe`: entered for non-finite geometry or a configured maximum active-time timeout; reset is required before re-entry.
- `disabled`: used when `enabled` is false.

The state-machine candidate makes hold/release conditions explicit. It holds through a configured minimum active dwell and then releases to `vpp` when the wider hysteretic hold guard clears. Release thresholds default to 30°, 3300 m, and 40 m/s from 25°, 3000 m, and 50 m/s entry defaults. This deadband prevents boundary chatter. Reset always returns to `vpp` and clears dwell state.

## Telemetry contract
Every event exposes prior state, resulting state, transition, reason, active-step count, effective mode-switch state, VPP/direct-track source, and effective guidance mode. Reasons distinguish entry, hold, dwell hold, release, reset, disabled, invalid geometry, and timeout.

## Candidate comparison and gate
The evidence compares: (1) the frozen permanent latch, which stays active until reset; (2) an immediate releaseable latch; and (3) the explicit state-machine candidate. Deterministic tests cover non-trigger, entry, hold, release, re-entry, reset, absent and explicit configuration, crossing, no-chatter dwell, timeout, and telemetry.

The current F16 gate remains `needs_more_evidence`. Runtime implementation requires paired frozen-matrix candidate runs, per-seed chatter/safety/tracking telemetry, strict-JSBSim evidence without fallback, and an approved protected-path allowlist. Existing permanent-latch behavior remains the frozen baseline until then.
