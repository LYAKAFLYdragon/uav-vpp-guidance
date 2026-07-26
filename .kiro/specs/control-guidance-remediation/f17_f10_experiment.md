# F17/F10 Offline Evidence Specification

## Scope
This evidence stage is analysis-only. It preserves frozen runtime `HybridGuidance` dwell/hysteresis and low-level-controller defaults. It changes neither protected code nor `config/**`.

## F17 baseline and candidates
The frozen range switch starts in PN. With `range_threshold_m=3000`, `hysteresis_m=500`, and `min_dwell_steps=3`, it requests LOS only below 2750 m, returns to PN only above 3250 m, and requires three consecutive pending requests before changing law. The evidence measures switch delay, travel during delay at 100 and 300 m/s closing speeds, chatter count, and missed opportunities for steady-closing, boundary-jitter, and post-entry-opening traces.

Candidates are explicitly named only: one-step dwell and 750 m hysteresis. They are not runtime-selected. Any adoption requires frozen-matrix task/safety evidence and a no-chatter conclusion.

## F10 baseline and candidate
The frozen low-level protection sequence is: guidance command, dynamic stall upper-limit clipping, bank recovery, altitude-hold correction, then final clipping. The offline sweep records qbar, altitude and AoA gain scales, saturation ceiling, bank/altitude deltas, flags, and roll command attribution for nominal, low/high-qbar, high-altitude, high-AoA, and combined bank/altitude samples.

The named candidate is `qbar_altitude_aoa_gain_scheduled_controller`. It is an analytic comparison only. F04 normal-load, roll-rate, and throttle limits remain unchanged.

## Gate
Both F17 and F10 remain `needs_more_evidence`: no frozen checkpoint matrix or paired strict-JSBSim run is available. Runtime changes require identical baseline/candidate seeds, artifacts, horizon, backend provenance, telemetry, safety/stability/task outcomes, and a specifically approved protected-path allowlist.
