# F03/F11 dynamics experiment — offline baseline

**Status:** evidence-only; no runtime or config change is authorized. This experiment measures the frozen command-chain approximation and candidate command transformations without invoking a plant.

## Frozen topology

`config/guidance.yaml` enables `mode_switch`, so the PN path can be selected. PN applies a LOS-rate first-order filter (`alpha=0.3`, `dt=0.2 s`); `CloseRangeTrackingEnv` then applies its three-channel command filter (`alpha=0.3`). On the JSBSim backend, the unspecified low-level controller selects `LowLevelController`, which applies one more first-order command filter (`alpha=0.3`) before `JSBSimActuatorInterface` maps and saturates commands. The interface has mapping and amplitude limits but no explicit actuator slew-rate state.

For the frozen LOS default, LOS internal filtering is disabled and the command post-processor is disabled. Lift compensation is also disabled. These disabled paths are recorded rather than treated as active delay stages.

## Measurements and candidates

The offline stage applies deterministic step and 0.1–2.0 Hz sinusoidal commands to every enabled stage. It records per-stage final response, aggregate amplitude/phase/apparent delay, peak command, saturation count, and peak output slew. It evaluates exactly one change per candidate: PN filter alpha 0.5, removal of the low-level filter, an offline 2 units/s actuator slew limiter, and 45-degree load/bank compensation. F04 limits remain `nz [-2,7]`, roll rate `[-1.5,1.5] rad/s`, and throttle `[0.4,0.9]` in all traces.

The model is not a closed-loop or actuator-validity claim: it excludes JSBSim plant/actuator response, sensor delay, communication delay, realistic engagement commands, tracking, and stability. The strict-JSBSim scenario matrix and its required checkpoints are unavailable in the frozen baseline, so no candidate can be selected from this evidence. The resulting gate is `needs_more_evidence`, preserves every runtime default, and has an empty protected-path allowlist.

## Required next evidence

Measure commanded-to-actual JSBSim actuator traces and all configured communication/sensor delays, then run paired frozen-matrix baseline/candidate evaluations with the same seeds, horizon, artifacts, safety limits, and backend provenance. Reject any candidate with safety, command-rate, stability, or tracking regression. Only an explicit reviewed approval may add a protected-path allowlist.
