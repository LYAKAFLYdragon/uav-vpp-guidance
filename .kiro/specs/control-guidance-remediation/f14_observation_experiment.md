# F14 Observation Information Audit

The existing policy vector has an immutable ordered 16-feature base. Existing opt-in segments already expose gains (+2), virtual-point tracking error (+3), saturation flags (+3), prediction state (+14), opponent stage (+2), and task type (+1). Candidate signals absent from the policy vector are AoA/beta/roll/angular rates, command-tracking error, target turn rate, and causal history.

For every candidate, this audit defines source, units, update timing, normalization requirement, availability/fallback, and leakage restrictions. It proposes fixed-protocol comparisons against the current baseline but does not execute training/evaluation because frozen checkpoints and strict-JSBSim evidence are absent.

F14 remains `needs_more_evidence`. No observation schema, base ordering, config flag, provenance field, or checkpoint behavior changes. A passing future experiment must open a separate migration task with a schema-version bump, deterministic appended names, include flag, provenance/tests, README/AGENTS documentation, and checkpoint handling.