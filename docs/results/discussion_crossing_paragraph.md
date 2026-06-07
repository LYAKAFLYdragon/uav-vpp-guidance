# Discussion: Sim-to-Real Gap in Crossing Scenarios

## Paper-safe paragraph (300–500 words)

After correcting a scenario-initialization bug identified in Stage 10.1, simple-backend-trained PPO policies achieve **partial zero-shot transfer** to the JSBSim F-16: head-on scenarios succeed in **100 %** of episodes (10/10 for both `no_prediction` and `gain_only`), whereas crossing scenarios fail in **100 %** of episodes (0/10 each). We stress that this is a *partial* transfer result: the policy generalizes to the high-fidelity aircraft when the required maneuver is small (head-on requires < 5° heading change), but fails when a large heading reversal is demanded.

Telemetry reveals that the failure mechanism is not controller instability or actuator saturation. In crossing episodes, the policy commands directionally correct maneuvers—peak load factors up to 4.16 g and roll rates up to 1.50 rad/s (≈ 86°/s)—yet the aircraft diverges from the target until `out_of_bounds` termination at a median final range of 12,038 m. Critically, **actuator saturation is absent**: `nz_cmd_saturation_rate` is 0 % across all crossing episodes, and roll-rate saturation occurs only occasionally (12 %) and transiently. The root cause lies in the F-16's sustained turn-rate envelope. At 5,000 m and Mach ~0.8, the F-16 delivers approximately **12–15°/s** sustained turn rate, whereas intercepting the crossing target from the initial 8 km geometry requires roughly **20–25°/s**. The simple backend used for training does not model F-16 turn-rate or energy limitations, so the policy never learned energy-management or lead-pursuit strategies for crossing geometry.

To verify that the failure is airframe-limited rather than policy-limited, we evaluated three classical baseline controllers (hold, direct PN, low-gain direct) and three guidance modes (LOS-rate, PN, hybrid) on identical crossing scenarios. **All six configurations failed**: direct PN even crashed in one crossing episode after commanding 14.5 g, far beyond structural limits. This convergence of evidence—policy, optimized gains, classical guidance, and alternate guidance laws all failing identically—supports the conclusion that crossing failure is a **geometry-dynamics mismatch** between the training distribution and the F-16's physical envelope.

We therefore treat crossing scenarios as an opportunity rather than a methodological flaw. Future work should explore (i) training directly inside JSBSim so the policy learns energy-aware maneuvers, (ii) energy-management strategies such as altitude exchange or speed modulation to extend the turn-rate envelope, and (iii) reduced crossing angles to identify the feasibility boundary. Until then, crossing results are reported in Discussion only and are **excluded from the main-results table**.

---

## Metadata

| Field | Value |
|-------|-------|
| Source analysis | `docs/stage10_3_crossing_failure_analysis.md` |
| Evidence commit | `fa9dbb2` |
| Written | 2026-06-07 |
| Word count | ~380 |
| Paper-safe claims | Section 5.1 of source analysis |
