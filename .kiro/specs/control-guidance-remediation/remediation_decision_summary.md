# Control-Guidance Remediation Decision Summary

| Finding(s) | Decision | Evidence / current disposition |
|---|---|---|
| F01/F02 | `needs_more_evidence` | Opt-in runtime path is allowlisted; closed-loop matrix, explicit engineering review, and strict-JSBSim evidence remain absent. |
| F03/F11 | `needs_more_evidence` | Offline filter/actuator budget is bundled; no plant/actuator or strict-JSBSim comparison. |
| F04 | `by_design_no_change` | Conservative F04 safety envelope retained. |
| F05 | `needs_more_evidence` | Fixed-metre VPP semantics retained pending paired tactical/safety/tracking and migration evidence. |
| F06 | `needs_more_evidence` | Offline schedule/delay model validates legacy trace; runtime cadence is unchanged. |
| F07 | `implemented_and_verified` | Canonical 3D contract is implemented with explicit legacy handling and portable validation. |
| F08 | `needs_more_evidence` | Fixed-budget seeded surrogate comparison only; no rollout or strict-JSBSim result. |
| F09 | `by_design_no_change` | Metric is labeled as a within-batch empirical score gap; training/rollback behavior is unchanged. |
| F10/F17 | `needs_more_evidence` | Offline arbitration and dwell/hysteresis sweeps completed; defaults retained. |
| F12/F13 | `needs_more_evidence` | Deterministic reward decomposition/counterfactual completed; no policy ablation. |
| F14 | `needs_more_evidence` | Signal availability/leakage audit completed; 16-feature base schema/order untouched. |
| F15 | `disproven_no_change` | Sine/cosine encoding premise remains disproven; no angle-magnitude addition. |
| F16 | `needs_more_evidence` | Explicit candidate state-machine evidence complete; runtime latch retained pending matrix/strict-JSBSim gate. |
| F18 | `needs_more_evidence` | Transfer preflight locks paired-run contract; required main checkpoints and paired runs are absent. |

All new analytic stages include manifests, artifact contracts, resolved configs, hashes, and verified portable bundles. No commit or push was made. Regression verification passed for 54 runnable targeted remediation tests, including preservation. F01/F02 property tests remain blocked because the isolated Python environment has no `hypothesis` package; strict-JSBSim/frozen-matrix evidence is also unavailable. The remaining blocked decisions require the frozen checkpoint artifacts, paired fixed-matrix evaluation, strict-JSBSim runs without fallback, and an environment with the recorded property-test dependency.
