# Final Weaving Target Pilot — VPP vs No-VPP vs E2E (JSBSim)

## Experiment

Re-evaluated the best checkpoints from the 10-seed JSBSim training matrix
(`outputs/jsbsim_10seed_matrix`) on a **zero-shot maneuvering target**:
- `target_mode: sinusoidal_weaving`
- `weaving_amplitude_g: 5.0`
- `weaving_frequency_rad_s: 1.0`
- 1 checkpoint per method × 4 scenarios × 3 eval seeds × 5 episodes = 60 episodes / method

Methods compared:
- `vpp_s0`   (hierarchical VPP + LOS-rate guidance)
- `no_vpp_s0` (hierarchical zero-offset + LOS-rate guidance)
- `e2e_s0`   (end-to-end direct control)

## Results

| Method | favorable | neutral | challenging | disadvantage | overall |
|---|---:|---:|---:|---:|---:|
| VPP     | 100 % | 100 % | 100 % | 0 % (OOB) | **75.0 %** |
| No-VPP  | 100 % | 100 % | 100 % | 0 % (OOB) | **75.0 %** |
| E2E     | 100 % | 100 % | 100 % | 0 % (crash) | **75.0 %** |

- VPP and No-VPP are **numerically identical** even on the stronger weaving target.
- The only difference is failure mode in `disadvantage`: E2E crashes, VPP/No-VPP go out-of-bounds.
- `disadvantage` remains impossible for all three methods under the current geometry/guidance law.

## Conclusion

Aggressive sinusoidal weaving (5 g) **does not differentiate VPP from No-VPP**.
The robust finding is instead:

> **The hierarchical guidance interface (VPP or No-VPP) is more stable than end-to-end PPO**, with comparable or slightly better success rates; the learned VPP offset itself is not the decisive factor.

## Paper narrative implication

The JSBSim evidence chain should be reframed from
- "VPP > No-VPP > E2E"

to
- "Hierarchical decomposition (VPP/No-VPP) > end-to-end control; VPP offset is optional/non-critical in the tested close-range tracking conditions."

This aligns with the simple-backend narrative in `docs/revised_narrative_plan.md`.

## Outputs

- Summary: `outputs/jsbsim_re_eval_weaving_pilot/summary.json`
- Per-episode logs: `outputs/jsbsim_re_eval_weaving_pilot/{vpp_s0,no_vpp_s0,e2e_s0}/episodes.csv`
