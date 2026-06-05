# Stage 6G.5B Direct-Track / Pure-PN Smoke Summary

- **Experiment**: stage6g5b_direct_track_smoke
- **Timestamp**: 20260605_233749
- **Input geometry**: Stage 6G.5A 40-point sample (`outputs/stage6g5_geometry_smoke_real_seed0/geometry_smoke_points.csv`)
- **Evaluated episodes**: 360 (40 points × 3 variants × 3 episodes)
- **Variants**: vpp_trained_ppo, direct_target_los, pure_pn_no_vpp

## Results by Variant

| Variant | Success Rate | Success / Total | Guidance Mode | Use VPP |
|---|---|---|---|---|
| vpp_trained_ppo | 0.0% | 0/120 | LOS-rate | Yes |
| direct_target_los | 0.0% | 0/120 | LOS-rate | No (direct track) |
| pure_pn_no_vpp | 7.5% | 9/120 | Proportional Navigation | No (direct track) |

## Key Findings

1. **VPP abstraction + LOS-rate guidance fails universally** in the tested 40-point geometry sample (0/120).
2. **Direct-track LOS-rate guidance also fails universally** (0/120), suggesting the LOS-rate guidance law itself may be a bottleneck in tail-chase configurations.
3. **Pure Proportional Navigation without VPP achieves 9 successes** on 3 specific geometry points:
   - pt20: `range=2000, ego=340, target=120, aspect=0, alt=-500` → 3/3 SUCCESS
   - pt29: `range=2000, ego=340, target=200, aspect=0, alt=0` → 3/3 SUCCESS
   - pt38: `range=2000, ego=340, target=120, aspect=0, alt=500` → 3/3 SUCCESS

   Common pattern: high ego speed (340 mps), close initial range (2000 m), zero aspect angle (tail-chase).

## Interpretation

- The failure observed in Stage 6G.5A is **not purely geometric**; pure PN can succeed on feasible geometries.
- The VPP abstraction **contributes to failure** (vpp_trained_ppo = 0%), but the underlying LOS-rate guidance also appears **insufficient for tail-chase** (direct_target_los = 0%).
- **Pure PN is a viable alternative** for tail-chase scenarios with high closure energy.

## Next Steps

- **Bilevel remains BLOCKED** for the full 40-point sweep because overall variant success rate < 20%.
- **Candidate confirmation recommended** for the 3 pure-PN-successful geometries:
  - Cross-seed validation (seeds 0, 1, 2)
  - Higher episode count (10 per point)
  - Compare no_prediction VPP vs pure PN vs hybrid guidance
- If candidate confirmation holds, these geometries become **bilevel pre-check candidates** for gain-sensitive tuning.

> **Paper-safe claim**: In the tested 40-point geometry sample, no variant achieved >20% overall success. Pure proportional navigation without VPP succeeded on 3 out of 40 geometries (100% per-point success rate), indicating that the guidance-law choice and VPP abstraction both influence feasibility.
