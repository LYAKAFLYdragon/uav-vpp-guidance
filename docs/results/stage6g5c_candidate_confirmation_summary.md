# Stage 6G.5C Pure-PN Candidate Confirmation Summary

- **Experiment**: stage6g5c_candidate_confirmation
- **Timestamp**: 20260606_000608
- **Candidate points**: pt20, pt29, pt38 (from Stage 6G.5B pure-PN successes)
- **Common geometry**: initial_range=2000 m, ego_speed=340 m/s, aspect=0°, tail-chase
- **Evaluated episodes**: 450 (3 points × 5 variants × 3 seeds × 10 episodes)

## Variant Results

| Variant | Success Rate | Success / Total | Crash Rate | Mean Min Range (m) | Mean Final Range (m) | Capture Time (s) |
|---|---|---|---|---|---|---|
| vpp_trained_ppo_los | 0.0% | 0/90 | 100% | 1710 | 10727 | — |
| direct_target_los | 0.0% | 0/90 | 100% | 1684 | 10103 | — |
| **pure_pn_no_vpp** | **100.0%** | **90/90** | **0%** | **11.0** | **87.6** | **11.5** |
| hybrid_no_vpp | 0.0% | 0/90 | 100% | 1640 | 10114 | — |
| vpp_policy_pn_guidance | 0.0% | 0/90 | 100% | 1334 | 10721 | — |

## Cross-Seed Stability

| Point | Variant | Seed 0 | Seed 1 | Seed 2 | Stable? |
|---|---|---|---|---|---|
| pt20 | pure_pn_no_vpp | 100% | 100% | 100% | **Yes** |
| pt29 | pure_pn_no_vpp | 100% | 100% | 100% | **Yes** |
| pt38 | pure_pn_no_vpp | 100% | 100% | 100% | **Yes** |

All other variants show 0% success across all seeds.

## Key Findings

1. **pure_pn_no_vpp is 100% cross-seed stable** on all 3 candidate geometries.
2. **Direct-track LOS-rate fails completely** (0/90), confirming LOS-rate guidance is a primary bottleneck in tail-chase.
3. **VPP + PN also fails completely** (0/90), confirming the VPP offset/policy anchor is harmful in tail-chase even when paired with a successful guidance law.
4. **Hybrid guidance fails completely** (0/90), suggesting the switching/blending logic does not rescue tail-chase.
5. **Mean min range for pure PN is ~11 m** (excellent capture), while all failing variants stall at ~1300–1700 m before crashing.

## Interpretation

- The 3 candidate geometries are **feasible** under pure proportional navigation.
- The failure is **not geometric** — the same geometries that crash under VPP+LOS succeed perfectly under pure PN.
- **Both LOS-rate guidance and VPP abstraction contribute to failure**:
  - LOS-rate alone (direct_target_los) crashes
  - VPP+PN (vpp_policy_pn_guidance) crashes
  - Only pure PN without VPP and without LOS-rate succeeds

## Next Steps

- **Bilevel remains BLOCKED** for VPP+LOS because no VPP variant shows success.
- **Recommended**: Stage 6G.5D — PN/VPP mechanism redesign or mode-switch probe.
  - Options: tail-chase-specific VPP constraints, VPP offset clipping, automatic mode-switch to pure PN when aspect ≈ 0° and range < threshold.
- **Candidate geometries** (pt20, pt29, pt38) can serve as a **regression test suite** for any guidance redesign.

> **Paper-safe claim**: In the tested high-energy tail-chase subset (ego 340 m/s, range 2000 m, aspect 0°), pure proportional navigation without VPP achieved 100% cross-seed success, while all VPP-based and LOS-rate variants achieved 0% success. This indicates that both the VPP abstraction and the LOS-rate guidance law contribute to observed tail-chase failure in the tested scenarios.
