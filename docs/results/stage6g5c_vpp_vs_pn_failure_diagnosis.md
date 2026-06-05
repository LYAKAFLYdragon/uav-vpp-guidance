# Stage 6G.5C VPP vs PN Failure Diagnosis

## Variant Summary

| Variant | Success Rate | Success / Total | Crash Rate | OOB Rate | Mean Min Range (m) |
|---|---|---|---|---|---|
| vpp_trained_ppo_los | 0.0% | 0/90 | 100.0% | 0.0% | 1710.1 |
| direct_target_los | 0.0% | 0/90 | 100.0% | 0.0% | 1684.2 |
| pure_pn_no_vpp | 100.0% | 90/90 | 0.0% | 0.0% | 11.0 |
| hybrid_no_vpp | 0.0% | 0/90 | 100.0% | 0.0% | 1639.9 |
| vpp_policy_pn_guidance | 0.0% | 0/90 | 100.0% | 0.0% | 1333.8 |

## Cross-Seed Stability

| Point | Variant | Seed 0 | Seed 1 | Seed 2 | Stable? |
|---|---|---|---|---|---|
| pt20 | vpp_trained_ppo_los | 0% | 0% | 0% | No |
| pt29 | vpp_trained_ppo_los | 0% | 0% | 0% | No |
| pt38 | vpp_trained_ppo_los | 0% | 0% | 0% | No |
| pt20 | direct_target_los | 0% | 0% | 0% | No |
| pt29 | direct_target_los | 0% | 0% | 0% | No |
| pt38 | direct_target_los | 0% | 0% | 0% | No |
| pt20 | pure_pn_no_vpp | 100% | 100% | 100% | Yes |
| pt29 | pure_pn_no_vpp | 100% | 100% | 100% | Yes |
| pt38 | pure_pn_no_vpp | 100% | 100% | 100% | Yes |
| pt20 | hybrid_no_vpp | 0% | 0% | 0% | No |
| pt29 | hybrid_no_vpp | 0% | 0% | 0% | No |
| pt38 | hybrid_no_vpp | 0% | 0% | 0% | No |
| pt20 | vpp_policy_pn_guidance | 0% | 0% | 0% | No |
| pt29 | vpp_policy_pn_guidance | 0% | 0% | 0% | No |
| pt38 | vpp_policy_pn_guidance | 0% | 0% | 0% | No |

> **Paper-safe note**: Results are limited to the 3 candidate geometries (pt20, pt29, pt38) tested under cross-seed evaluation. No universal claims about tail-chase feasibility are made.
