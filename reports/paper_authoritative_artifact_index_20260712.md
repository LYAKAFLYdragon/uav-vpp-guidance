# Paper Authoritative Artifact Index

## Status

This is the active single source of truth for the corrected manuscript and
supplement. A numerical or protocol statement is submission-ready only when it
matches the source ID listed here.

## Citation Rules

1. Headline outcome numbers may cite only CAN-20260705.
2. Crossing preservation, training-seed, and mechanism claims retain their own
   source ID and must not be pooled with CAN-20260705.
3. A source marked EXCLUDED must not appear as evidence until its exact raw
   artifact location and provenance are restored.
4. Do not cite post_merge_recovery_route_b_evidence_package.md as an
   authoritative evidence index. It names a superseded 2026-07-03 SHA.

## CAN-20260705: Canonical Headline Evidence

| Field | Value |
|---|---|
| Claim status | Only source for main-text headline outcomes |
| Frozen branch | codex/canonical-post-merge-recovery-manifest60-fwdneg |
| Frozen commit | 9a9f9bf6d68560afa81560f5260b78f318dcd9b1 |
| Methods | oracle_task_gate, commander_post_merge_recovery3 |
| Tasks | head_on, crossing_feasible |
| Opponents | expert, end_to_end |
| Backend | JSBSim |
| Crossing attack-zone limit | 60 deg |
| Contract | Both runs: status=completed, paper_safe=true, valid artifact contract |

| Split | Authoritative directory | Required files |
|---|---|---|
| Expert | E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_expert_20260705\ | run_manifest.json, resolved_config.yaml, summary.csv, aggregate/method_task_summary.json, raw episodes |
| End-to-end | E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_end_to_end_20260705\ | run_manifest.json, resolved_config.yaml, summary.csv, aggregate/method_task_summary.json, raw episodes |

### Permitted Headline Numbers

Win rate is wins divided by wins plus losses. Draws are retained in total
episode accounting but excluded from this denominator.

| Opponent | Task | Oracle W/L/D | Oracle rate | PPO W/L/D | PPO rate | Permitted wording |
|---|---|---:|---:|---:|---:|---|
| Expert | Head-on | 34/26/0 | 0.5667 | 39/21/0 | 0.6500 | PPO is no longer weaker; no superiority claim |
| Expert | Crossing-feasible | 3/1/0 | 0.7500 | 3/1/0 | 0.7500 | Four-scenario scope check only |
| End-to-end | Head-on | 55/4/1 | 0.9322 | 54/6/0 | 0.9000 | PPO remains slightly below Oracle |
| End-to-end | Crossing-feasible | 3/1/0 | 0.7500 | 3/1/0 | 0.7500 | Four-scenario scope check only |

### Frozen Component Identity

| Component | Exact identity | Fact boundary |
|---|---|---|
| High-level checkpoint | outputs/diagnostics/hierarchical_commander_mvp_2mode_task_type_mode_dominance_gated_alignment_shaped_v2_oracle_imitation_warmstart_balanced_tail_checkpoint_retrospective_expert_10seed_20260701_checkpoint_snapshots/best.pt | 19-D input, 2 learned actions, 128/128 hidden layers, checkpoint step 8,192 |
| Head-on checkpoint | outputs/experiments/prediction_vpp_ppo_jsbsim_compare_long_lh1p0_reset075_no_mode_switch_longscale00_tactical_basis_headon_mvp_combat_finetune_mixed_opponent_aware_32k_task_type_headon_weighted/checkpoints/best.pt | 19-D input, 3-D VPP action, selected checkpoint at 8,192 steps |
| Crossing checkpoint | outputs/experiments/prediction_vpp_ppo_crossing_v3/checkpoints/last.pt | 18-D input, 3-D VPP action, 65,536 steps |
| Evaluation registry | resolved_config.yaml -> commander.modes | Three evaluation modes, but mode 2 is a deterministic post-merge recovery profile over the head-on checkpoint, not a third learned low-level checkpoint |
| High-level cadence | resolved_config.yaml -> env.high_level_dt=0.2; commander.macro_action_repeat_steps=12 | 5 Hz high-level decision cadence |
| Episode horizon | resolved_config.yaml -> env.max_high_level_steps=512 | At most 102.4 s at 0.2 s per high-level step |

## Supporting Evidence: Separate Claim Strata

| Source ID | Scope and permitted claim | Exact source |
|---|---|---|
| RQ1-20260711 | 24-scenario crossing preservation against fixed head-on VPP; practical -0.05 screen, not superiority | E:\uav-vpp-guidance-clean-rq1-crossing\reports\rq1_crossing_generalization_result_20260711.md and rq1_crossing_generalization_v1 outputs |
| RQ1-SEEDS-20260711 | Three clean-import PPO seeds meet the six-cell practical preservation screen; descriptive seed robustness only | E:\uav-vpp-guidance-clean-rq1-crossing\reports\rq1_ppo_seed_robustness_result_20260711.md and six rq1_seed_robustness_v1 outputs |
| G4-20260711 | End-to-end three-method fixed-reference counterexample retained in Supplementary Table S3; context only, not a headline or superiority claim | reports/g4_end_to_end_three_method_ablation_20260711.md; input run `flight_envelope_manifest60_end_to_end_clean_20260711` at clean commit `2465b6520cb61c95e9ef260267dbb9e48570d2da` |
| RQ2-20260712 | Second finite crossing envelope; preservation replication only | E:\uav-vpp-guidance-clean-rq2-crossing\reports\rq2_crossing_independent_envelope_result_20260712.md and rq2_crossing_independent_envelope_v1 outputs |
| RQ3-20260712 | Task-aware initial crossing selection with bootstrap and lock disabled; not task-label-free geometry recognition | E:\uav-vpp-guidance-clean-rq3-crossing\reports\rq3_crossing_locked_unlocked_mechanism_result_20260712.md |
| RQ4-20260712 | State-conditioned post-merge head-on/crossing re-selection with deterministic recovery constraints; mechanism evidence only | E:\uav-vpp-guidance-clean-rq4-routing\reports\rq4_postmerge_dynamic_routing_result_20260712.md |

## Excluded or Stale Sources

| Source | Status | Reason |
|---|---|---|
| reports/post_merge_recovery_route_b_evidence_package.md | Historical/supporting only | The document now explicitly defers all headline claims to CAN-20260705; its 2026-07-03 paths remain diagnostic history only |
| outputs/diagnostics/post_merge_recovery_route_b_artifacts_20260703/ | Supporting/historical only | May illustrate diagnostics, but cannot supply headline numbers |
| Former G1 / Supplementary Tables S3-S5 short-name artifacts | Removed from submission-facing material | The files remain excluded unless their exact raw source path, run manifest, and hash are restored in a later revision |
| Former inference-latency paragraph | Removed from manuscript | No source-data benchmark artifact is present in the authoritative bundle |

## Completed 2026-07-12 Sync

The corrected manuscript cites this index in its reproducibility section and
uses CAN-20260705 for every headline numerical statement. RQ1 evidence is
explicitly separated from CAN-20260705; RQ2--RQ4 remain absent from the
submission-facing result narrative unless separately introduced with their own
source IDs.
