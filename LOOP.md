# Loop Engineering: Publication-Safe Operation

**Updated:** 2026-07-12  
**Phase:** submission packaging and evidence preservation, not canonical policy tuning.

## Operating Rule

The active paper route is `CAN-20260705`, not the current dirty development root.
Every paper-facing action starts from:

1. `E:\uav-vpp-guidance-can20260705` at `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`.
2. `E:\uav-vpp-guidance-artifacts\CAN-20260705` for formal raw data and external assets.
3. `reports/paper_authoritative_artifact_index_20260712.md` for source hierarchy.

## Allowed Loops

| Loop | Allowed action | Required check | Prohibited action |
|---|---|---|---|
| Evidence verification | Verify hashes, artifact contracts, and paper-source claims | CAN bundle verifier and exact-worktree preflight | Re-running canonical training or changing checkpoints |
| Submission packaging | Build `drones/submission_dist/ast_vpp_can20260705.zip` | Submission bundle verifier | Uploading any unlisted draft or figure |
| Manuscript maintenance | Edit only the v5 source and S3 supplement after source audit | `paper_single_source_of_truth_reaudit_20260712.md` | Reintroducing excluded historical ablations or obsolete supplement numbering |
| Non-canonical research | Work in a separate clean worktree/output root | Explicit lane label and stop rule | Modifying canonical configs, registry, or formal artifacts |

## Required Commands

```powershell
# Verify the frozen code, assets, raw episodes, hashes, and checkpoint shapes.
D:\Anaconda3\envs\jsbenv\python.exe scripts\can20260705_artifact_bundle.py verify `
  --bundle-root E:\uav-vpp-guidance-artifacts\CAN-20260705 `
  --canonical-worktree E:\uav-vpp-guidance-can20260705

# Verify exact canonical dependency closure.
D:\Anaconda3\envs\jsbenv\python.exe E:\uav-vpp-guidance-can20260705\scripts\preflight_formal_reproducibility.py `
  --repo-root E:\uav-vpp-guidance-can20260705 `
  --config config/experiment/jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot.yaml

# Build the only current AST upload package.
D:\Anaconda3\envs\jsbenv\python.exe scripts\build_submission_bundle.py build --overwrite
```

## Claim Guardrails

- `expert/head_on`: PPO is no longer weaker than Oracle; no superiority claim.
- `end_to_end/head_on`: PPO remains slightly below Oracle.
- `crossing_feasible`: canonical N=4 scope check; RQ1 evidence is in Supplementary Table/Figure S3 and remains finite-envelope preservation evidence.
- The system is task-aware and has deterministic recovery constraints; it is not task-label-free geometry recognition or independently learned recovery routing.

## Historical Material

Prior current-entry documents were copied to `reports/historical/*_pre_CAN20260705_sync_20260712.md`. They are retained for traceability but must not be used as current submission instructions.
