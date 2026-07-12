# UAV VPP Guidance

This repository studies a hierarchical virtual-pursuit-point (VPP) maneuver interface for close-range air combat in JSBSim. The active publication route is a **positive but bounded** PPO high-level routing result, not an open-ended tuning branch.

## Current Canonical Evidence

| Opponent | Task | Oracle | PPO | Claim boundary |
|---|---|---:|---:|---|
| expert | head-on | 0.5667 | 0.6500 | PPO is no longer weaker; Wilson intervals overlap. |
| expert | crossing-feasible | 0.7500 | 0.7500 | N=4 scope-preservation check. |
| end-to-end | head-on | 0.9322 | 0.9000 | PPO remains slightly below Oracle. |
| end-to-end | crossing-feasible | 0.7500 | 0.7500 | N=4 scope-preservation check. |

The only headline source is `CAN-20260705` at commit `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`. See `reports/paper_authoritative_artifact_index_20260712.md`.

## Start Here

- Canonical code worktree: `E:\uav-vpp-guidance-can20260705`
- Immutable formal artifacts: `E:\uav-vpp-guidance-artifacts\CAN-20260705`
- Current manuscript: `drones/dfar_ast_final_v5.tex`
- Current supplement: `drones/supplementary_material_v4.tex`
- Current submission manifest: `drones/submission_manifest.yaml`
- Project review and next-step rationale: `reports/project_system_review_and_roadmap_20260712_zh.md`

## Verification

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\can20260705_artifact_bundle.py verify `
  --bundle-root E:\uav-vpp-guidance-artifacts\CAN-20260705 `
  --canonical-worktree E:\uav-vpp-guidance-can20260705

D:\Anaconda3\envs\jsbenv\python.exe E:\uav-vpp-guidance-can20260705\scripts\preflight_formal_reproducibility.py `
  --repo-root E:\uav-vpp-guidance-can20260705 `
  --config config/experiment/jsbsim_hrl_oracle_vs_commander_post_merge_recovery_manifest60_pilot.yaml
```

## AST Submission Package

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\build_submission_bundle.py build
```

This command compiles the v5 manuscript, title page, Supplementary Table/Figure S3 material, copies only listed figures and reproducibility ledgers, verifies package checksums, and creates `drones/submission_dist/ast_vpp_can20260705.zip`.

## Lane Policy

- **Canonical:** `oracle_task_gate` versus `commander_post_merge_recovery3`; do not retune it.
- **Supporting RQ1--RQ4:** finite-envelope or mechanism evidence; do not pool it with headline results.
- **Non-canonical:** prediction-guided reward and `v14b` work stays in separate worktrees/output roots until clean replication.
- **Historical:** DQN, SAC, early PPO, old Route B and old manuscript variants are retained for traceability only.

See `docs/worktree_lane_policy.md` for retention and ownership rules.
