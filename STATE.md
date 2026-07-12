# UAV VPP Guidance State

**Last audited:** 2026-07-12  
**Current phase:** M3 submission packaging  
**Canonical evidence ID:** `CAN-20260705`

## Milestones

| ID | Status | Current definition |
|---|---|---|
| M0 | Done | Core evaluation and artifact contracts are operational. |
| M1 | Done | Canonical comparison remains `oracle_task_gate` versus `commander_post_merge_recovery3`. |
| M2 | Done | Headline claims are anchored to the two paper-safe CAN-20260705 formal runs. |
| M3 | In progress | Submission package is manifest-driven; final author portal upload remains human work. |

## Single Source Of Truth

- Canonical commit: `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`.
- Detached canonical code worktree: `E:\uav-vpp-guidance-can20260705`.
- Immutable external artifact bundle: `E:\uav-vpp-guidance-artifacts\CAN-20260705`.
- Evidence ledger: `reports/paper_authoritative_artifact_index_20260712.md`.
- Current manuscript source: `drones/dfar_ast_final_v5.tex`.
- Current supplementary sources: `drones/supplementary_material_v4.tex` and `drones/supplement_index.tex`.
- Current AST package manifest: `drones/submission_manifest.yaml`.
- Current generated upload archive: `drones/submission_dist/ast_vpp_can20260705.zip`.

## Canonical Results

| Opponent | Task | Oracle | PPO | Permitted interpretation |
|---|---|---:|---:|---|
| expert | head-on | 0.5667 | 0.6500 | PPO is no longer weaker; no superiority claim. |
| expert | crossing-feasible | 0.7500 | 0.7500 | Four-scenario preservation check only. |
| end-to-end | head-on | 0.9322 | 0.9000 | PPO remains slightly below Oracle. |
| end-to-end | crossing-feasible | 0.7500 | 0.7500 | Four-scenario preservation check only. |

## Current Controls

1. Do not reopen or retune the canonical commander family.
2. Do not promote non-canonical, historical, or dirty-worktree outputs into the headline table.
3. Use `Supplementary Table/Figure S3` for RQ1 crossing evidence; no other legacy supplement numbering is current.
4. Verify CAN-20260705 before manuscript changes:

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\can20260705_artifact_bundle.py verify `
  --bundle-root E:\uav-vpp-guidance-artifacts\CAN-20260705 `
  --canonical-worktree E:\uav-vpp-guidance-can20260705
```

## P0 Closure (2026-07-12)

| Item | Status | Evidence |
|---|---|---|
| P0-1 exact code and artifact freeze | Done | CAN-20260705 bundle verification and exact-worktree preflight pass. |
| P0-2 one-command submission package | Done | `scripts/build_submission_bundle.py build` generated and verified the AST zip. |
| P0-3 current-document synchronization | Done | Current docs name CAN-20260705, v5 sources, and S3; previous versions live under `reports/historical/`. |
| P0-4 worktree lane classification | Done | `reports/worktree_inventory_classification_20260712.md` and `docs/worktree_lane_policy.md`. |

## Watch List

- One optional independent head-on envelope can strengthen external validity; it must be pre-registered, evaluation-only, and run from a new clean worktree.
- `v14b` remains a non-canonical low-level-routing research lane until it passes a clean, two-opponent replication.
- Current root-worktree changes are not reproducibility evidence. Use the CAN bundle for paper-facing verification.
