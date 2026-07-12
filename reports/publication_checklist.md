# Publication Execution Checklist

**Current evidence contract:** `CAN-20260705`  
**Canonical comparison:** `oracle_task_gate` versus `commander_post_merge_recovery3`  
**Do not reopen canonical tuning while executing this checklist.**

## P0. Evidence and Submission Freeze

| Item | Status | Acceptance evidence |
|---|---|---|
| P0-1 exact canonical worktree | Complete | `E:\uav-vpp-guidance-can20260705` is detached at `9a9f9bf6`. |
| P0-1 immutable raw-artifact bundle | Complete | `E:\uav-vpp-guidance-artifacts\CAN-20260705`; hash/shape/run-contract verifier passes. |
| P0-2 manifest-driven AST package | Complete | `drones/submission_manifest.yaml` and `scripts/build_submission_bundle.py`. |
| P0-2 generated archive | Complete | `drones/submission_dist/ast_vpp_can20260705.zip` verifies after build. |
| P0-3 current documentation | Complete | STATE, LOOP, README, readiness checklist, and submission README use CAN-20260705, v5, and S3. |
| P0-4 worktree lanes | Complete | `docs/worktree_lane_policy.md` and `reports/worktree_inventory_classification_20260712.md`. |

## P1. Paper Quality

- [x] Main table reports total/resolved episodes, wins/losses, and Wilson 95% CIs.
- [x] Win-rate denominator and unresolved-draw semantics are explicit.
- [x] Expert/head-on is not overclaimed as superiority.
- [x] End-to-end/head-on residual and three paired counterexamples are disclosed.
- [x] Crossing N=4 caveat is retained.
- [x] Method facts are aligned: two learned PPO actions, two frozen checkpoint specialists, and a deterministic recovery profile.
- [x] Unsupported latency and historical unindexed-ablation claims are excluded.
- [ ] Optional: run one independent, pre-registered, evaluation-only head-on envelope. This is a quality upgrade, not a submission prerequisite.

## P2. Human Submission Tasks

- [ ] Check the generated v5 manuscript, supplement, and title-page PDFs visually.
- [ ] Confirm author names, affiliations, funding, competing interests, and acknowledgements.
- [ ] Add suggested and opposed reviewers in the portal.
- [ ] Upload only the files enumerated by `drones/submission_manifest.yaml`.
- [ ] Record the final zip SHA-256 and portal submission date.

## Stop Rules

- Do not substitute non-canonical v14b results for CAN-20260705.
- Do not restore excluded historical ablation or unindexed latency numbers to the manuscript.
- Cite only the current S1--S3 supplement and Figure S3 numbering.
- Do not train, retune, or add a new comparator to repair wording or statistics.

## Current References

- `reports/paper_authoritative_artifact_index_20260712.md`
- `reports/paper_single_source_of_truth_reaudit_20260712.md`
- `reports/project_system_review_and_roadmap_20260712_zh.md`
- `drones/submission_manifest.yaml`
