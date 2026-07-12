# Worktree Lane Policy

This policy classifies all dirty and untracked files without deleting them.
It is an operational boundary, not a claim that every file is ready to merge.

| Category | Owner | Lane | Retention policy | Merge policy |
|---|---|---|---|---|
| `canonical` | Paper/reproducibility steward | CAN-20260705 and v5 submission package | Preserve with source and provenance review | Merge only through a minimal reviewed change set after CAN verification |
| `noncanonical` | Research-lane owner | Future-geometry reward, v14b, or experimental RL | Preserve in its named lane and separate output root | Never overwrite canonical configs, checkpoint registry, or formal artifacts |
| `historical` | Archive steward | DQN/SAC/legacy stages, prior manuscripts, superseded Route B material | Retain read-only for traceability | Not eligible for current paper claims |
| `build` | Release/build steward | PDFs, TeX auxiliaries, generated bundles, caches, temporary unpacking | Regenerate from a manifest; do not treat as source evidence | Do not merge unless explicitly needed as a release deliverable |
| `review-required` | Repository steward | Unclassified files | Preserve unchanged until a human assigns a lane | No merge or paper citation before assignment |

## Canonical Protection Rules

1. The current root worktree is not evidence for CAN-20260705 merely because it contains similarly named files.
2. Canonical verification must use `E:\uav-vpp-guidance-can20260705` and `E:\uav-vpp-guidance-artifacts\CAN-20260705`.
3. A file classified as non-canonical or historical must never silently replace a canonical config, checkpoint, figure, or manuscript input.
4. Build files are retained for convenience but are not a source of truth; `drones/submission_manifest.yaml` is the only current submission input list.
