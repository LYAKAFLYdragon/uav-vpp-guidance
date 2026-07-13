# Five-State Shared-Intent v1 P0 Closure

## Completed controls

- Clean worktree: `E:\uav-vpp-guidance-thesis-five-state-v1`, branch `research/thesis-five-state-shared-intent-v1`, created from Taxonomy30 evidence commit `56782b0ee2b73d8cd6d61f8bbbf0b9dff4967082`.
- The worktree intentionally excludes the main repository's dirty/untracked changes and keeps the canonical baseline read-only.
- Four baseline/predictor assets were restored into `E:\uav-vpp-guidance-thesis-five-state-v1-assets`, SHA-256 locked in the asset manifest, and marked read-only.
- Output policy reserves a separate root `E:\uav-vpp-guidance-thesis-five-state-v1-results`; it requires at least 120 GB free space, keeps only top-3 checkpoints, and prohibits full raw telemetry for training.
- `scripts/preflight_thesis_five_state_p0.py` validates SHA-256, declared checkpoint shape, output-root safety, retention policy, and third-opponent readiness.

## Blocking result

The third PPO/VPP opponent audit is complete but **does not pass**. The only historical self-play PPO candidate is a 64-step, 19-D-to-2-mode commander rather than an independently trained continuous 3-D VPP opponent. P0 therefore closes with asset integrity and storage controls passing, but with `training_ready=false` by design.

## Required next decision

Do not start the four-skill or high-level training stages. First establish a separate independent PPO/VPP opponent preparation lane and complete its capability card. After a qualified asset is frozen, update the manifest, rerun the P0 preflight with `--require-training-ready`, and only then enter P1.
