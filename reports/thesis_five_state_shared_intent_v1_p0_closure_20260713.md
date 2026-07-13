# Five-State Shared-Intent v1 P0 Closure

## Completed controls

- Clean worktree: `E:\uav-vpp-guidance-thesis-five-state-v1`, branch `research/thesis-five-state-shared-intent-v1`, created from Taxonomy30 evidence commit `56782b0ee2b73d8cd6d61f8bbbf0b9dff4967082`.
- The worktree intentionally excludes the main repository's dirty/untracked changes and keeps the canonical baseline read-only.
- Six baseline/predictor/opponent assets were restored into `E:\uav-vpp-guidance-thesis-five-state-v1-assets`, SHA-256 locked in the asset manifest, and marked read-only.
- Output policy reserves a separate root `E:\uav-vpp-guidance-thesis-five-state-v1-results`; it requires at least 120 GB free space, keeps only top-3 checkpoints, and prohibits full raw telemetry for training.
- `scripts/preflight_thesis_five_state_p0.py` validates SHA-256, declared checkpoint shape, output-root safety, retention policy, and third-opponent readiness.

## Third-opponent resolution

The initial historical self-play candidate was correctly rejected because it was a 64-step, 19-D-to-2-mode commander rather than a continuous VPP opponent. A separate preparation lane then trained and audited `independent_ppo_vpp_opponent_v1_step_32768.pt`: a no-warm-start, 16-D-to-3-D PPO/VPP policy. Its fixed-horizon checkpoint passed all four six-episode training-evaluation lanes and a 48-action JSBSim target-side deployment probe. The detailed boundary is recorded in `thesis_five_state_shared_intent_v1_third_opponent_capability_audit_20260713.md`.

## Required next action

Rerun the P0 preflight with `--require-training-ready`. If it passes, P0 is fully closed and P1 observation-contract implementation may begin. The third-opponent result is a preparation qualification only; it does not alter any canonical paper result or make a general opponent-strength claim.
