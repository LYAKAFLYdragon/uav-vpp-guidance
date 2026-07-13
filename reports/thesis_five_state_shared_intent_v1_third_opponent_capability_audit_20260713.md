# Third PPO/VPP Opponent Capability Audit

**Lane:** `noncanonical_thesis_extension` / `thesis_five_state_shared_intent_v1`
**Date:** 2026-07-13
**Decision:** `QUALIFIED_LIMITED_PREPARATION_ENVELOPE`

## Acceptance contract

The third opponent must be a frozen, independently auditable PPO policy that emits a continuous 3-D VPP action through the same JSBSim guidance/PID interface. It must have a recoverable training/config provenance, fixed checkpoint hash, known decision cadence, and a capability card evaluated separately from the ego policy.

## Candidate audit

| Candidate | Checkpoint facts | Decision |
|---|---|---|
| `exploratory_ppo_commander_self_play_test/checkpoints/best.pt` | SHA-256 `47690eb0fc3b5e859abe6868d236470cfd17f2afdfebbf730cbdaefabc24f53c`; `CommanderPPOAgent`; `obs_dim=19`; `action_dim=2`; `total_timesteps=64`. | **Rejected.** This is a historical exploratory high-level two-mode commander, not a continuous 3-D VPP opponent. Its 64-step smoke-scale training and missing independent capability evaluation also fail provenance and strength requirements. |
| Canonical head-on specialist | `obs_dim=19`, `action_dim=3`; SHA locked in the v1 asset manifest. | **Rejected as opponent.** It is an ego baseline specialist reused from the canonical family, not an independent adversary. |
| Canonical crossing specialist | `obs_dim=18`, `action_dim=3`; SHA locked in the v1 asset manifest. | **Rejected as opponent.** It is an ego baseline specialist and its 18-D legacy contract cannot establish an independent opponent condition. |
| Existing end-to-end neural opponent | Existing fixed neural opponent used by canonical evidence. | **Not a third PPO/VPP opponent.** It remains the second required opponent but does not satisfy the new pool's independent PPO/VPP requirement. |

## Qualified candidate

| Field | Frozen value |
|---|---|
| Checkpoint | `E:\uav-vpp-guidance-thesis-five-state-v1-assets\independent_ppo_vpp_opponent_v1_step_32768.pt` |
| SHA-256 | `43cd37fc86c0eb01d96a693f68d30894de7e768de01aed6dba41fa4761edfc60` |
| Source lane | `E:\uav-vpp-guidance-thesis-five-state-opponent-prep`, commit `2de69150bb1f493be8dc427f9ec43532fec29776` |
| Selection rule | Predeclared fixed final horizon, `step_32768.pt`; not the earlier same-score `best.pt`. |
| Interface | Role-reversed base geometry `16-D -> 3-D` continuous VPP action; target aircraft executes through JSBSim VPP/guidance/PID. |
| Training scope | Balanced `head_on/crossing_feasible x expert/end_to_end`, 32,768 PPO steps; no warm-start, task bit, opponent-stage bit or prediction reward. |
| Capability gate | Four final lanes each had six evaluation episodes, at least one win, crash/OOB `<= 0.25`; JSBSim target-side probe emitted 48 finite VPP actions without exception. |
| Raw audit | `E:\uav-vpp-guidance-thesis-five-state-opponent-prep-results\experiments\thesis_independent_ppo_vpp_opponent_v1\capability_audit.json` (SHA-256 `0b2ef40c3bae119fe8e3934bf0ce455c18c9276fd0039b8adf74053df2934f1d`). |

## Claim boundary

This checkpoint is qualified only as the third frozen PPO/VPP opponent for the documented preparation envelope. It is not an Elo-ranked adversary, a universal strength claim, or evidence that predicted-target VPP input is unnecessary: this opponent intentionally uses the same VPP/guidance/PID execution chain with a `current_target` anchor, whereas the five-state ego system retains its frozen predicted-target VPP interface.

## Consequence

The third-opponent condition is now met. The five-state P0 preflight must be rerun with `--require-training-ready`; only a passing result permits P1 observation-contract implementation. The rejected 64-step historical candidate remains documented as negative provenance evidence.
