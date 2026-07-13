# Third PPO/VPP Opponent Capability Audit

**Lane:** `noncanonical_thesis_extension` / `thesis_five_state_shared_intent_v1`
**Date:** 2026-07-13
**Decision:** `BLOCKED_NO_ELIGIBLE_INDEPENDENT_PPO_VPP_OPPONENT`

## Acceptance contract

The third opponent must be a frozen, independently auditable PPO policy that emits a continuous 3-D VPP action through the same JSBSim guidance/PID interface. It must have a recoverable training/config provenance, fixed checkpoint hash, known decision cadence, and a capability card evaluated separately from the ego policy.

## Candidate audit

| Candidate | Checkpoint facts | Decision |
|---|---|---|
| `exploratory_ppo_commander_self_play_test/checkpoints/best.pt` | SHA-256 `47690eb0fc3b5e859abe6868d236470cfd17f2afdfebbf730cbdaefabc24f53c`; `CommanderPPOAgent`; `obs_dim=19`; `action_dim=2`; `total_timesteps=64`. | **Rejected.** This is a historical exploratory high-level two-mode commander, not a continuous 3-D VPP opponent. Its 64-step smoke-scale training and missing independent capability evaluation also fail provenance and strength requirements. |
| Canonical head-on specialist | `obs_dim=19`, `action_dim=3`; SHA locked in the v1 asset manifest. | **Rejected as opponent.** It is an ego baseline specialist reused from the canonical family, not an independent adversary. |
| Canonical crossing specialist | `obs_dim=18`, `action_dim=3`; SHA locked in the v1 asset manifest. | **Rejected as opponent.** It is an ego baseline specialist and its 18-D legacy contract cannot establish an independent opponent condition. |
| Existing end-to-end neural opponent | Existing fixed neural opponent used by canonical evidence. | **Not a third PPO/VPP opponent.** It remains the second required opponent but does not satisfy the new pool's independent PPO/VPP requirement. |

## Consequence

The P0 asset and capacity preflight can pass, but `training_ready=false` until a new independent PPO/VPP opponent checkpoint and capability card are frozen. The prescribed next action is **not** to begin five-state training with two opponents or to relabel a canonical specialist as an opponent. Instead, create and audit an independent PPO/VPP opponent in a separately named preparation lane, then update the asset manifest only after its checkpoint, config, cadence, action shape, fixed reference-set behavior, terminal mix, and SHA-256 are available.
