# P2-B3 Phase-Observability Preflight 完成审计

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-PREFLIGHT-V1`
**最终判定：** `normalized_phase_input_contract_no_go`
**执行状态：** 已一次性完成，授权关闭；不得重跑或修改后覆盖。

## 关键结果

| 合同 | 结果 | 解释 |
|---|---:|---|
| strict JSBSim structural contract | 90/90 通过 | fresh environment、有限 action/state 与无 fallback 均通过 |
| Scenario application receipt | 90/90 通过 | raw backend own/target 位置、速度、航向和高度均与请求场景在预注册容差内一致 |
| Phase receipt replay | 90/90 通过 | persisted tracker input、内部状态和 output 可由独立 PhaseTracker 回放 |
| step-0 `pre_merge` 语义 | 0/90 | 所有 episode 在 step-0 被标为 `post_merge` |
| Tracker raw input | `range_m=0.75997--0.76003` | 与约 3,800 m 的请求场景不在同一单位 |
| Requested raw range | `3,800 m` | reset receipt 证明 scenario 已正确注入 JSBSim |

三个 opponent 的 phase matrix 都只有 `post_merge`：expert 7,495 steps、end-to-end 7,412 steps、independent PPO/VPP 7,658 steps；`pre_merge=0`、`re_entry=0`。这不是对手强度、策略性能或技能库能力的结论。

## 根因

`src/uav_vpp_guidance/envs/observation.py::build_observation` 将 policy vector 的 `range_m` 除以 `5,000`，将 `range_rate_mps` 除以 `200`。B3 使用 `base_observation(observation)` 从该**归一化 policy vector**取值，却将其直接输入 `PhaseTracker(merge_range_m=1000, reentry_rate_mps=-25)` 的 SI 阈值。因此约 `3,800/5,000=0.760` 的归一化 range 被错误地判为小于 1,000 m，导致 tracker 在 reset 即进入 post-merge。

场景应用、JSBSim 后端、checkpoint、控制链与 phase tracker 本身均不是本次失败的已证实原因；证据指向 phase-input unit contract 的实现错误。B3 的 raw receipts 正好把这一问题与“训练步数不足”或“对手轨迹差异”区分开。

## 证据与边界

- `phase_observability_gate.json` SHA-256：`e61e223288c85a721e43b6a1a28e31463d5e30a1e1c20142826a3987ef9e10bf`。
- `run_manifest.json` SHA-256：`0fafadb7dce642fd6c1a2beb2561c5f3d23a0ebe1f610b4ec87da8c7ce48ebb0`。
- B3 raw output：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_b3_phase_observability`。

下一个且唯一允许的研发动作是新 Source ID 的 P2-B4：使用 `observation.relative_state` 中原始 SI `range_m/range_rate_mps` 驱动 PhaseTracker，同时把 normalized policy vector 明确标注为诊断副本。B4 必须使用不重合 manifest/seed；四共享技能、高层 PPO 和 formal held-out 继续锁定。
