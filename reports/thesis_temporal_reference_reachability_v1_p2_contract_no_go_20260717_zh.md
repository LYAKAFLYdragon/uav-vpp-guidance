# Temporal Reference Reachability V1 P2 输入契约 No-Go

**Source ID：** `THESIS-DISADVANTAGE-TEMPORAL-REFERENCE-REACHABILITY-V1`  
**阶段：** `P2_offline_envelope_screen`  
**最终判定：** `p2_input_contract_incompatible`  
**P3 状态：** `NOT STARTED / NOT PERMITTED`

## 结论

P2 不能使用现存的 12 条 replay-qualified continuous handoff 作为 P1 时域参考编译器的离线包线输入。P1 固定要求 `tactical_basis_v1`，即每步必须有当前三轴 tactical-basis world vectors；而 12/12 条历史 handoff 的 active/configured action semantics 均为 `cartesian_offset`。

因此，不存在合法的 basis 矩阵可供 P1 编译器计算请求、投影残差或实际参考位移。将旧 `cartesian_offset` 重新解释成 tactical-basis 坐标会改变 action semantics，属于未预注册的新接口，而不是 P2 的只读分析。

这不是“候选机动无效”、训练不足、奖励设计问题、对手强度问题，或 JSBSim 后端故障。它是**历史连续轨迹与新接口之间的输入契约不兼容**，故不能用于判断 physical-or-behavior reachability。

## 已执行事项

| 项目 | 结果 |
|---|---|
| P2 baseline-only replay | 已启动；首条 strict-JSBSim replay 在取得 candidate basis 前 fail-closed |
| 候选 action 写入 JSBSim | `0` 次；未执行 candidate action |
| P2 compiler matrix | `0` 行；没有合法 tactical-basis 输入 |
| 12 条 cohort semantics 审计 | `12/12` strict JSBSim，`0/12` tactical-basis，`12/12` cartesian offset |
| P3 runtime smoke | 未启动；无输出目录，按 P2 stop rule 禁止 |

## 证据与复现

- P2 config SHA-256：`e290dba563a93c9dc82c93464080f6b8fe5d4e4834b31e019f1bf00c02a4a4d3`
- P2 implementation commit：`5f354b0`
- 首次 fail-closed output：[failure.json](E:/uav-vpp-guidance-disadvantage-capability-v1-results/temporal_reference_reachability_v1/p2_offline_envelope/failure.json)  
  SHA-256：`ab2376602d948bc43b3536467614eaba82a5195bf735ef10c2ced6e07054ae0f`
- 12-record read-only audit：[p2_contract_audit.json](E:/uav-vpp-guidance-disadvantage-capability-v1-results/temporal_reference_reachability_v1/p2_offline_envelope/p2_contract_audit.json)  
  SHA-256：`82e5bcc466a6486017e2b32dfd759d4f8bc9b1916eef612b199b29e443de46ea`
- 审计命令：

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\audit_thesis_temporal_reference_reachability_v1_p2_contract.py `
  --config config\experiment\thesis_disadvantage_temporal_reference_reachability_v1_p2.yaml `
  --output E:\uav-vpp-guidance-disadvantage-capability-v1-results\temporal_reference_reachability_v1\p2_offline_envelope\p2_contract_audit.json
```

## 决策边界

- 不运行 P3，不训练新 specialist，不调时域、幅度、reward、路由或旧 VPP pulse。
- 不在当前 Source 内替换 historical cohort、补采样或将 `cartesian_offset` 投影为 tactical-basis。
- P0/P1 仍然有效：它们定义并测试了新编译器，但尚未获得同语义的连续 handoff 输入。
- 唯一可继续的路线必须使用**新的 Source ID**：从合法 reset 开始，以当前 `tactical_basis_v1` frozen head-on baseline 连续运行，先只记录并验证真实 post-merge/re-entry handoff、基底、reference action、prediction、VPP/guidance/PID telemetry；只有该新的 collector 在至少两对手下通过 phase + contract gate，才可重新设计 P2。

## 论文边界

该 no-go 是开发性输入契约负证据，`paper_safe=false`。它不能写成“时域参考轨迹在物理上不可行”，只能说明现有历史 replay cohort 无法为这一新接口提供可验证的 physical-or-behavior reachability 证据。
