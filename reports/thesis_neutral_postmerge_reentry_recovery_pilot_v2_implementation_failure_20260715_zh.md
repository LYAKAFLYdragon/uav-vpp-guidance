# Neutral Post-Merge Reentry-Recovery Pilot V2 实现失败归档

**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2`  
**最终状态：** `implementation_failure_not_experimental_evidence`  
**授权配置提交：** `8ff8aafe83ade58bbce128b413952ac475444343`  
**执行实现祖先：** `6f35453f091a88d2d51ec4a9d9d3cbbd979e50d0`

## 执行结果

一次性授权 preflight 通过：V2 design、P3 encoder、三对手、两个 frozen
specialist、66-D -> 3-D VPP contract、manifest pairing、磁盘门槛和 fresh
output root 均通过。随后 strict JSBSim 运行开始，完成全部 `dev12` frozen
baseline：3 个 opponent x 2 个方法 x 12 个场景，共 72 条持久化 baseline
record 与 72 份逐回合 telemetry。

在进入第一条 train scenario 时，V2 全局 episode serializer 对 train scenario
调用 `pair_key(scenario)`，并触发：

```text
NeutralPostMergeV2PairingError(
  'scenario lacks geometry_cell_id or integer scenario_seed'
)
```

`pair_key` 设计上适用于需要 paired delta 的 dev/heldout manifest；但冻结的
连续 train sampler 仅产生 `scenario_signature + scenario_seed`。因此这不是
策略、物理后端、reward、safety 或 low-level skill 的结果，而是 train identity
contract 未被纳入 V2 adapter round-trip test 的实现错误。

## 精确归因

- [V2 ledger](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/training/thesis_neutral_postmerge_reentry_recovery_pilot_v2.py:50) 对所有 episode 强制 `pair_key`。
- [V1 train sampler](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/training/thesis_neutral_postmerge_reentry_recovery_pilot.py:441) 没有输出 `geometry_cell_id` 或序列化 `pair_key`。
- [训练循环](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/src/uav_vpp_guidance/training/thesis_neutral_postmerge_reentry_recovery_pilot.py:716) 将该 train scenario 传入同一个 V2 episode wrapper；ledger 序列化在 transition 返回前失败。

因此，V2 现有的 manifest-only pairing test 不能覆盖 train/evaluation identity
差异。执行前的设计 preflight 也没有运行一条 train scenario 的 end-to-end
serialization probe。这是唯一的实现失败归因。

## 已保存证据与边界

- output root：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_pilot_v2`。
- `runner_failure_manifest.json` SHA-256：`d54da1d3769421b623470730b4357273a6f0eeed39567bf63deaff16cc2c7b7e`。
- root 共 82 个文件、约 94.43 MB；含 authorization preflight、resolved config、6 份 dev baseline record、72 份 baseline telemetry、baseline summary 与 failure manifest。
- 不含 checkpoint、`train/training_summary.json`、candidate record、candidate telemetry、heldout record、heldout decision 或 `pilot_summary.json`。
- 第一条 candidate train episode 可能已在内存中运行到 ledger，但没有持久化 transition、没有调用 PPO update、没有生成 checkpoint，故不能产生任何 candidate performance、safety 或 skill-capability 结论。
- 本 Source ID 的所有输出均为 `paper_safe: false`，不能写入论文、主结果、对手能力比较或共享技能库结论。

## 基线覆盖观察（非比较性结果）

`dev/baseline_summary.json` 是在实现错误前已实际写入的 frozen baseline
telemetry 汇总。它不比较 candidate，不能用于宣称 reentry skill 的优劣；但它是
V2 `dev12` 在固定 run-in 下是否具备下一轮 pilot 所需 phase support 的直接检查：

| Opponent | Head-on handoff / 12 | Crossing handoff / 12 | 预注册最小 2 条 | claim-ready 8 条 |
|---|---:|---:|---:|---:|
| expert | 1 | 1 | 未达到 | 未达到 |
| end-to-end | 0 | 0 | 未达到 | 未达到 |
| independent PPO/VPP | 1 | 1 | 未达到 | 未达到 |

候选技能仅在 frozen run-in handoff 后才接管，因此它不能补回上述 handoff 缺失。
这说明 V2 的 `dev12` 不是三对手下可完成预注册选择门的运行时 phase-feasible
包线。该观察仅约束未来设计；它不说明对手“更强”、reentry 能力不存在，或任何
方法不安全。

## 决策与唯一允许的后续

本 Source ID 已消耗并冻结。不得修改后重跑 V2、覆盖其 output root、添加训练
步数、改变 reward/场景/encoder/VPP/guidance/PID，或将其 baseline 文件挑选为
论文证据。

若要继续该单一机制，必须另建未授权的 `V3` 设计线，且先满足全部条件：

1. 仅 dev/heldout 使用 `pair_key`；train scenario 使用不参与 paired delta 的显式 `train_episode_key`。
2. 为 train sampler、V2-style ledger 和 result serializer 新增真实 round-trip test；该 test 必须覆盖第一条 train episode，而不只覆盖 manifest。
3. 在授权前以 frozen run-in 对新的 dev 包线逐 opponent 实测 handoff；每个对手至少达到预注册的 2 条最小覆盖，且最好在授权前验证能达到 8 条 claim-ready coverage。
4. 使用新的 Source ID、互不重叠的 manifests、fresh output root、独立预注册、独立复核和新的单次授权。

在这些条件完成前，四共享技能、combat finetune、高层 PPO、P5/P6/P7 和任何 V3
训练均继续锁定。
