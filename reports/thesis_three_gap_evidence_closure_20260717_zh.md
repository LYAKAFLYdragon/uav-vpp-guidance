# 硕士论文三项证据缺口闭合矩阵

## 结论

本矩阵以冻结的 `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` 为唯一共同
评估 cohort，并以新的只读可判别性 ledger 筛选可用于论文的比较范围。三个
原始缺口已经获得可复核的、但有边界的证据：

1. **场景过少**：由 60 个冻结 JSBSim 初始条件的五态势包线补足；每个
   `Advantage`、`Head-on`、`Disadvantage`、`Neutral`、`crossing entry`
   初始态势各 12 个条件。它是明确的评估包线，不是全部空战态势或完整飞行包线。
2. **对手不透明**：由三个独立 opponent capability card 补足；它们在同一
   60 场景、同一 strict-JSBSim 接口、同一 M0--M2 共同参考控制器下单独报告，
   不再把对手合并成一个未定义的平均条件。
3. **消融不完整**：由“方法是否真的产生动作/模式差异”作为先决条件补足。
   M0--M2 保留为全包线比较；M3 仅在无 guard 的动态重选子集上讨论；M4
   被正式排除，不再被写作有效消融。

这不支持“所有场景均优于所有对手”或“纯 PPO 普适路由”的结论；它支持一篇
范围清楚的硕士论文：**冻结双技能 VPP 接口在一个五态势、三对手的严格 JSBSim
包线中的表现、对手条件性，以及可观察的动态路由边界。**

## 共同 Cohort 认证

| 项目 | 证据 | 认证结果 | 可写入论文的边界 |
|---|---|---|---|
| 初始场景 | Taxonomy60 manifest | 60 个冻结条件，五态势各 12 个 | 初始 taxonomy 覆盖，不等于所有几何/能量条件 |
| 工程执行 | `formal_execution_manifest.json` | 3 对手 x 5 方法 x 60 条件 = 900 条 episode，三条 child run 均完成 | 只证明该仿真链完整 |
| 后端与 raw telemetry | `formal_verification.json` 与 raw digest | 900/900 strict JSBSim，verdict `pass`；raw joint digest 为 `fca9f9eedeffa0ce191f656f52e48e39cdc9f32c2cb2ffa2bea28f96fdbaa76e` | 不是现实飞行器安全认证 |
| 动态可观测性 | `dynamic_taxonomy_phase_episode_index.csv` | 保存逐步 `(AA, ATA)` taxonomy、phase、VPP、guidance、AoA、`n_z` command/actual、terminal 语义 | phase 是分析标签，不是额外训练任务 |

正式 artifact 位于
`E:\uav-vpp-guidance-thesis-evidence\THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`。

## 对手 Capability Card

三张 capability card 使用共同参考 M0 fixed head-on、M1 fixed crossing、M2
static gate，报告 W/L/D、HP advantage、target attack-zone fraction 与
post-merge/re-entry episode 数。它们描述对固定我方接口施加的**操作压力**，
而不是 Elo、绝对强度或训练数据复原。

| Opponent | 角色与来源 | 共同参考下 target attack-zone fraction | 溯源边界 |
|---|---|---:|---|
| expert | primary；rule-based `ExpertOpponent` | 0.188--0.189 | 无 checkpoint，规则实现可定位 |
| end-to-end | primary；冻结 neural projected-base16 checkpoint | 0.146--0.147 | checkpoint SHA 已记录；训练配方不由本 Source 重建 |
| independent PPO/VPP | supporting；冻结 continuous-VPP projected-base16 checkpoint | 0.165--0.168 | checkpoint SHA 已记录；provenance-limited，不能替代主对手复现 |

完整卡片：[opponent_capability_cards_v3.md](E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1/paper_closure_materials_v2/opponent_capability_cards_v3.md)。所有结果按对手分别呈现，禁止 pooled win rate 或“最强对手”排序。

## 可判别紧凑矩阵

M0 为 fixed head-on，M1 为 fixed crossing，M2 为 static initial-label gate，
M3 为 canonical PPO，M4 为 no-bootstrap PPO。下表来自对全部 900 条 raw
record 的只读逐步重放分析；两主对手为 expert 与 end-to-end，第三对手为支持性
复核。

| 比较 | 首个决策可区分 pair / 60（每个对手） | 全程动态可区分 pair | 论文处置 |
|---|---:|---:|---|
| M0 vs M1 | 60 / 60 / 60 | 60 / 60 / 60 | 保留：两个冻结 specialist 的明确对照 |
| M0 vs M2 | 12 / 12 / 12 | 32 / 28 / 29 | 保留：static gate 仅在其选择 crossing 的条件下改变初始动作 |
| M1 vs M2 | 48 / 48 / 48 | 56 / 52 / 58 | 保留：与上项互补，说明 static gate 的状态条件选择 |
| M2 vs M3 | 0 / 0 / 0 | 25 / 32 / 28；其中无 guard 15 / 20 / 16 | 仅保留无 guard 的动态重选作为 routing observability；不写作纯 PPO 性能优越 |
| M3 vs M4 | 0 / 0 / 0 | 0 / 0 / 0 | 排除：treatment-identical，停止正式 M3/M4 消融 |

每行数字顺序为 expert / end-to-end / independent PPO/VPP。M3/M4 的独立 48
条 runtime probe 也为三对手各 `0/4`，与 Taxonomy60 的 `0/60` 结论一致。

M2/M3 的 68 条 guard-confounded record 不进入纯路由归因。其余无 guard 的动态
可区分 pair 分布为：expert 15 条（Neutral 12、Disadvantage 3）；end-to-end
20 条（Neutral 12、Disadvantage 8）；independent PPO/VPP 16 条（Neutral 12、
Disadvantage 3、Advantage 1）。这支持“在有限 cohort 中存在状态后续驱动的
模式重选”，但不支持 general-purpose geometry recognition 或跨态势优越性。

派生 ledger（不改变任何 raw result）：

- [decision.json](E:/uav-vpp-guidance-disadvantage-capability-v1-results/taxonomy60_discriminability_closure_r1/decision.json)
- [discriminability_summary.csv](E:/uav-vpp-guidance-disadvantage-capability-v1-results/taxonomy60_discriminability_closure_r1/discriminability_summary.csv)
- [pair_ledger.csv](E:/uav-vpp-guidance-disadvantage-capability-v1-results/taxonomy60_discriminability_closure_r1/pair_ledger.csv)

## 论文写作与 Stop Rule

**可写主张**：五态势 taxonomy 提供了一个明确的研究包线；三类对手在共同
参考接口上施加可量化且不同的压力；两个冻结 specialist 与 static gate 的作用
在相应条件中可区分；canonical PPO 仅在无 deterministic recovery override
混杂的动态重选子集中具有可观察接口差异。

**必须保留的限制**：`Disadvantage` 仍是跨主对手的能力边界，不能归因于已
证明的特定低层技能缺失；timeout HP advantage 不是击杀率、气动优越性或安全
认证；independent PPO/VPP 只作支持性证据；M3/M4 不构成有效因果消融。

**冻结决定**：不重跑 Taxonomy60，不调奖励/VPP/制导/PID，不改 M3/M4 的
task-bit 或 bootstrap 后再次比较。若未来必须证明纯学习的任务接口贡献，必须
从新的机制假设、新 Source ID 和独立可判别性 preflight 开始，不能回改本
cohort。
