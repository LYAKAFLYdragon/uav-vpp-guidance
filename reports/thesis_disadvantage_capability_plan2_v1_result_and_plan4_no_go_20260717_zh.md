# Disadvantage Capability Plan 2 V1 结果与 Plan 4 Go/No-Go 决策

**Source ID：** `THESIS-DISADVANTAGE-CAPABILITY-PLAN2-V1`  
**权威输出：** [`plan2_motif_capacity_v1_clean_repro`](E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan2_motif_capacity_v1_clean_repro)  
**独立 verifier：** [`plan2_verification.json`](E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan2_motif_capacity_v1_clean_repro/plan2_verification.json)  
**最终判定：** `current_envelope_physical_or_behavioral_reachability_gap`  
**Plan 4：** `NO-GO`，不授权训练单技能或共享技能库。

## 1. 执行与溯源

正式矩阵为 `5 methods x 3 opponents x 12 new disadvantage scenarios = 180` 条 strict-JSBSim record。三个对手子运行均为 `completed`，`failures.json` 为空，warm-up hash 均与同场景 frozen head-on reference 完全一致。

首次输出根在外层工具超时后发生过短暂的同 run-id 手动恢复冲突，故只保留为诊断副本，不作为结论依据。随后在新的空输出根按完全相同的 clean SHA `e4ffa05`、config、manifest、checkpoint、seed 和门槛完成一次不变的技术复现；复现 verifier JSON 与诊断副本的 SHA-256 一致。下述结论仅引用 clean reproduction。

这仍是 `formal-small` 研发证据，`paper_safe=false`，不得写入 AST 主结果或替代 Taxonomy60 的论文结果。

## 2. 执行链是否有效

三个 motif 在 end-to-end 与 independent PPO/VPP 下都通过执行门：20-step intervention 完整、预测器有效且无 fallback、active tactical basis rank 为 `3`、最大 condition number 约 `4.319`、action clipping 为 `0`、VPP 与 guidance 的成对差异均超过预注册下限。

这排除了“动作没有进入 VPP/guidance”“横向 basis 缺失”或“JSBSim 后端没有运行”的解释。expert 下三个 motif 均比 fixed head-on 多 `1/12` ego crash/OOB（`+0.0833`），因此 expert 的执行安全门失败；这一风险本身也不支持直接训练候选技能。

| Motif | 执行传递（end-to-end / independent） | 主要几何结果 | 结论 |
|---|---|---|---|
| `lateral_break_same_side` | actual world delta `504 m`；VPP delta 约 `440 m`；guidance delta 约 `0.094` | 对手攻击区 fraction delta 为 `0.0 / 0.0`，未达 `<= -0.10` | 横向 pulse 被执行，但未降低即时压迫 |
| `range_extension_los` | actual world delta `572 / 586 m`；guidance delta `0.126 / 0.128` | range-rate delta `+0.542 / +0.567 m/s`，远低于 `+5 m/s` | 距离方向可传递，但未形成实用 opening |
| `vertical_separation` | actual world delta `198 m`；guidance delta 约 `0.046` | 高度差增加 `+0.355 / +0.345 m`，远低于 `+50 m` | 垂向命令未在 4 s 窗口形成可用 separation |

## 3. 对 disadvantage 失败的四分归因

| 假设 | 现有证据 | 结论边界 |
|---|---|---|
| 高层路由问题 | Plan 1 V2 的预注册 H-to-C schedule 在三对手均未过实用门；expert 还多 `+0.0833` ego crash/OOB | 不支持继续调机械切换、阈值或 guard；不排除尚未训练的复杂状态条件路由 |
| 现有技能库缺少某个明确 specialist | 三个候选低层原语均未在至少两个 opponent 形成安全的目标几何改善 | **未证明**缺少 defensive-break、range-extension 或 vertical-separation specialist；不能直接增加技能数量 |
| VPP 接口失效 | rank-3 tactical basis、0 clipping、数百米 VPP 实际变化与可测 guidance response 均存在 | 不是“动作被吞掉”或简单维度缺失；但接口的当前单点、4 s、有界 action envelope 未建立有用几何控制权威 |
| 物理可达性 | 三种方向都可传递，却没有跨对手降低压力、拉开闭合或形成垂向间隔 | 支持**当前冻结动作包线、时域与 opponent 条件下**的 physical-or-behavior reachability 缺口；不证明任意更长时域或全新接口在物理上不可能 |

因此，最准确的论文/硕士论文表述是：

> 在本研究冻结的单点 3-D VPP 接口和 4 s 连续机动窗口内，opening-pressure disadvantage 尚未出现跨对手、可安全利用的局部低层控制方向。当前失败不能归因于“PPO 没有切换专家”，也不能据此声称已确定缺少某一新 specialist。

## 4. Plan 4 决策

Plan 4 的预条件是 Plan 2 出现至少一个 `candidate_low_level_motif_exists`，即同一原语在至少两个 opponent 同时通过执行与对应几何门。实际 `candidate_motifs=[]`，故 Plan 4 单技能训练不具备因果依据。

**禁止：**

- 不训练 `defensive_extension`、lateral-break、range-extension 或 vertical-separation 的新 checkpoint；
- 不重跑本 Source、更改 pulse 幅度/窗口/门槛、增加训练步数或修改 reward；
- 不启动四共享技能、combat finetune、高层 PPO 或把 fixed crossing 的零星胜利解释为能力证明。

**若未来另起新研究线，唯一合理的下一假设是：** 改变时间扩展参考轨迹或 VPP action-envelope 参数化能否先建立跨对手几何控制权威。它必须使用新 Source ID、全新不重合 manifest，并在训练前以非学习 gate 证明至少两个 opponent 中存在安全的压力/闭合/垂向联合改善；否则应停止低层训练线。
