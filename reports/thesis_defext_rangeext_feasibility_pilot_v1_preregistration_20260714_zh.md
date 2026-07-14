# Defensive-Extension / Range-Extension Feasibility Pilot V1 预注册

## 1. 研究问题与边界

**Source ID：** `THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1`  
**Output root：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/defensive_extension_range_extension_feasibility_v1`  
**当前状态：** design-only；训练、baseline evaluation、heldout evaluation 均未授权。

本 pilot 只回答一个问题：在 first-pass 后的 dynamic `disadvantage`、`post_merge/re_entry` 连续物理状态下，一个固定 `defensive_extension` skill 在固定 `range_extension` profile 下，能否比现有 frozen head-on/crossing specialist 产生更好的脱离几何，并保持 ego crash/OOB 非劣。

本 pilot 不训练高层 PPO，不选择 profile，不改变 P3 encoder、预测器、VPP、制导或 PID，不回答五态势整体架构是否有效，也不使用 win rate 作为主要优化或通过依据。

## 2. 冻结实现与前置证据

- Implementation clean SHA：`8cec8f0c0b3cac1c50adbfc1581b06d45df09d8e`。
- Implementation manifest：[thesis_defext_rangeext_feasibility_pilot_v1_implementation_freeze.json](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_defext_rangeext_feasibility_pilot_v1_implementation_freeze.json)。
- P3 encoder SHA-256：`385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`，全程冻结且禁止 fallback。
- Single-motif collector 的 dirty-run 只作为研发正证据，证明 exact `66-D -> 3-D` 数据契约成立；不升级为 paper-safe，也不作为本 pilot 的性能结果。

## 3. 可证伪假设

### H1：新技能库缺口

候选 defensive-extension 在至少两个对手下，相对 head-on 及两种既有 specialist 中表现更好的那个，均达到预注册 practical improvement，同时第三个对手保持非劣和安全非劣。

### H2：现有技能已足够，但 routing/composition 不足

Frozen crossing 在相同 handoff 与相同场景下匹配或优于候选 defensive-extension，且无额外安全代价。此时不得声称缺少新技能，应回到高层选择或技能组合问题。

### H3：defensive-extension 假设不成立

候选技能无法相对 primary head-on baseline 保持几何非劣，或无法在至少两个对手下产生 practical improvement。结果冻结为负证据，不追加训练步数或调 reward。

## 4. 物理连续 Handoff

所有方法从同一合法 reset 开始，先由冻结 `legacy_static_oracle_task_gate`、`head_on` task key 完成相同 run-in。首次同时满足以下条件时发生一次性 handoff：

1. `first_pass_complete=true`；
2. dynamic taxonomy 为 `disadvantage`；
3. P4 phase 属于 `post_merge` 或 `re_entry`；
4. validity mask 允许 `(defensive_extension, range_extension)`；
5. 已形成 10 帧真实 16-D history，无 padding。

handoff 不 reset、不恢复 snapshot、不注入 future state。handoff 前 transition 不进入候选训练 buffer；handoff 后本 episode 的 skill 固定，不存在 routing。

## 5. 数据划分与泄漏控制

### Train

使用 [continuous train distribution](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_defext_rangeext_feasibility_pilot_v1_train_distribution.yaml)。初始 range 固定在 `1250--1450 m` 连续支持内，三对手 balanced round-robin。训练支持显式排除 v2 的两个 package signature 和全部 dev/heldout fixed package。

### Dev12

[Dev12 manifest](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/manifests/thesis_defext_rangeext_feasibility_pilot_v1_dev12.yaml) 包含 2 个新 distance/speed package x 3 个高度条件 x 2 个镜像方向，共 12 场景。仅用于固定 `10k/20k/30k/40k/50k` checkpoint 中的选择与 safety stop，不产生论文声明。

### Heldout24

[Heldout24 manifest](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/manifests/thesis_defext_rangeext_feasibility_pilot_v1_heldout24.yaml) 包含 4 个未见 distance/speed/angle package x 3 个高度条件 x 2 个镜像方向，共 24 场景。checkpoint 选定后只评估一次；禁止根据 heldout 结果重训、换场景或重跑相同 Source ID。

Dev、heldout、v2 与 train support 的 package signature 均不重合。

## 6. 固定方法矩阵

| Method | 角色 | Handoff 后控制 | Routing |
|---|---|---|---|
| Candidate fixed defensive-extension | 实验方法 | 新训练的 `defensive_extension + range_extension` | 禁止 |
| Frozen fixed head-on | Primary baseline | canonical head-on specialist | 禁止 |
| Frozen fixed crossing | Auxiliary existing-library baseline | canonical crossing specialist | 禁止 |

不加入 Oracle task gate：在固定 `head_on` task key 下，它与 fixed head-on baseline 信息重复，且不能帮助区分技能缺口与 routing 缺口。

## 7. 固定训练预算

- 单一训练 seed：`2026071402`。
- 总训练步数：`50,000`，禁止追加。
- 三对手 balanced round-robin。
- P3 encoder、预测器、VPP、guidance、PID 全冻结。
- PPO 与 geometry reward 参数在 [pilot config](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/thesis_defext_rangeext_feasibility_pilot_v1.yaml) 中固定，训练开始后禁止修改。
- checkpoint 仅按 Dev12 固定时间点选择；heldout 不参与选择。

这是 feasibility pilot，不用于宣称 seed 稳健性。若 pilot 为正，后续正式实验必须另行预注册多 seed replication。

## 8. 指标定义

### Primary metric

`normalized_range_extension_intent_loss_auc20`：从 handoff 后首个有效 target-motif step 开始，取前 20 个有效 step 的固定 `range_extension` normalized intent loss 均值。越低越好。

所有比较使用相同 opponent、scenario signature 和 seed 的 paired delta：

`delta = candidate - baseline`。

### Mechanism metrics

- 前 20 step 的 range-opening fraction；
- 第 20 step specific-energy-height delta；
- target attack-zone exposure fraction；
- post-merge/re-entry 有效持续步数；
- 三维 VPP bias；
- actual `n_z` / AoA response。

### Secondary only

Win rate、HP advantage 和 terminal mix 只作背景，不可替代 primary geometry gate。

## 9. 预注册 Gate

### 每个对手独立的数据契约门

- 至少 2 条 paired qualifying episode；
- 至少 20 个有效 target-motif step；
- 至少 2 个 scenario signature 和两个镜像方向；
- 所有 66-D observation 与 3-D action 有限；
- 无 padding、prediction fallback、reset、backend fallback 或 checkpoint fallback。

为避免仅凭极少样本作技能结论，进入最终 H1/H2/H3 判定还要求每个对手至少 8 条 paired qualifying episode 和 160 个有效 step。

### Safety 与效果门

- Candidate 相对 fixed head-on 的 ego crash/OOB rate delta `<= +0.05`，每个对手分别满足。
- Candidate 相对 fixed head-on 的 paired intent-loss delta `<= +0.02`，每个对手分别满足。
- 至少两个对手相对 fixed head-on 达到 `delta <= -0.05`。
- 至少两个对手相对该对手下表现更好的 existing specialist 达到 `delta <= -0.02`。
- 剩余对手相对 best existing specialist 不得差于 `+0.02`。

不进行 opponent pooling，也不将 `-0.05` 解释为统计显著性门；它是预注册 practical-improvement threshold。

## 10. Safety Stop Rule

以下任一事件立即停止并冻结负证据：

- non-finite 66-D observation；
- non-finite 或越界 3-D action；
- JSBSim backend fallback；
- checkpoint fallback；
- 异常 reset、history padding、snapshot restore 或 future-state injection；
- 固定 Dev12 检查中，任一对手 ego crash/OOB 超过 fixed head-on `+0.05`。

停止后禁止调 reward、加训练步数、替换 encoder/checkpoint、修改场景或使用相同 Source ID 重跑。

## 11. 结果解释规则

| 结果 | 唯一允许结论 |
|---|---|
| Candidate 通过完整 H1 门 | 现有库存在 defensive-extension 新技能缺口的初步支持；仍需多 seed 正式复核 |
| Frozen crossing 匹配或优于 candidate | 现有技能可能已足够，主要问题是 routing/composition；不新增技能 claim |
| Candidate 未过 head-on 非劣门 | defensive-extension 假设不获支持，停止该技能线 |
| 任一 safety/contract gate 失败 | pilot NO-GO，不作技能能力结论 |

## 12. 当前授权状态

本预注册只完成设计与独立复核。`training_permitted=false`、`pilot_execution_permitted=false`、`baseline_evaluation_permitted=false`。任何训练或评估必须在新的用户授权后，先验证 clean SHA、fresh output root 和所有资产 hash，再开始执行。
