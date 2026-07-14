# P4 v1 Go/No-Go 正式决策

**日期：** 2026-07-14  
**适用分支：** `research/thesis-five-state-shared-intent-v1`  
**决策对象：** `GEO-20260713-R1` 的四共享技能 geometry pretrain  
**决策级别：** 训练停止决策；不改变既有 gate 或任何冻结结果。

## 1. 决策

| 项目 | 决策 | 含义 |
|---|---|---|
| P4 v1 geometry pretrain | **NO-GO** | 本轮结果冻结为正式负证据；不得重训、加步数、改 reward、放宽 gate 或改 dev30 以追求通过。 |
| combat finetune | **NO-GO** | `all_skills_ready_for_combat_finetune = false` 保持有效。 |
| P5/P6/P7 | **NO-GO** | 高层 PPO、消融与 heldout60 均继续锁定。 |
| P4-v2 sampler-feasibility | **设计 GO，执行未授权** | 仅允许完成新的非学习预检设计；不得据此自动启动 JSBSim probe 或任何训练。 |

P4 v1 的结论不是“共享技能架构不可行”，也不是“某个低层策略已被证明没有能力”。它证明的是：**当前 v1 sampler 与 fixed `initial_class × phase` readiness contract 未能为多个已声明 cell 提供可判定覆盖，因此本轮训练不具备进入 combat finetune 的证据资格。**

## 2. 冻结证据

可移植的只读 evidence bundle 已由 [exporter](../scripts/export_p4_evidence_bundle.py) 校验。其 gate 和四个 skill summary 的 SHA256、registry snapshot，以及 `26 declared / 26 emitted / diff 0 / extra 0` 对账均可复现。

- 归因审计：[P4 gate failure attribution audit](thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md)
- 机器可读矩阵：[P4 attribution matrix](thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json)
- 证据包：[P4 evidence bundle](p4_evidence_bundle_20260714/evidence_manifest.json)
- 可移植性闭合：bundle verifier 与 `tests/test_p4_gate_failure_attribution.py` 均通过。

仍须保留的 provenance boundary：run commit、`training_plan.json`、`resolved_geometry_config.json` 与 dev30 manifest 未被找回。因此该证据支持的是**后验 gate 归因复核**，不是对原训练运行的完整再现。

## 3. 失败归因

`geometry_phase_coverage` 的失败并非 key 缺失。现有 gate 对每个 declared state-phase cell 以 `.get(..., 0)` 显式发出计数；本轮所有 26 个 declared cell 都已发出。

| Skill | 零覆盖 cell | 归因 |
|---|---|---|
| `defensive_extension` | `disadvantage/neutral × post_merge/re_entry` | train 与 dev 均无可用的 fixed-initial-class 覆盖；training-support gap。 |
| `lead_intercept` | `advantage:post_merge`, `neutral:post_merge` | 前者在 train 出现但 dev 为零，属于 dev coverage gap；后者在 train/dev 都不足。 |
| `pursuit_conversion` | `advantage/neutral × post_merge` | train 与 dev 覆盖不足。 |
| `reentry_recovery` | `advantage/neutral × post_merge/re_entry` | train 与 dev 覆盖不足；其 intent progress 虽为 0.5820，但不能抵消 coverage failure。 |

三项 skill 的 aggregate intent progress 也低于 0.55：`defensive_extension=0.5325`、`lead_intercept=0.4977`、`pursuit_conversion=0.5193`。该指标只有 skill 级聚合值，**不能**被反推为任一零覆盖 cell 的行为失败。

最重要的语义限制是：现有 telemetry 以场景的固定 `initial_class` 记录 phase step，而非按逐步动态五态势重新标注。因此它只能证明 fixed `initial_class × phase` 覆盖，不能证明 post-merge 或 re-entry 中的动态态势覆盖。

## 4. 不应做的事

- 不以更多 PPO steps 处理零覆盖。
- 不将 P4 v1 的 `minimum_geometry_phase_coverage_per_declared_cell=2` 事后放宽。
- 不将 P4 v1 dev30 场景加入训练，或用新的 heldout 替换失败 cell。
- 不将 `reentry_recovery` 的单项 intent pass 解释为四技能 readiness pass。
- 不修改 66-D observation、冻结 P3 encoder、预测目标 VPP 接口、guidance/PID、reward、三 opponent registry 或 canonical 双技能主线。

## 5. 允许的唯一后续

下一步仅为 [P4-v2 sampler-feasibility preflight 设计](thesis_five_state_p4_v2_sampler_feasibility_preflight_design_20260714_zh.md)。它必须先回答：在不伪造 JSBSim 状态、不过度依赖场景名称、也不改动策略学习目标的前提下，是否存在可重复的物理 sampler，使每个新的 declared **dynamic-state × phase** cell 在 train 与 dev 支持域中实际可观察。

只有该预检被单独授权、实际执行并通过全部 stop rule 后，才可另行请求一个新的 P4-v2 geometry-pretrain run。该请求不由本决策自动授予。
