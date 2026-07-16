# Taxonomy60 No-Go 路线决策器 v3

**适用 Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`
**取代关系：** 保留 v2 为历史合同；v3 只补全结果分类到研究路线的机器可读映射，不改变场景、方法、资产、评估数或任何性能变量。

## 审核结论

v2 已正确限制了“有效负结果不能自动变成下一轮训练”，但其 `classification.precedence` 有 10 个类别，而 `allowed_routes` 只显式列出其中 4 个。因此，P4 正式 verifier 在遇到 guard 混杂、对手依赖、routing-sensitive、容量边界、crossing 约束或 terminal 语义限定时，仍可能需要临时选择路线。

v3 要求每个分类只能映射到一个 `selected_study_route`。除已验证的工程无效或其唯一 R1 恢复外，所有有效的正面、混合和负面结果均进入 `paper_closure`。这意味着它们会被完整报告为能力图谱、对手能力卡、消融或边界，而不会触发调参、补场景、重训或新控制器。

## 固定切换表

| 首个适用分类 | 唯一研究路线 | 允许的下一动作 | 明确禁止 |
| --- | --- | --- | --- |
| `implementation_invalid` | 工程恢复 | 在 disposable smoke 中复现并修一个工程根因 | 分析 partial 性能、替换科学变量 |
| `technical_recovery_r1` | 同变量 R1 | 完整重跑一次 900 条，且必须先完成 smoke 复现和独立 diff 审核 | R2、调参、换场景或 checkpoint |
| `paper_two_opponent_only` | 历史双对手写作 | 用既有 Taxonomy30 与 Calibration v2 完成论文 | 从无效 Taxonomy60 记录挑选结果或声称三对手结论 |
| `mixed_interface_guard_bound` | `paper_closure` | 报告部署混合接口与 guard telemetry | 把性能差归因于 PPO routing |
| `opponent_dependent_or_non_discriminative` | `paper_closure` | 保留 split-specific cell、能力卡和完整矩阵 | 合并 opponent 或补新场景寻找差异 |
| `routing_sensitive_candidate` | `paper_closure` | 写 mode/VPP/terminal 因果审计 | routing patch 或重训 |
| `two_skill_capacity_boundary_candidate` | `paper_closure` | 报告双技能当前边界；论文闭合后最多起草独立 memo | 声称已证明第三技能必需，或自动启动新算法 |
| `crossing_constraint_mechanism` | `paper_closure` | 将 bootstrap 定位为确定性保持约束 | 声称无任务标签几何识别 |
| `terminal_semantic_qualification` | `paper_closure` | 报告 terminal decomposition 和安全语义 | 将 win rate 写成 kill rate 或安全认证 |
| `paper_closure` | `paper_closure` | 只读分析、artifact bundle 与论文写作 | 训练、调参、替换场景、结果驱动重跑 |

## P4 输出合同

正式路线 JSON 必须由 **P4 formal verifier** 生成，而非已完成的 P2 smoke verifier。除原有 hash、有效性、证据和 guard 字段外，P4 必须写入：

- `route_candidates_in_precedence_order`：逐项列出检查顺序与命中状态。
- `selected_study_route` 与 `route_transition_reason`：证明路线来自预注册映射，而非结果后选择。
- `post_closure_eligibility`：只有容量边界满足全部独立条件时才可为 memo 候选，且不授权实验。
- `recovery_state`：记录是否发生正式 Source、是否已用 R1、以及是否被送往历史双对手闭合。

**预期正确路线：** 结合既有 V1--V22 的负证据，Taxonomy60 最可能出现的是“正式有效但跨场景表现混合”的情况。这不是失败；正确出口是完整的 Taxonomy60 capability atlas、三张 opponent capability card 和收紧的双技能系统边界。只有跨两个主对手重复、且排除 guard/执行链/早期对手轨迹混杂的容量边界，才允许在论文闭合后另起一份不自动训练的接口 memo。
