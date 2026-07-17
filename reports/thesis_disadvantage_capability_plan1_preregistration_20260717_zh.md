# Disadvantage Capability Plan 1 预注册

**Source ID:** `THESIS-DISADVANTAGE-CAPABILITY-PLAN1-V1`  
**性质:** 独立研究线、评估专用；不修改 Taxonomy60、AST 主稿或任何既有正式结果。

## 问题

Taxonomy60 显示 `disadvantage` 是当前双技能接口在两个主对手上的共同失败区域，但该事实不能区分“现有技能库缺能力”和“现有 head-on/crossing 只是没有在合适阶段组合”。本试验只检验后一种解释的一个可证伪上界。

## 冻结方法和场景

- 场景：新生成的 `Disadvantage12`，两组未见距离/速度、三种高度条件、双镜像；仅包含由 `AA/ATA` 几何规则验证的 disadvantage 初始态势。
- 对手：`expert`、`end_to_end`、`independent_ppo_vpp`，分别报告；前两个是主要 gate。
- 方法：冻结 canonical PPO、固定 head-on、固定 crossing，以及 phase-rule H-to-C。
- phase-rule：初始至已观测 `first_pass_complete` 前固定选择 head-on；该事件第一次为真后固定选择 crossing。它不读取未来状态、未来对手动作、奖励或终端结果，最多切换一次。
- 后端：严格 JSBSim；所有低层 checkpoint、预测器、VPP、guidance、PID、奖励和对手保持冻结。

## 主要判定

每个主对手分别将 phase-rule 与该场景下终端 ordinal 最好、再按对手攻击区暴露最小的固定 H/C 控制比较。只有两个主对手都同时满足下列门槛，才支持“路由/组合解释”：

- 至少两个场景中 phase-rule 相对最佳固定技能取得严格的 resolved win 改善。
- 平均减少至少 `1.0 s` 的“对手处于攻击区”时间。
- phase-rule 的 ego crash/OOB 比例不比最佳固定技能高 `0.05`。

不满足时，结论只能是“该预注册 H-to-C 组合不支持纯路由解释”；不得据此调路由、换阈值、重选场景或增加训练。下一步只允许进行新的、非学习的低层 motif 传递/可达性 probe。
