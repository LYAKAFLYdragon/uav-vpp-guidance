# Disadvantage Capability Plan 1 V2 预注册

**Source ID:** `THESIS-DISADVANTAGE-CAPABILITY-PLAN1-V2`  
**性质:** 独立、评估专用研究线；不改变 Taxonomy60、AST 主稿或任何既有正式输出。

## 触发背景

V1 的 non-Source engineering smoke 显示 opening-pressure disadvantage 场景不会达到 `first_pass_complete`，故基于该事件的 phase-rule 从不切换，不能形成有效的路由干预。V2 不把这一工程事实写成性能结论，也不使用其胜负结果选择方法。

## 冻结设计

- 场景：全新的 `Disadvantage12-V2`，使用未出现过的场景名、seed、距离、速度、方位角和高度组合。
- 对手：`expert`、`end_to_end`、`independent_ppo_vpp`；前两者单独组成主 gate。
- 方法：冻结 canonical PPO、fixed head-on、fixed crossing、temporal H-to-C schedule。
- temporal schedule：前 10 个 high-level step 使用 head-on，第 11 步及之后使用 crossing。10 步与已冻结预测器的 9-step warm-up 对齐；它不读取未来状态、未来对手动作、奖励、终端或任何 performance telemetry。
- 不允许训练、奖励/VPP/guidance/PID 改动、checkpoint 替换、场景替换或结果导向重跑。

## 判定和分流

对每个主对手，temporal schedule 必须同时满足：

- 至少两个场景相对该场景最佳 fixed H/C 取得严格的 resolved win 改善。
- 平均减少至少 `1.0 s` 的对手攻击区暴露。
- ego crash/OOB 比例不比最佳 fixed H/C 高 `0.05`。

两个主对手均通过，才可称“该特定、机械预注册的 H-to-C 组合支持 routing/composition 解释”，并停止新低层技能研发。任一主对手失败，则不能通过调路由补救；下一步只允许进入独立的非学习 motif feasibility/传递实验，以判断是否存在可训练的低层能力缺口。
