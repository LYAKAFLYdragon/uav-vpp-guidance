# Disadvantage Capability Plan 1 V2：路由上界结果与决策

**Source ID：** `THESIS-DISADVANTAGE-CAPABILITY-PLAN1-V2`  
**性质：** 研究诊断；只评估、不训练、不调参。  
**最终决策：** `routing_explanation_not_supported`；禁止继续修改或搜索双技能路由规则，进入独立的非学习 Plan 2 motif-capacity discriminator。

## 1. 要回答的问题

Taxonomy60 表明，初始 `disadvantage` 是当前双技能接口在主对手下的失败区域；但低胜率本身不能区分“高层没有选择合适的既有技能”和“现有两种技能均不足”。

Plan 1 V2 预先指定一个唯一、非学习的组合上界：前 10 个高层决策步固定使用 frozen head-on specialist；第 11 步起固定使用 frozen crossing specialist。十步 warm-up 与冻结预测器的九步 warm-up 对齐。该规则不读取未来状态、奖励或终端，且最多切换一次。

这不是“所有可能路由策略”的检验。它只检验一个可复现的命题：**已有 head-on/crossing 专家能否仅通过一项预先规定的、预测器对齐的 H-to-C 切换，实用地缓解 opening-pressure disadvantage。**

## 2. 冻结执行与完整性

- 新的 Disadvantage12：两组未见距离/速度包、三种高度条件、双镜像方向；不复用 V1 的场景或 seed。
- 对手独立报告：`expert`、`end_to_end`、`independent_ppo_vpp`。
- 方法：canonical PPO、fixed head-on、fixed crossing、temporal H-to-C；每个 opponent 为 `12 x 4 = 48` 条，共 `144/144` 条 strict-JSBSim records。
- 所有 temporal episode 均在第 11 个高层决策步发生一次且仅一次切换；无 backend fallback、缺失 raw trajectory 或执行失败。
- 独立 verifier：[`plan1_verification.json`](E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan1_v2_route_upper_bound_r1/plan1_verification.json)。执行回执：[`execution_receipt.json`](E:/uav-vpp-guidance-disadvantage-capability-v1-results/plan1_v2_route_upper_bound_r1/execution_receipt.json)。

运行以 `formal-small` 标记，`paper_safe=false`；它可用于研发归因，不得替代 Taxonomy60 的论文正式结果或被写成投稿主表。

## 3. 预注册门槛与结果

每个主对手须同时满足：

1. temporal H-to-C 相对同场景最佳 fixed H/C 至少有 2 条 resolved improvement；
2. 平均对手攻击区时间至少减少 `1.0 s`；
3. ego crash/OOB 相对最佳 fixed H/C 的增量不超过 `+0.05`。

| Opponent | Resolved improvement vs. best fixed | Mean target attack-zone reduction | Ego crash/OOB delta | Gate |
|---|---:|---:|---:|---|
| expert | 0/12 | -0.033 s | +0.0833 | fail |
| end-to-end | 1/12 | -0.600 s | 0.0000 | fail |
| independent PPO/VPP | 0/12 | -0.050 s | 0.0000 | fail |

负的“攻击区时间减少”表示 temporal schedule 比最佳 frozen specialist 暴露更久。三个 opponent 均未通过；expert 还出现了预注册不允许的安全退化。

## 4. 可支持与不可支持的结论

**可支持：** 当前双技能库中，预先规定的 predictor-aligned H-to-C 组合不构成 opening-pressure disadvantage 的实用修复。不能再把“再调一次切换时间、阈值或 guard”作为低成本优先路线。

**不可支持：**

- 不能称所有状态条件路由均无效；本实验未训练或搜索状态依赖的路由器。
- 不能称现有 specialist library 必然不足；H-to-C 失败也可能来自 VPP-to-guidance 传递、动作包线或对手条件。
- 不能把本 Source 的结果并入 Taxonomy60 或用于论文中的统计优越性声明。

## 5. 强制后续路线

下一步只能进入独立的 **Plan 2 non-learning motif-capacity discriminator**。其任务不是再切换 H/C，而是对预先定义的 defensive break/unload、range/energy extension、vertical-plane separation 逐一验证：

`bounded VPP reference -> guidance/PID actual response -> pressure geometry`。

Plan 2 的判定必须把四种原因分开：

| Plan 2 观察 | 后续归因 |
|---|---|
| 某个参考机动在至少两个 opponent 中安全地降低攻击区压力 | 对应机动是唯一允许训练的低层技能候选 |
| VPP 指令高裁剪，或未转化为预期 n_z/AoA/roll 响应 | VPP/guidance/flight-control 接口缺口 |
| 执行响应正常，但所有候选机动均无法改善几何 | 当前冻结包线下的 physical/behavior reachability 缺口 |
| 仅单一 opponent 有一致改善 | opponent-dependent sensitivity；不得称为通用技能缺口 |

如 Plan 2 未给出跨对手的安全正方向，则停止扩大低层技能库与四技能共享库研发；不通过增加训练步数、奖励或场景替换继续尝试。
