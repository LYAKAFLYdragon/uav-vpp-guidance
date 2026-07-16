# Runtime Smoke 与三锚点路径包线 Gate（2026-07-16）

## 决策

工程 runtime smoke 已通过，但三锚点动态参考路径的只读可执行包线 gate 未通过。因此，
**不得创建第 3 步的 fixed-head-on vs. projected three-knot path 非学习对比 Source ID**，更不得训练
新技能、高层 PPO 或调节 reward。

这不是性能失败，也不否定动态参考轨迹这一更高层假设；它只说明：在当前冻结的 world-frame raw
3-D VPP action head 下，已经没有足够的未裁剪动作余量来诚实测试所选三锚点候选。

## 1. 工程 Smoke：PASS

- 身份：`ENGINEERING-RUNTIME-SMOKE-NOT-A-SOURCE-ID`，不属于科学 Source ID，不可用于论文或性能结论。
- 场景：冻结 V17 场景 `head_on__v17_path_context_a__co_altitude__negative::seed=199102`。
- 对手：expert、end-to-end、independent PPO/VPP；每个对手一条完整 strict-JSBSim 连续 episode。
- 输出：[smoke_20260716_v2](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/outputs/diagnostics/runtime_smoke/smoke_20260716_v2)。
- 独立复核：`3/3` records，reported gate 与重算结果一致，`passes=true`。

三对手均在第 10 个高层步发生唯一的 observer-only release；每条均保存 261 个 observer receipt 和
240 个 VPP/guidance/PID actual-response telemetry step。raw-SI first-pass/re-entry 顺序、handoff state
hash、VPP 三轴、guidance command、由 JSBSim body velocity 推导的实际 AoA、实际 `n_z`、anchor mode
以及 verifier 可读性均通过。

首个 disposable smoke 输出 `smoke_20260716_v1` 保留为工程失败：统一 JSBSim state 没有直接暴露
`alpha_rad`。该问题已被 fail-closed 检出，v2 使用既有 shared-skill evaluator 相同的
`atan2(body_velocity_z, body_velocity_x)` 实际 AoA 定义；这不修改 simulator、VPP、guidance 或 PID。

## 2. 三锚点包线：NO-GO

只读分析重放 V16 `progress_triggered_reference_path` 的 14 条 qualifying episode（expert `5`、
end-to-end `3`、independent PPO/VPP `6`）和 280 个 20-step telemetry row。它对六个候选 escape extent/
connector geometry 重算当前 action head 的 requested/applied action 与逐轴 clipping，不运行 JSBSim，
不估计未来 AA/ATA/能量响应。

通过门为：同一候选必须在至少两个对手中各有 `>=4` 条全程无裁剪 episode，且各自无裁剪步比例
`>=0.90`。结果 `passing_candidates=[]`。

最能说明问题的是理论下界候选 `lower_bound_escape`（escape extent = 1 m）：

| 对手 | 无裁剪步比例 | 全程无裁剪 episode | 裁剪轴计数 x/y/z |
|---|---:|---:|---|
| expert | 0.500 | 1 / 5 | 12 / 49 / 0 |
| end-to-end | 1.000 | 3 / 3 | 0 / 0 / 0 |
| independent PPO/VPP | 0.717 | 3 / 6 | 15 / 32 / 0 |

因此，问题不只是旧 V16 的 `1200 m` escape path 过宽。即便路径 escape 项近乎为零，expert 与
independent PPO/VPP 下仍有大量 forward/lateral action 被裁剪；end-to-end 虽未裁剪，但只有 3 条
qualifying episode，不满足新 interface 的最小 4 条分母。完整矩阵见
[三锚点包线报告](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_three_knot_envelope_feasibility_v1_20260716_zh.md)。

## 3. 后续边界

1. 第 3 步的 JSBSim 六维联合改善比较保持锁定；不得通过放宽 clipping 门、删除对手或事后缩小分母解锁。
2. 任何后续时间扩展接口必须首先明确改变什么：例如新的 trajectory-level action parameterization 或
   action-envelope-aware compiler；不能只是再次改变 VPP offset、尺度、reward 或窗口长度。
3. 新接口实现后，必须先运行 `thesis_runtime_smoke_v1.yaml` 的等价 smoke，再用全新且不相交的 manifest
   做非学习 action-to-trajectory-to-six-intent 传递 probe。
4. 只有新的接口在至少两个对手下通过无裁剪可执行性和六维联合改善门，才可讨论任何训练。
