# V16 动态参考路径接口可行性决策

## 决策

V15 已经以完整 144 条 strict-JSBSim 记录否定了“单点 VPP 上的短 10-step profile 走廊”在冻结包线内产生跨对手实用改善的假设。下一条研发线若继续，必须是**动态参考路径接口**，而不是继续修改 reward、单点 VPP scale、profile、窗口或训练步数。

本文件仅确定接口选择与前置证据，不授权 V16 实现、probe 或训练。

## 现有代码审计

| 现有组件 | 能力 | 为什么不能直接成为 V16 combat 接口 |
|---|---|---|
| `VirtualPointGenerator.action_to_virtual_point` | 将 3-D action 映射为一个 world VPP position。 | 返回的是单个点；没有 path segment、切线、曲率、路径进度或前瞻点选择契约。 |
| `HybridGuidance.compute_command` | 以当前单个 `virtual_point` 的相对位置计算 PN/LOS command。 | 不接受 reference trajectory 或多点 path，无法区分“沿路径前进”与“追逐某个瞬时点”。 |
| `VirtualPointSmoother` | 对单个点做 EMA。 | 仅是 temporal smoothing，且当前标记 TODO；不生成预测路径或路径跟踪语义。 |
| `TrajectoryPredictorAdapter.predict` | 返回一个预测位置、方差与信息。 | 当前 API 是 single future position，不提供 K 个未来点或一个带不确定度的预测轨迹。 |
| `MultiWaypointTrackingEnv` | 合成五个 waypoint 并顺序切换。 | 使用 synthetic target、覆盖 combat target、重定义终止条件并包含任务专属 command shaping；直接复用会改变对手、任务动力学和胜负语义。 |

结论：仓库没有一个可不加改动就用于真实三对手空战的多点参考路径接口。将 `MultiWaypointTrackingEnv` 作为“多点 VPP 证明”会造成任务替换，不能用于支持五态势空战结论。

## V16 的最小合格接口

V16 必须以一个新的 Source ID 建立以下独立契约：

1. **路径来源：** 在每个连续 JSBSim step，仅基于当时和历史 target state 构造有序 `K=3` reference anchors；不得使用 future simulator state。第一版可以比较冻结预测器的多时域 rollout 与确定性运动学 rollout，但必须逐点记录来源、horizon、方差/有效性与 fallback。
2. **路径消费者：** 新的 path-to-guidance adapter 从该序列选择一个可解释的 lookahead point，并至少记录 path tangent、progress index、选择的 anchor 和传给现有 guidance/PID 的最终单点 VPP。它不能修改 guidance/PID 的安全约束。
3. **边界隔离：** 真实 backend target、三类 opponent、terminal/win semantics 和 existing VPP baseline 不变；禁止使用 `MultiWaypointTrackingEnv` 的 synthetic target、waypoint completion 或 command shaping。
4. **观测契约：** 66-D base/history/P3 不变。若路径信息将进入策略输入，必须通过新 schema version 或明确附加 contract；第一轮 non-learning probe 只记录路径，不训练任何模型。

## 必经的非学习预检

在任何新 reward 或训练之前，新的 V16 preflight 至少要在三个对手下分别验证：

- 完整连续 run-in 后存在至少两个与当前 target 不同且有限的 future anchors，且各 anchor 的 source/horizon 可复算；
- 选择的 lookahead VPP 与 single predicted-anchor baseline 有非零、受限的 world delta，并在 VPP、guidance command 和 actual response 三层留下可重放 telemetry；
- 无 snapshot/future-state injection/backend/prediction/checkpoint fallback；无 reset at handoff；
- 相同 scene/seed 下 path 与 baseline handoff hash 一致，且满足预注册的 ego crash/OOB stop rule；
- 只有当至少两个对手下存在明确、方向一致的 mechanism signal，才可起草新的低层训练 feasibility-pilot。

若 V16 的非学习 path-transfer probe 仍未能建立该机制信号，就应停止以“共享 intent 训练”追求全局优势的研发线，并将 V13--V15 作为方法边界，而不是继续堆叠 reward 或专家数量。
