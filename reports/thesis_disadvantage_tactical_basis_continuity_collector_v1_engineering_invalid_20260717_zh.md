# Tactical-Basis Continuity Collector V1 工程无效记录

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V1`
**运行提交：** `a8aca6eab0e3bac3e7d52174f36763e1380b711e`
**输出根目录：** `E:/uav-vpp-guidance-disadvantage-capability-v1-results/tactical_basis_continuity_collector_v1`
**最终归类：** `engineering_invalid_before_first_jsbsim_step`

## 已发生的事实

- 已按冻结配置尝试 12 场景 x 3 对手，共 36 条 non-learning record-only 记录。
- 36/36 在 `RawSIFirstPassConfig` 初始化时停止，错误为 `TypeError: unexpected keyword argument 'consecutive_closing_steps'`。
- 36/36 的 `continuity_trace` 长度为 0；没有任何 JSBSim step、VPP action、basis、prediction、guidance 或 PID telemetry 被采集。
- 独立 verifier 已读取全部 36 个 JSON、复核哈希和 gate 重算一致；这只说明空记录的 artifact 结构完整，并不验证物理链路。

## 不能得出的结论

- 不能称 tactical-basis baseline 无法到达 `disadvantage -> post_merge/re_entry`。
- 不能称 temporal reference 不可执行、basis 不满秩、预测器失败、对手过强，或当前接口缺少控制权威。
- 不能以 V1 的 0 条 qualifying handoff 作为训练、论文或能力缺口结论。

## 唯一后续处理

V1 不得重跑。新建 `THESIS-DISADVANTAGE-TACTICAL-BASIS-CONTINUITY-COLLECTOR-V2`，只修复 raw-SI dataclass 的字段名：

- `consecutive_closing_steps` -> `minimum_closing_steps`
- `consecutive_opening_steps` -> `minimum_opening_steps`

V2 必须重新冻结代码和授权后才可执行；场景、资产、动作语义、40-step window、逐对手 gate、无训练/无奖励/无 routing 的边界均不得改变。
