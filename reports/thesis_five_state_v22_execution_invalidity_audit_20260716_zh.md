# V22 observer-release probe 执行有效性审计（2026-07-16）

## 结论

`THESIS-FIVE-STATE-V22-OBSERVER-RELEASE-PROBE` 的 72 条 strict-JSBSim 输出必须标记为
**execution-invalid，不构成正证据或负证据**。它不能支持“observer-only warm-up 已隔离控制”或
“三对手均无法 release”的结论。

本次结论只针对 V22，不改变 V21 已经成立的因果发现：原始 `current_target` warm-up 会阻止环境调用
predictor，因此以 predictor-valid 作为 release 条件会产生接口循环依赖。

## 已执行对象

- Source ID：`THESIS-FIVE-STATE-V22-OBSERVER-RELEASE-PROBE`
- 实现 SHA：`43f0a04f000b95d46478123eecd3e2c386fa4132`
- 输出根：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_v22_observer_release_probe`
- 执行量：3 个对手 × 24 个冻结场景 = 72 条记录；无训练、无调参。

## 无效原因及可复核证据

`Episode` 在该 SHA 中继承了 V22 mixin，但遗漏了 `_observe()` 覆盖。于是每次环境观测更新后，
`_v22_after_observation()` 从未运行。该钩子本应：固定 warm-up anchor、记录 observer receipt，
并在十帧有效 receipt 后切换 anchor。

逐文件检查显示：

- 72/72 条记录的 `observer_trace` 均为空；
- 72/72 条记录的 `anchor_switches` 均为空；
- `released=false`、`warmup_control_isolated=true` 和 `post_release_valid_steps=0` 因而是初始默认值，
  不是协议测量；
- 独立 verifier 因 `V22 observer trace is missing` 正确拒绝该输出。

因此，输出中的 `gate.json` 只反映了错误实现产生的默认汇总，不得解释为 sampler、预测器或三对手的
物理/行为负证据。

## 保留与后续边界

1. 保留该输出目录及其 `run_manifest.json`，不得覆盖、删除或混入论文安全证据。
2. V22 Source ID 已消耗，禁止以同一 Source ID 重跑，也不得修改其 `gate.json` 伪装为有效结果。
3. 已在实现中恢复 `_observe()` 集成，并新增四项单元测试，防止后续 collector 静默产出默认值。
4. 是否以**新的、独立预注册 Source ID**执行修正后的 observer-release probe，必须在重新审视五态势
   路线后单独决定；本审计不自动创建后续运行，也不授权训练。
