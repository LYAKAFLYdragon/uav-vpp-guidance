# P1 R4 JSON Telemetry 实现失败归档

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R4-FRESH-ENV-REPRO-V1`
**状态：** `implementation_failure_not_experimental_evidence`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r4_fresh_environment`

## 事实

R4 在首个 `expert / advantage / own_below / negative / forward` child 的 raw telemetry 写盘阶段终止，父 runner 的 `completed_episode_count` 为 `0`。异常为：

```text
TypeError: Object of type ndarray is not JSON serializable
```

partial temporary file 显示未序列化字段位于 `steps[0].own_state.attitude_rpy`；该值来自 JSBSim state telemetry，类型为 `numpy.ndarray`。该异常发生在 episode 运行后、artifact 原子替换前，故没有可用于 gate 的完整 episode artifact。

## 固定证据

| 文件 | SHA-256 |
|---|---|
| `runner_failure_manifest.json` | `fcab9635ec535237654ae2b5c68e13b50b48d307bfa1ecb2ebb1e465341855e0` |
| `episodes/expert/repeat_0_forward/tmpxg7k978r.tmp` | `367ecbd5cac345fd894b4d7c1f83379b0eb141a0faf0764b898303bd81ed1d63` |
| `launches/expert_r0_five_state_dev30_dev_outside_train_midrange_advantage_own_below_neg.json` | `bd7789849d711b5adafeaa877b3b9879d4e7111ca6a29d13ec0322742f1a0a54` |

## 结论与边界

- R4 不支持任何关于策略、对手强度、JSBSim、VPP、制导、PID、安全或可重复性的正/负结论。
- R4 output root 保留且不可覆盖；不得重跑失败 child 或其余 269 条 episode。
- 唯一允许的后续修复是新 Source ID 的 R5：在 raw telemetry 原子写盘前采用既有 R4 `canonicalize` fail-closed 逻辑，将 `numpy.ndarray` 递归转换为 JSON 数组；其它非有限或不支持状态仍应失败。
