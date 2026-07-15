# V4 Runtime-Feasibility 结果与 Phase 语义审计（2026-07-15）

## 证据身份

- Source ID：`THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-RUNTIME-FEASIBILITY-V4`
- 运行提交：`884af19f136b7f2588421f53b195d5d1a75dcc3f`
- 输出根：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v4_runtime_feasibility`
- 独立验证：[independent_verification.json](E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_v4_runtime_feasibility/independent_verification.json)

严格 JSBSim 共完成 `48 x 3 = 144` 条 record。独立 verifier 复算了完整 `opponent x pair_key` universe、manifest/config/overlay 快照哈希、raw-SI phase replay、66-D/3-D telemetry、reward 字段隔离和 per-opponent gate；结果为 `pass`。未出现 backend/telemetry contract failure，也没有 `runner_failure_manifest.json`。

## 预注册 Gate 结果

| Family | Expert valid steps / qualifying | End-to-end valid steps / qualifying | Independent PPO/VPP valid steps / qualifying | 通过 |
|---|---:|---:|---:|---|
| close_balanced | 236 / 3 | 0 / 0 | 43 / 1 | 否 |
| close_high_closure | 77 / 1 | 0 / 0 | 0 / 0 | 否 |
| extended_balanced | 42 / 1 | 0 / 0 | 62 / 1 | 否 |
| extended_high_closure | 0 / 0 | 0 / 0 | 34 / 1 | 否 |

每个 family 都未满足任一 opponent 下的 `>=8` qualifying episode、`>=160` valid target step、双镜像、三高度和零 contract failure 的联合门。因此唯一有效结论是：

```text
v4_no_cross_opponent_runtime_feasible_family
```

V4 不解锁单技能 pilot、四共享技能、combat finetune、高层 PPO 或论文级 heldout 结论。

## Raw-SI 归因

V4 的 target handoff 同时依赖环境字段 `first_pass_complete`、neutral dynamic state、post-merge phase、有效预测和 validity mask。逐 episode 只读审计得到：

| 对手 | raw range-rate 符号翻转 | 环境 first-pass | 实际 handoff | raw first-pass 后存在 neutral + prediction-valid window |
|---|---:|---:|---:|---:|
| expert | 40 / 48 | 6 / 48 | 5 / 48 | 25 / 48 |
| end-to-end | 42 / 48 | 0 / 48 | 0 / 48 | 16 / 48 |
| independent PPO/VPP | 38 / 48 | 5 / 48 | 4 / 48 | 24 / 48 |

这证明两点：

1. end-to-end 的零 qualifying 不是“从未发生物理 first pass”。在 42/48 条 episode 中，raw-SI `range_rate` 从闭合转为张开，但环境字段始终为 false，阻断了 handoff。
2. V4 也不能据此声称完整跨对手数据已建立。raw event 后的 neutral window 对 end-to-end 只有 16/48，且在既定 family 内未达双镜像/三高度和 8 episode 门。

因此 V4 是**collector phase-semantics mismatch 的正向诊断证据**，但仍是运行时可行性 gate 的负结果；它既不能支持训练，也不能被改写为对五态势技能库的能力结论。

## 后续决策

不得重跑或补改 V4。下一条研发线必须使用新的 Source ID、全新且与 V4 不重叠的五态势 manifest，并将 first-pass 定义为预注册 raw-SI event（连续闭合后稳定张开），而非环境特定的 `first_pass_complete` 标志。

V5 的第一任务不是训练，而是验证五种初始态势在三对手下的 raw-SI phase event 与动态 taxonomy sidecar 是否形成连续、真实的 66-D -> 3-D VPP 数据契约。只有该任务通过，才有资格启动共享技能的几何预训练。
