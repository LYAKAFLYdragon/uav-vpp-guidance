# V18 Phase-Telemetry Audit 失败归档

## 正式决定

`THESIS-FIVE-STATE-V18-PHASE-TELEMETRY-AUDIT` 在首条 strict-JSBSim episode 的第一个
collector telemetry step 即按预注册停止规则终止。输出目录只含 `failure.json`，
`completed_records=0`；没有训练、没有策略选择、没有性能结果、没有可用于论文的比较数据。

`failure.json` 的 SHA-256 为
`d20abdc195f11b2fd60c388e9b0cfa0df4507f4708dfb21fc486c80f97704c6e`。

## 失败归因

V18 collector 将每一个 step 的 `prediction_valid != true` 与 `prediction_fallback == true`
共同视为“fallback”。但环境在 predictor history 尚未形成时初始化为：

```text
prediction_valid = false
prediction_fallback = false
```

这表示**预测尚未就绪**，不是回退，也不是预测器或 JSBSim 后端失败。冻结 V17 的原始合同只在
`history == 10`、re-entry handoff 候选处要求 `prediction_valid == true` 且无 fallback；它并不要求
episode 的 warm-up 前缀每一步预测有效。

所以 V18 的“全步 prediction-valid”要求与其声明复用的 V17 连续接口不兼容。该结论只说明 V18 的
观测合同定义错误，不能归因于 end-to-end 对手、飞行器动力学、VPP、guidance、PID 或五态势架构。

## 不可执行事项

- 不修补、不重跑 V18，不替换场景，不增加记录数。
- 不将 V18 作为 V17 phase-occupancy 的反证或支持。
- 不因本失败启动 reward、skill、profile、PPO 或 combat finetune。

## 后续边界

若继续诊断 V17，必须建立新的 Source ID，并把 telemetry 分为两个明确窗口：

1. `warmup`：完整记录 `prediction_valid` 和 `prediction_fallback`，允许 valid 为 false，但禁止 fallback。
2. `analysis_ready`：仅在 10-frame history 已满且预测有效时要求完整的 phase/VPP/guidance/PID
   telemetry，并与 V17 handoff 语义对齐。

这将成为 V19 的独立设计，不会改写或复用 V18 的 gate。
