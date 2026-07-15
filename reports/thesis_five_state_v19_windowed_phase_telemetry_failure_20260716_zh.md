# V19 Windowed Telemetry 失败归档

V19 在第一条 strict-JSBSim episode 的 warm-up 阶段停止，`completed_records=0`。它没有训练、没有
性能输出，也不能用于比较或论文结论。

环境实现表明，当 `anchor_mode=predicted_target` 但预测位置尚不可用时，会将 VPP anchor 回退为
`current_target`，并设置 `prediction_fallback=true`。这在 predictor warm-up 是预期的运行语义；V17
只在 history 已满且 re-entry handoff 处要求 prediction valid/no-fallback。V19 虽允许 warm-up 的
`prediction_valid=false`，但仍错误禁止 warm-up fallback，故仍不兼容冻结接口。

按 stop rule，不修补或重跑 V19。唯一允许的后续是新的非学习 Source ID：warm-up 记录 fallback
及其原因但不参与 analysis gate；analysis-ready 后才严格禁止 fallback。该修正不能被解释为放宽
V17/V18/V19 的性能门，因为它只改变 telemetry 分区语义。
