# V20 Warm-up-Fallback Telemetry 设计

V20 仅为解释 V17 phase occupancy 而设计，不训练、不调参、不改变 VPP、guidance、PID、policy 或场景。
它复用 V17 24-scene × 3-opponent 包线，以便逐对手复现 handoff 数。

窗口语义固定为：

- `warmup`：history 小于 10 或 prediction invalid。记录 `prediction_fallback` 与 reason，允许
  `fallback_to_current_target`，但该窗口不计入 analysis gate。
- `analysis_ready`：history 等于 10、prediction valid 且 no-fallback。完整记录双方能量、taxonomy、
  phase、VPP/guidance/PID telemetry；任意 fallback 立即失败。

三对手各 24 条 record 必须精确复现 V17 的 handoff 与有效 path-context step 数。若 V20 仍无法在
analysis-ready 窗口取得每对手至少 20 个完整 step，则冻结为“现有 V17 接口无法建立可解释连续数据”
的负证据，停止这条路径/奖励/技能训练线。
