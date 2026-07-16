# Temporal-Corridor Transfer V2 工程 Smoke Gate（2026-07-16）

## 结论

V2 的工程 build-smoke 已通过，可作为建立一次性正式授权的**工程必要条件**。
它证明修复后的 config 可以通过 V16 record-only builder 对三个冻结对手构造 strict-JSBSim
环境，并在 predictor warm-up 后读出有限的预测、VPP、guidance 与实际过载遥测。

它不构成 temporal-corridor 的性能证据，也不构成 phase-reachability 或 handoff 可达性证据。
因此通过 smoke 只允许授权一次正式 V2 矩阵，不能解锁训练、调参或论文结论。

## 可复核执行

| 项目 | 值 |
|---|---|
| 工程 smoke ID | `ENGINEERING-TEMPORAL-CORRIDOR-TRANSFER-V2-BUILD-SMOKE-NOT-A-SOURCE-ID` |
| 受测 Source | `THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2` |
| 实现 SHA | `05a944370b21e4387db97b7a74719d5953041c8b` |
| 输出目录 | `outputs/diagnostics/temporal_corridor_transfer_v2_runtime_smoke/smoke_20260716_runtime_mapping_v3` |
| 场景 | V2 manifest 的第一个冻结场景，仅作 engineering reset |
| 对手 | expert、end-to-end、independent PPO/VPP，分开执行 |
| records | `3/3` |
| gate | `passes=true` |

每个对手均满足：`backend=jsbsim`、19-D fixed-specialist observation 有限、260 个 V16
run-in step 已执行、预测目标位置有限、`backend_fallback_occurred=false`、
`prediction_fallback=false`，且 `vp_world_offset_m`、三轴 guidance command 与 actual `n_z`
均可读且有限。

## 重要边界

三个 engineering episode 在 260-step 上限内均未达到 re-entry handoff。这不是 gate 失败，
因为本 gate 的预注册职责只是验证 runtime mapping、连续 run-in 和 telemetry 可读性；它没有
声称所选工程 reset 必须进入目标 phase。这个事实同时意味着：V2 的 108 条正式矩阵仍必须按
预注册的 qualifying-pair 规则独立裁决，不能把 smoke 误写为 corridor 已获得控制权威的正证据。

早期 `runtime_mapping_v1` 与 `runtime_mapping_v2` 输出分别发现 predictor warm-up 缺失和
receipt JSON 序列化缺陷；两者均没有产生完整 smoke record 或任何正式 Source record。它们已由
V3 的完整 run-in 与三对手可读 receipt 闭合，且不改变 V2 的动作、场景、门槛或方法矩阵。
