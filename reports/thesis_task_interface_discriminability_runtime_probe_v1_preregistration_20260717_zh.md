# Task-Interface Runtime Probe V1 预注册

- Probe ID：`ENGINEERING-TASK-INTERFACE-DISCRIMINABILITY-RUNTIME-PROBE-V1-NOT-A-PAPER-SOURCE-ID`
- 类型：一次性、非学习、工程性接口可判别性 probe；不是论文正式 Source。
- 前置条件：静态 preflight V1 已通过，且其 receipt 的 config/manifest hash 必须仍与当前设计匹配。

## 冻结矩阵

- 4 条物理 reset，每条复用相同位置、速度、航向、高度与 seed，分别赋予 legacy `head_on` 与 `crossing_feasible` context。
- 两种方法：M3 `canonical_ppo`，M4 `ppo_unbootstrapped_crossing`，共用同一 commander checkpoint。
- 三个对手分别运行：expert、end-to-end、independent PPO/VPP。
- 总数：`4 x 2 contexts x 2 methods x 3 opponents = 48`；每条恰 20 个高层步，`dt=0.2 s`。

## 验收

每个 record 必须有 strict-JSBSim、无 backend fallback、19-D initial observation、物理 reset receipt、20 步 VPP/guidance/AoA/`n_z` telemetry，以及第 10 步起无 prediction fallback。

同一 physical pair 在同一方法/对手下，两个 context 的 reset receipt 与 19-D vector 必须完全相同，唯一向量差异只能是 `is_crossing: 0 -> 1`。

在 expert 与 end-to-end 中，至少各 3/4 个 crossing-context pair 必须使 M3/M4 的首个 effective mode 或第一条 VPP action 不同。M3 必须记录 active bootstrap，M4 必须记录 inactive bootstrap。

## Stop Rule

- 若任何 telemetry、配对等价性或完整性条件失败，整个 probe 作为 engineering-negative evidence 冻结。
- 若任一主对手的有效差异少于 3 个 pair，停止后续 M3/M4 正式消融；结论仅为该接口在本包线内不可判别。
- 即使通过，本 probe 也不得支持胜率、优势、泛化或战术标签正确性的声明；它只允许单独设计新的正式 Source。
- 禁止调整场景、标签、checkpoint、奖励、VPP、制导、PID、对手、seed 或阈值后重跑。
