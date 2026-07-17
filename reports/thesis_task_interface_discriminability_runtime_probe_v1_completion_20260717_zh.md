# Task-Interface Runtime Probe V1 完成备忘录

## 结论

`ENGINEERING-TASK-INTERFACE-DISCRIMINABILITY-RUNTIME-PROBE-V1-NOT-A-PAPER-SOURCE-ID`
已按预注册一次性完成，并经独立只读 verifier 复核。两名主对手均为
`0/4` 个可区分 physical pair，低于预注册的每名主对手至少 `3/4` 个
pair 门槛。因此 verdict 为 **fail**：停止后续 M3/M4 正式消融；不重跑、
不调整场景、标签、checkpoint、奖励、VPP、制导、PID、对手、seed 或门槛。

这是一项 engineering-negative evidence，不是论文正式 Source，也不产生
胜率、优势、泛化或战术标签正确性的结论。

## 已执行矩阵与证据

- 输出根目录：`E:\uav-vpp-guidance-disadvantage-capability-v1-results\task_interface_discriminability_runtime_probe_v1`
- 运行 SHA：`726d4e5`。
- 48 条完整 record：`4 physical resets x 2 contexts x 2 methods x 3 opponents`。
- 每条 record 固定 20 个高层步、`dt=0.2 s`，共 960 条 raw telemetry step。
- 方法：M3 `canonical_ppo`；M4 `ppo_unbootstrapped_crossing`；共用同一 commander checkpoint。
- 对手：`expert`、`end_to_end`、`independent_ppo_vpp`，不合并统计。
- `verification.json` 由只读 verifier 从现有 `records.json` 生成；不重跑 JSBSim。

独立复核确认：48/48 条记录均为 strict JSBSim、无 backend fallback；每条
均具备有限的 19-D initial observation、raw physical reset receipt、20 步
3-D VPP/raw-position/AoA/`n_z` telemetry；prediction warm-up 与之后的有效
prediction receipt 符合预注册规则。对每个 opponent、method、physical reset，
`head_on` 与 `crossing_feasible` 的 reset 完全相同，初始向量仅
`is_crossing: 0 -> 1` 不同。

## Gate 结果

| Opponent | M3 bootstrap active / M4 inactive | 首个 effective mode 不同 | 首个 3-D VPP action 不同 | 任一行为差异 pair | 门槛 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| expert | 4/4 | 0/4 | 0/4 | 0/4 | 3/4 | fail |
| end-to-end | 4/4 | 0/4 | 0/4 | 0/4 | 3/4 | fail |
| independent PPO/VPP | 4/4 | 0/4 | 0/4 | 0/4 | 3/4 | fail |

四个 physical reset 分别为：`pressure_postmerge_short_tight_own_below_negative`、
`pressure_postmerge_short_tight_own_below_positive`、
`pressure_postmerge_medium_wide_co_altitude_negative` 和
`pressure_postmerge_medium_wide_co_altitude_positive`。所有 12 个
crossing-context M3/M4 对中，M3 bootstrap telemetry 为 active、M4 为
inactive；但首个有效 mode 均为 `crossing_specialist`，且首个 VPP action
逐值相同。因此开关虽被正确记录，却未在本包线中形成可观测的下游决策差异。

## Verifier 修复与可复核性

首次 verifier 失败来自回执读取契约，而非运行失败：runner 将 initial
observation 写在 record 顶层 `feature_names`/`vector`，而 verifier 错误地只
读取嵌套的 `initial_observation`；backend provenance 同样按 episode 顶层写入，
而非重复写入每一个 telemetry step。修复仅让 verifier 兼容这两种既有 receipt
粒度，不改变 runner、配置、策略、原始 records 或任何性能计算。

相关回归测试：

```powershell
python -m pytest tests/test_thesis_task_interface_discriminability_runtime_probe_v1.py -v --tb=short
python scripts/verify_thesis_task_interface_discriminability_runtime_probe_v1.py --output-root E:\uav-vpp-guidance-disadvantage-capability-v1-results\task_interface_discriminability_runtime_probe_v1
```

测试结果为 5 passed。第二条命令的非零退出码是预注册的 gate fail，表示
`stop_m3_m4_ablation`，不是 verifier 或 JSBSim 执行异常。

## 解释边界与后续限制

本结果只证明：在冻结的四个 reset、20-step 窗口与三对手条件下，当前 M3/M4
变体没有产生“首个 effective mode 或首个 VPP action”层面的可判别差异。它
不证明 task bit 在所有状态、后续时刻或不同 checkpoint 中永远无效，也不证明
任一方法性能更差。

按 stop rule，后续不得以该 probe 为基础开展或声称 M3/M4 模块贡献。后续研究
若需要研究任务标签或接口效应，必须从新的、独立的机制假设与 Source ID 出发，
而不能修改本 probe 后重跑。
