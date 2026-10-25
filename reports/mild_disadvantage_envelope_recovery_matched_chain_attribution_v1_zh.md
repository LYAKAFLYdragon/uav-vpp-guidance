# Matched-contract 12 条连续轨迹链路归因

## 1. 分析范围

- 分析标识：`MILD-DISADVANTAGE-MATCHED-CHAIN-ATTRIBUTION-READONLY-V1-20260903`
- 输入 Source ID：`THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-CONTRACT-PROBE-V1-20260903`
- 输入目录：`E:\uav-vpp-guidance\outputs\mild_disadvantage_envelope_recovery_v1\matched_probe_v1_retry`
- 轨迹数量：12 条
- 分析方式：只读，不重跑 JSBSim，不训练，不修改任何控制参数。
- 观察时点：从每条轨迹的第 0 步计算 4、12、48 步后的变化；第 4、12 步位于 crossing 恢复段，第 48 步是 48 步预算耗尽后的首次 H0 交接点。

## 2. 链路可评估性

| 链路层级 | 当前证据 | 判定 |
|---|---|---|
| crossing 编译动作 | `action_0`、`action_1`、`action_2` | 12/12 条轨迹均有有限值，已观测 |
| 实际 VPP 位置或偏置 | 未发现实际 VPP 数值字段 | 不可评估 |
| 制导数值指令 | 未发现过载、滚转速率和油门数值指令字段 | 不可评估 |
| 滚转状态响应 | 未发现滚转角或滚转速率字段 | 不可评估 |
| 几何结果 | `range_m`、`range_rate_mps`、`ata_deg`、`aa_deg`、`altitude_m` | 可作描述性分析 |

> 重要边界：现有文件只能证明 crossing 动作被记录，以及几何量随后发生了变化；不能证明动作已经通过 VPP、制导或滚转环节，也不能据此计算各层的响应保真率或延迟。

## 3. 4/12/48 步几何变化

下表为同一条轨迹内的端点差值，定义为“第 N 步值减去第 0 步值”。这些差值不是动作干预的因果效应，因为没有同状态的无干预轨迹。

| 对手 | 初始点 | 时域 | 距离变化（m） | 距离率变化（m/s） | ATA 变化（°） | AA 变化（°） | 高度变化（m） |
|---|---|---:|---:|---:|---:|---:|---:|
| end_to_end | x625_y200 | 4 | -271.479 | 89.593 | 23.595 | 23.661 | -0.255 |
| end_to_end | x625_y200 | 12 | -83.544 | 740.655 | 129.148 | 129.309 | 8.107 |
| end_to_end | x625_y200 | 48 | 3004.955 | 798.329 | 121.516 | 139.304 | 65.142 |
| end_to_end | x750_y250 | 4 | -278.745 | 55.818 | 16.566 | 16.615 | 0.026 |
| end_to_end | x750_y250 | 12 | -294.231 | 677.232 | 113.671 | 115.113 | 11.282 |
| end_to_end | x750_y250 | 48 | 2725.712 | 795.775 | 121.366 | 140.009 | 75.047 |
| expert | x625_y200 | 4 | -271.773 | 89.499 | 23.815 | 23.758 | -0.128 |
| expert | x625_y200 | 12 | -80.289 | 744.551 | 129.791 | 129.411 | 10.024 |
| expert | x625_y200 | 48 | 3088.689 | 826.214 | 142.594 | 135.354 | 83.265 |
| expert | x750_y250 | 4 | -279.140 | 55.265 | 16.736 | 16.678 | 0.123 |
| expert | x750_y250 | 12 | -289.805 | 686.199 | 115.659 | 115.204 | 12.933 |
| expert | x750_y250 | 48 | 2829.703 | 826.905 | 141.578 | 136.565 | 94.658 |

## 4. 归因结论

1. crossing 原始三轴动作在 12 条轨迹中均被记录，且第 0--47 步均保持 crossing 模式；第 48 步统一进入 H0，说明恢复预算耗尽后发生了预定交接，而不是没有调用 crossing 专家。
2. 现有轨迹文件没有实际 VPP、制导数值指令和滚转状态字段，故差异消失层不能进一步定位到 VPP、制导或滚转环节。
3. 距离、距离率、ATA、AA 和高度的 4/12/48 步变化可以作为结果演化记录，但不能单独解释为 crossing 动作造成的几何改善或恶化。
4. 终端结果仍受对手目标运动、制导闭环和 JSBSim 物理响应共同影响；仅凭这 12 条文件不能建立跨对手的局部因果控制方向。

## 5. 决策

> 当前最准确的结论是：crossing 动作层和几何结果层可观测，但中间执行链不可评估。因此不能据此授权继续训练恢复技能、修改低层接口或扩大 PPO。若要继续做一次最小补强，应只补采同一接口下的实际 VPP、数值制导指令、滚转角/滚转速率，并保留动作前状态哈希；补齐后再做链路延迟和差异保留率分析。

## 6. 文件

- 机器可读归因：`E:\uav-vpp-guidance\outputs\mild_disadvantage_envelope_recovery_v1\matched_probe_v1_retry/chain_attribution_v1.json`
- 时域差值：`E:\uav-vpp-guidance\outputs\mild_disadvantage_envelope_recovery_v1\matched_probe_v1_retry/chain_attribution_horizon_deltas.csv`
- 本报告：`E:\uav-vpp-guidance\reports\mild_disadvantage_envelope_recovery_matched_chain_attribution_v1_zh.md`
