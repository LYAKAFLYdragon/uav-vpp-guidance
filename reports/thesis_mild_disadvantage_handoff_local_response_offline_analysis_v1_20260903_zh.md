# 劣势态势真实交会正负脉冲离线分析记录

## 1. 分析目的

本分析只读取
`outputs/mild_disadvantage_handoff_local_response_v1/` 中已经生成的条件结果和逐条件轨迹，不重新运行 JSBSim、不训练模型，也不修改控制器。分析目标是确认三轴正负脉冲是否按预注册的对称动作契约真实执行；只有契约通过后，才计算局部几何响应、反对称敏感度和跨对手方向一致性。

## 2. 输入范围

- Source ID：`THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V1-20260903`
- 真实 handoff：12 条
- 条件：基线以及三个动作轴的正、负脉冲，共 84 条
- 响应时刻：4、12、48 步
- 对手：`expert`、`end_to_end`
- 预期脉冲：单轴、幅值 `0.08`，正脉冲为 `+0.08`，负脉冲为 `-0.08`

## 3. 门禁结果

| 检查项 | 结果 |
|---|---:|
| 非基线脉冲条件 | 72 |
| 动作契约通过 | 36 |
| 动作契约失败 | 36 |
| 失败条件 | 全部为 `axis*_minus` |
| 正脉冲 | 36/36 通过 |
| 严格 JSBSim、handoff hash 和安全字段 | 原采集 verifier 报告通过 |
| 有效反对称响应分析 | 拒绝 |

逐条件审计结果见：
`outputs/mild_disadvantage_handoff_local_response_v1/offline_analysis/pulse_contract_audit.csv`。

## 4. 失败原因

逐条件轨迹显示，负脉冲被记录为负号，但实际动作差值仍为正值。例如 `axis0_minus` 的记录为：

```text
recorded_sign=-1
expected_delta_0=-0.08
actual_delta_0=+0.08
```

采集器中的 `HandoffPulsePolicy` 使用了未乘以符号的表达式：

```python
candidate = base + axis_vector * amplitude
```

因此，当前批次的 `axis*_plus` 和 `axis*_minus` 实际执行了同向动作。虽然运行链、状态哈希、后端和响应窗口满足工程门禁，但正负配对的因果含义不成立。

## 5. 结论边界

本批次不能支持以下结论：

- 任一 VPP 动作轴具有正负对称的局部几何响应；
- 某一动作轴能够稳定改变距离、距离率、攻击区角、进入角、比能或高度；
- 两个对手之间存在方向一致的局部控制权；
- 当前问题属于技能缺失、制导闭环限制或物理响应限制。

可以保留的结论只有：真实 handoff 的回放、严格 JSBSim 后端和 4/12/48 步遥测链路已经生成，但动作符号契约未通过，因此这批数据是“工程链路证据”，不是“局部响应证据”。

## 6. 后续门槛

在修正采集器并建立新的独立 Source ID 前，不计算或引用当前批次的反对称敏感度、热力图和跨对手方向一致性。修正后的批次必须逐条件证明：

1. `axis*_plus` 的实际单轴差值为 `+0.08`；
2. `axis*_minus` 的实际单轴差值为 `-0.08`；
3. handoff hash、strict JSBSim、无 fallback、无 reset、无裁剪和 4/12/48 步响应继续通过；
4. 上述契约全部通过后，才允许进行离线正负响应和跨对手比较。

截至本分析完成，未启动新技能训练、编译器运行时 probe、PPO 扩展或参数调整。
