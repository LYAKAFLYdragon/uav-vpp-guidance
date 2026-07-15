# V9 Action-Authority Probe 实现失败归档

**Source ID：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V9`
**最终状态：** `implementation_failure_not_experimental_evidence`
**授权提交：** `e062038`
**运行实现提交：** `bd90d62875269ecb581083b1fb6f85c7d5168d8b`

## 发生了什么

V9 通过了静态 preflight：12 个物理不相交场景、7 个冻结动作方法、3 个对手、252 条预定
strict-JSBSim record、输出根为空、约 290 GB 可用磁盘。一次性授权运行启动了 JSBSim，但在
第一个内存 record 写入前即停止：

```text
ValueError: Out of range float values are not JSON compliant: nan
```

输出根仅有 `authorized_config.yaml` 与 `failure.json`；没有 episode JSON、gate、run manifest、
checkpoint、训练 transition 或 heldout 结果。`failure.json` 的 `completed_records=1` 是 record
已在内存列表 append、但尚未成功持久化时的计数，不能当作任何实验样本或性能结果。

## 精确归因

V9 的 JSON writer 正确使用 `allow_nan=false`，但 runner 在缺失时把六项飞控 telemetry 的默认值
写为 `math.nan`。已完成的 V4 reward-free runtime evidence（144 episode、36,689 step）显示：

| 字段 | V4 可用性 | V9 影响 |
|---|---|---|
| `vp_forward_bias_m` | 全部有限 | 可作为必需 response telemetry |
| `vp_lateral_bias_m` | 全部有限 | 可作为必需 response telemetry |
| `nz_cmd` / `actual_nz_g` / `ego_attack_aoa_deg` | 全部有限 | 可作为必需 guidance/PID telemetry |
| `vp_vertical_bias_m` | 全部为 `null` | 不能伪造为 `NaN` 或宣称存在直接垂直 VPP bias telemetry |
| `guidance_command` | 均为非空 mapping | 可保存为结构化 command receipt |

因此，这次停止反映 **V9 记录器没有把“后端未暴露 vertical-bias 字段”编码成可审计的缺失语义**。
它不是动作方向无效、PPO 训练不足、JSBSim fallback、VPP/guidance/PID 不稳定、对手差异或安全失败。

## 不可做事项

1. 不重跑 V9，不修改 V9 manifest、动作 grid、门槛、奖励、VPP、guidance 或 PID。
2. 不从内存计数或失败前 JSBSim 启动日志推导任何动作可控性或性能结论。
3. 不把 `vp_vertical_bias_m=null` 改写为“垂直动作无效”。它只说明当前 backend telemetry 不直接公开该内部量。

## 唯一后续方向

若继续研究动作可控性，必须建立新 Source ID 的 V10：使用全新且物理不相交的 manifest，
保持同一非学习、冻结动作 grid 的研究问题，但把 telemetry contract 改为：

- 所有 JSON 使用显式 finite/value-present receipt，禁止把缺失字段写为 NaN。
- forward/lateral VPP bias、`n_z` command/actual、AoA 和 guidance receipt 必须有限；否则 V10 fail-closed。
- `vp_vertical_bias_m` 明确标为 backend-unobserved；垂直轴只通过**实际应用动作、own altitude / altitude-difference 响应和 `n_z` response**审计。
- V10 仍只回答“当前 3-D VPP 接口是否存在安全的局部几何动作权威性”，不解锁训练或论文性能声明。
