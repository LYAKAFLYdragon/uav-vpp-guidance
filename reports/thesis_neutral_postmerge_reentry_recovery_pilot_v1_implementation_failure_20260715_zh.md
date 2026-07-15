# Neutral Post-Merge Reentry-Recovery Pilot V1 实现失败归档

**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1`
**最终状态：** `implementation_failure_not_experimental_evidence`
**执行授权提交：** `5dad3e8444d4092a1204b200ec7fbdcf183bec37`
**执行实现祖先：** `c50c554a659c9d68a68c1e01cb3b2afca6c39e3b`

## 发生了什么

一次性授权执行通过了授权 preflight，并启动 strict JSBSim。首个 dev baseline episode 在 runner 写入第一份 telemetry ledger 时终止：

```text
KeyError('scenario_signature')
```

新 manifest 的唯一配对键名为 `metadata.geometry_cell_id`，而 runner 错误读取了并不存在的 `metadata.scenario_signature`。这是 metadata contract 未在真实 manifest 上进行 round-trip 测试造成的实现错误。

## 证据边界

- output root：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/neutral_postmerge_reentry_recovery_pilot_v1`。
- `runner_failure_manifest.json` SHA-256：`2705e00003b542d676fc27523c53e6acd1e36e46a31805e83440c1456024e2f8`。
- root 只含 `authorization_preflight.json`、`resolved_config.json` 和 `runner_failure_manifest.json`。
- 未生成 checkpoint、训练 summary、dev record、heldout record 或逐步 telemetry；因此 `completed_episode_count=0`。
- JSBSim 的启动日志仅证明后端开始初始化，不能证明任何方法性能、物理安全或技能有效性。

## 决策

该 Source ID 已消费并冻结，不得修补后重跑，不得覆盖 output root，不得将它作为新技能失败、对手强度或 safety 结果。

若继续该机制，下一步必须先独立设计 `V2`：修正 metadata pairing contract、添加真实 manifest round-trip test、使用新的 output root，并按新的 Source ID 重新进行预注册与一次性授权。是否沿用 V1 的 dev/heldout manifest 必须在 V2 预注册时明确说明，因为 V1 未产生任何性能观察，但同源重跑仍被禁止。
