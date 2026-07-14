# Global-Advantage P1 R1 实现失败记录

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-RUNIN-TELEMETRY-R1`
**状态：** `implementation_failure_not_experimental_evidence`

## 事实

R1 在第一条 reference rollout 完成后，向新 output root 写入 raw telemetry 时遇到 `ndarray is not JSON serializable`。失败发生在记录器序列化层，尚未成功落盘任何 episode artifact，`completed_episode_count = 0`。

保留的失败目录为：

`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_runin_telemetry_r1`

其中 `runner_failure_manifest.json` 只记录该实现错误和零完成 episode。R1 不构成 JSBSim、控制器、P3 encoder、五态势 taxonomy 或性能的正/负证据，也不改变已有 pilot 的 `safety_or_contract_no_go`。

## 后续边界

- R1 输出保留，不删除、不覆盖、不纳入统计。
- R2 修复 JSON serialization 后使用新的 Source ID 和全新 output root。
- R2 仍为非学习 P1 reproducibility/telemetry preflight，不训练、不调参、不运行 formal held-out。
