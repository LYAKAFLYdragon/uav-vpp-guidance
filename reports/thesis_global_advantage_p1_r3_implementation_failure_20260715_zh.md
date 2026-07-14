# P1 R3 实现失败归档

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R3-FRESH-ENV-REPRO-V1`
**状态：** `implementation_failure_not_experimental_evidence`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r3_fresh_environment`

## 事实

R3 在第一个 `expert / advantage / own_below / negative` launch 的 reset-runtime snapshot 阶段终止，`completed_episode_count=0`。保留的 `runner_failure_manifest.json` SHA-256 为：

`044f13a9d2c0379b1dc1bb5ca65af7743fd1d1c905a8b1a530af8ccabe4d877a`

失败原因是 `CombatHPManager.last_info.combat_time_to_kill` 使用 `NaN` 表示“尚未终局或未击杀”，而 R3 的通用 finite-only serializer 尚未为这一个具名、已定义的语义哨兵提供编码。

## 解释边界

- R3 未完成任何 episode，不能构成 JSBSim、策略、对手、VPP、P3 encoder 或性能的正/负证据。
- R3 输出根保留，不删除、不覆盖、不重跑。
- 修复仅将这个具名 `NaN` 编码为 `not_terminal_or_not_killed`；未放宽其它 NaN/Inf 的 fail-closed 规则。
- 纠正后的执行使用新 Source ID `THESIS-GLOBAL-ADVANTAGE-V1-P1-R4-FRESH-ENV-REPRO-V1` 与全新输出根，仍须重新冻结并授权。
