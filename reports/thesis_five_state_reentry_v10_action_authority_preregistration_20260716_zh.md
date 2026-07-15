# V10 Action-Authority Probe 遥测合同预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10`
**前序：** V9 仅为 JSON-safe telemetry 实现失败，未产生可用 episode 或 action 结果。

V10 以全新的物理不相交 12-scenario manifest 重做**同一非学习研究问题**：冻结 head-on run-in 后，
对其 3-D VPP action 施加六个预注册的 `+/-0.25` 单轴 delta，并与无扰动 reference 比较 20 步
`range_extension` intent-loss、动作实际差、VPP/guidance/PID 响应和 ego crash/OOB。

V10 不把后端未公开的 `vp_vertical_bias_m` 伪造成 NaN 或零。该字段被显式标注为
`backend_unobserved`；vertical action 的机制证据只能来自实际 applied action、观测高度差、
`n_z` command 和 actual response。forward/lateral VPP bias、`n_z` command/actual、AoA 及 guidance
command 三项必须有限，任一缺失均 fail-closed。

训练、奖励、PPO、路由、VPP/guidance/PID 参数、对手、snapshot/future-state injection 和 history
padding 全部禁止。通过仅允许起草新的 action-conditioned warm-start 机制；失败则禁止以继续调 reward
或追加步数替代对当前动作接口的否定证据。
