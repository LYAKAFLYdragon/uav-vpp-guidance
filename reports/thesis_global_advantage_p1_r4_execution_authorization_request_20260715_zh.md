# P1 R4 一次性执行授权请求

**状态：** `not_authorized`
**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R4-FRESH-ENV-REPRO-V1`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r4_fresh_environment`

## R3 后的唯一修复

R3 的零 episode 实现失败来自 `CombatHPManager.last_info.combat_time_to_kill=NaN` 的具名业务语义。R4 仅增加此值的显式 snapshot 表示：`not_terminal_or_not_killed`。其它非有限值仍使 snapshot fail-closed；训练、奖励、VPP、guidance、PID、对手、场景与 checkpoint 均不变。

## 授权前条件

- [ ] R4 implementation 与测试已在新的 clean SHA 冻结。
- [ ] R4 config 的 source/checkpoint/code-file SHA 已填写并独立复核。
- [ ] R4 output root 不存在，磁盘空闲空间不少于 120 GB。
- [ ] 一次性授权仅允许 30 dev 场景 x 3 对手 x 3 repeat 的非学习复核。

R4 的通过门仍为 90/90 cell 的完整 reset envelope、首个双方动作、trajectory、boundary 与 terminal 等价；任何失败都为 `runin_protocol_not_reproducible_do_not_train`，不允许 rerun 或训练。
