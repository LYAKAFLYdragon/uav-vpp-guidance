# Terminal-Reward Pilot V1 训练后完成审计与正式决定

## 正式结论

`THESIS-REENTRY-TERMINAL-REWARD-PILOT-V1` 的 20,000-step Dev-only pilot 已完成，
不是一次科学实验或安全门失败。`failure.json` 被归因为**训练完成后的 stdout 输出通道异常**。
但该 pilot 只满足安全门，**没有满足预注册的实用改善门**；因此 `heldout24` 继续锁定，
同一 Source ID 下不得加步数、调奖励或重跑。

这是一条开发集负/不确定证据：在本次冻结的 continuous re-entry、66-D -> 3-D VPP、
`range_extension` profile、三对手和 20,000 步预算下，末步 terminal reward 没有提供足够大的
几何准备改善。它不证明终端奖励在所有任务中无效，也不构成五态势系统的正式性能结论。

## 完成性与异常归因

审计输入是只读的结果目录
`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/thesis_reentry_terminal_reward_pilot_v1`，
独立重算结果保存在
`reports/thesis_reentry_terminal_reward_pilot_v1_post_completion_audit_20260715.json`。

| 检查项 | 结果 |
|---|---|
| 授权运行 SHA | `2be1888bb903c0950fdac9551c39bf4b781d5b86` |
| 训练记录 | 20,000 / 20,000 steps，351 attempts，未触发 safety stop |
| Dev 记录 | `step_10000.json` 与 `step_20000.json` 都存在且逐对重算一致 |
| checkpoint 选择 | minimax Dev 规则选择 20k；`best.pt` 与 `step_20000.pt` SHA-256 相同 |
| heldout | `heldout_executed=false`，未执行 |
| 时间顺序 | `training_summary.json` 先于 `run_manifest.json`；`failure.json` 最后写入 |
| `failure.json` | 仅含 `OSError(22, 'Invalid argument')` |

历史运行版本的 runner 先写 `training_summary.json`，再写 `run_manifest.json`，随后才在同一
`try` 块中打印完整 JSON。该 stdout 写入失败会落入异常处理并生成 `failure.json`。因此，
它不能覆盖已经落盘且可独立复核的训练、Dev 评估和 checkpoint 选择事实。原始
`failure.json` 保留，不删除、不覆盖，也不作为性能结果使用。

为避免未来已完成任务被同类外部管道错误误标为失败，runner 现将最佳努力 stdout 输出移到
持久化异常边界之外，并仅忽略该输出阶段的 `OSError`。此修复不回写、不改变本次运行的
历史 SHA、配置、结果或预注册结论。

## 独立 Dev 重算

指标为 candidate 相对 frozen fixed-head-on 的 paired mean `intent_loss_auc20` delta，负值较好。
三个对手分别报告，不合并统计量。

| Checkpoint | Expert | End-to-end | Independent PPO/VPP | 最差对手分数 | Safety |
|---|---:|---:|---:|---:|---|
| 10k | -0.000208 (N=6) | -0.000923 (N=4) | +0.000501 (N=6) | +0.000501 | pass |
| 20k / best | +0.000014 (N=6) | -0.000974 (N=4) | +0.000309 (N=6) | +0.000309 | pass |

20k 在冻结 minimax 选择规则下优于 10k，因此 `best.pt` 指向 20k checkpoint。三个对手的
ego crash/OOB 相对 fixed-head-on 的 delta 均为 `0.0`，满足 `<= +0.05` safety 门。

预注册的 practical-improvement 门为每个对手 `delta <= -0.05`，且至少两个对手必须通过。
本次没有任何对手通过该门，故不具备运行 heldout24 的资格。

## 奖励语义核对

本结论沿用 V1 已冻结的末步奖励：胜利 `+1.0`、失败 `-1.0`、平局 `-0.25`，我方
crash/OOB 额外 `-1.0`，故我方 crash/OOB 总 terminal term 为 `-2.0`。对方 crash/OOB 和
`target_killed` 均为合法胜利，只获得 `+1.0`，绝不误加惩罚。该语义由已通过的非学习
preflight 验证，本审计不修改任何 reward 数值。

## Go/No-Go 决定

- **No-go**：不得执行 V1 heldout24，不得继续这个 Source ID 的训练、奖励调节、场景替换或 rerun。
- **保留**：保留 V8 连续相位合同、V1 preflight 和本次 Dev 负证据，作为后续机制选择的边界条件。
- **不允许的推断**：不能把“安全”写成“有效”，也不能把一次 Dev-only 结果写成跨对手、跨态势或全局性能结论。
- **后续方向**：回到主五态势路线的独立机制/技能库假设；任何新 pilot 必须使用新 Source ID、全新 train/dev/heldout 划分和独立预注册，而非在 V1 上继续调参。
