# Temporal-Corridor Physical Transfer V2 结果归档（2026-07-16）

## 结论

`THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2` 已在授权 SHA
`0c8c2ef40422445bf47b59513427d8d02a3411c5` 上完成唯一一次非学习执行。全部 `108/108`
strict-JSBSim records 成功写入，独立 verifier 重放结果与 runner gate 完全一致；但没有任何
temporal-corridor candidate 通过预注册门槛。

因此 V2 是**有效的机制负证据**：在当前冻结 V16 run-in/phase/telemetry 链、fixed head-on
specialist、20-step window 与 moderate left/right corridor 下，接口物理可执行且没有新增我方
crash/OOB，却没有建立跨对手的实用六维局部控制权威。不得在 V2 下调整 corridor extent、
slew、奖励、场景、窗口、对手或训练步数，也不得启动技能训练。

## 可复核性

| 项目 | 结果 |
|---|---|
| Source ID | `THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2` |
| 输出根 | `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/temporal_corridor_transfer_v2` |
| 方法矩阵 | fixed head-on / moderate-left / moderate-right x 3 opponents x 12 scenarios |
| 完成记录 | `108/108` |
| runner gate | `passes=false`, candidates `[]` |
| 独立 verifier | `108` records，reported gate matches recomputed `true` |
| JSBSim / fallback | 全部 contract-passed；无 backend 或 prediction fallback |
| safety | 三个 candidate x opponent cell 的 ego crash/OOB delta 均为 `0.0` |
| compiler envelope | hard-bound steps 均为 `0`；平均投影残差为 `0--2.859 m`，低于 `25 m` 门槛 |

## 失败归因

| 对手 | qualifying handoff pair | moderate-left weighted delta | moderate-right weighted delta | 主要阻塞 |
|---|---:|---:|---:|---|
| expert | 5 / 12 | +0.000207 | -0.000164 | 改善远小于 `-0.01`；0 个改善分量 |
| end-to-end | 2 / 12 | +0.000639 | -0.001817 | 覆盖不足（门槛为 4），且改善不足 |
| independent PPO/VPP | 5 / 12 | -0.000148 | +0.000480 | 改善远小于 `-0.01`；0 个改善分量 |

这不是 action clipping 或安全 stop 导致的失败：corridor action 被物理链接受，terminal mix 也在
三个方法间完全相同。每个对手的 12 条结果分别保持为 expert `11 timeout HP advantage + 1 target
crash/OOB`、end-to-end `11 timeout HP advantage + 1 target killed`、independent PPO/VPP
`9 timeout HP advantage + 2 target crash/OOB + 1 timeout HP disadvantage`。

最合理的边界内解释是：在可获得 handoff 的少数轨迹上，moderate corridor 的作用量虽可被执行，
却没有转化为预注册定义的几何目标改善；而 end-to-end 的连续 re-entry/handoff 覆盖本身仍不足。
这不能证明任何更强的时间扩展轨迹一定无效，也不能归因于训练不足；它只排除了本 Source 的
bounded moderate-left/right compiler 作为跨对手六维控制方向。

## 后续决策

V2 到此关闭。最小且诚实的后续动作不是训练或重复 corridor，而是把 V15 与 V2 合并进尝试账本：
二者均已排除短时确定性 corridor 在当前冻结执行接口中的实用收益。若仍继续研究，必须先提出
新的、可预先验证的 action-envelope/interface 假设，并以新的 Source ID、独立 manifest、非学习
可控性 gate 开始；否则应停止该五态势 temporal-corridor 分支，把结果作为研究边界而非继续调参。
