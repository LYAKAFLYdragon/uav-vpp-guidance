# 局部动作控制权威图谱 V1 预注册

**Source ID：** `THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V1`  
**状态：** design only；未实现、未授权、未运行  
**性质：** 非学习、非奖励、非性能排序的分支因果诊断；不可作为论文主结果

## 1. 问题与证据位置

V15 与 V2 已排除两种短时确定性 corridor 在当前冻结接口下取得跨对手实用收益的可能，
但没有直接回答一个更基础的问题：在真实进入 re-entry 的连续状态上，3-D VPP action head
是否存在足够强、可重复且安全的局部几何作用方向。

本 Source 只回答该局部问题。它不训练 policy、不改变 reward、VPP、guidance、PID、预测器、
对手或任务动力学，不比较 win rate，也不声称全局轨迹或任务性能改善。

## 2. 冻结候选状态

候选仅来自 V15/V2 的 `frozen_fixed_head_on` 原始 episode，且历史 record 同时满足
`handoff_reached=true`、`contract_passed=true`、`window_steps=20`：

| 历史 Source | expert | end-to-end | independent PPO/VPP | 合计 |
|---|---:|---:|---:|---:|
| V15 | 7 | 3 | 6 | 16 |
| V2 | 5 | 2 | 5 | 12 |
| 合计 | 12 | 5 | 11 | 28 |

新的 cohort manifest 必须逐项保存：历史 Source ID、原始 episode 相对路径及 SHA-256、
scenario signature、seed、opponent、历史 handoff hash、handoff step（若历史 artifact 提供）
和原始 config/manifest 哈希。三个 opponent 永远分开报告，不合并统计量。

## 3. 连续重放与动作分支

不使用 snapshot restore、future-state injection、手工 post-merge reset 或轨迹复用。JSBSim
封装没有足以同时恢复飞行器、预测器、滤波器、PID 和 opponent 内部状态的可靠完整恢复接口。

每个 branch 都从原始 Source 的合法 reset、相同 scenario、相同 seed 和 fixed-head-on policy
完整连续 run-in。只有当新 run-in 达到 re-entry handoff 且其 source-specific handoff hash 与
历史 artifact 完全相等时，才允许开始 20-step local window；否则该 state 标为
`replay_mismatch`，不注入动作且阻断对应 cell 的控制权威结论。

固定方法矩阵为每个有效 state 七条独立连续重放：

| Method | Handoff 后 step 0--2 | step 3--19 |
|---|---|---|
| reference | frozen fixed-head-on action | frozen fixed-head-on action |
| longitudinal plus / minus | reference action plus / minus `0.08` on axis 0 | reference action |
| lateral plus / minus | reference action plus / minus `0.08` on axis 1 | reference action |
| vertical plus / minus | reference action plus / minus `0.08` on axis 2 | reference action |

动作先计算 reference，再执行 `clip(reference +/- 0.08 e_i, -1, 1)`。任何 pulse step clipping、
non-finite action、backend/prediction/checkpoint fallback、异常 reset 或 telemetry 缺失均使该
axis-state pair 无效；不得改小/改大 pulse 后重跑本 Source。理论计划量为 `28 x 7 = 196` 条。

## 4. 指标与局部敏感度

每一 local step 记录 AA、ATA、range、range-rate、specific energy height、own altitude、
action clipping、VPP world offset、guidance command、actual `n_z`、phase 与 terminal reason。
以第 20 step 与同 state reference branch 的差作为局部作用量，不把不同 state 的初始几何混合。

对动作轴 `i` 与几何指标 `j`，定义反对称局部响应：

```text
S(i, j) = (m_plus(i, j, H=20) - m_minus(i, j, H=20)) / (2 * scale(j))
```

其中 angle scale 为 `180 deg`，range 为 `3000 m`，range-rate 为 `200 m/s`，specific energy
height 与 altitude 均为 `1000 m`。该定义测量“该轴能否在当前动态状态改变几何”，而非预设
哪一个方向必然战术更优。

## 5. 预注册 Gate 与决策

一个 axis-metric direction 只有同时满足以下条件才称为 `local_authority_candidate`：

1. 每个相关 opponent 至少有 2 个 replay-hash-matched、无 clipping 的对称 pair。
2. 每个相关 opponent 内，`median(abs(S)) >= 0.01`，且至少 75% state pair 的符号一致。
3. 至少两个 opponent 的 median sign 相同；不得以 pooled opponent 平均替代此条件。
4. 在这些 pair 上，无 candidate-only ego crash/OOB、无 fallback，且 20-step telemetry 全有限。

若至少一个 axis-metric candidate 在两个 opponent 上通过，下一步仅允许起草新的
state-conditioned compiler 预注册：compiler 必须由实时几何选择方向，且与本图谱的有效 axis
和符号有可追溯对应。若没有 candidate，正式结论为“当前冻结 VPP action envelope 尚未显示
跨对手、可供 compiler 利用的局部控制方向”，停止 temporal-corridor/共享技能训练线。

## 6. 解释边界

即使 gate 通过，它也只证明局部几何可控性，不证明 win-rate 改善、策略可学习性或普适五态势
泛化。即使 gate 不通过，也只排除本 Source 的 `0.08`、三步脉冲和二十步观察窗，不能声称
所有时间扩展接口都无效。两种结果都应进入尝试账本和硕士论文的方法边界章节，而不改写 AST
主稿的双技能层级 VPP 结论。
