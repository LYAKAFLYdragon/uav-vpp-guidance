# P2-B6 Opponent-Conditional Reachability 完成报告

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-OPPONENT-CONDITIONAL-REACHABILITY-B6-R1`
**结论：** `b6_pilot_input_candidate_identified`
**执行状态：** `completed_once_execution_closed`

## 1. 执行与完整性

- 60 个预注册五态势场景 x 3 frozen opponent 已一次性完成，共 180 条 strict-JSBSim record。
- 所有 180 条 record 的 scenario application、raw-SI phase replay 和 step-0 `pre_merge` 语义均通过；180 个逐 episode artifact 均在 run manifest 中列出 SHA-256。
- Gate SHA-256：`dddc31dd385584aa41ea0999d9c8fac73c250a3476f24b96d0ff00b82a4d2e11`。
- Run-manifest SHA-256：`e8ffb640b37f99930e252b1fb35b5785904d5654f2a1a973fc11c0d0c4417bf7`。
- 本轮只证明 profile-free data/observability contract，`paper_safe=false`，不能作为候选方法的性能结果或安全认证。

## 2. 预注册选择结果

| Selected cell | Expert | End-to-end | Independent PPO/VPP | 跨对手下界 |
|---|---:|---:|---:|---:|
| neutral -> post_merge valid steps | 1,461 | 1,247 | 1,430 | 1,247 |
| qualifying episodes | 19 | 16 | 16 | 16 |
| distinct signatures | 6 | 6 | 6 | 6 |
| distinct mirrors | 2 | 2 | 2 | 2 |

按照运行前冻结规则，`neutral -> post_merge` 以最大的三对手最小有效步数获选为唯一 `pilot-input candidate`。其它也满足最小 gate 的 cell 包括 `head_on -> re_entry`、`disadvantage -> post_merge/re_entry` 与 `crossing_entry -> post_merge/re_entry`，但不得在本轮后改选或并行扩张。

## 3. 合同与终端语义

所有对手的 60/60 record 都完成了 ready-state structural contract；发生合法 first-pass 连续 handoff 的 episode 为 expert/end-to-end/independent 的 19/17/16。ego crash/OOB 为 2/60、1/60、2/60，属于固定 reference rollout 的终端 readout，不等价于候选方法安全比较或飞控安全认证。

此次正向结果不表示新技能、profile、PPO routing 或五态势方法优于任何 baseline。它只表示：在冻结 B6 包线内，三个 opponent 都能够为 `neutral -> post_merge` 提供连续、有效、可追溯的 profile-free 54-D 与 profile-composable 66-D 输入及冻结 3-D VPP action。

## 4. 允许与禁止的后续动作

唯一允许的后续研发动作是起草一个新的、单一 `neutral -> post_merge` pilot 输入预注册：必须使用新的 Source ID、全新 train/dev/heldout continuous run-in manifest、固定 frozen reference baselines、三个 opponent 分开 gate 与独立安全 stop rule。

本结果不授权 training、reward/VPP/guidance/PID 修改、四共享技能、P5/P6、formal held-out，或 B6 重跑。任何实际 pilot 仍需单独的 clean SHA、资产哈希、空输出根与一次性 execution authorization。
