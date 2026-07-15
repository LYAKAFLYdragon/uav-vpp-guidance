# Re-entry Terminal Reward Pilot V1 预注册

## 研究问题

在 V8 已证明连续 re-entry phase 可观测的前提下，增加小幅、仅末步生效的终端结果信号，
能否改善单一 re-entry 候选的几何准备，同时不增加 ego crash/OOB？

这不是“五态势系统已全面占优”的试验，也不训练高层 PPO。它只检验一个可解释的低层
训练变量：终端奖励是否能在冻结 VPP/guidance/PID/encoder/对手条件下提供有用的稀疏反馈。

## 固定奖励

`r = r_geometry + I(terminal) [ +1.0 I(win) -1.0 I(loss) -0.25 I(draw) -1.0 I(ego crash/OOB) ]`

其中原有 `terminal_ego_crash_oob_penalty` 设为 `0.0`，避免与新增 terminal 项双重计数；
因此 ego crash/OOB 的总新增终端影响为 `-2.0`。对方 crash/OOB 与 target killed 均为合法
胜利，只获得 `+1.0`，绝不施加惩罚。timeout HP advantage/disadvantage/draw 依据环境已经
冻结的 `win/loss/draw` 语义映射，不在结果出现后改写。

## 方法与门槛

- Candidate：新训练的单一 continuous 66-D -> 3-D VPP re-entry policy，固定 `range_extension` profile。
- Primary baseline：frozen fixed-head-on；辅助 baseline：frozen fixed-crossing。
- Three opponents 分开训练采样和分开报告，不合并统计量。
- 连续物理 run-in、raw physical first-pass-aware phase、strict JSBSim、无 reset/fallback/padding。
- 固定训练预算 `20,000` step；dev-only checkpoint 选择；heldout 只运行一次。
- 若任何 contract 失败，或 candidate ego crash/OOB 高于 fixed head-on 超过 5 pp，立即停止并冻结为负证据。

## 当前状态

配置已冻结为设计文档，`execution_permitted=false`。在实现 preflight 建立 V8 phase tracker、
新 disjoint manifest、终端 reason 单元测试与真实 reward-term ledger 前，不得启动训练。
