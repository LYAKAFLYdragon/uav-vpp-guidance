# P2-B3 Phase-Observability Preflight 设计预注册

**拟议 Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-OBSERVABILITY-PREFLIGHT-V1`
**状态：** `design_only_not_authorised`
**前置证据：** P2-B2 的 `physical_pass_phase_observability_no_go`

## 目标与非目标

P2-B3 只回答一个实现与测量问题：冻结 scenario 的几何是否进入 JSBSim reset，以及 raw telemetry 是否足以独立重建 `PhaseTracker` 的 phase 判断。它不是 P2-B2 的 rerun，不复用其 source ID 或 seed，不训练、不调参、不加载候选五态势策略，也不输出 win rate、优越性或 opponent 排名。

P2-B2 已证明 reference chain 的 180 条记录在 strict JSBSim 下可完成，但 manifest 的 180 条 `pre_merge` reset 与 raw 的 180 条首个 `post_merge` 标签矛盾，且 raw 缺少 range/range-rate。因此 B3 必须先建立**测量契约**，再讨论任何 phase coverage。

## 冻结方法

| 项目 | B3 约束 |
|---|---|
| Reference controller | 冻结 `run_in_head_on` specialist，3-D VPP action 不变 |
| Opponent | `expert`、`end_to_end`、`independent_ppo_vpp` 分开报告，不 pooled |
| Backend | strict JSBSim、fresh environment per episode、无 backend fallback |
| 学习 | training/tuning/candidate-policy/profile/encoder 微调均为 false |
| 样本 | 新的 non-overlapping diagnostic manifest；不重用 P2-B2 scenario instance 或 seed，不消耗 heldout240 evaluation seed |
| 规模 | 先冻结 `5 initial states x 3 height conditions x 2 mirrors = 30` 条诊断场景，三 opponent 共 90 条；只允许一次执行 |
| 输出 | reset receipt、step-0 receipt、逐 step phase input/output、scenario-application receipt、终端与 fallback；不写性能聚合 |

## 必须持久化的可复算字段

每条记录在 `env.reset()` 立即之后写入 reset receipt，第一步控制前写入 step-0 receipt，之后逐 high-level step 写入同一组基础字段。

| 层级 | 最小字段 |
|---|---|
| Requested scenario | 完整 own/target position、speed、heading、altitude、seed、initial state、height、mirror、物理 signature |
| Applied reset | own/target position/velocity/heading/altitude、derived 3-D range、relative range-rate、observation schema 与 16-D base observation |
| Phase input | `range_m`、`range_rate_mps`、`merge_range_m`、`reentry_rate_mps`、`seen_merge_before/after`、`post_merge_steps_before/after` |
| Phase output | `phase_before_step`、`phase_after_step`、first-pass flag、first-pass completion step |
| Control chain | frozen specialist SHA、normalized 3-D action、backend、prediction valid/fallback、own/target finite-state receipt |
| Provenance | config/manifest/registry/checkpoint SHA、clean git SHA、output hash index |

`PhaseTracker.update()` 的输入必须来自已持久化的 raw base observation，而非只保存其返回标签。reset receipt 与 step-0 receipt 也必须分开，防止将第一次 `env.step()` 后的状态误当作 reset 状态。

## 预注册 gate

| Gate | 通过条件 | 失败的归因 |
|---|---|---|
| Scenario application | 90/90 requested-vs-applied receipt 在预先固定的位置、速度、航向、海拔容差内一致 | reset/sampler contract gap |
| Step-0 phase | 90/90 step-0 `range_m/range_rate_mps` 可重算，且 tracker input/output 与 persisted raw 一致 | phase instrumentation gap |
| 基础物理合同 | 90/90 strict JSBSim、有限 3-D action/state、无 backend/prediction fallback，至少 20 steps | backend/control contract gap |
| 初始 pre-merge 语义 | 若 manifest 声明 `pre_merge`，step-0 tracker output 必须为 `pre_merge`；否则不再把该 manifest 描述为 pre-merge envelope | manifest-to-runtime semantic gap |
| 覆盖读数 | `state x opponent x phase` 只作 readout，不把 re-entry 的零样本当作 instrumentation failure | physical/behavior coverage gap（仅在前四 gate 通过后） |

没有“必须到达 re-entry”的通过门槛。B3 的目的在于可观测、可归因，而非通过增加 episode 数量制造 phase 覆盖。

## 停止规则与后续决策

- 任一 hash、clean-worktree、输出根、scenario receipt、有限值或 fallback gate 失败，立即归档为负证据；不改 sampler、阈值、场景或控制器后重跑同一 Source ID。
- 若 scenario application 与 phase instrumentation 都通过，但三个 opponent 下持续缺少目标 phase，这才是可解释的 physical/behavior coverage gap；之后可设计新的 phase-feasible sampler，仍不得直接训练。
- 若 reset receipt 或 phase 输入输出不一致，则优先修复**测量契约**，建立新 source ID 后重新 preflight；不能将问题归因于技能库或 PPO。
- 只有 B3 通过前四个 gate，才允许起草后续 phase-feasible sampler 的执行授权。四共享技能、P5 与 formal held-out 均继续锁定。
