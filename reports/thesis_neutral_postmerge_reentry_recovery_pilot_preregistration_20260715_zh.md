# Neutral Post-Merge Reentry-Recovery 单技能 Pilot 预注册

**拟议 Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V1`
**状态：** `preregistration_draft_not_authorised`
**前置数据契约：** B6 `neutral -> post_merge` candidate，三个 opponent 的最小覆盖为 1,247 valid step、16 qualifying episode、6 signature、2 mirror。

## 1. 单一问题

在 B6 已证明共同输入数据契约存在的限定条件下，一个新训练但固定的 `reentry_recovery` 低层技能，使用全局
`reentry_preparation` profile，能否比冻结 head-on/crossing specialist 更好地实现 neutral post-merge 的可解释再进入准备几何，且不增加 ego crash/OOB？

这不是四共享技能训练、不是高层 PPO 实验，也不是五态势全场景优越性检验。routing 关闭，profile 固定，P3 encoder 冻结。

## 2. 冻结动作与意图

- **Candidate skill：** `reentry_recovery`，统一 66-D 输入到 normalized 3-D VPP action。
- **Fixed profile：** `reentry_preparation`，其物理 target 为 `(AA*=30 deg, ATA*=130 deg, range*=2200 m, range-rate*=-30 m/s, specific-energy-delta*=100 m, altitude-delta*=0 m)`，权重为 `(0.8, 0.8, 1.0, 0.8, 0.7, 0.4)`。
- **Reference chain：** 预测器、VPP、guidance、PID、P3 checkpoint、JSBSim 版本均冻结；无 prediction reward、无 action fallback、无 history padding。
- **Baselines：** frozen fixed head-on specialist（主基线）和 frozen fixed crossing specialist（现有技能库相邻能力对照）。不使用 Oracle，因为固定任务下它与 fixed-head-on 不提供独立比较。

## 3. 新包线与数据隔离

pilot 必须在新的 clean worktree 与新 output root 中进行，且 train/dev/heldout 三个 continuous run-in manifest 与 B2--B6、heldout240、dev30、heldout60 的物理 state signature 全部不相交。

- train：连续随机分布，仅用于训练；不得用 B6 episode、seed、raw trajectory 或 output 作训练样本。
- dev：新的 12 个固定 scenario，只用于固定 checkpoint schedule 的选择与 safety stop。
- heldout：新的 24 个固定 scenario，三 opponent x candidate/head-on/crossing 一次性评估；不得用于 checkpoint 选择。
- 每个 evaluation episode 必须从合法 pre-merge reset 连续进入 first-pass 后的 neutral post-merge target window；禁止 reset、snapshot restore 与 future-state injection。

## 4. 奖励与主指标

对每个 valid neutral post-merge step 定义归一化 profile intent loss：

```text
L_intent(t) = sum_j w_j [clip((x_j(t) - x*_j) / s_j, -1, 1)]^2 / sum_j w_j
```

其中 `x=(AA, ATA, range, range-rate, specific-energy-delta, altitude-delta)`，
`s=(180 deg, 180 deg, 5000 m, 200 m/s, 1000 m, 1000 m)`，`x*` 和 `w` 为上节固定 profile。训练中仅使用冻结的 profile-weighted geometry-progress reward 与预定义 ego crash/OOB、物理 command saturation penalties；开始后禁止改 reward 或系数。

主比较指标为每个 episode first 20 个 valid target step 的 `normalized_reentry_preparation_intent_loss_auc20`，candidate 减 frozen head-on 的 paired delta，数值越低越好。攻击区暴露、specific energy、range/range-rate、VPP bias、实际 n_z/AoA response 和 terminal mix 为机制 readout；win rate 仅为次要结果。

## 5. 预注册 Gate 与 Stop Rule

每个 opponent 独立要求至少 2 条 qualifying paired episode、20 valid target step、2 signature、2 mirror，且无 non-finite 66-D/action、JSBSim fallback、checkpoint fallback、异常 reset 或 padding。

Dev safety stop：candidate 的 ego crash/OOB rate 高于 frozen head-on 超过 0.05 时立即冻结为负证据。Heldout 时，candidate 必须在三个 opponent 都满足 safety 与 head-on intent-loss noninferiority（paired delta <= 0.02），并在至少两个 opponent 同时满足对 head-on 的实用改善（delta <= -0.05）与对较好现有 specialist 的改善（delta <= -0.02）；剩余 opponent 对较好现有 specialist 不得差于 0.02。

任一 immediate safety/contract stop 或 heldout failure 均禁止调 reward、增加步数、更换场景、替换 encoder/checkpoint 或重跑同一 Source ID。

## 6. 可证伪解释

- candidate 通过全部 gate：仅支持“在预注册 neutral post-merge 包线内，现有库缺少一个有用的 reentry-recovery 行为候选”；仍不支持四技能系统或全五态势优越性。
- frozen crossing 匹配/优于 candidate：优先归类为已有技能库的 routing/composition 问题，不宣称缺技能。
- candidate 无改善或安全失败：拒绝 reentry-recovery skill-gap 假设，并停止该 training lane。

## 7. 当前允许动作

当前只允许实现 manifest builder、训练/评估 config、contract tests、artifact schema 和执行前 preflight。实际 pilot 必须另有 clean SHA、资产 hash、独立审查与一次性 execution authorization。
