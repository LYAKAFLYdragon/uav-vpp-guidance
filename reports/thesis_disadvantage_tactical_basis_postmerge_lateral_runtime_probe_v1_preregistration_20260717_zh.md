# Selected Lateral-Extension Post-Merge Runtime Probe V1 预注册设计

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-LATERAL-RUNTIME-PROBE-V1`  
**状态：** `design_frozen_execution_not_authorized`  
**性质：** 独立、非学习、配对、strict-JSBSim runtime probe，`paper_safe=false`

## 问题与唯一候选

P2 离线筛选已证明 8 个冻结时域参考在两个主要对手的真实 V3 post-merge 窗口中都可无裁剪编译，但没有让候选动作进入 JSBSim。因此，本 Source 只回答一个问题：

> P2 唯一选择的 `lateral_extension:lat+1:vert+1`，在同一合法 reset、同一场景和种子、同一 frozen fixed-head-on run-in 后，能否在不破坏执行链的前提下产生有益的 post-merge 防御几何变化？

唯一候选固定为 40 个高层 step（8 s）的 `lateral_extension:lat+1:vert+1`；每一步的 basis 必须从该步真实状态计算，reference action 必须来自该步冻结 fixed-head-on specialist。程序结束后恢复该相同的 fixed-head-on specialist。不得测试其它模板、符号、幅度、reward、路由或 checkpoint。

## 配对协议

- 方法仅两个：`fixed_head_on` baseline 与 `lateral_extension_lat_pos_vert_pos` candidate。
- 对每个 `opponent × scenario` 分别进行两次独立但相同的合法 reset；不在 handoff reset、不 snapshot restore、不注入 future state。
- 两条轨迹在首次合格 handoff 前必须使用逐步相同的 frozen fixed-head-on action。只有 `scenario`、`seed`、首次 handoff step 和 pre-handoff hash 全部相同，才构成因果配对。
- 首次 handoff 的条件是实时 `disadvantage -> post_merge`、10-frame history 完整、预测器无 fallback、strict JSBSim、tactical basis rank=3、condition 不超过 100。candidate 从该步起执行固定 40 step 程序；baseline 始终执行 frozen fixed-head-on。
- 采用 P2 对齐的 12 个初始场景、3 个对手，共计划 72 个完整 episode；这不是新的 held-out 泛化实验，不能用于论文泛化声明。
- `end_to_end` 与 `independent_ppo_vpp` 是 gate 对手；`expert` 仅诊断，因为 V3 中只有 3 个有效 signature。

## 记录与指标

每个 candidate step 必须记录当前 basis、frozen reference action、compiler receipt、实际 VPP 偏置、guidance、AoA、`n_z` command/actual、saturation、预测器状态、phase、taxonomy、range、range-rate、AA、ATA、速度、高度、specific energy、双方 attack-zone score/HP 和 terminal reason。baseline 记录相同字段，但不产生 candidate receipt。

主指标是 40-step window 内 `target_attack_score` 的配对均值差（candidate − baseline，越小越好）。只有 baseline 压力均值至少 0.01 的 pair 才可进入主指标；这防止把“对手本来不构成压力”误写成防御改善。

主指标的最小实用改善固定为 `<= -0.05`。同时必须至少满足一个二级几何改善：平均 range-rate 增加至少 `5 m/s`（正值表示目标正在远离）、末步 range 增加至少 `50 m`，或末步我方 specific energy 增加至少 `25 m`。胜率、HP 和 terminal 仅作二级描述，不作为通过门。

## Gate 与停止规则

一个 required opponent 要通过，必须有至少 4 个有效 pair、4 个 scenario signature、双镜像，并至少有 3 个 baseline-pressure-exposed pair。该对手还必须满足主指标和至少一个二级几何指标，且 candidate 的 ego crash/OOB 数不高于 baseline。

只有两个 required opponents 都通过，才能称为 `cross_opponent_authority_established`，且**只允许起草**一个单技能 pilot 的预注册。若仅一个对手通过，只能称为 `opponent_conditional_authority`，不训练通用技能；若两者都不通过但执行链完整，结论为 `behavioral_or_control_authority_not_established`，停止 temporal-reference/shared-skill training line。

任何 candidate 非有限 action、box/slew clipping、basis rank/condition 违规、backend/predictor/checkpoint fallback、异常 reset 或 pre-handoff hash 不一致，都触发 `execution_invalid` 或 `runtime_contract_failure`，立即冻结 Source：不调 reward、不加步数、不换场景、不换模板、不选择性重跑。

## 独立验证与后续

正式执行前必须另行实现 runner 和不导入 runner 的 verifier。verifier 必须从 episode JSON 重新构造 pairing、逐步检查 40-step runtime contract、重新计算 pressure/geometry delta 与 gate，并比对 `pairing_audit.json`、`paired_geometry_deltas.csv` 和 `runtime_probe_gate.json`。本文件和 config 只冻结设计，不授权任何 JSBSim 运行。
