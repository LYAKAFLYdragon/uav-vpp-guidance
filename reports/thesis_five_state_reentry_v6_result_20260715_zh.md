# V6 Re-entry Reachability Control Matrix 结果

> **2026-07-15 phase-contract correction.** The frozen-controller results are
> preserved, but their legacy `re_entry=0` label is not physical negative
> evidence: the tracker received normalized observations against physical-unit
> thresholds and had no first-pass prerequisite. This report must not be used
> to claim that re-entry was absent or infeasible. See
> `reports/thesis_five_state_v5_v7_phase_contract_audit_20260715_zh.md`.

## 结论

在与既有 evaluation/train manifest 物理不相交的 12 个场景中，V6 完成 `12 scenarios x 3 opponents x 2 frozen controllers = 72` 条 strict-JSBSim record。固定 head-on 和显式 18-D 投影后的固定 crossing 均未在 raw-SI first pass 后产生任何 `re_entry`：所有 6 个 controller x opponent 单元都是 `0` qualifying episode、`0` re-entry 66-D step。

这是一条可复现的负证据，不是训练不足、隐式 crossing 截断、JSBSim fallback 或对手池化造成的假阴性。V6 的 crossing 输入只显式删除 `is_crossing` task bit，拒绝所有其它 truncation/padding；72 条均使用连续 reset、冻结 P3、固定 VPP/guidance/PID 和三对手独立 gate。

## 证据边界

- 输出：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v6_control_matrix`
- Source ID：`THESIS-FIVE-STATE-REENTRY-REACHABILITY-CONTROL-MATRIX-V6`
- Gate：每个 controller x opponent 至少 2 条 re-entry episode、20 个 re-entry 66-D step、双镜像和至少两种高度。
- 结果：六个单元均为 `0/12` re-entry episode，所有 gate 为 false。

它只能说明：**在当前冻结二技能控制库和已定义的连续物理包线中，re-entry 不可作为跨三对手共享技能训练的输入契约。** 它不能证明不存在其它控制器、其它飞行包线或其它任务设计能够形成 re-entry，也不能比较任何新方法的胜率。

## 正式决策

1. 不启动四共享技能 P4、combat finetune、P5 高层 PPO、P6 消融或 P7 heldout60。
2. 不把 V6 失败解释为“多训练一些就会解决”，也不修改 V6 场景、门槛、奖励或控制器重跑。
3. 五态势主线只能继续为 **post-merge shared-intent interface** 的新 family；任何包含 `reentry_recovery` 的全阶段系统必须先有新的、独立的物理任务定义与数据契约，而不是在当前证据上宣称完成。
