# V12 Six-Objective Intent-Controllability Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V12`
**状态：** design-only；尚未授权执行
**性质：** 非学习、非奖励、机制诊断；不是论文主结果

## 1. 来由

V11 已经证明 relative-frame semantic compiler 的动作可以产生数百米 VPP 位移并改变 guidance command，因此“动作被后端、VPP 或 guidance 吞掉”不再是主要解释。但固定 `range_extension` 完整 profile 没有在两个对手同时改善聚合 intent-loss；而 V11 只保存了聚合 loss，不能判定 AA、ATA、range、range-rate、specific-energy 和 altitude 中究竟哪些相互冲突。

V12 只回答：**冻结的单点 VPP 接口是否存在一个安全的局部 relative-frame semantic 方向，能同时改善多个 range-extension intent 分量而不显著损害 range/range-rate？** 它不会训练任何策略，也不会将成功自动转化为训练或性能结论。

## 2. 冻结条件

- 固定 V11 的 compiler scales、20-step local window、world-NEU raw VPP interface、frozen fixed-head-on specialist、P3 encoder、预测器、guidance/PID 与 strict JSBSim。
- 每条 episode 只 reset 一次；只在 raw-SI first-pass 后的连续 re-entry handoff 记录；禁止 reward recording、训练、调参、snapshot restore、future-state injection、history padding 和任意 fallback。
- 全新 `manifest12` 采用 7050 m、300/350 m/s 与 `seed=192100`，并显式记录与 V1--V11、Dev30、Heldout60、Taxonomy30 的物理不相交性。
- 三对手独立报告：`expert`、`end_to_end`、`independent_ppo_vpp`；禁止 pooled gate。

## 3. 全因子动作矩阵

保持 `range_extension` profile 和 V11 定义：range 轴沿 LOS 向我机侧移动虚拟点；turn 轴沿目标速度侧向单位向量；vertical-energy 轴沿 NEU up。八个方法构成 `2^3` 全因子，不选择运行后方向：

| Method | Range | Turn | Vertical-energy |
|---|---:|---:|---:|
| `frozen_fixed_head_on` | 0 | 0 | 0 |
| `range_only` | 1 | 0 | 0 |
| `turn_only` | 0 | 1 | 0 |
| `vertical_only` | 0 | 0 | 1 |
| `range_turn` | 1 | 1 | 0 |
| `range_vertical` | 1 | 0 | 1 |
| `turn_vertical` | 0 | 1 | 1 |
| `range_turn_vertical` | 1 | 1 | 1 |

计划记录数固定为 `8 x 3 x 12 = 288`。每个 semantic window 记录 actual compiler receipt、VPP world offset、guidance command、actual n_z、saturation，以及六维物理 intent vector 和相对 profile target 的归一化绝对误差。

## 4. 预注册诊断门

每种非参考方法、每个对手单独比较相同 handoff-hash 的 pair。定义六维分量误差差为 `method_after_error - reference_after_error`，负值较好。某个对手的 local multiobjective compatibility pass 需要：

1. 至少 4 个完整 20-step qualified pair，handoff hash 完全一致，且无 contract failure。
2. 高优先级 `range_target_m` 与 `range_rate_target_mps` 分量均不恶化超过 `+0.02`。
3. 任一分量不恶化超过 `+0.05`；至少两个分量改善 `>=0.01`。
4. profile-weighted 六维误差相对参考改善 `>=0.01`，且 ego crash/OOB 不高于参考 `+0.05`。

仅当同一 action combination 在至少两个对手分别通过，才称为 `local_multiobjective_candidate`。即使出现 candidate，本轮也只允许写 decision memo，不能自动训练；若不存在 candidate，则不得通过改变 V12 scale、window、profile 或 scene 补跑。

## 5. 入口

- Builder：`scripts/build_thesis_five_state_reentry_v12_manifest.py`
- Design config：`config/experiment/thesis_five_state_reentry_v12_intent_controllability_design.yaml`
- Runner：`scripts/run_thesis_five_state_reentry_v12_intent_controllability.py`
- Verifier：`scripts/verify_thesis_five_state_reentry_v12_intent_controllability.py`
- 输出：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v12_intent_controllability`

V12 不改变 V8--V11 的任何结论，也不将局部可控性误写为五态势系统在未定义空战状态中的优势。
