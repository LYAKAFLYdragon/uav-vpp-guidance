# V15 Fixed-Encoder Temporal-Corridor 结果与 Go/No-Go 决定

**Source ID：** `THESIS-FIVE-STATE-REENTRY-FIXED-ENCODER-TEMPORAL-CORRIDOR-V15`
**运行提交：** `ff36d60a3b637690d4c8643dadaa19af392033a8`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v15_fixed_encoder_temporal_corridor`
**最终判定：** `no_practical_temporal_corridor_candidate_under_preregistered_gate`

## 1. 完整性与独立复核

V15 完成了预注册的 `4 methods × 3 opponents × 12 scenarios = 144/144` 条 strict-JSBSim 连续记录。输出无 `failure.json`，包含 144 条 episode record、gate、run manifest 和 `independent_verification.json`。

独立 verifier 从逐步 telemetry 重放以下条件，结果均一致：

- 144 条完整 `method × opponent × scenario` 宇宙，且 report gate 与重算 gate 完全一致；
- 所有 handoff 后 66-D target/weight 段固定为 `range_extension`，没有将动态 action/objective profile 偷渡进 observation contract；
- step 0--4 为 escape objective、step 5--19 为 re-entry objective；各方法的 semantic axes 与固定时间走廊一致；
- 三对手单独统计，无 pooled gate；训练与 reward recording 均为禁止状态。

这说明 V14 的 `KeyError('profile')` 已被新的、明确的固定编码锚点 contract 正确排除，V15 是可解释的物理机制结果，而非实现失败。

## 2. 主要结果

主要候选为 `escape_then_reentry_corridor`；负 delta 表示相对 `frozen_fixed_head_on` 改善。通过条件要求每个对手至少 4 个配对，且 total/re-entry delta 均 `<= -0.01`、escape range-rate 不恶化超过 `+0.02`、ego crash/OOB delta `<= +0.05`；主要候选必须在至少两个对手独立通过。

| 对手 | 合格配对 | Total weighted-error delta | Re-entry delta | Escape range-rate delta | Ego crash/OOB delta | 判定 |
|---|---:|---:|---:|---:|---:|---|
| expert | 7 | +0.000682 | +0.001382 | +0.000084 | 0.0000 | 未达到改善门槛 |
| end-to-end | 3 | -0.000095 | -0.000183 | -0.000055 | 0.0000 | 配对数不足，且幅度远小于 0.01 |
| independent PPO/VPP | 6 | +0.006261 | +0.013061 | -0.003034 | 0.0000 | re-entry/total 恶化 |

`static_range_extension` 与 `escape_only_corridor` 也没有成为候选。所有方法的 ego crash/OOB 相对参照差值为 `0.0`，所以这不是安全退化；但安全持平不等于控制收益。最终 `corridor_candidates=[]`、`passes=false`。

## 3. 可证伪的解释

V15 排除了两个较弱解释：

1. **不是 profile 缺失造成的假阴性。** 固定编码 profile 已逐步复核，144 条真实 JSBSim 记录正常完成。
2. **不是完整后端或安全链失效。** 所有三对手均完成计划记录，handoff hash 一致，且无新增 ego crash/OOB。

在本次冻结包线中，证据支持的结论是：将 `range_extension` 后接 `reentry_preparation` 的短 10-step semantic VPP 走廊，并未形成跨两个对手的、可供学习利用的实用局部下降方向。它不证明任意更长的 reference trajectory、不同任务动力学或新低层控制接口无效；也不允许将“安全持平”夸大为性能提升。

## 4. 正式决定

- **No-go：** 不得启动 reward rerun、单技能/四技能训练、combat finetune、P5--P7 或 V15 held-out；不得在 V15 内调整 encoder profile、动作尺度、窗口、场景、对手或门槛后重跑。
- **保留：** V13 的单点接口负证据、V14 的零记录实现失败和 V15 的完整机制负证据应共同保留，形成对“单点或短走廊 VPP 语义接口尚未建立实用控制权威”的可复核边界。
- **若继续：** 下一研发线不能再是 reward 调节或短走廊补跑。它必须先提出一个新的、可在非学习 probe 中验证的几何接口，例如受约束的多点 reference path 或新的连续 re-entry task dynamics，并以新 Source ID、全新物理 manifest 和独立预注册开始。
