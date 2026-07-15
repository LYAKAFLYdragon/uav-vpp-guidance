# V15 Fixed-Encoder Temporal-Corridor Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-FIXED-ENCODER-TEMPORAL-CORRIDOR-V15`
**状态：** design-only；尚未授权执行
**性质：** 非学习、非奖励、非论文主结果的接口兼容性机制探针

## 1. 与 V14 的关系

V14 在首条 JSBSim record 的 66-D handoff 组装处因 `KeyError('profile')` 停止，`completed_records=0`。这不是关于时间走廊的物理或性能证据，但揭示了一个必须显式处理的 contract：五态势 66-D 输入的第 22--33 维包含 profile-derived target/weight，而动态走廊不能删除这一编码条件。

V15 使用全新的 Source ID、`7650 m / 320 / 352 m/s / seed=197100` 物理不相交 manifest 和新输出根。它不重跑或修补 V14。唯一的新接口定义是：

- **固定编码 profile：** 所有方法、所有时间步的 66-D target/weight 段固定为 `range_extension`，仅作为冻结 observation schema 的可追溯锚点。
- **动态 objective/action profile：** 评价目标和 semantic VPP action 仍独立地在 step 0--4 使用 `range_extension`、step 5--19 使用 `reentry_preparation`；它们不改变 66-D 维数、history、P3 embedding 或未来信息。

独立 verifier 必须逐步重放时间走廊，并核对每个 66-D 向量的 target/weight 段等于固定 `range_extension` 锚点。故 V15 不能通过隐性切换观测 contract 取得结果。

## 2. 冻结方法与场景

场景为 12 个连续 raw-SI first-pass 后 re-entry handoff 条件；对手 `expert`、`end_to_end`、`independent_ppo_vpp` 分开报告，禁止 pooled gate。每条 episode 严格 JSBSim、单次 reset、连续 run-in；禁止训练、reward recording、VPP/guidance/PID 改动、snapshot restore、future state、history padding 与各类 fallback。

固定比较矩阵为 `4 methods × 3 opponents × 12 scenarios = 144` 条记录：

| 方法 | Step 0--4 | Step 5--9 | Step 10--19 |
|---|---|---|---|
| `frozen_fixed_head_on` | 无 semantic offset | 无 semantic offset | 无 semantic offset |
| `static_range_extension` | `range_extension` | `range_extension` | 无 semantic offset |
| `escape_only_corridor` | `range_extension` | 无 semantic offset | 无 semantic offset |
| `escape_then_reentry_corridor` | `range_extension` | `reentry_preparation` | 无 semantic offset |

评价目标对所有方法相同，故不能以改变评分目标制造收益。输出逐步记录 phase、固定锚定的 66-D、六维 physical intent error、semantic receipt、VPP offset、guidance command、actual `n_z` 和 saturation。

## 3. 预注册门槛与解释

每个非参照方法仅同相同 `pair_key`、同 handoff hash、完整 20-step 的 `frozen_fixed_head_on` record 配对。total weighted-error 以 escape 与 re-entry 两阶段等权汇总，避免 15 个 re-entry step 在聚合中压倒 5 个 escape step。

`escape_then_reentry_corridor` 要在至少两个对手分别同时满足：12 条记录完整、至少 4 个合格配对、handoff hash 一致、total delta `<= -0.01`、re-entry delta `<= -0.01`、escape range-rate error delta `<= +0.02`、ego crash/OOB delta `<= +0.05`。其他三种方法仅用于归因，不可替代主要候选。

通过只允许起草新的训练 feasibility-pilot 预注册；不自动允许训练或 held-out 性能声明。若不通过，结论只限于：在该冻结的短时走廊接口、指定包线和三对手下，尚未建立跨对手的实用局部控制权威。

## 4. Stop Rule

执行前必须通过 manifest/hash/runtime/66-D-anchor 预检、目标测试和独立 verifier 测试；输出根必须不存在且至少有 120 GB 可用空间。运行中任一 contract failure、非有限 telemetry、fallback、记录数不足 144、gate 不一致或 safety stop，均将 V15 冻结为负证据或实现失败。失败后不得调整 V15 的 profile、窗口、动作尺度、场景、对手、门槛或训练预算，不得启动 reward rerun、技能训练、combat finetune 或 P5--P7。
