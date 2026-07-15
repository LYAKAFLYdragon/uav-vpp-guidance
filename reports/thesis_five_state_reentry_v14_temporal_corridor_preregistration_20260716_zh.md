# V14 Temporal-Corridor Controllability Probe 预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-TEMPORAL-CORRIDOR-V14`
**状态：** design-only；尚未授权执行
**性质：** 非学习、非奖励、非论文主结果的机制探针

## 1. 研究问题与边界

V13 已经排除了“语义动作未进入 VPP/guidance/PID”这一解释：三条语义轴均产生数百米量级的 VPP 位移，且没有新增 crash/OOB 或饱和。但是，在冻结的单点 3-D VPP 接口与 20-step 窗口内，没有任何轴组合在至少两个对手下形成安全、共同的六维意图下降方向。

V14 只检验一个新的、时间扩展但仍完全冻结的接口假设：**前半段先执行可解释的脱离几何，后半段再执行再进入准备，是否比单一静态意图更能降低阶段匹配的 physical-intent error。**

V14 不训练策略、不记录或修改 reward、不改动作尺度、不延长观测窗口、不改 P3 encoder、预测器、VPP、guidance、PID、对手或场景后补。即使通过，也只说明这一时间化接口值得另行起草训练 pilot；不证明五态势系统、共享技能库或任何未定义空战状态已经优越。

## 2. 冻结实验宇宙

- 新 manifest：12 个连续 first-pass 后 handoff 场景；`7350 m` 初始距离、我方/目标速度 `312/342 m/s`、`seed=195100`，与 V1--V13、Taxonomy30、Dev30 和 Heldout60 明确不相交。
- 对手分别报告：`expert`、`end_to_end`、`independent_ppo_vpp`；禁止 pooled gate。
- 每条记录：严格 JSBSim、单次 reset、连续 run-in 与 handoff；禁止 snapshot restore、future-state injection、history padding、backend/checkpoint/prediction fallback。
- 输入/输出：冻结 66-D observation 与 3-D VPP action；每个 handoff 后最多保留现有 20-step 连续窗口，记录 phase、66-D、六维 physical intent error、VPP offset、guidance command、实际 `n_z` 与饱和标记。
- 预期记录数固定为 `4 methods × 3 opponents × 12 scenarios = 144`；输出根目录在执行前必须不存在，磁盘空闲空间至少 120 GB。

## 3. 固定方法矩阵

四种方法在同一场景、同一 seed、同一冻结 fixed-head-on specialist 与相同 run-in 下比较；只有 handoff 后的 semantic VPP profile 时间表不同：

| 方法 | Step 0--4 | Step 5--9 | Step 10--19 | 作用 |
|---|---|---|---|---|
| `frozen_fixed_head_on` | 无 semantic offset | 无 semantic offset | 无 semantic offset | 主要参照 |
| `static_range_extension` | `range_extension` | `range_extension` | 无 semantic offset | 检验持续脱离而不切换 |
| `escape_only_corridor` | `range_extension` | 无 semantic offset | 无 semantic offset | 检验仅前段脱离 |
| `escape_then_reentry_corridor` | `range_extension` | `reentry_preparation` | 无 semantic offset | 唯一的主要候选 |

所有方法使用相同**评价目标**：step 0--4 为 `range_extension`，step 5--19 为 `reentry_preparation`。因此，候选方法不能仅凭更换评价目标取得优势。走廊是固定的时间索引参考序列，不是 learned policy，也不是训练 teacher。

## 4. 预注册门槛

针对每一个非参照方法、每一个对手，先仅配对相同 `pair_key` 且具有相同 handoff hash 的完整 20-step episode。为避免 15 个 re-entry step 在聚合中压倒 5 个 escape step，`total` 指标按 episode 内两个阶段均值**等权**聚合。

某个对手下的候选通过必须同时满足：

1. 12 条记录完整，至少 4 个合格配对，且所有参与配对的 handoff hash 完全一致。
2. 相对 `frozen_fixed_head_on`，stage-balanced total weighted-error delta `<= -0.01`。
3. re-entry stage weighted-error delta `<= -0.01`。
4. escape stage 的 range-rate component-error delta `<= +0.02`，避免通过牺牲脱离阶段取得表观收益。
5. ego crash/OOB rate delta `<= +0.05`。

V14 只有在 `escape_then_reentry_corridor` 于**至少两个对手分别通过**时才记为 `passes=true`。其他三种方法仅作机制对照，不能替代主要候选的通过条件。

## 5. 可复现性与停机规则

执行前必须先通过 design preflight、目标单元测试与独立 verifier 测试；随后把 manifest、设计配置、runner、verifier 和所有依赖文件 SHA-256 写进单独的 hash-bound authorized overlay。Runner 只接受干净 worktree，且授权实现 SHA 必须是当前 `HEAD` 的祖先，同时逐文件哈希必须匹配。

执行后由 `scripts/verify_thesis_five_state_reentry_v14_temporal_corridor.py` 独立重放每个窗口的阶段、profile、semantic axes、有限 66-D/telemetry 契约，并从 144 条记录重算门禁。报告 gate 与重算 gate 不一致即视为失败。

以下任一情况均立即冻结为负证据：runtime/contract failure、非有限 telemetry、fallback、输出/哈希漂移、少于 144 条记录、任一已定义安全 stop，或主要候选未在两个对手通过。失败后不得修改 V14 的 action scale、窗口长度、profile、场景、对手、门槛或重跑；更不得据此启动 reward rerun、单技能/四技能训练、combat finetune、P5--P7。

## 6. 执行后的唯一解释范围

- **通过：** 允许起草一个全新 Source ID 的训练 feasibility-pilot 预注册，仍不自动授权训练或 held-out 结论。
- **未通过：** 说明在该冻结接口、包线与三对手范围内，短 10-step escape/re-entry 走廊也未建立跨对手的实用局部控制权威；应停止单点和短走廊 VPP 训练线，而不是奖励调参或增加训练步数。
