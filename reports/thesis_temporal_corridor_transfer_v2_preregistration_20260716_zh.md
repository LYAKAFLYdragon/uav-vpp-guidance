# Temporal-Corridor Physical Transfer V2 预注册

**Source ID：** `THESIS-FIVE-STATE-TEMPORAL-CORRIDOR-TRANSFER-V2`  
**状态：** executed once; preregistered negative mechanism evidence
**性质：** 非学习、非奖励、非论文主结果的物理传递机制 probe

## 1. V1 的边界与 V2 的唯一修复

V1 在首个 `build_record_only_env` 前因缺少 `execution.runtime_template` 停止，
`completed_records=0`。它是工程执行无效证据，不能被覆盖或补跑。

V2 使用新 Source ID、新输出根和新 `8870 m / 334 / 364 m/s`、seed `208200--208211`
manifest。唯一接口修复是：设计 config 同时冻结 `inputs.runtime_template` 与
`execution.runtime_template`，runner 在构造环境前要求二者路径和 SHA-256 完全相等，
并只传递这个已验证的完整 Source config 给 V16 record-only environment builder。
这不是动作尺度、走廊窗口、奖励、策略、VPP、制导或 PID 的修改。

## 2. 工程 Smoke Gate

正式 V2 授权前，必须在该冻结实现 SHA 上执行一个不占 Source ID、不可用于论文结论的
engineering smoke。它对 expert、end-to-end、independent PPO/VPP 分别使用 V2 manifest
中的同一冻结 reset：

1. 实际调用 `build_record_only_env` 构造 strict-JSBSim 环境。
2. 完整执行冻结 V16 run-in 填满预测历史；随后同步冻结预测器，并检查预测目标三维位置有限且无 fallback。
3. 使用冻结 fixed head-on specialist 走一步，读取 VPP world offset、guidance command 与
   actual `n_z` 响应。
4. 任一对手出现 backend/prediction fallback、非有限 observation/action/telemetry 或缺字段，
   gate 即失败，V2 不得授权。

三对手均通过，且每条记录都具有 JSBSim backend、有限 VPP/guidance/PID telemetry，才可建立
hash-bound one-time authorization overlay。smoke 只验证工程可执行性，不预判或改变 108 条正式
物理传递的结果。

该 gate 已在 `05a944370b21e4387db97b7a74719d5953041c8b` 上通过。完整 receipt 与其
解释边界见 `reports/thesis_temporal_corridor_transfer_v2_smoke_gate_20260716_zh.md`。正式 108 条
执行与独立复核均已完成，但没有 candidate 通过 gate；结果见
`reports/thesis_temporal_corridor_transfer_v2_result_20260716_zh.md`。

## 3. 正式矩阵与解释门

在 smoke 通过后，V2 的唯一正式执行矩阵是
`frozen_fixed_head_on / moderate_left / moderate_right × 3 opponents × 12 scenarios = 108`。
run-in、raw-SI phase tracker、prediction/VPP/guidance/PID 链、固定 head-on specialist 与
20-step window 均继承 V16；只有 handoff 后 temporal-corridor action compiler 可改变动作。

每个候选必须独立按对手满足至少 4 个 matching-handoff pair、平均投影残差不超过 `25 m`、
hard-bound step 为 `0`、我方 crash/OOB 相对基线不增加超过 `0.05`、六维加权 error delta
不高于 `-0.01`、至少两个分量改善且无任一分量恶化超过 `+0.02`。同一候选必须在至少两个
对手分别通过；禁止 pooled gate。

任一运行时 contract failure、fallback 或记录不足均冻结 V2，不在本 Source ID 下调 extent、
slew、reward、窗口、场景、对手或训练步数。即使 V2 通过，也只允许起草下一步训练 feasibility
设计，不能自动启动技能训练或形成 held-out 论文结论。
