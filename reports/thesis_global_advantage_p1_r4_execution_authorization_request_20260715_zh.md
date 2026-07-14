# P1 R4 一次性执行授权请求

**状态：** `one_shot_execution_authorized_not_started`
**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R4-FRESH-ENV-REPRO-V1`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r4_fresh_environment`

## 已冻结的实现

- **Implementation SHA：** `5af60fa1a2f7fb0d3722601be393b7008a29b712`
- **合同测试：** `tests/test_global_advantage_p1_r3_contract.py` 为 `6 passed`。
- **非执行预检：** `validated_not_executed`；`execution_permitted=false`、30 个场景、270 个计划 episode。
- **运行时文件：** runner、R4 contract、run-in contract、pilot adapter、P3 encoder adapter、tracking environment 与 command post-processor 的 SHA-256 已写入 R4 config；执行时逐文件复核。
- **空间与输出：** R4 output root 当前不存在；E: 可用空间约 292.8 GB，高于 120 GB 下限。

## R3 后的唯一修复

R3 的零 episode 实现失败来自 `CombatHPManager.last_info.combat_time_to_kill=NaN` 的具名业务语义。R4 仅增加此值的显式 snapshot 表示：`not_terminal_or_not_killed`。其它非有限值仍使 snapshot fail-closed；训练、奖励、VPP、guidance、PID、对手、场景与 checkpoint 均不变。

## 独立静态复核

- [x] **90/90 cell 形成：** runner 对 30 个冻结 dev 场景和 3 个对手逐一启动 3 个独立 child process；比较器按 `opponent x scenario_signature` 分组，要求每组恰有 repeat `0/1/2`。
- [x] **等价判定：** 每个 cell 同时核对完整 reset envelope、首个 reference/opponent action、逐步 telemetry trajectory、first-pass（若无则末步）boundary envelope 与 terminal reason；还要求三条记录来自不同 OS process、telemetry 完整且无 backend/runtime prediction fallback。
- [x] **状态覆盖：** reset envelope 包含 observation/schema、真实 history、phase tracker、环境运行时字段、guidance/PID/filter/post-processor、predictor、opponent、HP manager 和 JSBSim FDM 属性。`combat_time_to_kill=NaN` 是唯一经定义的业务哨兵，其余 NaN/Inf 仍拒绝序列化。
- [x] **失败语义：** child timeout、异常、缺失/重复 artifact 或任一 hash 不一致均使 gate 为 `runin_protocol_not_reproducible_do_not_train`；不会生成或解释任何策略性能结论。
- [x] **命名边界：** runner 和内部 config key 中的 `r3` 是历史文件名；运行时 `SOURCE_ID`、默认 config、输出根和全部 gate artifact 都锁定为 R4，不能与 R3 failure root 混用。

## 授权前条件

- [x] R4 implementation 与测试已在新的 clean SHA 冻结。
- [x] R4 config 的 source/checkpoint/code-file SHA 已填写并独立复核。
- [x] R4 output root 不存在，磁盘空闲空间不少于 120 GB。
- [x] 一次性授权仅允许 30 dev 场景 x 3 对手 x 3 repeat 的非学习复核。

R4 的通过门仍为 90/90 cell 的完整 reset envelope、首个双方动作、trajectory、boundary 与 terminal 等价；任何失败都为 `runin_protocol_not_reproducible_do_not_train`，不允许 rerun 或训练。

授权时仅允许以下三项一致修改：将 `execution_permitted` 改为 `true`，把 `status` 改为已授权状态，并在本文件与 goal checklist 记录授权时间。任何代码、资产、场景、对手、阈值或输出根变化都会使授权失效。

**授权记录（2026-07-15）：** 执行范围固定为 30 个 dev 场景、3 个对手和 3 个顺序 repeat，共 270 条 non-learning fresh-process episode。`training_permitted`、`tuning_permitted`、`policy_change_permitted`、`vpp_change_permitted`、`guidance_change_permitted`、`pid_change_permitted` 与 `heldout_evaluation_permitted` 均持续为 `false`。运行结束后无论 PASS 或 NO-GO 都冻结输出，不重跑任何 cell。
