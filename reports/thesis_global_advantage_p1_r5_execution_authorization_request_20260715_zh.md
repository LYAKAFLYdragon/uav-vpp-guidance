# P1 R5 JSON Telemetry 一次性执行授权请求

**状态：** `not_authorized`
**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R5-JSON-TELEMETRY-REPRO-V1`
**输出根：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r5_json_telemetry`

## R4 后的唯一修复

R4 的首个 child 在 raw telemetry 由 `numpy.ndarray` 写入 JSON 时失败。R5 只在 `_write_json_atomic` 前调用已存在的 R4 `canonicalize`：递归将 `numpy.ndarray` 和 NumPy scalar 变为 JSON 结构，并继续拒绝未定义的 NaN/Inf、循环对象或不支持运行时类型。该修复不改变 episode、对手、场景、seed、run-in specialist、P3 encoder、VPP、guidance、PID、奖励、终端规则或 gate。

## 授权前条件

- [ ] R5 implementation 与新 writer regression test 已在 clean SHA 冻结。
- [ ] R5 config 的 source/checkpoint/code-file SHA 已填写并独立复核。
- [ ] R5 output root 不存在，磁盘空闲空间不少于 120 GB。
- [ ] 一次性授权仅允许 30 dev 场景 x 3 对手 x 3 repeat 的非学习复核。

R5 仍要求 90/90 cell 的 reset envelope、首个双方动作、trajectory、boundary 和 terminal 等价。任何 serialization、child、fallback 或 hash 问题均为 `runin_protocol_not_reproducible_do_not_train` 或 implementation failure，不允许在同一 Source ID 下重跑或训练。
