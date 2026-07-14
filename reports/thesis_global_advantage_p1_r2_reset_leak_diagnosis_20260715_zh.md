# P1 R2 跨回合状态泄漏归因记录

**状态：** `confirmed_reset_contract_defect_not_a_replacement_for_r3`
**关联 Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-RUNIN-TELEMETRY-R2`
**方式：** 代码与回归测试归因；不重跑 R2，不训练，不改变任何 R2 artifact。

## 已确认的事实

1. R2 的环境构造基于 `config/experiment/jsbsim_hrl_comparison.yaml`，其中 `guidance.post_process.enabled=true`，并启用了 `enable_lift_compensation=true` 和一阶滤波系数 `0.3`。
2. `CloseRangeTrackingEnv.reset()` 已有“若 processor 有 `reset()` 则调用”的逻辑。
3. 修复前 `CommandPostProcessor` 没有 `reset()`：其 `_lift_compensation_state` 会跨 episode 保留，且 `set_safety_mode(False)` 的修改不会在下一 reset 恢复到 `start_in_safety_mode`。
4. 因而，在同一 environment 实例连续评估不同场景时，物理初始状态即使相同，第一步 command 仍可受上一回合的 bank/lift compensation 历史影响。

这是一条已证实的 reset-contract 缺陷，且与 R2 “reset base-state hash 一致、轨迹常在第 1 步分歧”的模式相容。

## 最小修复与验证

修复仅在 `CommandPostProcessor` 中：保存配置的初始 safety mode，并增加 `reset()`，将 safety mode 和 lift-compensation filter state 恢复到 episode 初始值。它不改变 reward、VPP、guidance law、PID 参数、场景、checkpoint 或任何已冻结结果。

| 验证 | 结果 |
|---|---|
| `tests/test_overload_rollrate.py` | 25 passed |
| `tests/test_tracking_env_no_prediction.py` | 124 passed, 2 skipped |
| R2 只读审计 | 保持 `runin_protocol_not_reproducible_do_not_train`，270 artifacts / 90 cells 不变 |

## 证据边界

该缺陷证明了一条实际存在、足以造成跨 episode command/trajectory 差异的共同执行链机制；但 R2 没有持久化完整 controller、predictor、FDM reset state。因此不能诚实地将 R2 的全部 90 个 mismatch 单独归因于此缺陷，也不能把修复后的代码写成 R2 已通过。

P1 R3 仍是必需步骤：它使用独立子进程与新环境实例，并保存完整 reset/runtime/FDM envelope。R3 通过前，P2、技能训练、combat finetune、P5 和 formal held-out 继续锁定。
