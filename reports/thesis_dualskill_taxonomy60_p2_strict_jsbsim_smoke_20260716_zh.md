# Taxonomy60 P2 Strict-JSBSim Smoke Record

**Checklist：** [thesis_master_evidence_closure_checklist_v2_20260716_zh.md](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_master_evidence_closure_checklist_v2_20260716_zh.md)

**正式 Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1` 未创建。

**Smoke Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE`，工程验证专用，`paper_evidence=false`。

**决定：** `PASS WITH FROZEN WARM-UP BOUNDARY - P3 独立复核获准开发；900 条正式 Source 仍未授权。`

## P2-R0：无效工程尝试

首次运行使用了 current worktree 默认的 JSBSim root。严格后端在任何 episode 开始前失败：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1/envs/JSBSim/data` 不存在。expert 子运行中的五个方法均没有产生 trajectory 或性能记录。

该结果是单一、可复现的工程根因，属于 No-Go 决策器允许的 smoke repair。未修改场景、checkpoint、策略、奖励、VPP、guidance、PID 或任何正式输出根。

## P2-R1：实际 strict-JSBSim 结果

R1 将 `--jsbsim-root` 显式固定为 `E:/uav-vpp-guidance-can20260705`。该数据目录的冻结 manifest SHA-256 为 `18fa1c4107fbcccb41ecdcf31e565ec5809fac16f45d1e1c006c6f691c42c10a`，与 P0 的 JSBSim F-16 asset contract 一致。

| 项目 | 验证值 |
| --- | --- |
| 输出根 | `E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE-R1` |
| 运行代码 SHA | `295c474` |
| 矩阵 | `5 methods x 3 opponents x 1 disposable crossing-entry scene = 15` |
| 每个 child run | 5 records、20 high-level steps、`status=completed`、`paper_safe=false` |
| 后端 | 三个 child run 均为 strict `jsbsim`，backend fallback 均为 0 |
| runner failures | expert、end-to-end、independent PPO/VPP 均为空 |
| artifact contract | 三个 child run 均为 valid |
| execution manifest SHA-256 | `18ec28540f9dd01958024c92a2f84bffc84885bc7da6046066d704ffc5ca4485` |
| independent verification SHA-256 | `fae0096c510d72ccc317438e29d162af37cd078821df3c37c8d8290ddbabed42` |

独立 verifier 已确认：三对手中每个方法恰有一条记录，crossing-v3 的 18-D adapter 可用，VPP/guidance/`n_z`/prediction telemetry 可读，canonical PPO 的首步 bootstrap 为 active，而 `ppo_unbootstrapped_crossing` 的首步 bootstrap 显式 disabled/inactive。该检查不读取胜负、HP 或任何比较性能指标。

## 冻结预测 warm-up 边界

所有 15 条 R1 episode 的 pattern 完全一致：总长 20 个 high-level steps，其中 9 个 step 记录 `prediction_fallback=true`，随后 11 个 step 记录 `prediction_valid=true`。这反映历史窗口尚未填满时的冻结预测器接口状态；它不是 backend fallback、checkpoint fallback 或运行时异常。

因此 P2 只能支持以下工程结论：预测 receipt 完整且可审计，**但 predicted-target anchor 并非在 episode 的前 9 个 high-level steps 全程有效**。后续论文不得写成“预测全程可用”。

P3 必须在创建正式 Source 前把下列边界写入 verifier/decision contract：

- 允许且必须记录前 9 个 high-level steps 的固定 prediction warm-up fallback。
- 从第 10 个 high-level step 起，任何 prediction fallback、非有限 prediction telemetry、backend fallback 或 checkpoint substitution 都使该正式 record 无效。
- 正式图表和 Discussion 必须将 warm-up 作为部署接口限制，而不是把它解释为性能收益或安全保证。

这是一项 receipt/语义收紧，不是新的控制器、训练、奖励或场景版本。

## 下一步

P3 仅可建立正式配置、full-universe verifier 与独立 preflight。它必须复核 60 场景、五方法、三对手、资产和输出根，并将上述 warm-up boundary hash 绑定到 manifest；通过前不得执行 900 条正式 episode。
