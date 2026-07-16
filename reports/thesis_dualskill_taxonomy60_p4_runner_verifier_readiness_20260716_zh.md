# Taxonomy60 P4 Formal Runner / Verifier Readiness Record

**Source：** `THESIS-DUALSKILL-TAXONOMY60-3OPP-V1`
**范围：** 实现 formal runner、独立 verifier 和运行时 telemetry 合同；不执行 900 条正式 episode，不读取任何正式性能结果。

## P4 决定

`PASS FOR FORMAL-EXECUTION AUTHORIZATION REVIEW ONLY`。

P4 runner 已能构造唯一的 `3 opponents x 5 methods x 60 scenarios = 900` 记录计划，并且默认不创建输出目录。真实执行还要求同时满足：显式 `--execute --acknowledge-formal-source`、clean Git worktree、空 formal output root 和 P3 preflight pass。当前开发 worktree 未冻结为 clean SHA，因此本记录不授权执行 900 条正式 Source。

## 交付物与合同

| 交付物 | 已验证的作用 |
| --- | --- |
| [formal runner](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/run_thesis_dualskill_taxonomy60_formal.py) | 生成 3 个严格 JSBSim child runs，每个 300 条；拒绝脏工作树、非空 output root 和未确认的正式执行。 |
| [formal verifier](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/scripts/verify_thesis_dualskill_taxonomy60_formal.py) | 先验证完整 900-record universe、哈希、后端、checkpoint receipt、warm-up 和 telemetry；仅通过后才读取 outcome 并生成 `taxonomy60_route_decision.json`。 |
| [No-Go v3](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/config/experiment/decision_contracts/thesis_dualskill_taxonomy60_no_go_v3.yaml) | verifier 输出全部 10 个 precedence 分类候选、唯一 `selected_study_route`、恢复状态和后闭合 memo 边界。 |
| [telemetry contract test](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/tests/test_thesis_dualskill_taxonomy60_telemetry_contract.py) | 固定 3-D VPP action、physical AoA、guidance/PID、`n_z` command/actual、clipping 与 terminal aliases。 |

有效的 900-record universe 默认选择 `paper_closure`，并不因 routing-sensitive 或 capacity-boundary candidate 自动打开训练。任何缺字段、backend/checkpoint fallback、prediction warm-up 漂移、bootstrap/lock 漂移或矩阵不完整均选择 `implementation_invalid`，不得分析 partial outcome。

## P4 Telemetry Smoke

P4 发现原有 recorder 没有显式 3-D `vpp_action`、结构化 guidance/PID、physical AoA、clipping 和 terminal aliases。该问题在新的 15 条 disposable strict-JSBSim smoke 中被单独验证：

| 运行 | 结果 | 解释 |
| --- | --- | --- |
| `P2-SMOKE-R2` | strict base smoke `15/15` pass；formal telemetry fail | `aoa_deg` 非有限。它是记录字段缺失，不是控制失败。 |
| `P2-SMOKE-R3` | base smoke `15/15` pass；formal telemetry `15/15` pass | JSBSim state 现由 body-axis `atan2(w, u)` 输出 `alpha_rad`，recorder 输出有限 `aoa_deg`。 |

R3 的固定工程事实：三个对手、五个方法、每条 20 high-level steps；每条都有 9 个 warm-up fallback 和 11 个 valid prediction steps；无 backend/checkpoint fallback。两轮均为 `paper_evidence=false`，未查看或比较 outcome。

**输出根：**

- `E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE-R2`
- `E:/uav-vpp-guidance-thesis-evidence/THESIS-DUALSKILL-TAXONOMY60-3OPP-V1-P2-SMOKE-R3`

## 下一步

1. 将 P1--P4 代码、manifest 和收据固定到一个 clean SHA，不推送远程。
2. 在该 clean SHA 重跑 P3 独立 preflight，确认 output root 仍为空且 assets/data hash 未漂移。
3. 独立复核 P4 plan、verifier 和 No-Go v3；通过后才可执行唯一一次 900 条正式 Source。
4. 正式结果出现后只运行 verifier 和只读分析，不重训、不换 checkpoint、不替换场景。
