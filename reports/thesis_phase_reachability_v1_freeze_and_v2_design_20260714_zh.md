# Phase Reachability v1 冻结与 v2 设计线

## v1 负证据冻结

`THESIS-FIVE-STATE-PHASE-FEASIBLE-DISADVANTAGE-V1` 已完成三对手 strict-JSBSim
reference rollout。expert 为 `6/12` 合格，但 end-to-end 为 `4/12`、independent
PPO/VPP 为 `5/12`，未达到每对手至少 `0.50` 的连续 post-merge/re-entry 门槛。

因此 `phase_feasible_envelope_not_established` 是不可重跑的负证据：不得修改 v1
的场景、seed、VPP、制导、PID、checkpoint 或门槛后重跑；不得由此启动共享技能训练、
combat finetune 或高层 PPO。

## v2 边界

v2 是独立的设计验证线，不是 v1 repair。它只允许路线 A：冻结、非学习的
`legacy_static_oracle_task_gate` 从合法 JSBSim reset 连续运行至 first-pass，随后
同一 simulator episode 交给 **observe-only** phase sampler sidecar。

禁止 reset、物理 snapshot restore、future-state injection 和 action replacement。v2
必须逐 handoff 验证 episode/JSBSim lineage、observation schema、history、预测器、VPP、
guidance 与 PID state 的连续 hash；任一不连续即 fail closed。

设计 preflight 后，独立执行线 `THESIS-PHASE-REACHABILITY-V2-RUN-IN-HANDOFF-R1` 已完成
36 个 strict-JSBSim evaluation-only episode，并在三对手下通过 phase gate 与 handoff
continuity。正式结果、哈希、观测边界和“不得自动启动训练”的决策见
[v2 R1 执行结论](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_phase_reachability_v2_r1_execution_20260714_zh.md)。
该正向 reachability 证据仅允许未来提出并审核一个独立的单技能 feasibility pilot；当前
`defensive-extension` / `re-entry` 训练、四共享技能、combat finetune 与 P5 均仍未授权。

随后完成的只读 [v2--P4 sampler 对齐审计](E:/uav-vpp-guidance-five-state-heldout-envelope-v1/reports/thesis_phase_reachability_v2_p4_sampler_alignment_audit_20260714_zh.md)
以 `sampler_mismatch` 为唯一归因：v2 的三对手均可物理进入目标 phase，但 P4-v2 physical
sampler 仅通过 `80/156` dynamic-state x phase support 行。该审计同时确认现有 v2 raw
episode 无法精确复构 `66-D -> 3-D VPP` 学习对，因此只作为观测边界而非第二归因；四共享技能
与最小 pilot 继续锁定。
