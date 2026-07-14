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

当前只完成 design preflight，不存在授权的 v2 JSBSim execution。只有未来 v2 在三个对手
下通过相位 gate 且 handoff continuity 全绿，才可提出单一 `defensive-extension` 或
`re-entry` 低层技能的最小 pilot 授权申请。
