# V17 End-to-End 相位占用只读归因审计

## 结论

本审计只统计冻结 V17 record 已保存的 first-pass、phase trace、handoff 与 terminal reason。
它不含逐步 target speed、altitude、specific energy 或 attack-zone telemetry，故不能将覆盖不足
归因于对手强度、目标早期轨迹或飞控执行链。

end-to-end 的缺口发生在 first-pass 之后：所有 24 条 episode 都锁存了 first-pass，但仅 4 条
进入 post-merge/re-entry 并完成 handoff；另 20 条 first-pass 后始终没有 post-merge phase。
因此 V17 的失败是 frozen envelope 中 post-first-pass phase occupancy 不足，而不是 first-pass
检测失败或 74-D 数值合同失败。

## 各对手汇总

| Opponent | Records | First pass latched | Post-merge episodes | Re-entry episodes | Handoff reached | First-pass without post-merge |
|---|---:|---:|---:|---:|---:|---:|
| end_to_end | 24 | 24 | 4 | 4 | 4 | 20 |
| expert | 24 | 24 | 9 | 9 | 9 | 15 |
| independent_ppo_vpp | 24 | 24 | 11 | 11 | 11 | 13 |

## 证据边界

- 本审计不重跑、不训练、不改变 path context、reward、VPP、guidance 或 PID。
- `first_pass_without_post_merge` 仅是标签轨迹事实，不能证明相位定义、target policy 或物理动力学中的任何单一原因。
- 如需进一步解释原因，必须以新 Source ID 预注册一个带 target/own kinematics 和 phase telemetry 的独立非学习 collector；不能修改或补跑 V17。
