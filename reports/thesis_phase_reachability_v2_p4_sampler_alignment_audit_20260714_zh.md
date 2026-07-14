# V2 Raw Episode 与 P4 Sampler 对齐审计

## 唯一归因

`sampler_mismatch`：Physical first-pass is observed in every v2 episode, while the separately frozen P4-v2 physical sampler fails its dynamic-state x phase support gate.

这不是训练结论，也不改变 P4 v1/P4-v2 的既有负证据。

## 直接证据

- v2 raw episode：36 条；first-pass 连续可观测：36/36。
- P4-v2 physical sampler：通过覆盖行 80/156；`all_cells_supported=False`。
- P4 v1 defensive-extension 零覆盖 cell：disadvantage:post_merge, disadvantage:re_entry, neutral:post_merge, neutral:re_entry。
- exact 66-D -> 3-D VPP pair：0/36 条；因此不能用本 raw corpus 起草 pilot 预注册。

## 三对手分别统计

| Opponent | Episodes | First pass | P4 post-merge | Dynamic disadvantage post/re-entry steps | Exact 66-D pairs |
|---|---:|---:|---:|---:|---:|
| expert | 12 | 12 | 12 | 1391 | 0 |
| end_to_end | 12 | 12 | 12 | 933 | 0 |
| independent_ppo_vpp | 12 | 12 | 12 | 1381 | 0 |

## 预测与执行遥测

| Opponent | Prediction valid | Fallback | Mean VPP forward (m) | Mean VPP lateral (m) | Max abs n_z command | Mean abs roll-rate command | Mean throttle | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| expert | 0.9654 | 0.0346 | 2022.9 | 1012.2 | 4.757 | 0.117 | 0.709 | 0.0000 |
| end_to_end | 0.9654 | 0.0346 | -828.2 | 1857.6 | 4.094 | 0.123 | 0.708 | 0.0000 |
| independent_ppo_vpp | 0.9652 | 0.0348 | 1953.6 | -39.8 | 4.906 | 0.128 | 0.708 | 0.0000 |

## P4 Zero-Cell 排查

| 检查项 | 证据 | 结论 |
|---|---|---|
| 物理 phase 不可达 | v2 三对手 36/36 均记录 first-pass；所有对手均有动态 disadvantage post-merge/re-entry steps。 | 排除为主归因。 |
| 提前终止 | v2 36 条均有 first-pass；P4-v2 physical preflight 的 120 条均为 horizon timeout。 | 没有证据表明早终止是零 cell 的主因。 |
| 采样 horizon / 场景 / 行为不同 | P4-v2 使用 20 reset signature、两种非学习 reference 和 80-step horizon；v2 使用 oracle、12 个独立 run-in 场景和 260-step horizon。 | sampler mismatch 的直接组成。 |
| Phase 语义不同 | P4 从首次 range <= 1000 m 开始 post-merge；v2 以 close-range 后 range opening 的 first-pass 为边界。 | sampler mismatch 的直接组成。 |
| 态势记账不同 | P4 v1 gate 以 fixed initial_class 计数；v2/P4-v2 sidecar 按逐步动态 taxonomy 计数。 | sampler mismatch 的直接组成。 |
| 66-D 输入/动作复构 | v2 raw 没有保存 base vector、history、P3 embedding、profile 或 normalized action。 | 审计证据边界，阻止 pilot，不作为第二归因。 |

## 边界与决定

v2 的 19-D raw telemetry 足以审计动态 taxonomy、P4 phase 语义、first-pass、预测有效性及 VPP/PID 响应；但未保存原始 16-D observation vector、十帧 history、P3 embedding、intent profile 与原始 normalized 3-D action。该事实是本审计的 evidence boundary，不是第二个根因标签。

P4 的训练 gate 按固定 initial_class 记账，P4-v2 physical preflight 则按动态 state x phase 记账；其场景、reference controller 和 80-step horizon 又不同于 v2 的 oracle、12 场景和 260-step horizon。因此 v2 的可达性不能回填 P4 失败 cell，也不能解锁训练。

结论：四共享技能和最小 pilot 继续锁定。下一动作仅可为独立、先审议的 sampler-contract design，不得复用本审计结果启动训练。
