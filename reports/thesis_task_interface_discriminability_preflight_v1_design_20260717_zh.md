# Task-Bit / Interface 可判别性 Preflight V1

## 问题

刚完成的 pressure-exposed M0--M4 Source 有效但不具备 M3/M4 消融解释力：所有物理场景的 legacy task key 都是 `head_on`，所以 crossing bootstrap 从未被调用，M3/M4 的 VPP action 在三对手下均完全相同。

## 唯一目标

先验证“相同物理 reset 下，仅切换 legacy task bit 与 bootstrap 开关”是否产生可观察的接口处理差异。该 preflight 不测胜率、不比较性能、不把 `crossing_feasible` 当作这四个 disadvantage 几何的真实标签；它是受控反事实接口测试。

## 冻结设计

- 物理初态：从既有 pressure cohort 预先选定 4 条物理 scenario，覆盖两个签名与正负镜像。
- 成对上下文：每条物理初态复制为 `head_on` task bit=0 和 `crossing_feasible` task bit=1；位置、速度、航向、高度、seed 必须逐字段相同。
- 方法：仅 `canonical_ppo` 与同 checkpoint 的 `ppo_unbootstrapped_crossing`。
- 对手：expert、end_to_end、independent PPO/VPP，始终分开记录。
- 规模：未来若单独授权 runtime probe，为 `4 physical x 2 contexts x 2 methods x 3 opponents = 48` 条 20-step 非学习 episode。

## Runtime Gate

每个主对手（expert、end-to-end）必须满足：

- 三个以上 physical pair 的 reset receipt 除 task context 外完全一致。
- 19-D initial observation 只有命名为 `is_crossing` 的特征发生 0/1 改变。
- crossing context 下，M3 的 bootstrap telemetry 为 active，M4 为 inactive。
- M3/M4 至少三个 pair 的首个 effective mode 或首个 VPP action 有差异。
- strict-JSBSim、prediction warm-up、原始物理状态、VPP/guidance/AoA/`n_z` telemetry 均完整，无 fallback/reset。

任一主对手未通过，正式 M3/M4 消融被禁止；结论是当前 task-bit/bootstrap 接口在该物理包线内不可判别，而不是 PPO 或专家库性能失败。通过后也只允许起草新的正式 Source，不自动执行性能实验。
