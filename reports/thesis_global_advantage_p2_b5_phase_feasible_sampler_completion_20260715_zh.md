# P2-B5 Phase-Feasible Sampler 完成与负证据归档

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1`
**结论：** `phase_feasible_data_contract_not_established`
**执行状态：** `completed_once_execution_closed`

## 1. 执行完整性

- 计划与实际均为 12 个独立 `disadvantage` continuous run-in 场景 x 3 frozen opponent，即 36 条 strict-JSBSim record。
- 未发现 `runner_failure_manifest.json`；输出包含 36 个逐场景 record、gate 与 run manifest。
- 全局 scenario-application、raw-SI phase replay 和 step-0 `pre_merge` 语义均通过。
- Gate SHA-256：`8beaa43bb2c36128555821161aca7c7a657ac9e80c7d43bba6dd3080b290257d`。
- Run-manifest SHA-256：`9db8dc1f3a2225750889fc91a887b3fcc91d56c3c69c82b29a9fcaa600ec6778`。
- 输出只用于研发决策，`paper_safe=false`；它不是候选方法性能结果。

## 2. 分对手 Gate

| Opponent | Episodes | Qualifying episodes | Valid target steps | Signatures | Mirrors | Gate |
|---|---:|---:|---:|---:|---:|---|
| expert | 12 | 1 | 12 | 1 | 1 | fail |
| end-to-end | 12 | 3 | 118 | 3 | 2 | pass |
| independent PPO/VPP | 12 | 0 | 0 | 0 | 0 | fail |

Gate 按 opponent 独立判定，不能以 end-to-end 的通过覆盖 expert 或 independent PPO/VPP 的失败。结果为 `training_unlocked=false`，也不允许起草 defensive-extension pilot 执行授权。

## 3. 只读归因

这不是 backend、单位、66-D 构造或 action fallback 的失败：所有已出现的 target motif step 都同时满足有限 66-D、有限 action、strict JSBSim、prediction valid、无 prediction fallback、无 reset/padding 与连续 handoff。因此失败不应归因为 P3 encoder、VPP action contract 或 JSBSim 后端失效。

跨对手的物理/行为覆盖才是缺口：

- expert 只有一个 first-pass 连续 episode，first-pass 后出现 12 个 `disadvantage:post_merge` step，低于 2 episode、20 step、2 signature 和 2 mirror 的全部门槛。
- end-to-end 有 4 个连续 first-pass episode，其中 3 个提供 118 个有效 target step，覆盖了 3 个 signature 和两个镜像方向；这只证明该 opponent 条件下的可达性。
- independent PPO/VPP 有 2 个连续 first-pass episode，但 first-pass 后分布为 neutral、crossing-entry 与 advantage/re-entry，没有任何 `disadvantage:post_merge/re_entry` target step。它是当前跨对手数据契约无法成立的决定性反例。

## 4. 正确的研究结论

本轮不能证明“缺少 defensive-extension 技能”，也不能证明“现有技能库足够”。它只证明：在本次冻结的 head-on continuous run-in、目标 motif 和三个对手组合下，尚未建立能够用于跨对手训练/比较的 `66-D -> 3-D VPP` disadvantage post-merge/re-entry 数据契约。

因此不得重跑 B5、替换场景、提高训练步数、改变 reward 或提前开始四共享技能/P5/P6/formal held-out。若要继续五态势路线，下一步必须是新的、独立 Source ID 的对手条件目标几何可达性设计与预注册，而不是修补本次 B5 负证据。
