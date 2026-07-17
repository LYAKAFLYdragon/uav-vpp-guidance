# Pressure-Exposed M0--M4 紧凑消融完成记录

- Source ID：`THESIS-DISADVANTAGE-PRESSURE-EXPOSED-M0M4-3OPP-V1`
- 执行状态：完成且独立验证通过。
- 输出根目录：`E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1`
- 冻结矩阵：3 个对手 x 16 个初始 `disadvantage` 场景 x 5 个名义方法 = 240 条 strict-JSBSim episode。
- 分析边界：这是共享压力 cohort 上的描述性模块比较，不是五个独立处理的充分因果消融，也不支持普适优越性声明。

## 有效性

- `formal_verification.json` 为 `pass`，完整覆盖 240/240 条预注册身份；三条 child run 均为 80 条、无记录失败。
- 原始轨迹逐步具备 `(AA, ATA)` taxonomy、JSBSim 本机/目标位置、3-D VPP action、tactical-basis world offset、guidance、AoA、`n_z` command/actual 与 clipping telemetry。
- 动态 taxonomy 并非静止：各对手均经历 disadvantage、transition、neutral、head_on 与 advantage 步。因此，此结果不是“场景标签或遥测未进入真实交战”的工程失败。

## 结果与边界

- 三个对手、全部 15 个 method x opponent cell 均为 16/16 `timeout_hp_disadvantage`。没有 win 或 draw，故终端胜负对该 cohort 没有判别性。
- M3 `canonical_ppo` 与 M4 `ppo_unbootstrapped_crossing` 在三个对手中均为 16/16 VPP action 完全相同。原因是全部场景的固定 task key 为 `head_on`，M4 关闭的 crossing bootstrap 不会被调用。
- M0 `fixed_head_on` 与 M2 `static_oracle_gate` 均使用 head-on 分支；虽可出现很小的数值轨迹差异，但终端类别完全相同，不能据此归因于静态 gate 的有效选择。
- M1 `fixed_crossing` 的几何轨迹不同，但也没有改变终端类别。它可作为现有 crossing specialist 的受压描述性对照，不能证明新技能库缺口或 routing 优越性。

## 正式决策

将本 Source 冻结为“**有效、可复核、但终端非判别且 M3/M4 treatment-identical 的负证据**”。不允许修改其场景、任务键、奖励、政策、VPP、制导、PID、对手或 seed 后重跑。

后续若需补足模块消融，只能新建独立 Source：先用 non-learning preflight 证明同一物理 cohort 在至少两个不同 task/interface 条件下都具有可观测且非退化的动作差异，再比较相应的消融开关。不得把这 240 条结果写成“PPO 在 disadvantage 中的五方法优势”。

## 产物

- [验证回执](E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1/formal_verification.json)
- [执行回执](E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1/formal_execution_manifest.json)
- [逐 episode 指标](E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1/analysis/episode_metrics.csv)
- [逐对手方法汇总](E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1/analysis/method_opponent_summary.csv)
- [配对 PPO 差值](E:/uav-vpp-guidance-disadvantage-capability-v1-results/pressure_exposed_m0m4_compact_ablation_v1/analysis/paired_ppo_deltas.csv)
