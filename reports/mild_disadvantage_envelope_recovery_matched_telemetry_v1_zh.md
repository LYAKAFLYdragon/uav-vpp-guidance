# 劣势包线恢复探针：控制链遥测补采报告

## 1. 目的与范围

本次工作只为补齐原有 12 条连续轨迹中缺失的控制链遥测，不训练、不调参、不修改专家、对手、奖励、PID、滤波器、动作接口或 JSBSim。补采使用与 Crossing 训练一致的运行契约，新增记录实际虚拟瞄准点、制导指令、执行器输出、滚转响应以及动作执行前状态哈希。

- Source ID：`THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-MATCHED-TELEMETRY-COLLECTOR-V1-20260903`
- 输出目录：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1`
- 运行脚本：`E:/uav-vpp-guidance/scripts/run_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py`
- 验证脚本：`E:/uav-vpp-guidance/scripts/verify_mild_disadvantage_envelope_recovery_matched_telemetry_v1.py`
- 对手：`expert`、`end_to_end`，分别统计
- 初始点：`(625 m, 200 m)`、`(750 m, 250 m)`
- 每个“对手—初始点”重复 3 次，共 12 条回合
- 后端：JSBSim 1.2.3

## 2. 输出和运行完整性

| 检查项 | 结果 | 说明 |
|---|---:|---|
| 回合数 | 12/12 | 两个对手、两个初始点、每格 3 回合 |
| 轨迹总行数 | 4113 | 所有轨迹文件非空 |
| 后端为 JSBSim | 12/12 | `backend_effective=jsbsim` |
| 后端回退 | 0/12 | `backend_fallback_occurred=false` |
| 每回合初始 reset | 12/12 | 汇总记录均为 1 次；未执行中途 reset |
| 有限遥测 | 12/12 | 动作、控制链、滚转和几何字段均可解析 |
| 动作前状态哈希 | 4113/4113 | 每行均为 64 位 SHA-256 字符串 |
| VPP 限幅标记 | 0 行触发 | `clipping_rate` 在记录中为 0 |

状态哈希在 `env.step(action)` 之前，根据本机状态、目标状态、相对状态和观测向量计算，未把当前动作写入哈希输入。因此它可用于复核“动作前状态”，而不是动作结果的哈希。

## 3. 补采字段

每条轨迹均包含以下字段组：

| 控制链环节 | 主要字段 | 用途 |
|---|---|---|
| 动作 | `action_0`、`action_1`、`action_2` | 记录 Crossing 策略输出的三轴动作 |
| 虚拟瞄准点 | `vp_position_neu`、`anchor_pos`、`vp_offset`、`vp_world_offset` | 记录实际虚拟瞄准点、锚点和偏置 |
| 制导 | `raw_nz_cmd`、`raw_roll_rate_cmd`、`raw_throttle_cmd`；`guidance_*` | 比较原始制导指令与传入低层控制器的指令 |
| 执行器 | `elevator_cmd`、`aileron_cmd`、`rudder_cmd`、`throttle_actual` | 记录执行器侧输出 |
| 飞行响应 | `roll_rad`、`p_rps`、`nz_g`、`alpha_rad`、`beta_rad` | 记录滚转角、滚转速率、过载和气动状态 |
| 几何结果 | `range_m`、`range_rate_mps`、`ata_deg`、`aa_deg` | 观察控制动作之后的交战几何演化 |
| 运行溯源 | `state_hash_before_action`、`backend`、`prediction_fallback`、`terminal_reason` | 复核状态连续性和运行有效性 |

其中 `p_rps` 为机体系滚转角速度，`roll_rad` 为滚转角，`nz_g` 为过载；制导指令的具体物理单位沿当前运行时接口定义，未在本次补采中改变其语义。

## 4. 预测器预热情况

独立 verifier 返回 `ok=false`，唯一实际阻塞项为预测器预热阶段的回退：每条轨迹前 9 步 `prediction_valid=false`、`prediction_fallback=true`，共 `108/4113` 行，占 `2.63%`；从第 9 步起预测有效，共 `4005/4113` 行，占 `97.37%`。未发现运行中预测器异常回退、JSBSim 回退或中途重置。

由于本 Source 在配置中明确规定“不允许任何 prediction fallback”，因此本批结果不能标记为严格无回退证据。这里的失败是证据契约失败，不是控制器性能失败。补采数据仍可用于描述控制链字段是否存在，但不能把它升级为无回退的正式物理传递结论。

## 5. 控制链和几何变化

下面给出各单元 3 条回合的平均端点差值。差值定义为第 N 个高层步的值减去第 0 步值，不是同状态反事实干预效应。

| 对手 | 初始点 | 距离变化 4/12/48 步（m） | 距离率变化 4/12/48 步（m/s） | ATA 变化 4/12/48 步（°） | AA 变化 4/12/48 步（°） | 滚转角变化 4/12/48 步（°） |
|---|---|---|---|---|---|---|
| expert | (625, 200) | -271.773 / -80.289 / 3088.689 | 89.499 / 744.551 / 826.214 | 23.815 / 129.791 / 142.594 | 23.758 / 129.411 / 135.354 | 9.103 / 30.932 / 73.710 |
| expert | (750, 250) | -279.140 / -289.805 / 2829.703 | 55.265 / 686.199 / 826.905 | 16.736 / 115.659 / 141.578 | 16.678 / 115.204 / 136.565 | 9.126 / 31.399 / 77.048 |
| end-to-end | (625, 200) | -271.479 / -83.544 / 3004.955 | 89.593 / 740.655 / 798.329 | 23.595 / 129.148 / 121.516 | 23.661 / 129.309 / 139.304 | 9.109 / 30.971 / 73.712 |
| end-to-end | (750, 250) | -278.745 / -294.231 / 2725.712 | 55.818 / 677.232 / 795.775 | 16.566 / 113.671 / 121.366 | 16.615 / 115.113 / 140.009 | 9.130 / 31.440 / 77.114 |

4 步内四个单元均表现为距离继续减小、滚转角约增加 9°；12 步时距离率和视线几何已发生明显变化；48 步时距离重新增大，且 ATA、AA 已大幅偏离初始值。这说明轨迹中确实存在可观测的滚转和几何演化，但它不能单独证明恢复策略带来了有利的几何变化。

## 6. 原始制导指令到执行器侧的记录

在预测有效行上，原始指令与记录的制导指令的平均绝对差如下。该统计用于说明接口差异的量级，不等同于控制保真率。

| 对手 | 初始点 | `nz_cmd` 平均绝对差 | `roll_rate_cmd` 平均绝对差 | `throttle_cmd` 平均绝对差 |
|---|---|---:|---:|---:|
| expert | (625, 200) | 0.05527 | 0.02515 | 0.00264 |
| expert | (750, 250) | 0.06193 | 0.01796 | 0.00196 |
| end-to-end | (625, 200) | 0.05313 | 0.01725 | 0.00250 |
| end-to-end | (750, 250) | 0.05363 | 0.01893 | 0.00196 |

实际副翼指令、滚转角和 `p_rps` 在全部轨迹中均有记录。`aileron_cmd` 在部分样本达到归一化边界 -1 或 1；而当前 `clipping_rate` 只反映虚拟瞄准点动作的限幅标记，不能把它解释为“执行器从未饱和”。因此本批数据可以确认执行器字段已补齐，但尚不足以给出执行器饱和比例或完整延迟估计。

## 7. 按对手的结果摘要

- `expert`：6 条回合中，`(625,200)` 三条均运行至 512 步，`(750,250)` 三条在约 275 步终止；稳定包线进入为 0/6。
- `end_to_end`：6 条回合在约 271--314 步之间终止；稳定包线进入为 0/6。
- 两个对手均为 JSBSim，均无后端回退，均记录了三轴动作、VPP、制导、执行器、滚转和几何字段。
- 两个对手都没有通过预先设定的“进入稳定包线并保持 10 步”恢复门，合计为 0/12。

## 8. 结论与后续边界

本次补采完成了原有记录缺失的控制链遥测：现在可以沿“动作—虚拟瞄准点—制导—执行器—滚转—几何”顺序读取数据，也可以用动作前状态哈希定位每一个控制步。它解决了“中间链路完全不可见”的工程问题。

但严格证据门未通过，原因是固定的预测器预热回退；同时本批没有同一动作前状态下的无干预配对轨迹。因此，本次结果只能支持以下结论：

> 在两个主对手和两个不可行初始点上，控制链字段可以连续记录，滚转和交战几何也发生了可观测变化；由于存在预热回退且缺少同状态反事实，当前数据不能证明恢复策略形成了跨对手稳定的有利几何控制方向。

据此，暂不训练共享技能、不扩大 PPO、不调奖励、PID 或滤波系数，也不把恢复门失败直接解释为“缺少新的防御技能”。若继续推进，应先单独建立不含预热回退的 observer-release 契约，或将预热阶段作为预注册的独立阶段处理，并重新核验其可比性；在此之前，本批保持为研发证据，不升级为论文安全的严格运行结果。

## 9. 文件索引

- 汇总结果：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1/episode_results.csv`
- 独立验证：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1/verification.json`
- 总清单：`E:/uav-vpp-guidance/outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1/pilot_manifest.json`
- 对手清单：`.../matched_telemetry_v1/expert/run_manifest.json`、`.../matched_telemetry_v1/end_to_end/run_manifest.json`
- 逐回合轨迹：`.../matched_telemetry_v1/<opponent>/trajectories/*.csv`

