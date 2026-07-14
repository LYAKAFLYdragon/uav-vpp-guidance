# Defensive-Extension Pilot V1 失败机制归因审计

**Source ID：** `THESIS-DEFEXT-RANGEEXT-PILOT-V1-ATTRIBUTION-R1`
**模式：** 只读；不训练、不调参、不重跑 Heldout24。
**Formal simulation SHA：** `c2ee6d3af2094c827b74419e8631533226f1fef9`

## 1. 单一结论

本轮能够确认候选策略存在可重复的、方法特异的 terminal safety signal，但无法从冻结产物识别该信号究竟起源于 VPP 动作、制导转换、PID 响应还是闭环对手轨迹分歧。正式归因结论为 **`causal_mechanism_not_identifiable_from_frozen_artifacts`**。

该结论不改变原 pilot 的 `safety_or_contract_no_go`，也不支持‘现有技能库明确缺少 defensive-extension’。

## 2. 新发现：预 handoff 配对并不等价

三个方法在 handoff 之前使用相同 run-in controller，理论上同一 opponent/scenario/seed 应具有一致的 handoff metadata。冻结记录却显示多处 handoff 到达状态、进入步数或未到达时的 terminal reason 不同，因此按 scenario signature 配对不等于从相同物理状态开始比较。

| Opponent | 场景数 | Handoff mismatch | Candidate-head loose/strict | Candidate-crossing loose/strict |
|---|---:|---:|---:|---:|
| end_to_end | 24 | 15 | 11/7 | 12/6 |
| expert | 24 | 10 | 9/5 | 11/6 |
| independent_ppo_vpp | 24 | 7 | 6/3 | 6/4 |

按最低限度的 handoff metadata 等价条件筛选后，三组对手的有效数量均低于预注册 claim-ready 门槛 `8`。冻结记录没有 handoff 时刻的完整状态哈希，因此 strict subset 仍不等同于已证明物理状态逐值一致；这些 delta 只作为敏感性诊断，不用于替换或重算原 formal gate。

| Opponent | Δ vs head loose/strict | Δ vs crossing loose/strict |
|---|---:|---:|
| end_to_end | -0.0108/-0.0154 | -0.0280/-0.0290 |
| expert | -0.0081/-0.0053 | -0.0143/-0.0357 |
| independent_ppo_vpp | -0.0082/-0.0035 | -0.0305/-0.0443 |

## 3. Candidate-specific safety cases

以下 episode 在三个方法中具有相同 handoff step；candidate 发生 ego crash/OOB，而两种 frozen baseline 均存活。因此可以确认风险发生于 handoff 后的候选闭环，但不能进一步定位到 VPP/guidance/PID 中的某一层。

| Opponent | Scenario | Speeds own/target | Handoff | Candidate terminal | Head-on terminal | Crossing terminal |
|---|---|---:|---:|---|---|---|
| expert | defext_hold_d|co_altitude|positive | +225/+430 | 51 | ego_crash_or_out_of_bounds | timeout_hp_disadvantage | timeout_hp_disadvantage |
| independent_ppo_vpp | defext_hold_c|own_above|positive | +215/+420 | 47 | ego_crash_or_out_of_bounds | timeout_hp_disadvantage | timeout_hp_disadvantage |
| independent_ppo_vpp | defext_hold_d|co_altitude|positive | +225/+430 | 52 | ego_crash_or_out_of_bounds | timeout_hp_disadvantage | timeout_hp_disadvantage |

三条风险记录都位于 positive-mirror、高速度 `defext_hold_c/d` 条件；这是局部模式线索，不是统计泛化结论。

## 4. Telemetry evidence boundary

冻结 `heldout/records.json` 只保存 episode-level 汇总。以下逐步字段没有落盘：

`normalized_vpp_action`, `phase`, `dynamic_state`, `vp_forward_bias_m`, `vp_lateral_bias_m`, `vp_vertical_bias_m`, `nz_cmd`, `actual_nz_g`, `ego_attack_aoa_deg`, `own_speed_mps`, `own_altitude_m`, `range_m`, `range_rate_mps`, `specific_energy_height_delta_m`, `altitude_delta_m`, `target_in_attack_zone`, `ego_in_attack_zone`, `saturation_flag`。

因此无法恢复首个 action divergence、VPP 三轴变化、n_z command/actual、AoA、速度-高度裕度和 PID saturation 时间线。重新运行相同 Heldout24 会违反一次性使用和 stop rule，不能用于修补该证据缺口。

## 5. 第一性原理决策

- **技能库缺口：未证明。** Candidate 的几何变化幅度不足以跨对手达到预注册实用改善，同时存在安全退化。
- **组合/路由假设：仅为 plausible。** Frozen crossing 在失败场景中保持存活并提供相近几何能力，但现有记录不足以证明 phase-aware composition 能解决问题。
- **执行安全假设：存在 terminal signal，但机制未定位。** 不得直接归因于 PID、制导律或 VPP action。
- **下一道门：协议修复而非性能训练。** 任何新 pilot 前，必须先在全新 dev-only 场景证明 pre-handoff deterministic equivalence，并逐步持久化 VPP/guidance/PID telemetry。

四共享技能、combat finetune、P5/P6/P7 继续锁定；不得增加本候选训练步数、调整 reward、删除失败场景或重跑 Heldout24。
