# P4-v2 Sampler-Feasibility Preflight 设计

**状态：** 仅设计，未授权执行。  
**目的：** 在任何 P4-v2 训练前，检验物理 sampler、phase 语义和动态态势标注是否能为四共享技能提供真实、可审计且 train/dev 分离的覆盖支持。  
**禁止：** 不训练 PPO、不加载或保存新 checkpoint、不运行 combat finetune、不启动 P5/P6/P7、不修改 P4 v1 gate。

## 1. 设计动机

P4 v1 的失败是 declared cell 的零覆盖，而现有记录用固定 `initial_class` 与 phase 交叉计数。对于 post-merge 与 re-entry，初始态势并不等于该时刻的实际相对几何。因此 P4-v2 不能仅扩大训练步数或复制更多初始场景；它必须先验证：

1. 阶段进入是否在 JSBSim 物理轨迹中实际可达；
2. 进入后的动态五态势是否可由冻结 taxonomy 稳定标注；
3. train 与 dev 是否都能观察到每个要求的 skill × state × phase cell；
4. phase-entry handoff 是否能在不伪造飞机状态、不泄漏 heldout 信息的条件下供后续训练使用。

## 2. 冻结不变量

| 层级 | P4-v2 preflight 中的状态 |
|---|---|
| 66-D observation / 3-D normalized VPP action | 不变 |
| P3 v2 encoder | 冻结，且不在 preflight 中训练或微调 |
| 七个 profile、validity mask、三 opponent registry | 只读，不重定义 |
| predicted-target VPP、guidance law、PID、JSBSim、终端规则 | 不变 |
| P4 v1 output 与 gate | 只读负证据，不覆盖、不复跑 |
| canonical 双技能 family | 完全隔离 |

P4-v2 的新增内容只可作为 **sampler metadata sidecar**：`initial_class`、逐步 `dynamic_class`、`phase`、`phase_entry_step`、scenario signature、probe id、opponent id、reference-controller id 与 telemetry-validity flags。它不得进入 66-D observation，也不得以隐式 task bit 影响既有 checkpoint。

## 3. 推荐的 P4-v2 语义

P4-v2 的 coverage contract 改为 `dynamic_state × phase`，而不是 P4 v1 的 fixed `initial_class × phase`。四个 skill 的职责集合在设计上仍保持 26 个 cell：

| Skill | 动态 state | phase | Cell 数 |
|---|---|---|---|
| `pursuit_conversion` | advantage, head_on, neutral | pre_merge, post_merge | 6 |
| `lead_intercept` | advantage, neutral, crossing_entry | pre_merge, post_merge | 6 |
| `defensive_extension` | disadvantage, neutral | pre_merge, post_merge, re_entry | 6 |
| `reentry_recovery` | advantage, head_on, neutral, crossing_entry | post_merge, re_entry | 8 |
| **总计** |  |  | **26** |

每步 `dynamic_class` 必须由已有冻结 ATA/AA taxonomy 规则计算，并在 90 度边界保留既有 transition band。preflight 必须首先用已知姿态单元测试验证角度方向、LOS 定义、镜像对称性与标签稳定性；未通过时停止，不进入物理采样。

## 4. 两条可行的物理实现路径

P4-v2 不得假设 JSBSim 可以任意“传送”到 post-merge 状态。preflight 必须在下列路径中选择一条实际可验证的路径；两条均失败即 stop。

| 路径 | 方法 | 必须验证的事实 |
|---|---|---|
| A. Deterministic run-in + handoff | 从合法 reset 初态开始，用固定、非学习 reference controller 运行至 phase entry，再将控制权交给后续 sampler。 | handoff 后 history、prediction、VPP、backend provenance 连续；无隐式 future state 或 reset 泄漏。 |
| B. Physical snapshot restore | 仅在 JSBSim 支持完整且可验证的物理状态重建时，恢复由合法 run-in 产生的 snapshot。 | snapshot 包含全部必要动力学状态；restore 后一步响应与原轨迹一致；不得仅复制相对几何而遗漏飞行器状态。 |

reference controller 仅用于验证可达性，不是 teacher、Oracle、baseline 或可报告策略。至少报告 zero-VPP 与一个固定 profile-compiled VPP reference 的覆盖差异；它们不参与优化。

## 5. 预检流程与 stop rules

### G0: Evidence and contract preflight

- 读取 P4 evidence bundle 并校验 SHA256。
- 冻结 P3 SHA、66-D feature names、profile YAML、opponent registry 与 sampler source hash。
- 若任何输入缺失、backend fallback、schema mismatch 或 config override 未记录，立即 STOP。

### G1: Dynamic taxonomy sidecar

- 仅实现/验证纯分析 classifier，不修改环境 observation 或 policy input。
- 对五态势、镜像、90 度 transition band、高度差扰动写单元测试。
- 输出逐步 `dynamic_class` 与 `initial_class`，两者必须并存，禁止覆盖原字段。
- 任一已知姿态方向不一致或标签在 band 外抖动，立即 STOP。

### G2: Physical phase-entry reachability

- 对每个候选 scenario 运行非学习 reference rollout，逐高层 step 记录 phase、dynamic class、backend、history bootstrap 与 terminal reason。
- 仅接受 strict JSBSim、无 fallback、无 NaN、无非法 restore 的物理段。
- 若某 required cell 仅能由伪造 reset、future-state 注入或 heldout scenario 产生，立即 STOP。

### G3: Train/dev split and coverage support

train 与 dev candidate signature 必须在初态、mirror、height、distance-speed package、scenario seed 和 snapshot lineage 上完全不相交。每个 required cell 的预检支持门槛为：

- 三个 opponent 各至少出现 1 个有效 episode；
- 每个 opponent 至少记录 20 个有效高层 policy steps；
- 至少来自 2 个不同 scenario signature；
- train 与 dev 分别满足上述门槛。

这些是 sampler-support 门槛，不是胜率、reward 或安全优越性门槛。任一 cell 不满足即 STOP；不得用更多训练替代。

### G4: Handoff/reset integrity

- 若采用路径 A，验证 handoff 后 observation schema、10-step history、P3 embedding、prediction validity 与 provenance 连续。
- 若采用路径 B，验证 restore 后至少一个控制周期的实际 AoA/n_z、位置/速度与原 run-in continuation 的容差一致。
- 任一机制不能证明物理连续性，即 STOP。

## 6. 计划产物（未来执行时才创建）

| 类别 | 计划路径 |
|---|---|
| 设计配置 | `config/experiment/thesis_five_state_shared_skills_geometry_v2_sampler_feasibility.yaml` |
| candidate manifests | `config/experiment/manifests/thesis_five_state_p4_v2_{train,dev}_candidates.yaml` |
| runner | `scripts/run_thesis_p4_v2_sampler_feasibility.py` |
| metadata sidecar | `src/uav_vpp_guidance/analysis/five_state_dynamic_taxonomy.py` |
| tests | `tests/test_thesis_p4_v2_sampler_feasibility.py` |
| compact output root | `outputs/thesis_five_state_shared_intent_v1/p4_v2_sampler_feasibility_<run_id>/` |
| formal report | `reports/thesis_five_state_p4_v2_sampler_feasibility_<run_id>_zh.md` |

候选 ledger 只保存每个接受/拒绝 segment 的 signature、hash、coverage counts、terminal reason 与必要摘要；不得保存全量 raw trajectory。若需要 snapshot，则仅保存通过 G4 的最小可复核 state package，并纳入容量预算与 SHA manifest。

## 7. 预注册决策

| Preflight 结果 | 后续动作 |
|---|---|
| G0-G4 全部通过 | 仅可提出新的 P4-v2 geometry-pretrain 授权申请：新 run ID、新 output root、新 config snapshot；不得复用 P4 v1 output。 |
| 任一 G0-G4 失败 | 停止五态势 shared-skill training lane；将失败作为“当前 JSBSim/sampler/phase contract 不支持该主张”的负证据。 |
| P4-v2 geometry pretrain 后仍不通过 readiness gate | 冻结 P4-v2 结果；不得启动 combat finetune 或转入高层 PPO。 |

本设计本身不构成执行授权，也不改变 P4 v1 的 `combat_finetune_NOT_allowed` 结论。
