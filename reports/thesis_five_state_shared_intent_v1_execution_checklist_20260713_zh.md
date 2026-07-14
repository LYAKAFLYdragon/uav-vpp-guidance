# Five-State Shared-Intent v1 执行 Checklist

**版本：** v1.0（访谈冻结版）  
**日期：** 2026-07-13  
**研发身份：** `noncanonical_thesis_extension`，作为硕士论文拟议的新主线；在正式 held-out 通过前，**不得**替代或修改 `CAN-20260705` 双技能 canonical baseline。  
**前置证据：** [Taxonomy30 冻结式消融执行 Checklist](E:/uav-vpp-guidance/reports/thesis_taxonomy_frozen_ablation30_execution_checklist_20260713_zh.md)、[Taxonomy30 go/no-go 决策](E:/uav-vpp-guidance/reports/taxonomy30_go_no_go_decision_zh.md)、[三项核心证据缺口路线图](E:/uav-vpp-guidance/reports/master_thesis_three_core_gaps_and_action_plan_20260713_zh.md)。

## 0. 冻结的研究问题与边界

### 0.1 研究问题

在不改变 JSBSim、3-D VPP guidance law、预测目标 VPP 锚点和 PID/飞控安全链的前提下，验证“**共享技能库 + 全局 tactical-intent profiles + 时序几何表征**”能否比现有双技能 PPO 更完整地覆盖五类初始态势，并在不同对手条件下保持可解释、物理可执行的机动决策。

### 0.2 不可变约束

- [ ] 现有 `CAN-20260705` checkpoint、config、formal artifact、论文主结果和 canonical output 不修改、不覆盖、不重跑。
- [ ] 当前双技能 PPO 是冻结主要 baseline；旧 head-on/crossing checkpoints 仅可用作 baseline、demonstration teacher 或 feature-name weight transplant 的来源。
- [ ] 保留已冻结的 trajectory predictor 与 `predicted_target` VPP 输入接口；**v1 禁止加入 prediction reward、prediction routing reward 或调整 predictor**。
- [ ] VPP 动作头继续输出三维连续偏置；guidance law、PID、JSBSim aircraft、终端规则、AoA60 限制保持为可追溯的独立配置，不因训练结果回看修改。
- [ ] `Taxonomy30` 永远只作旧双技能系统的诊断 baseline，不用于新系统 checkpoint 选择或正式 held-out 统计。
- [ ] 所有新 config、checkpoint、report 和 output 均以 `thesis_five_state_shared_intent_v1` 或 `noncanonical_thesis_extension` 命名。

### 0.3 预注册的 v1 主成功门槛

以同场景、同 opponent、同 evaluation seed 下的冻结双技能 PPO 为配对 baseline；所有 opponent 分开报告，不跨 opponent 合并成“总体优越”。

| 门槛 | 正式判定 |
|---|---|
| 保留性 | 在每个 opponent 的 heldout60 中，Head-on 与 Crossing-entry 的 shared-intent PPO 对双技能 PPO 配对胜率差均不得低于 `-0.05`。 |
| 新态势实用增益 | Advantage、Disadvantage、Neutral 至少两类分别在不少于 `2/3` opponents 中达到配对胜率差 `>= +0.05`，且剩余 opponent 不低于 `-0.05`。 |
| 安全回归 | 每个 opponent 及所有 episode 合并审计中，ego crash/OOB rate 不得高于相应双技能 baseline `+0.05`。 |
| 证据完整性 | 三个训练 seeds 均完成、每个 cell 报告 `N_total/N_resolved`、terminal mix、paired outcome、Wilson CI 和 scenario-clustered paired-bootstrap CI。 |

`+/-0.05` 是预注册的实用门，不是统计显著性或普适优越性的门槛。若主门槛不通过，冻结并报告负证据；不得追加 reward、换 held-out 场景、事后调 mask 或继续训练来“修复”该轮结论。

## 1. 目标架构与术语契约

### 1.1 五态势与阶段：状态、任务和技能解耦

| 正交维度 | 冻结标签 | 用途 |
|---|---|---|
| 初始几何态势 | `advantage`、`head_on`、`disadvantage`、`neutral`、`crossing_entry` | 训练采样、分层评估、teacher 先验与结果分组。 |
| 动态阶段 | `pre_merge`、`post_merge`、`re_entry` | 时序状态描述、profile validity mask 与诊断。 |
| 二级条件 | 高度、LOS elevation、specific-energy-height | 采样平衡、mask 和安全/物理可执行性分析。 |

冻结 rule-based taxonomy 是训练、采样、评价和 teacher 的契约；它**不是**新策略唯一的输入 task-id。高层 PPO 必须基于实时几何、显式短时统计和时序 encoder embedding 学习决策。taxonomy/phase 只可生成可审计的 validity mask 和 warm-start teacher，且每步必须记录其影响。

### 1.2 四个共享低层技能

| `skill_id` | 名称 | 可解释职责 | 非职责边界 |
|---:|---|---|---|
| 0 | `pursuit_conversion` | 由迎头或中性关系转入可攻击的追击几何 | 不等同于“Head-on 专家”。 |
| 1 | `lead_intercept` | 在横向/交叉条件下建立 lead pursuit 与截获窗口 | 不等同于“Crossing 专家”。 |
| 2 | `defensive_extension` | 在劣势或能量不利时创造安全间隔、降低即时威胁 | 不承诺单独完成反杀。 |
| 3 | `reentry_recovery` | 在 post-merge 分离后恢复可控 re-entry 几何 | 不重用旧 deterministic recovery profile 作为独立 checkpoint。 |

四个技能共享一个 observation contract、同一个预测目标 VPP 接口、同一 3-D VPP action head 与 guidance/PID 链。每个技能均先独立预训练、再进行三 opponent 平衡对抗微调，随后冻结。

### 1.3 七个全局 tactical-intent profiles

| `profile_id` | 名称 | 主要几何意图 |
|---:|---|---|
| 0 | `front_intercept` | 形成前向截获与可控 closure。 |
| 1 | `rear_quarter_alignment` | 向后半球/尾追几何收敛。 |
| 2 | `lateral_displacement` | 建立安全且有目的的侧向几何位移。 |
| 3 | `defensive_break` | 缩短即时威胁暴露并形成转向脱离窗口。 |
| 4 | `range_extension` | 优先扩大间隔和重建能量/时间裕度。 |
| 5 | `energy_altitude_recovery` | 恢复 specific energy 与高度安全裕度。 |
| 6 | `reentry_preparation` | 在分离后为重新闭合准备几何。 |

每个 profile 通过确定性的 `ProfileIntentCompiler` 生成含权重的连续 intent：`AA*`、`ATA*`、`range*`、`closure*`、`specific_energy*`、`altitude*`。profile 不是新的离散 task，也不直接替换 VPP action；它作为低层技能的条件输入，影响技能如何生成三维 VPP 偏置。

### 1.4 高层与低层接口

```text
base geometry + predicted-target VPP context + explicit short history
  -> frozen temporal encoder z_t
  -> PPO factorized heads: skill_id, profile_id
  -> taxonomy/phase/energy validity mask (recorded, never silent)
  -> frozen selected skill(profile-conditioned observation)
  -> 3-D normalized VPP action
  -> existing predicted-target VPP + guidance law + PID + JSBSim
```

- [ ] 高层 action 使用 factorized `(skill_id, profile_id)` logits，而非 28 个互不共享的 task-action；mask 后至少保留一个有效组合。
- [ ] 低层 observation 不包含抽象 `task_id`；允许包含 profile compiler 输出、profile weights、实时几何、显式短时统计和 temporal embedding。
- [ ] 每高层步记录 proposed/effective skill、profile、mask、mask reason、intent vector、3-D VPP action、预测器有效性/回退和终端语义。
- [ ] v1 不复用 canonical recovery guard；若保留任何硬安全 guard，必须只作用于物理安全且被单独列为 deterministic override，不得称为 learned skill。

## 2. P0：隔离、可复现性、空间预算

### 2.1 新 lane 与资产冻结

- [ ] 从当前明确 SHA 创建干净 worktree，例如 `E:\uav-vpp-guidance-thesis-five-state-v1`；主仓库 dirty/untracked 内容不纳入该 lane。
- [ ] 建立 `reports/thesis_five_state_shared_intent_v1_asset_manifest.yaml`，记录 baseline、预测器、三 opponent、环境配置、训练数据和代码 SHA-256。
- [ ] 新 output root 固定为 `E:\uav-vpp-guidance-thesis-five-state-v1-results`；严禁写入 `outputs/jsbsim_hrl_comparison`、`CAN-20260705`、RQ1--RQ4、IND-HEADON 或 Taxonomy30 output。
- [ ] 对第三个 PPO/VPP opponent 先完成独立 capability audit，冻结 checkpoint/config/hash 后才能进入训练池；未通过时该阶段应阻塞，而非静默退回两 opponent。

### 2.2 容量与留存规则

当前 `E:` 剩余约 297.6 GB，而主仓库约 365 GB 几乎全部来自历史 `outputs`。本 lane 的目标是避免重复保存训练期 raw trajectories。

- [ ] 在 asset manifest 中声明硬容量预算：训练期 `<= 80 GB`、dev/ablation `<= 40 GB`、heldout/bundle 预留 `<= 70 GB`，并保留 `>= 100 GB` 系统安全余量。
- [ ] 训练期仅保留 metrics、config、asset hash、top-3 checkpoints 和每个阶段固定少量诊断 trajectory；不保存全量逐步 raw episode。
- [ ] 仅 dev30、独立 capability card、ablation 与 heldout60 保存完整 raw telemetry；输出按 split/method/seed 分目录，禁止复制同一资产。
- [ ] 每个 runner 开始前检查剩余空间；低于 120 GB 自动拒绝启动。每阶段结束后输出 size manifest 和清理候选清单，但不得自动删除文件。
- [ ] 先对历史超大 output 根做“可归档/必须保护”清单后再启动多 seed 训练；本 checklist 不授权删除任何已有 artifact。

### P0 验收

- [ ] clean worktree、asset manifest、hash verifier、output-root isolation 和容量 preflight 全部通过。
- [ ] baseline 与 predictor 可只读加载；第三 PPO/VPP opponent 的 capability card、checkpoint hash、接口和固定 reference-set 成绩完整。
- [ ] `git diff --exit-code` 在新 worktree 的 P0 freeze commit 后通过。

## 3. P1：统一 observation contract、profile compiler 与 validity mask

### 3.1 必须新增的代码与配置

| 类型 | 路径 | 最小职责 |
|---|---|---|
| Observation contract | `src/uav_vpp_guidance/hierarchy/five_state_observation_contract.py` | 定义 feature names、单位、归一化、缺失值、版本和 feature-name transplant 映射。 |
| Explicit history | `src/uav_vpp_guidance/hierarchy/temporal_geometry_features.py` | 从固定长度窗口计算角度变化、range-rate 趋势、closure、能量与高度趋势。 |
| Intent compiler | `src/uav_vpp_guidance/hierarchy/shared_intent_profiles.py` | 将七 profile 编译为连续 target/weight 向量。 |
| Validity mask | `src/uav_vpp_guidance/hierarchy/shared_intent_validity.py` | 按 taxonomy、phase、energy/altitude 给出可解释的 skill/profile mask 和 reason。 |
| 运行时路由 | `src/uav_vpp_guidance/hierarchy/five_state_shared_intent_policy.py` | factorized PPO 输出、mask、profile-conditioned specialist 调用与完整 telemetry。 |
| 主配置 | `config/experiment/thesis_five_state_shared_intent_v1_base.yaml` | 唯一 schema、版本、output 安全约束和 frozen predictor 引用。 |
| Profile 配置 | `config/experiment/thesis_five_state_shared_intent_v1_profiles.yaml` | 七 profile 的六维 intent target、weight、允许 phase/状态及版本 hash。 |

### 3.2 observation contract 要求

- [ ] 先列出现有 head-on 19-D、crossing 18-D 与新 contract 的 feature-name 映射；禁止按列截断/补零作为新训练的长期兼容方案。
- [ ] 新 contract 固定包含实时相对位置/速度/航向、range/range-rate、角度语义、own/target 高度与 specific energy、VPP/prediction validity context、显式短时统计、profile target/weight 和 temporal embedding。
- [ ] 每个 feature 有 source、单位、归一化区间、缺失策略和 leakage 说明；不得使用未来真实 target state，预测相关信息只能来自冻结 predictor 已提供的当前接口。
- [ ] temporal encoder 输入窗口固定为过去 `K=10` 个高层步（2.0 s）；不得跨 episode、跨 reset 或读取 future step。
- [ ] action head 保持 3-D normalized VPP semantics；profile compiler 只改变条件输入，不直接篡改 action 后处理。

### 3.3 feature-name weight transplant 规则

- [ ] 仅复制名称、单位、归一化完全一致的 first-layer columns；新 profile/embedding columns 零初始化或小尺度初始化，并生成 transplant report。
- [ ] 隐藏层及 action head 仅在 architecture/action semantics 完全一致时复制；否则明确重新初始化。
- [ ] 禁止使用 `strict=False` 掩盖 observation shape 不匹配。

### 3.4 P1 tests

- [ ] `tests/test_five_state_observation_contract.py`：feature uniqueness、shape、单位/归一化、missing/fallback、无 future leakage、旧 checkpoint 映射报告。
- [ ] `tests/test_temporal_geometry_features.py`：K=10 边界、reset 清空、趋势符号、无跨 episode history。
- [ ] `tests/test_shared_intent_profiles.py`：七 profile 完整、六维 target/weight 有限、compiler 单调/方向性、未知 profile 拒绝。
- [ ] `tests/test_shared_intent_validity.py`：每种 taxonomy/phase 至少有有效组合，明显不合理组合被屏蔽且 reason 可审计。
- [ ] `tests/test_five_state_shared_intent_policy.py`：factorized logits、mask 后采样、proposed/effective telemetry、预测器回退不改变 contract。

### P1 stop rule

若无法建立无泄漏、无静默截断、可由 feature names 解释的统一 contract，则停止训练四技能；先修复数据/接口契约，不以更多训练弥补接口不确定性。

## 4. P2：数据、对手与独立评估集

### 4.1 配置与构建器

| 产物 | 建议路径 |
|---|---|
| 连续随机训练分布 | `config/experiment/thesis_five_state_shared_intent_v1_train_distribution.yaml` |
| 训练分布构建器 | `scripts/build_thesis_five_state_train_distribution.py` |
| dev30 manifest | `config/experiment/manifests/thesis_five_state_shared_intent_v1_dev30.yaml` |
| heldout60 manifest | `config/experiment/manifests/thesis_five_state_shared_intent_v1_heldout60.yaml` |
| manifest 构建器 | `scripts/build_thesis_five_state_manifests.py` |
| 三 opponent 能力卡构建器 | `scripts/build_thesis_five_state_opponent_capability_cards.py` |
| trajectory dataset 构建器 | `scripts/build_thesis_five_state_trajectory_dataset.py` |

### 4.2 数据划分

- [ ] 训练采用连续随机分布，五态势、三档高度、镜像方向、phase transition 和三 opponent 平衡采样；完整 sampler seed 与分布 hash 写入 manifest。
- [ ] `dev30` 仅用于 checkpoint 选择和小规模 ablation，不与训练分布或 heldout60 full signature 重叠。
- [ ] `heldout60 = 5 initial states x 3 altitude conditions x 2 mirror signs x 2 unseen distance-speed packages`；60 个场景均为训练和 dev 未出现的 full signatures。
- [ ] heldout60 在开始多 seed formal 前写入 SHA-256；改变一个数值即视为新 manifest 版本，旧结果不得混合。
- [ ] 三 opponent 为 rule-based expert、现有 end-to-end neural opponent、独立审计后冻结的 PPO/VPP opponent；训练平衡采样，dev/heldout 每个 opponent 分开报告。
- [ ] 对手 capability card 必须包含 identity、SHA/checkpoint、action cadence、训练预算或 rule mechanism、reference-set terminal mix、AoA/n_z/speed/altitude/energy 和已知边界；不得用两三个结果包装为 Elo 排名。

### 4.3 P2 tests

- [ ] `tests/test_thesis_five_state_manifests.py`：dev30/heldout60 数量、五态势/高度/镜像平衡、transition-band exclusion、full-signature disjointness、hash freeze。
- [ ] `tests/test_thesis_five_state_train_distribution.py`：三 opponent 与五态势采样比例、连续变量范围、无 heldout leakage。
- [ ] `tests/test_thesis_five_state_opponent_registry.py`：三 opponent 全部可加载、asset hashes 固定、接口/cadence 一致性或 adapter 显式记录。
- [ ] `tests/test_thesis_five_state_trajectory_dataset.py`：仅训练轨迹、时间顺序、normalization fitting 仅用 train split、episode provenance 完整。

### P2 stop rule

若第三 PPO/VPP opponent 无法独立审计和冻结，或 heldout60 不能证明与 train/dev 不相交，则不启动三 opponent 训练；可退回完成数据/对手基础设施，但不得把“两 opponent 结果”写成已满足本 v1 计划。

### P2 执行记录（2026-07-13）

- [x] 已冻结 [连续训练分布](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/thesis_five_state_shared_intent_v1_train_distribution.yaml)：五态势、三高度、镜像、phase transition 与三 opponent 均衡采样；训练支持域固定为 range `1800--2500 m`、双方 speed `185--225 m/s`，不允许以 dev/heldout 数据拟合 normalization。
- [x] 已冻结 [dev30](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/manifests/thesis_five_state_shared_intent_v1_dev30.yaml)（payload SHA-256 `539d91632358d2ed26d064de65c36498ff0ca6379ca2d08a85a5c9df6ba032a6`）和 [heldout60](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/manifests/thesis_five_state_shared_intent_v1_heldout60.yaml)（payload SHA-256 `a43e304751c2f23d4be4975f07d781a2475eacef100c6ef336923ffed4001725`）。heldout60 精确平衡 `5 states x 3 heights x 2 mirrors x 2` 个训练域外 distance-speed packages。
- [x] dev30、heldout60、Taxonomy30、CAN-20260705 和 IND-HEADON-20260712 的已读取 full-state signatures 交集均为零；Taxonomy30 仍只允许作旧系统诊断，不得作新 family 的选择或正式统计。
- [x] 已冻结 [三 opponent registry](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/thesis_five_state_shared_intent_v1_opponent_registry.yaml) 和 [capability cards](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_opponent_capability_cards.yaml)：rule-based expert、existing end-to-end neural、independent PPO/VPP；训练采用均衡轮转，失败时 fail-closed，正式结果按 opponent 分开报告且禁止 Elo/总强度声明。
- [x] 已冻结 [trajectory provenance](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_trajectory_dataset_provenance.yaml)：当前状态为 `schema_frozen_no_training_trajectories_collected`；只允许后续 P3 收集 train-only 轨迹，dev30/heldout60/Taxonomy30 只能 transform、永不参与 statistics fitting。
- [x] P2 tests：`tests/test_thesis_five_state_manifests.py`、`test_thesis_five_state_train_distribution.py`、`test_thesis_five_state_opponent_registry.py`、`test_thesis_five_state_trajectory_dataset.py`，共 `10 passed`。本阶段没有启动四个低层技能或高层 PPO 训练，也没有创建训练 checkpoint。

## 5. P3：时序 encoder 自监督预训练

### 5.1 训练设计

- [ ] 新增 `src/uav_vpp_guidance/training/thesis_temporal_geometry_encoder.py`：仅编码过去 10 个高层步，输出固定 `z_t`，并保留显式短时统计旁路。
- [ ] 新增 `config/experiment/train_thesis_five_state_temporal_encoder_v1.yaml` 与 `scripts/train_thesis_five_state_temporal_encoder.py`。
- [ ] 使用仅来自训练 split 的冻结 raw trajectories，按态势/phase/opponent 平衡采样。
- [ ] 自监督目标固定为 masked-history reconstruction 加多 horizon（1、5 high-level steps）相对几何变化预测；标准化参数仅由 training split 拟合。
- [ ] v1 高层训练阶段 encoder 默认冻结；只在后续受控 ablation 中允许小学习率微调，且必须与 frozen-encoder 完整对照。

### 5.2 P3 验收

- [ ] train/dev loss 和各态势、各 opponent 的重建/预测误差完整保存；禁止以 heldout60 选择 encoder。
- [ ] synthetic sequence tests 显示 range-rate、角度/能量趋势方向可恢复，且 reset 后 `z_t` 不携带上个 episode 信息。
- [ ] encoder 输出在真实 dev30 无 NaN/Inf、无跨 split normalization leakage，显式短时统计在 encoder 禁用时仍可独立提供。

### P3 stop rule

若 encoder 不能通过无泄漏/稳定性检查，使用“显式短时统计 only”继续后续 pipeline，并将 encoder 标记为未通过的 optional representation；不得改为端到端 GRU/LSTM 来绕过此 stop rule。端到端 recurrent PPO 只属于后续独立 ablation。

### P3 执行记录（2026-07-13）

- [x] 已实现 `src/uav_vpp_guidance/training/thesis_temporal_geometry_encoder.py`、[P3 config](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/train_thesis_five_state_temporal_encoder_v1.yaml) 和 `scripts/train_thesis_five_state_temporal_encoder.py`：目标固定为 masked-history reconstruction + horizon `1/5` relative-geometry deltas，encoder 默认冻结，明确禁止 shared-skill / high-level PPO training。
- [x] 已以官方 JSBSim `v1.1.6` / commit `b477f6312bee2fd3af4c4e5ba1a3e732ed2b99d4` 运行 90 个 train-only rollout。三 opponent 均实际加载，90/90 为 JSBSim backend，产生 6,030 个紧凑 16-D 几何窗口；heldout60 未读取。
- [x] P3 phase coverage gate **失败并停止在 encoder 训练之前**：Head-on 与 Crossing-entry 覆盖三阶段，但 Advantage、Disadvantage、Neutral 未形成充分 post-merge/re-entry。详见 [P3 stop report](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p3_collection_stop_20260713_zh.md) 和外部 [collection report](E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_v1/collection_report.json)。
- [x] 已遵守 stop rule：未启动 80-epoch encoder training，未创建 P3 encoder checkpoint，未运行 dev30/heldout60，也未启动四 skill 或高层 PPO 训练。P4/P5/P6/P7 继续关闭。
- [x] 已在独立 `phaseconditional_v2` config 和全新 output root 冻结并运行 physical-marginal coverage contract：90 个五态势初始覆盖 rollout + 36 个独立 transition-provider rollout，未复用 v1 train data，未放松 physical coverage gate。126/126 train rollout 为 JSBSim；state x opponent、phase x opponent、provider post-merge/re-entry x opponent 均超过预注册下限。
- [x] v2 encoder 实际训练 80 epochs，选择 epoch 74 的 `best.pt`（dev total loss `0.3238515537`，SHA-256 `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`）。dev30 仅作 transform-only 诊断，heldout60 未使用；完整证据见 [P3 v2 report](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p3_phaseconditional_v2_20260713_zh.md)。
- [x] P3 通过 representation-readiness gate：encoder 默认冻结；合成趋势/episode-reset tests、JSBSim coverage、train-only normalization 和 finite dev embedding 均已验证。P4/P5/P6/P7 仍未启动。

## 6. P4：四共享技能的两阶段训练与冻结

### 6.1 必须新增的配置/脚本

| 产物 | 建议路径 |
|---|---|
| 几何预训练配置 | `config/experiment/train_thesis_five_state_shared_skills_geometry_v1.yaml` |
| 三 opponent 微调配置 | `config/experiment/train_thesis_five_state_shared_skills_combat_v1.yaml` |
| skill library registry | `config/experiment/thesis_five_state_shared_skill_registry_v1.yaml` |
| 训练入口 | `scripts/train_thesis_five_state_shared_skills.py` |
| checkpoint/shape verifier | `scripts/verify_thesis_five_state_shared_skills.py` |
| skill 报告 | `scripts/build_thesis_five_state_skill_report.py` |

### 6.2 两阶段训练

1. [ ] **几何能力预训练：** 对四技能分别使用与其职责匹配、但不等于固定 task name 的几何/phase/profile 条件；目标是学习稳定、方向正确、可执行的 VPP 行为。
2. [ ] **三 opponent 平衡对抗微调：** 固定 observation contract、profile compiler、predictor、VPP/guidance/PID；按三 opponent 平衡采样微调四技能。
3. [ ] **冻结：** 选择 checkpoint 只能依据 dev30 的预注册 skill-readiness 指标，之后保存 hash、config、seed 和 source trajectory provenance，禁止在高层训练期间继续更新。

### 6.3 skill-readiness gate（仅允许进入高层训练的必要条件）

- [ ] 4/4 skills 都通过 checkpoint shape、action semantics、predictor/VPP interface 与 deterministic inference preflight。
- [ ] 每个 skill 在其允许 profile 的 synthetic geometry templates 上达到至少 `95%` intent-direction agreement；不允许用场景名称替代几何判定。
- [ ] 每个 skill 在 dev30 对应子集、三个 opponent 分开统计时，ego crash/OOB 不高于双技能 baseline `+0.05`；任何一个 opponent 违反时不得冻结为 ready。
- [ ] 至少 3/4 skills 在对应子集相对于“零 VPP 偏置/不切换”reference 同时展示可审计的几何进展（attack-zone/re-entry/安全 separation 中与职责一致的一项）且无新的 terminal failure mode。
- [ ] 报告 profile usage、VPP forward/lateral/vertical 分布、n_z/AoA actual response 与 terminal mix；不能仅以训练 reward 选 checkpoint。

### P4 stop rule

若两个或更多 skills 未通过 readiness gate，不启动高层 PPO。先将失败归为 observation/profile/compiler/low-level capability 的哪一层，并只修复这一层；不得通过扩充 profile 数量、添加 prediction reward 或调整 held-out 场景掩盖失败。

### P4 readiness 执行记录（2026-07-13）

- [x] 已冻结 [P3 v2 `best.pt`](E:/uav-vpp-guidance-thesis-five-state-v1-results/p3_temporal_encoder_phaseconditional_v2/checkpoints/best.pt) 的 SHA-256 `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`、`16-D x K=10 -> 32-D` encoder contract、66-D observation contract、七个 global profiles 和三 opponent registry。
- [x] 已建立 [四共享技能 registry](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/thesis_five_state_shared_skill_registry_v1.yaml)：四个 skill 均为 `untrained_not_ready`，无 checkpoint、无 fallback；每个 skill 分别声明允许 profile、几何/阶段覆盖和独立 readiness gate。
- [x] 已建立 [geometry pretrain](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/train_thesis_five_state_shared_skills_geometry_v1.yaml) 与 [combat finetune](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/train_thesis_five_state_shared_skills_combat_v1.yaml) 配置；两者均固定为 `execution_mode: readiness_only` 及 `training_permitted: false`，使用新的 P4 空输出根。
- [x] P4 verifier 已通过：P3 SHA/load/determinism、66-D profile-conditioned -> 3-D normalized VPP interface、三 opponent 实际加载、fresh-output/retention guard 和一条 strict JSBSim reset/step probe 均通过。详见 [machine-readable report](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p4_readiness_20260713.json) 与 [readiness report](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p4_readiness_20260713_zh.md)。
- [x] 已实现真实 geometry-pretrain runner：每个 skill 使用独立 PPO actor-critic，固定 P3 `eval()` encoder、显式 9-step zero-VPP real-environment history bootstrap、profile round-robin、连续 train distribution、train-only 聚合 telemetry、dev30 三 opponent 分开评估和 strict checkpoint schema；完整训练 loop 仅在未来授权后运行。
- [x] 已实现逐 skill gate：检查允许 profile 覆盖、声明的 geometry/phase 覆盖、intent-progress fraction、相对同场景 zero-VPP reference 的每 opponent crash/OOB delta，以及 VPP/n_z command/JSBSim actual `n_z` 和 AoA response 摘要。若任一 skill 不通过，输出 `not_ready_do_not_start_combat_finetune`。
- [x] 当前 `--geometry-plan` 已证明 4 个技能分别有 `18/18/12/24` 个连续训练分布候选；`--run-geometry-pretrain` 在 `training_permitted: false` 时于创建 output root 前 fail-closed。frozen learned predictor 与 `predicted_target` VPP interface 已由 strict JSBSim probe 再次验证。
- [x] P4 implementation/readiness tests：`tests/test_thesis_five_state_shared_skill_geometry.py`、`test_thesis_five_state_skill_checkpoints.py`、`test_thesis_five_state_skill_readiness_configs.py` 与 P1/P3 关联测试共 `25 passed`。
- [x] 一次性授权 overlay `GEO-20260713-R1` 下的 non-smoke geometry pretrain **已完成**：输出根为 `E:/uav-vpp-guidance-thesis-five-state-v1-results/p4_shared_skills_geometry_v1_geo20260713_r1`。四个 skill（`pursuit_conversion`、`lead_intercept`、`defensive_extension`、`reentry_recovery`）均已训练至 `200,000` steps；本次 run 产出 `src/p4_geometry_pretrain_gate.json`，判定 `all_skills_ready_for_combat_finetune = false`、`next_stage = do_not_start_combat_finetune`。此即完成态、非“进行中”。
- [x] **最终 gate 判定（正式负证据）：** 四个 skill 的 profile-conditioning coverage 与 JSBSim safety 全部通过（4/4）；geometry_phase_coverage 全部失败（4/4），失败完全由声明 cell 以 0 值发出（`post_merge`/`re_entry` 的 `present_value_0`）驱动，而非任何缺失/absent cell。intent-progress 三个失败——`defensive_extension`（0.5325）、`lead_intercept`（0.4977）、`pursuit_conversion`（0.5193，均 < 0.55）；`reentry_recovery`（0.5820）通过 intent-progress，但仍因 geometry_phase_coverage 失败而被阻断。这是正式负证据；combat finetune / P5 高层 PPO / P6 消融 / P7 heldout60 全部保持锁定。P4 v1 的正式 **NO-GO** 决策见 `reports/thesis_five_state_p4_v1_go_no_go_decision_20260714_zh.md`；唯一允许的后续是未授权执行的 P4-v2 sampler-feasibility 设计，见 `reports/thesis_five_state_p4_v2_sampler_feasibility_preflight_design_20260714_zh.md`。

## 7. P5：高层 PPO 的三阶段训练

### 7.1 配置与代码

| 产物 | 建议路径 |
|---|---|
| 高层模型与 factorized action | `src/uav_vpp_guidance/agents/five_state_shared_intent_ppo_agent.py` |
| 高层训练环境 | `src/uav_vpp_guidance/envs/five_state_shared_intent_commander_env.py` |
| Geometry-intent teacher | `src/uav_vpp_guidance/hierarchy/five_state_geometry_intent_teacher.py` |
| Oracle imitation config | `config/experiment/train_thesis_five_state_commander_imitation_v1.yaml` |
| 平衡多任务 PPO config | `config/experiment/train_thesis_five_state_commander_balanced_v1.yaml` |
| 动态转移微调 config | `config/experiment/train_thesis_five_state_commander_transition_v1.yaml` |
| 训练入口 | `scripts/train_thesis_five_state_commander.py` |
| selection/report builder | `scripts/build_thesis_five_state_dev30_report.py` |

### 7.2 三阶段规则

1. [ ] **Oracle imitation warm-start：** teacher 由 taxonomy、phase、能量/高度与 validity mask 生成 `(skill, profile)` demonstration；只用作初始化，不称为最优 Oracle，也不得依据结果回看重标。
2. [ ] **平衡多任务 PPO：** imitation 权重按预注册 schedule 退火到零，三 opponent 与五态势连续随机采样平衡；四个低层 skills 和 encoder 均冻结。
3. [ ] **动态转移微调：** 固定前两阶段所有模块，仅提高 phase transition、post-merge 和 re-entry 在训练分布中的预注册比例；不改变 reward、predictor、VPP/guidance/PID 或 validity rules。

### 7.3 checkpoint selection

- [ ] 每个完整系统训练 `3` 个独立 seeds；每 seed 的 low-level skills、encoder、高层 PPO、sampler 和 evaluation seed 全部写入 provenance。
- [ ] 仅用 dev30 选择每个 seed 的 checkpoint；selection metric 为五态势/三 opponent 分层的预注册 score，禁止只选 pooled reward 或单一 opponent 最佳点。
- [ ] dev30 selection 同时要求 head-on/crossing 不低于冻结双技能 baseline `-0.05`，并把 crash/OOB `+0.05` 作为硬拒绝条件。
- [ ] 输出 teacher agreement、mask rate、invalid-action rejection、skill/profile entropy、first action、switch count、phase dwell、VPP bias 与 terminal reason；teacher agreement 不等于最终性能。

### P5 stop rule

若三个 seed 中少于两个能通过 dev30 的保留性和安全门，则停止在 dev30，冻结失败结果并进行 routing/library/representation 审计；不得进入 heldout60，也不得把 dev30 场景加入训练。

## 8. P6：最小但完整的模块因果消融

消融在 dev30 执行，用于解释模块作用，不得用其替代 heldout60 主结论。每个学习型变体训练预算、训练分布、三 opponent、seed 和 checkpoint selection 规则必须与 full v1 一致。

| ID | 方法 | 回答的问题 | 最低证据等级 |
|---|---|---|---|
| B0 | 冻结双技能 PPO baseline | 现有系统的共同基线 | L3 reference |
| B1 | full v1：4 skills + 7 profiles + explicit stats + frozen encoder | 总体效果 | L3 |
| B2 | no-encoder：4 skills + 7 profiles + explicit stats | encoder 的净增量 | L2/L3 |
| B3 | single-profile：4 skills + 一个冻结中性 profile | profile library 的净增量 | L2/L3 |
| B4 | fixed-skill：full state/profile，但固定为单 skill | 高层技能选择的净增量 | L2/L3 |
| B5 | no-mask audit：仅在物理安全允许时关闭非安全 validity mask | mask 是防止无效组合还是暗含 task routing | L2，仅限诊断 |

- [ ] B0--B4 至少完成三个 opponents 分开报告和 3 seeds 的 dev30；B5 只在预先定义安全沙盒中运行，若出现新 crash/OOB 立即停止。
- [ ] 每个模块分别标注 L1（正确调用）、L2（中间机制变化）、L3（预注册结果改进）；不得把 L1/L2 叙述为性能提升。
- [ ] 若 B1 未优于 B0，仍允许用 B2--B4 解释失败来源，但不得以单一正向子指标宣称系统成功。

## 9. P7：三 opponent、三 seed 的 heldout60 正式验证

### 9.1 正式方法集与工作量

primary formal methods 为：

1. [ ] `five_state_shared_intent_ppo_v1`（B1，每个 training seed 一个冻结系统）。
2. [ ] `canonical_two_skill_ppo_baseline`（只读冻结 baseline，以相同 scenario/evaluation seed 配对评估）。
3. [ ] `legacy_two_skill_static_oracle`（privileged reference，不称五态势最优 Oracle）。

最小 primary 工作量为 `60 scenarios x 3 opponents x 3 system seeds x 3 methods = 1,620 episodes`。若容量/时间不允许，不得削减某个 opponent 后仍宣称三 opponent 验证；应明确降级为 pilot。

### 9.2 正式运行规则

- [ ] 在新的 clean formal worktree 从 P0 asset manifest 开始 preflight，冻结 config/asset/manifest SHA 后启动。
- [ ] `heldout60` 对所有方法、所有 seeds 完全一致；每个 episode 记录 JSBSim backend、initial condition、method/system seed、opponent、terminal event 和 raw telemetry。
- [ ] 正式运行前不查看结果；任一失败 episode 只允许基础设施重跑并写入 run ledger，禁止替换 policy/checkpoint/mask。
- [ ] 每个 method/opponent/system-seed 必须有 60/60 completion；有 unresolved episode 时按冻结 resolved-win-rate 语义报告且保留 total/resolved 分母。
- [ ] formal 期间不更新 encoder、skills、高层 PPO、normalization、teacher、profile、mask、predictor 或 guidance/PID。

### 9.3 必须输出的结果

- [ ] 五态势 x opponent x method x seed：`N_total`、`N_resolved`、win/loss/draw/unresolved、Wilson 95% CI、paired outcome 和 scenario-clustered paired-bootstrap delta CI。
- [ ] terminal reason/safety table：ego/target crash/OOB、kill、HP timeout、draw timeout；明确 command 与 actual response、AoA/n_z/saturation 的意义边界。
- [ ] routing/intent table：first/mean proposed-effective skill/profile、mask reason、switch count、phase dwell、profile/skill usage entropy。
- [ ] geometry/physics table：attack-zone first entry/time fraction、range/range-rate、AA/ATA、specific energy、altitude、VPP 3-D bias、AoA/n_z command/actual response。
- [ ] 三 opponent capability cards 与 failure atlas；禁止将 opponent-specific 差异写成“通用能力”。
- [ ] artifact index、asset manifest、run ledger、raw telemetry index、figure/table index 与可验证 submission bundle。

### 9.4 最终决策

| 结果 | 决策与允许表述 |
|---|---|
| 三项主门槛全部通过 | v1 可成为硕士论文主结果；表述为“三 opponent、冻结 heldout60 范围内的实用改进与保留性”。 |
| Head-on/Crossing 保留但新态势增益不足 | 保留为结构化负/中性结果；论文可讨论能力覆盖，但不能声称五态势系统解决了新态势。 |
| 任一 opponent 安全门失败 | 保留负证据，回到失败归因；不得把 aggregate win rate 写成安全提升。 |
| 仅单一 opponent 通过 | 明确为 opponent-dependent，不替代双技能主线。 |
| formal 输入/资产/manifest 不可复现 | formal 无效；先修复 reproducibility，结果不得进入论文。 |

## 10. 最小 runner、分析与测试清单

### 10.1 运行与分析入口

- [ ] `scripts/run_thesis_five_state_dev30.ps1`：支持 `--dry-run`、phase 参数、容量 preflight、resume ledger 和不覆盖 output root。
- [ ] `scripts/run_thesis_five_state_heldout60.ps1`：拒绝未冻结 manifest/asset/config，拒绝非空 output root，生成 formal run ledger。
- [ ] `scripts/analyze_thesis_five_state_v1.py`：只读 raw telemetry，生成所有分层表、paired deltas、failure atlas 和 gate decision JSON。
- [ ] `scripts/build_thesis_five_state_formal_bundle.py`：将 config、hash、reports、CSV/JSON、figures 和日志索引打入唯一 bundle；不得复制无关 raw train dumps。

### 10.2 端到端测试

- [ ] `tests/test_thesis_five_state_configs.py`：所有 config 的 schema、冻结 predictor、3-D VPP、三 opponent、noncanonical output root、无 prediction reward。
- [ ] `tests/test_thesis_five_state_skill_checkpoints.py`：四技能同一 contract、profile-conditioned shape、feature-name transplant report、无 checkpoint fallback。
- [ ] `tests/test_thesis_five_state_commander_training.py`：teacher anneal、factorized action、mask telemetry、冻结 specialist/encoder、seed/provenance。
- [ ] `tests/test_thesis_five_state_dev30_runner.py`：三 opponent、30 scene completeness、capacity preflight、duplicate-root 拒绝。
- [ ] `tests/test_thesis_five_state_heldout60_runner.py`：60 scene completeness、train/dev disjointness、三 seeds配对、formal immutability。
- [ ] `tests/test_analyze_thesis_five_state_v1.py`：分层分母、Wilson CI、paired delta、主门槛、terminal semantics、opponent split 不混合。
- [ ] `tests/test_thesis_five_state_output_retention.py`：top-K checkpoint、raw telemetry 只在允许 split、size manifest、历史路径不可写。

## 11. 推荐执行顺序与每阶段交付

| 顺序 | 只做什么 | 交付物 | 进入下一步的唯一条件 |
|---:|---|---|---|
| P0 | worktree/asset/容量冻结 | asset manifest、space report、capability card plan | 资产可读、容量合格。 |
| P1 | contract/profile/mask | schema、单元测试、synthetic report | 无泄漏、无静默 shape adaptation。 |
| P2 | train/dev/heldout 与 opponent registry | 三个 manifest、hash、card | 3 opponent 可用且三 split 不相交。 |
| P3 | temporal encoder | encoder checkpoint、train/dev report | encoder 合格或降级为 explicit-stats-only。 |
| P4 | 四技能两阶段训练 | frozen skill registry、readiness report | 4/4 preflight 与 safety gate 通过。 |
| P5 | 高层 3 seeds | dev30 selection report | 至少 2/3 seeds 通过 dev gate。 |
| P6 | dev30 最小消融 | B0--B5 matrix | 模块证据等级明确。 |
| P7 | heldout60 formal | formal bundle、gate decision | 仅按预注册门槛做最终结论。 |

## 12. 当前状态

- [x] 研究目标、五态势 taxonomy、4 skills、7 profiles、预测器边界、三 opponent 计划、时序 encoder 方案、训练阶段、数据切分和成功门槛已通过访谈冻结。
- [x] 已有 Taxonomy30 诊断证据支持“不应继续对双技能 canonical family 做 routing patch”，但它不构成新系统的训练证据。
- [x] P0 的 clean worktree、[asset manifest](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_asset_manifest.yaml)、[容量策略](E:/uav-vpp-guidance-thesis-five-state-v1/config/experiment/thesis_five_state_shared_intent_v1_output_retention.yaml) 和 [preflight 记录](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p0_preflight_20260713.json) 已完成；六个 baseline/predictor/opponent assets 的 SHA/shape 均通过。
- [x] P0 第三 PPO/VPP opponent 审计已完成，结论详见 [capability audit](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_third_opponent_capability_audit_20260713.md)：独立 `16-D -> 3-D` PPO/VPP checkpoint 已通过 fixed-horizon training gate 与 JSBSim target-side probe，并已冻结进 asset bundle。
- [x] P0 的 `--require-training-ready` 最终 preflight 已通过：六个资产 SHA/shape、容量 policy 与第三 opponent readiness 均为 green。
- [x] P1 的 observation contract、显式 10-step history extractor、七个 profile compiler 与 `(skill, profile)` validity mask 已完成：`five_state_shared_intent_v1` 固定为 66-D，禁止 task/opponent bits、future state 和隐式 padding/truncation；profile YAML target/weight 已冻结并与代码一致性测试通过。
- [x] P2 已完成：连续 train distribution、dev30/heldout60、三 opponent registry、capability cards 与 train-only trajectory provenance 已冻结并通过 10 项测试；训练输入契约在 `GEO-20260713-R1` 启动前保持不变。
- [x] P3 v1 已作为缺相位 coverage 的负证据保留；P3 v2 已在全新物理 marginal sampler 上通过 representation-readiness gate，P3 encoder 仍冻结，详见 [P3 v2 report](E:/uav-vpp-guidance-thesis-five-state-v1/reports/thesis_five_state_shared_intent_v1_p3_phaseconditional_v2_20260713_zh.md)。
- [x] P4 geometry-pretrain 已完成：`GEO-20260713-R1` 四个 skill 均训练至 200k steps（4/4），四个 skill 全部未通过 readiness gate（geometry_phase_coverage 4/4 失败；intent-progress 三个失败，`reentry_recovery` 通过 intent-progress 但因 phase-coverage 失败而被阻断），构成正式负证据；combat finetune / P5 高层 PPO / P6 消融 / P7 heldout60 全部保持锁定。
- [x] `GEO-20260713-R1` 的 P4 gate 失败归因与正式 go/no-go 决策已完成，见 `reports/thesis_five_state_p4_v1_go_no_go_decision_20260714_zh.md`；P4 v1 终止于负证据，combat finetune / P5 高层 PPO / P6 消融 / heldout60 仍被禁止。P4-v2 sampler-feasibility 的 config、candidate manifests、纯函数 sidecar、design-only runner 与 tests 已落地，设计见 `reports/thesis_five_state_p4_v2_sampler_feasibility_preflight_design_20260714_zh.md`；runner 只验证 26-cell contract 且拒绝 `--execute`，未授权执行 JSBSim probe 或训练。
- [x] 单技能 `THESIS-DEFEXT-RANGEEXT-FEASIBILITY-PILOT-V1` 已按一次性授权完整执行并冻结为预注册负证据（formal simulation SHA `c2ee6d3af2094c827b74419e8631533226f1fef9`；output root `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/defensive_extension_range_extension_feasibility_v1`）：固定 50,000-step 训练后由 minimax Dev 规则选择 20k checkpoint，一次性 Heldout24 的最终判定为 `safety_or_contract_no_go`。候选相对 fixed head-on 在三个对手下均未达到 `-0.05` 实用改善门；independent PPO/VPP 下出现 `2/24` ego crash/OOB（两种 frozen baseline 均为 `0/24`），且 paired qualifying coverage 仅 `6/6 < 8`。原始 `decision.json` 保持不变，配对覆盖修正见 `heldout/decision_corrected.json`；执行结论见 `reports/thesis_defext_rangeext_feasibility_pilot_v1_execution_20260714_zh.md`。本结果不支持“现有技能库明确缺少 defensive-extension”，四共享技能、combat finetune、P5/P6/P7 继续锁定，禁止在同一 Source ID 下调参、加步数或重跑 Heldout24。
- [x] `THESIS-DEFEXT-RANGEEXT-PILOT-V1-ATTRIBUTION-R1` 只读失败归因已完成，见 `reports/thesis_defext_rangeext_pilot_v1_attribution_audit_20260714_zh.md` 与 `reports/thesis_defext_rangeext_pilot_v1_attribution_matrix_20260714.json`。冻结记录确认 3 条相同 handoff step 的 candidate-specific ego crash/OOB（expert 1、independent PPO/VPP 2），但 Heldout 未保存逐步 VPP/guidance/PID telemetry，无法定位首个 action/response divergence；同时最低限度的 pre-handoff metadata 一致性检查发现 expert `10/24`、end-to-end `15/24`、independent `7/24` 场景不一致，严格 candidate-head/crossing 配对分别仅为 `5/6`、`7/6`、`3/4`，均低于 claim-ready `8`。正式结论冻结为 `causal_mechanism_not_identifiable_from_frozen_artifacts`，不改写原 `safety_or_contract_no_go`。任何新性能 pilot 前必须先在全新 dev-only 场景证明 run-in 可重复性、保存 handoff state hash，并逐步落盘 VPP 三轴、n_z command/actual、AoA、速度、高度与 saturation；在此之前继续禁止四共享技能、combat finetune 与 P5/P6/P7。
