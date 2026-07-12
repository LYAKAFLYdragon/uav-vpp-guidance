# UAV-VPP-Guidance 项目全盘审查与整改路线图

**审查日期：** 2026-07-12  
**审查目标：** 区分已被原始产物支持的结论、仍受限的科学问题，以及阻碍复现、论文收口和后续研发效率的工程问题。  
**审查边界：** 本文不重跑训练，不改变 canonical 策略、checkpoint 或正式结果；所有结论优先依据 clean-worktree 产物、run manifest、原始 episode 汇总和可执行测试。

---

## 1. 执行结论

项目已经具备一个**可投稿但结论边界必须严格受控**的 VPP 分层决策结果，而不是一个尚未跑通的原型：

1. 在 JSBSim、AoA 60、`head_on + crossing_feasible`、`expert + end_to_end` 的固定范围内，canonical PPO 高层策略在 `expert/head_on` 上达到 `0.6500`，高于 oracle task gate 的 `0.5667`；在 `end_to_end/head_on` 上为 `0.9000`，略低于 oracle 的 `0.9322`；两个 crossing cell 均为 `3/4`。
2. 该正面结果并不等于“普适的学习式空战决策”。crossing 的 canonical 样本仅为 `N=4`，且 RQ3 已证明其表现为**带 task bit 的 task-aware 选择**，不能写成不依赖任务标签的几何自主识别。
3. 当前最重要的技术发现不是“再增加一条 commander clamp 就能全面解决问题”，而是：当进入 recovery 后，性能更主要受**低层 VPP 几何质量与对手分布**限制。非 canonical `v14b` 也支持“选择性低层子策略路由”优于全局 clamp，但它仍是 dirty、非 paper-safe 的探索性结果。
4. 当前最紧迫的风险不是性能，而是**证据快照、工作树、配置、文档和投稿包的分叉**：canonical formal run 固定在 commit `9a9f9bf6`，但其声明的 clean worktree 当前已经移动到 `2465b65`。原始结果仍有效，资产也存在，但必须恢复一个精确的 `9a9f9bf6` 只读工作树或可校验 bundle，才能把“产物存在”提升为“可从同一代码和资产重建”。
5. 工程复杂度已超过继续堆叠局部 guard 的收益：`tracking_env.py` 约 7,539 行、comparison runner 约 6,583 行、模式约束文件约 1,407 行，并由 286 份实验 YAML（266 份含 includes）共同驱动。继续在 canonical family 中叠加条件分支将显著增加过拟合、不可解释性和回归风险。

**总判断：**

- 若目标是尽快提升 AST 投稿成功率，优先级应是**重建不可变证据包、统一投稿源、用一个严格预注册的独立 head-on envelope 补强外推性**，而不是重新训练或继续调 commander。
- 若目标是推进下一篇方法论文，应该冻结当前 canonical 线，将 `v14b` 迁入新的 clean non-canonical research lane，以“选择性低层恢复子策略”的可迁移性为唯一研发问题；不得将其直接写入当前主论文。

---

## 2. 当前系统的事实底座

### 2.1 方法结构

当前 paper-facing 方法是一个混合分层接口：

```text
19-D high-level observation
        |
        v
PPO high-level policy (two learned discrete actions)
        |
        +--> frozen head-on VPP specialist (19-D input)
        +--> frozen crossing-v3 VPP specialist (historical 18-D input)
        |
deterministic geometry guards
        |
        +--> post-merge recovery profile (reuses head-on checkpoint)
        |
        v
3-D VPP action -> guidance law -> PID/flight control -> JSBSim 6-DOF
```

关键事实：

- 高层 PPO checkpoint 输入为 19-D、两项 learned action、128/128 隐层；其决策周期为 0.2 s（5 Hz），每项动作保持 12 个 60 Hz JSBSim 积分步。
- episode 上限是 512 个高层步，即 102.4 s。
- mode 2 不是第三个 learned specialist：它复用 head-on checkpoint，附加 deterministic post-merge recovery profile。
- crossing checkpoint 的历史输入契约为 18-D；运行环境发出 19-D observation 后，wrapper 做兼容适配。这是受控兼容行为，但也是未来复现与重训时必须显式测试的接口边界。
- `is_crossing` task indicator 被高层 observation 使用。RQ3 表明，去掉 bootstrap/lock 后 PPO 仍在 96/96 个 crossing episode 首个宏步选择 crossing；这支持 **task-aware PPO routing**，不支持 task-label-free 几何识别。

### 2.2 权威 formal 证据

唯一允许作为主表 headline 的来源为 `CAN-20260705`：

- expert：`E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_expert_20260705`
- end-to-end：`E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_end_to_end_20260705`
- manifest 记录的 commit：`9a9f9bf6d68560afa81560f5260b78f318dcd9b1`
- 两条 run 均为 `completed`、`paper_safe=true`、artifact contract 有效。

| Opponent | Task | Oracle | PPO | 正确解释 |
|---|---:|---:|---:|---|
| expert | head-on | 34/60 = 0.5667 | 39/60 = 0.6500 | PPO 不再弱于 oracle；Wilson CI 重叠，不能声称显著优越。 |
| expert | crossing-feasible | 3/4 = 0.7500 | 3/4 = 0.7500 | N=4 保留性检查，不是泛化证明。 |
| end-to-end | head-on | 55/59 = 0.9322 | 54/60 = 0.9000 | PPO 略低于 oracle，差值 -0.0322。 |
| end-to-end | crossing-feasible | 3/4 = 0.7500 | 3/4 = 0.7500 | N=4 保留性检查，不是泛化证明。 |

### 2.3 终局语义与安全诊断

胜率不是单一“击落率”。它以 resolved episode 的 `wins / (wins + losses)` 定义，且不少胜负来自 timeout HP advantage 或 crash/OOB 事件。必须区分 ego 与 target terminal event。

| Split / method | 主要胜因 | 主要负因 | 关键安全信号 |
|---|---|---|---|
| expert/head-on Oracle | 25 timeout HP advantage，7 target crash/OOB，2 target killed | 14 timeout HP disadvantage，10 ego crash/OOB，2 ego killed | ego crash/OOB 10/60 = 0.1667 |
| expert/head-on PPO | 27 timeout HP advantage，10 target crash/OOB，2 target killed | 13 timeout HP disadvantage，8 ego crash/OOB | ego crash/OOB 8/60 = 0.1333 |
| end-to-end/head-on Oracle | 46 target crash/OOB，9 timeout HP advantage | 3 timeout HP disadvantage，1 ego crash/OOB，另有 1 draw | ego crash/OOB 1/60 = 0.0167 |
| end-to-end/head-on PPO | 44 target crash/OOB，10 timeout HP advantage | 3 timeout HP disadvantage，3 ego crash/OOB | ego crash/OOB 3/60 = 0.0500 |

正式 safety 汇总显示 backend fallback 与 aggregate command saturation 均为 0，但这只能说明评估没有退回 SimplePointMassEnv、命令限幅没有普遍触发；**不能推出气动 AoA、失速裕度、过载响应或真实飞行安全已被认证**。

### 2.4 扩展证据的正确位置

| 证据 | 已证明的内容 | 不能证明的内容 |
|---|---|---|
| RQ1 crossing，24 场景 | PPO 相对 fixed head-on VPP 的 practical preservation：expert +0.125，end-to-end -0.042，门槛 -0.05 | 优越性、普适 crossing 泛化；end-to-end 仅高于门槛 0.008。 |
| RQ2 crossing，另一 24 场景 | 第二个独立 envelope 复现 preservation：expert 0.000，end-to-end -0.042 | 更大范围的统计显著性或物理安全。 |
| RQ3 locked/unlocked | 在两个 envelope 中，解锁后 PPO 仍 96/96 首拍选 crossing | 不含 task bit 的 geometry-only recognition。 |
| RQ4 post-merge routing，12 场景 | 出现 state-conditioned PPO-effective crossing re-selection；recovery entry 由 guard 造成 | PPO 独立学会 recovery 或性能优越。 |
| non-canonical v14b | dirty exploratory manifest60/end-to-end/head-on 达 58/60；小 gate 2/2；expert 与同次 canonical 相同 | paper-safe、跨 split 泛化、可直接替代 canonical mainline。 |

---

## 3. 已经做对的部分

1. **结论边界开始被制度化。** 主结论已固定在 clean JSBSim 输出，而历史 DQN、早期 PPO、专项 small audit 和 dirty exploratory lane 已被降级。近期 `paper_authoritative_artifact_index_20260712.md` 与 B1--B9 correction 把方法事实和 paper evidence 对齐。
2. **机制证据比早期单纯 win-rate 更成熟。** RQ3 区分 bootstrap/lock 与 PPO proposal；RQ4 区分 PPO-effective switch 与 guard-forced recovery；这是较好的因果归因实践。
3. **负结果没有被简单抹去。** end-to-end/head-on 三个 oracle-only counterexample、G4 crossing counterexample、无效 external-import seed cohort 和 v14 系列的 split regression 都有记录。
4. **测试与 artifact contract 有实质价值。** 本次针对 formal reproducibility、hierarchical policy、specialist subpolicy routing 与 observation contract 的测试为 `183 passed, 2 skipped`；current clean worktree 的 formal preflight 识别 42 个 YAML 依赖和 12 个代码依赖，结果为 `Issues: none`。
5. **非 canonical 探索已经提供了有价值的第一性经验。** 14 次尝试显示：广泛 clamp 往往造成 split regression，而只在明确 motif 下选择性路由真实低层子策略更有效。这是下一阶段低层机制研究的合理起点。

---

## 4. 核心瓶颈、根因与改进方案

### B0. 精确复现快照漂移

**现象：** canonical run manifest 固定在 `9a9f9bf6`，但该输出路径所在的 `E:\uav-vpp-guidance-clean-formal-448155d` 当前 HEAD 是 `2465b652`。commit `9a9f9bf6` 仍可达，两个 specialist 和 prediction asset 也仍存在，current worktree 的 preflight 也通过；然而“当前可运行代码”不再等于“产生 formal 数字的代码”。

**根因：** 同一 clean worktree 被后续 G4/flight-envelope 证据工作复用，代码工作树与结果归档没有物理隔离或不可变 bundle。

**风险：** 高。审稿人或未来作者可能以 2465b65 复跑 9a9f9bf6 的记录，得到不同结果但无法判断是代码、资产还是配置漂移所致。

**改进：**

1. 新建只读 detached worktree，例如 `E:\uav-vpp-guidance-can20260705`，精确 checkout `9a9f9bf6`。
2. 复制或只读挂接两个 run 的 `resolved_config.yaml`、`run_manifest.json`、`artifact_contract.json`、raw episodes、四个外部 checkpoint，并生成统一 SHA-256 asset manifest。
3. 在该 worktree 仅运行 preflight 与 artifact verifier；不要在此 worktree 上做任何新实验或代码编辑。
4. 让主稿、supplement 和 submission manifest 只引用 `CAN-20260705`，而不是笼统引用“clean worktree”。

**验收：** detached worktree HEAD 为 `9a9f9bf6`、assets hash 与 run manifest 一致、preflight 通过、bundle 可在另一目录被验证。

### B1. Head-on 外推性不足，且 canonical 未给出多策略分布

**现象：** expert/head-on 的点估计为 +0.0833，但 canonical 仅有一份 PPO checkpoint；end-to-end/head-on 是 -0.0322；两者均未形成多 seed 的 head-on policy 分布。当前的 60 个场景也经历过 repair threshold 的迭代曝光。

**根因：** 项目把大量资源投入 scenario-specific repair，而不是在 policy-level 随机性与真正独立 head-on envelope 上建立外推证据。

**风险：** 高。当前 “no longer weaker” 是诚实且可用的说明，但尚不是强的泛化或算法稳定性证据。

**改进：**

- **投稿稳定路线：** 不再训练；在 Discussion 中明确“single frozen PPO checkpoint + iterative repair exposure”，只保留 bounded claim。
- **质量升级路线：** 仅做一次完全预注册、无训练、无调参的独立 head-on envelope（建议 24--32 场景，两个 opponent split，冻结同一 checkpoint/asset/config）。事先定义：PPO-oracle 配对 resolved win-rate delta 的 practical margin、ego crash/OOB non-inferiority 条件和全部 terminal-reason 报告规则。
- **后续研究路线：** 若要讨论训练稳定性，再训练 3 个 high-level seed，并只在同一独立 head-on manifest 上报告均值、离散度和 paired outcome；不得把 seed 平均值与 canonical 单 checkpoint 主表混池。

**验收：** 任何新 head-on 结果都必须有新 manifest、clean SHA、asset hashes、预先固定的门槛，以及不因结果而修改策略/场景/阈值的 stop rule。

### B2. Crossing 的“学习式路由”创新受 task bit 与 deterministic preservation 限制

**现象：** RQ3 中 unlocked PPO 仍 96/96 首拍选择 crossing，但 observation 保留 `is_crossing`；RQ1/RQ2 中 PPO 从首个高层步一直使用 crossing，零切换。canonical crossing 仅 N=4。

**根因：** 当前设计有意使用 task-aware input 与 bootstrap/lock 保证 crossing specialist 不退化。这是合理的工程接口约束，但不是 task-label-free tactical perception。

**风险：** 中高。若论文把该结果称作自主几何识别或普适 expert routing，会被 RQ3 直接反驳。

**改进：**

1. 论文与图中固定用语改为 `task-aware PPO routing with deterministic preservation constraints`。
2. 把 RQ3 保留为机制透明度证据，而不是作为 superiority evidence。
3. 如果下一篇论文要主张几何驱动路由，必须建立独立新 lane：移除 task bit、冻结低层策略、使用未见 initial geometry，预注册“geometry-only first action”和“post-merge re-selection”两类指标。这个实验不应在当前 canonical paper 中临时加入。

**验收：** 当前论文中不出现 `task-label-free`、`autonomous geometry recognition`、`general-purpose routing` 等超出证据的表述。

### B3. Recovery 的主要短板是低层几何与 opponent split，而非缺少更多高层 clamp

**现象：** canonical PPO 在 expert/head-on 减少 ego crash/OOB（0.1333 vs 0.1667）并增胜；在 end-to-end/head-on 却增加 ego crash/OOB（0.0500 vs 0.0167）且少 2 个 win。三条 formal oracle-only counterexample 中至少两条已正确进入 recovery，仍然失败。v3/v9 等探索也反复出现“某一 split 提升、另一 split 崩溃”。

**根因：** mode 2 沿用 head-on checkpoint，只改变 profile/geometry clamp；它没有独立可学习的 recovery primitive。一个全局 recovery 几何规则必须同时面对不同 opponent 的 post-merge energy、altitude 和 re-engagement 分布，因而天然容易产生 split-specific trade-off。

**风险：** 高。继续往 `commander_mode_constraints.py` 叠加条件会降低可解释性、增加测试负担，并进一步污染 canonical evaluation set。

**改进：**

- canonical family 继续冻结，不再加 commander clamp。
- 将 `v14b` 作为独立 research family 的种子：只允许“显式 low-level subpolicy + 显式 routing predicate + off-lane abstention”，不允许全局替换 mode 0 或增加未经验证的 reward hack。
- 首个 clean gate 必须同时覆盖 `expert/head_on`、`end_to_end/head_on` 与独立 head-on envelope；每个 split 都报告 PPO、oracle、canonical、candidate 的 paired outcome、ego crash/OOB、timeout terminal mix、mode/subpolicy activation rate。
- 若 candidate 任一 split 相对 canonical 退化超过预注册 practical margin，或 ego crash/OOB 增加超过预注册阈值，立即冻结为负证据；不得再以新 guard 覆盖。

**验收：** candidate 只有在 clean paper-safe、两 split 不退化、独立 manifest 不失效、且 activation 只发生在预定义 motif 时，才可升级为正式 research family。

### B4. 终局指标被 timeout 与 crash/OOB 混合，限制战术解释与飞控安全论证

**现象：** expert/head-on 的 60 episode 中，Oracle 有 39 个 timeout、PPO 有 40 个 timeout；两者大部分胜利来自 HP advantage timeout。end-to-end/head-on 的胜利主要来自 target crash/OOB。aggregate saturation 为零，但并未直接测量实际载荷响应、aerodynamic alpha 或 stall margin。

**根因：** 现有评价目标将战术优势、强制对方越界、击落与自身失控同时映射到 win/loss。对于战术问题这是可接受的 composite outcome，但对于“低层飞控质量”解释不足。

**风险：** 中高。若把 win-rate 改写成击杀能力、追踪质量或安全裕度，会产生不成立的强解释。

**改进：**

1. 所有主表外补充 `terminal-reason decomposition`：HP advantage/disadvantage、ego crash/OOB、target crash/OOB、target/ego killed、draw。
2. 将飞控指标与战术指标严格拆开：命令饱和、实际 $n_z$、aerodynamic alpha、速度、高度、边界距离、控制裕度应是独立 safety/feasibility appendix，不能由 attack AoA 代替。
3. 若希望发表“物理可执行性”，新增不训练的 telemetry envelope：速度--高度--AoA--$n_z$ 实际可达域、命令-响应误差、限幅持续时间，并明确这是 simulation validation 而非 airworthiness certification。

**验收：** 每个 performance claim 与对应 terminal mix 同时可追溯；安全段不再从 `ego_attack_aoa_deg` 推断 aerodynamic alpha 或失速裕度。

### B5. 模式约束与配置空间已经形成“规则堆叠”风险

**现象：** `commander_mode_constraints.py` 包含多个 target-threat、secondary、overdeep、geometry-quality 等 clamp；`hierarchical_commander_policy.py` 中多次串联它们。`tracking_env.py` 同时承载 observation、prediction、VPP geometry profile、backend 等大量职责。286 份 YAML 使实验命名和 includes 链很容易偏离真实执行配置。

**根因：** 每轮 residual audit 都以“局部修复”落到新 YAML、guard 或条件分支；缺少将 guard 视为可声明、可枚举、可测试的 policy layer 的统一抽象。

**风险：** 高。不同 config 的隐式叠加、mode 2 与 mode 0 的实际差异、以及同名 checkpoint 的依赖很难通过人工阅读保证一致，极易形成“可运行但不可解释”的系统。

**改进：**

1. 在冻结 bundle 后，把 guard chain 重构为一个声明式 `ConstraintPipeline`：固定执行顺序、输入 state、输出 effective mode、override reason、优先级和 trace。
2. 为每个 canonical config 自动生成 `resolved_policy_card.json`，列出 learned action space、所有 deterministic overrides、specialist checkpoint hash、observation adapter、active profile 和关闭项。
3. 为 YAML includes 建图并在 CI 中禁止 cycle、未声明 override、同一 key 的静默覆盖；canonical config 只允许白名单字段发生变化。
4. `tracking_env.py` 先做职责切片，不在 canonical freeze 前大规模重构：优先把 telemetry/profile calculation 移至独立 module，保持 observation order 与 backend contract 不变。

**验收：** 任一 episode 都能用单条 telemetry 记录回答“PPO proposal 是什么、哪个 guard 改写了它、最后哪个 specialist/profile 输出了 VPP”。

### B6. 工作区污染与投稿包多版本并存

**现象：** 本次审查起点的主工作区有 381 项未提交内容（53 modified、327 untracked），其中包括 54 个已跟踪文件的约 8,402 行新增与 1,971 行删除。`drones/` 同时存在多版 TeX/PDF；`README_submission_ast.md` 仍指向旧 `dfartv2_ast_anonymized.pdf` 与旧 G1 图，而近期主稿是 `dfar_ast_final_v5.tex`。`STATE.md`、AST readiness checklist、publication checklist 仍有 `Table/Figure S7` 与 20260703 artifact 引用。

**根因：** 论文修订、图生成、探索算法导入、历史文件和运行产物都在同一根目录累积，缺少明确的 submission manifest 与归档策略。

**风险：** 极高。即使实验正确，也可能上传错误 PDF、错误补充材料或旧图表；审稿复现者会面对相互矛盾的路径。

**改进：**

1. 建立唯一 `submission_manifest.yaml`：列出主稿 TeX/PDF、title page、highlights、supplement、figure 文件、CAN-20260705 artifact index 及 SHA-256。
2. 将所有旧稿、旧 build、旧 figure 迁至 `drones/historical/` 或在 manifest 中显式标记 `excluded`；不删除历史证据。
3. 以 `dfar_ast_final_v5.tex` / `supplementary_material_v4.tex` / `supplement_index.tex` 为唯一 current submission source，更新 README、STATE、LOOP、readiness checklist 的 S3 编号与 canonical source。
4. 新研究一律在独立 worktree/output root 进行；主仓库只接受经过 review 的 minimal import。

**验收：** 一条 `build_submission_bundle` 命令从 manifest 生成唯一 zip，且 zip 内没有未引用的旧稿、G1、20260703 headline 或 historical DQN 叙事。

### B7. 外部资产虽已恢复，但还没有被当作可版本化依赖管理

**现象：** head-on checkpoint、crossing-v3 checkpoint 和 prediction model 当前可见，preflight 也通过；但它们不在 Git 中，过去曾发生过本地缺失与重建时 action-dimension/warm-start 不一致的问题。

**根因：** checkpoint 的训练谱系、代码 SHA、观测维度、asset hash 与运行引用没有集中成一个 immutable registry。

**风险：** 高。它会让“同名 checkpoint”在不同工作树中实际代表不同二进制文件，从而破坏跨机器复现。

**改进：**

- 为每个外部 asset 固定 `asset_id`、SHA-256、size、producer commit、observation/action dimensions、training config、selected-step、保存位置和使用证据 run。
- preflight 必须比较实际 hash 而非只检查路径存在；任何 mismatch 直接 `paper_safe=false`。
- 为 canonical bundle 保存只读资产副本或受控 artifact store 地址，禁止 runner 自动从主工作区“同步最新版”。

**验收：** 删除本机 outputs 后，按 registry 在新目录还原 assets 能通过 hash/shape/preflight 三重检查。

---

## 5. 当前最合理的决策分流

### 路线 A：以投稿稳健性为第一目标（推荐）

**不重新训练，不改 canonical policy。**

1. 完成 B0 的精确 snapshot/bundle。
2. 完成 B6 的唯一投稿 manifest 与文档同步。
3. 可选但高信息增益：一次无训练、预注册的独立 head-on envelope。
4. 以现有 RQ1--RQ4 作为分层 supporting evidence：RQ1/RQ2 说明有限 envelope preservation，RQ3 说明 task-aware selection，RQ4 说明 constrained dynamic re-selection。

**适用情形：** 目标是三周内可靠投稿，宁可缩窄 claim，也不再让研究叙事被新 tuning 打断。

### 路线 B：以方法创新为第一目标

**canonical 永久冻结；另开 clean research lane。**

1. 将 v14b 定义为 `selective_low_level_recovery_routing_v1`，不使用未来几何奖励的版本号作为论文方法名。
2. 用相同时间点的 canonical、candidate、oracle 和 fixed specialist，在两 opponent split + 独立 head-on manifest 上验证。
3. 将评价从总 win-rate 扩展到 activation precision、off-lane abstention、paired terminal transition、ego crash/OOB 和 low-level geometry windows。
4. 若第一轮跨 split 不稳定，则停止“路由新 specialist”路线，转而研究低层 primitive/flight-control mechanism，不再增加高层 guard。

**适用情形：** 目标是下一篇更强的方法论文，而不是改写当前 AST 主稿。

### 明确不推荐的路线

- 不要继续在 canonical mainline 上增加 commander clamp、阈值或训练步数。
- 不要把 v14b 的 dirty 58/60 直接替换 canonical 0.9000。
- 不要把 RQ3/RQ4 提升为 task-label-free 或纯学习式 recovery 的证据。
- 不要用 aggregate zero saturation 宣称飞控安全认证。
- 不要再扩充对比算法，直到证据快照和 submission manifest 收口；新增算法会加重工作树和叙事分叉，不能解决当前核心风险。

---

## 6. 分阶段行动清单

### P0：72 小时内完成的证据与工程收口

| 优先级 | 动作 | 不应做什么 | 验收 |
|---|---|---|---|
| P0-1 | 创建 `9a9f9bf6` detached canonical worktree 与 immutable artifact bundle | 不重跑、不卡住主论文 | hash/shape/preflight/artifact verifier 全绿 |
| P0-2 | 建立 `submission_manifest.yaml` 与 bundle builder | 不继续维护多份“current” PDF | 一条命令生成唯一投稿 zip |
| P0-3 | 同步 STATE/LOOP/README/readiness/checklist 到 CAN-20260705、S3、v5 source | 不改主结果数字 | 文本扫描无 S7、旧 G1、错误 current TeX |
| P0-4 | 将主仓库 dirty/untracked 内容按 canonical / noncanonical / historical / build 分类 | 不删除用户文件 | 每类有 owner、lane、保留策略 |

### P1：一周内完成的科学质量增强

| 优先级 | 动作 | 预注册门槛 | 结果解释 |
|---|---|---|---|
| P1-1 | 独立 head-on envelope，无训练、双 opponent | PPO-oracle paired delta、ego crash/OOB、terminal mix 均事先定义 | 通过则强化外推；失败则缩窄论文声明，不调参 |
| P1-2 | 在 supplement 加 terminal-reason 与 safety-semantic 表 | 区分 command 与 actual response，区分 ego/target terminal | 使 win-rate 不被误读为 kill rate 或安全认证 |
| P1-3 | 把 RQ1--RQ4 做为分层证据矩阵 | 每个 RQ 独立 source ID，不合并统计量 | 强化透明机制解释，不夸大创新 |

### P2：下一研究周期的可控研发

| 优先级 | 动作 | Stop rule |
|---|---|---|
| P2-1 | v14b clean import 与 cross-split replication | 任一 split 对 canonical 的性能/安全 practical gate 失败即冻结 |
| P2-2 | Guard pipeline 声明化、policy card 自动化 | 不在复现 bundle 建立前重构 canonical 代码 |
| P2-3 | geometry-only routing 新 lane | 未通过 task-bit removal baseline 时，不做“自主识别”主张 |

---

## 7. 可验收的项目健康指标

| 维度 | 当前状态 | 下一门槛 |
|---|---|---|
| Canonical 数字 | 有效且 paper-safe | 形成精确代码+资产 bundle |
| Head-on 泛化 | 单 checkpoint，iterative exposure 风险 | 独立冻结 envelope 或明确降级表述 |
| Crossing | 两个 24 场景 envelope + 3 seed supporting evidence | 保持 task-aware、有限 envelope 定位 |
| Dynamic routing | 有受限机制证据 | 不把 guard recovery 计作 PPO 独立学习 |
| 低层 recovery | 有 v14b 正向探索 | clean、两 split、独立 manifest 后才升级 |
| Safety | 无 fallback/aggregate saturation | 增加实际响应与终局语义诊断，不做认证声明 |
| 代码质量 | 关键测试通过，但核心模块过大 | bundle 后再做声明式 guard/telemetry 解耦 |
| 投稿包 | 多稿、多 index、多版本仍并存 | manifest 驱动的唯一交付物 |

---

## 8. 本次审查的可追溯来源

- `reports/paper_authoritative_artifact_index_20260712.md`
- `reports/paper_single_source_of_truth_reaudit_20260712.md`
- `reports/canonical_commander_mainline_freeze_20260705.md`
- `reports/post_merge_recovery_route_b_safety_report.md`
- `reports/g4_end_to_end_three_method_ablation_20260711.md`
- `reports/noncanonical_future_geometry_prediction_reward_attempt_log_20260711.md`
- `reports/noncanonical_future_geometry_prediction_reward_routed_family_v14b_closure_run_20260711.md`
- `E:\uav-vpp-guidance-clean-rq1-crossing\reports\rq1_crossing_generalization_result_20260711.md`
- `E:\uav-vpp-guidance-clean-rq2-crossing\reports\rq2_crossing_independent_envelope_result_20260712.md`
- `E:\uav-vpp-guidance-clean-rq3-crossing\reports\rq3_crossing_locked_unlocked_mechanism_result_20260712.md`
- `E:\uav-vpp-guidance-clean-rq4-routing\reports\rq4_postmerge_dynamic_routing_result_20260712.md`
- canonical dual-split `run_manifest.json`、`method_task_summary.json`、`summary.csv`、`combat_geometry_diagnostics.json`

## 9. 本次审查执行的验证

- targeted contract tests：`183 passed, 2 skipped`；三条 JSBSim fallback warning 来自显式模拟 backend failure 的测试用例。
- current clean worktree preflight：42 YAML dependencies、12 code dependencies、2 methods、`Issues: none`。
- 该 preflight 的当前工作树 HEAD 为 `2465b652`，不是 formal run manifest 的 `9a9f9bf6`；因此它验证了依赖可用性，不能替代 B0 所要求的精确 canonical reconstruction。

