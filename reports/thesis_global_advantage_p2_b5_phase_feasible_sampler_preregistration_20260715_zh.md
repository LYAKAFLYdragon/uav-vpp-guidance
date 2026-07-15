# P2-B5 Phase-Feasible Sampler 设计与预注册

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1`
**状态：** `implementation_complete_execution_not_authorised`
**性质：** 非 canonical、只读、无学习的数据契约检验，不是性能实验。

## 1. 问题与边界

P2-B4 已证明 phase tracker 必须从 `observation.relative_state` 读取原始 SI 单位的
`range_m/range_rate_mps`。它消除了将归一化 policy observation 当成米制数据的错误，
但不能保证每个对手都存在 `disadvantage -> post_merge/re_entry` 的连续样本。

B5 只检验下列问题：冻结 head-on specialist 的真实、连续 JSBSim rollout，能否在三个
opponent 下分别提供足够的目标 motif 66-D observation 与原始 3-D VPP action。它不比较胜率，
不训练新技能，不修改 reward、VPP、guidance、PID 或高层路由。

## 2. 冻结设计

- 参考控制器：`frozen_fixed_head_on_specialist`，registry key 为 `run_in_head_on`；SHA-256 为 `0aeadd11cd8c7723c8963bcc5da21dd97c365e64f234d07b86c2b8f0d7ec5fb6`。
- P3 temporal encoder：冻结 16-D 到 32-D checkpoint；SHA-256 为 `385663281a9c48c0416ea4d1a1bb8dc08ed64a0cfebb0f01083cbbb8bcfbdc59`。
- 观测合同：真实 16-D base、10 帧无 padding 历史、6-D 显式短时统计、固定 `range_extension` intent profile、6-D targets、6-D weights、32-D P3 embedding，合成为 66-D。
- 动作合同：记录参考控制器输出的 normalized 3-D VPP action；collector 只比较 action 前后哈希，不替换 action。
- 相位合同：只有 raw-SI `observation.relative_state.range_m/range_rate_mps` 可以驱动 `PhaseTracker`；policy vector 的 `range_m/5000` 和 `range_rate/200` 只作为诊断对照。
- Taxonomy 合同：动态分类器严格以 `(ATA, AA)` 顺序接收我机视线角与目标姿态角；已用 `ATA=160°、AA=20° -> disadvantage` 的逐步 collector 回归测试锁定，禁止按 policy vector 的特征排列颠倒两者。
- 连续性：每 episode 只 reset 一次。只有在真实 first-pass 后的物理下一步才记为 handoff；禁止 reset、snapshot restore、future-state injection 和 history padding。

## 3. 新包线与独立性

Manifest：`config/experiment/manifests/thesis_global_advantage_v1_p2_b5_phase_feasible_sampler12.yaml`

- Manifest Source ID：`THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-MANIFEST12-R1`。
- 12 个 scenario：单一初始 `disadvantage`，两组新的距离/速度包、三种高度条件、两个镜像方向。
- package C：3,200 m，own/target speed 为 230/360 m/s；package D：3,800 m，250/390 m/s。
- 新 seed：15,101--15,112；与 P2-B2、P2-B3、P2-B4、heldout240、dev30、旧 heldout60 和旧 phase-v2 manifest 的物理状态签名均不相交。
- 每个 opponent 每场景一次，共计划 36 条 record；三个 opponent 永不合并成强度排名或 pooled gate。

完整 manifest payload SHA-256：`76379d9f7959e7358b669892395d1f26c2c913633239a1b14028685b23146cb9`。
Manifest 文件 SHA-256：`63acf19e57d5488f33fdef48df9b59ec9c7f892c5a70e1465cea9622bf3f798a`。

## 4. 预注册 Gate

每个 opponent 必须**分别**满足：

1. 至少 2 条 qualifying episode。
2. 至少 20 个连续有效 `disadvantage -> post_merge/re_entry` target step。
3. 至少 2 个不同 scenario signature 和两个镜像方向。
4. 所有 qualifying record 都是 strict JSBSim，无 backend fallback、无 reset、无 future-state injection、无 padding，raw-SI phase 可回放，且 66-D 与 3-D 均有限。

所有三个 opponent 均通过时，只允许起草一个新的 defensive-extension pilot 执行授权；`training_unlocked` 在 B5 中始终为 `false`。任何 opponent 失败即为 B5 的预注册负证据，不能用其它两个 opponent 的结果覆盖，也不能改场景、调参、重跑或开始训练。

## 5. 已完成验证

- 设计 preflight 已通过：manifest/runtime/registry/P3/reference SHA 均匹配，输出根不存在，E 盘可用空间约 292.05 GB，高于 120 GB 门槛。
- `python scripts/preflight_thesis_global_advantage_p2_b5_phase_feasible_sampler.py` 只读验证通过；未调用 JSBSim。
- `python scripts/run_thesis_global_advantage_p2_b5_phase_feasible_sampler.py` 只输出 `validated_not_executed`；`execution_permitted=false`。
- 新增 B5 5 项合同测试，并与 B4 raw-SI 和已有 single-motif collector 回归共运行 13 项：`13 passed`。

## 6. 下一步

唯一可行的下一步是独立复核本预注册的场景包、门槛和 stop rule；复核通过后，另建一次性 execution authorization，冻结 clean SHA、授权文件哈希和空输出根。未经该单独授权，不得执行 B5，也不得进入 defensive-extension pilot、四共享技能、P5/P6 或 formal held-out。
