# V17 Path-Context Schema 与采样 Preflight 预注册

**Source ID：** `THESIS-FIVE-STATE-PATH-CONTEXT-PREFLIGHT-V17`
**状态：** schema implementation only；未授权采集、训练、调参或 held-out 评估。

## 1. 触发依据

V16 已在 expert 与 independent PPO/VPP 下证明：当前 state、冻结 predictor 与 range-rate 构造的动态参考路径能在不增加 ego crash/OOB 的条件下，通过既有 3-D VPP、guidance 和 PID 链产生非零响应。但 V16 的有效配对 terminal reason 未改变，且路径 action 常被裁剪。因此不能直接训练，更不能宣称性能优势。

V1 的 66-D 输入没有当前 path anchor、path progress 或路径相对 VPP 可达域的信息。一个新策略只能间接猜测外部 path adapter 的目标，无法区分“策略本身选择的 VPP”与“接口追加的、可能饱和的路径偏置”。V17 先修复该可观测性缺口，再验证是否存在可用于训练的连续样本。

## 2. 冻结的 v2 输入合同

`five_state_shared_intent_v2_path_context` 保留 V1 的完整 66-D 前缀，并追加 8-D current-only path context，总维度为 `74`：

| Segment | Dim | 来源 | 禁止项 |
|---|---:|---|---|
| V1 prefix | 66 | 当前 16-D 几何、10-step history、冻结 profile、冻结 P3 embedding | task/opponent id、future state、隐式 padding |
| path delta | 3 | selected anchor 相对 frozen predicted-target anchor 的 world-NEU 偏置，按冻结 VPP 范围归一化并保留至 `[-2,2]` | policy 尚未输出的 action |
| path progress | 1 | 当前 range-rate 的确定性映射 | time-index schedule |
| anchor one-hot | 3 | 当前 escape / connector / predicted-reentry 选择 | 标签泄漏 |
| envelope excess | 1 | path delta 是否超出冻结 3-D VPP action range | post-action response |

旧 66-D checkpoint 的 transplant、截断或补零均禁止。V17 policy 必须从新的 `74-D -> 3-D normalized VPP` head 开始；冻结 P3 encoder 可以复用，但不微调。

## 3. 第一阶段：非学习 sampler preflight

在任何 PPO、reward 或 profile 调参前，必须生成一个与 V1--V16、Taxonomy30、dev30、heldout60 完全不相交的 **dev-only** 24 场景 continuous run-in manifest，并用 frozen fixed-head-on specialist 采集。三个 opponent 分开统计，禁止 pooled gate。

每个 opponent 必须同时满足：

1. 至少 8 条连续 handoff episode，且每条有 20 个完整 74-D / normalized 3-D VPP 对；
2. 至少 160 个有限 path-context step，无 prediction/backend/checkpoint fallback、无 reset、无 padding；
3. 至少 2 个镜像与 2 个场景 signature；escape 和 predicted-reentry 均至少出现一次；
4. 记录 requested/applied path delta、clip flag、VPP offset、guidance command、actual `n_z`、saturation、terminal reason；
5. 采集器中 ego crash/OOB 不得高于同场景 frozen fixed-head-on reference `+0.05`。

任何一项失败即冻结为 sampler/observability 负证据，禁止训练、加步数、缩窄 opponent 或用 V16 输出替代。

## 4. 后续许可边界

仅当 V17 sampler preflight 三个 opponent 全部通过，才允许另起 Source ID 起草一个 **dev-only path-conditioned residual-policy pilot**。该 pilot 必须预注册新 output root、训练预算、三对手平衡采样、冻结 reward、terminal/safety stop rule 与 paired mechanism gate；它仍不自动解锁四共享技能、P5--P7 或正式论文声明。
