# Disadvantage Capability Plan 2 V1 预注册：低层机动能力判别

**Source ID：** `THESIS-DISADVANTAGE-CAPABILITY-PLAN2-V1`  
**性质：** 独立、非学习、只评估的机制实验；不改变 Taxonomy60 或投稿主稿。  
**前置证据：** Plan 1 V2 未支持机械 H-to-C 路由解释，但未能区分技能库、VPP 接口与可达性。

## 1. 问题与范围

本实验只回答：在 opening-pressure `disadvantage` 中，现有 tactical-basis VPP action head 是否能将一种有界、可解释的低层几何原语安全传递为实际 guidance/flight-control 响应，并产生跨对手的目标几何改善。

它不训练 PPO、修改 reward、修改 VPP/guidance/PID、扩大 action envelope、选择终端后重跑，或把临时 primitive 当成可部署策略。Plan 1 中的 canonical PPO 与 H-to-C schedule 不在本矩阵中重复；fixed head-on/crossing 仅作为冻结控制。

## 2. 冻结方法

所有候选先使用同一 frozen head-on specialist 连续运行 10 个 high-level step，使预测器通过既有九步 warm-up。随后仅在连续的 20 个 high-level step 内，向该 specialist 的当步 tactical-basis action 加入一个预先规定的梯形 pulse；其余时段恢复相同 head-on action。

| Method | 原语 | 当前状态信息 | 预期检验 |
|---|---|---|---|
| `fixed_head_on` | 无 | 不适用 | 成对 reference |
| `fixed_crossing` | 无 | 不适用 | 既有技能库控制 |
| `lateral_break_same_side` | active IO basis 的同侧横向偏置 | 当前 tactical basis | 是否可降低 opponent attack-zone exposure |
| `range_extension_los` | 沿 LOS 向我机侧的 VPP 偏置 | 当前 own/target position | 是否可提高 range-rate |
| `vertical_separation` | 远离目标高度侧的垂向 VPP 偏置 | 当前 own/target altitude | 是否可增大垂向间隔而不显著损失能量 |

每个候选均用 active runtime tactical basis 将 world-frame request 逆投影到现有 3-D action head；每步记录 basis rank/condition、请求与实际 action、投影残差、VPP world offset、guidance command、actual `n_z`、AoA、saturation、prediction validity 与终端。这不是替换 action semantics 的新接口。

## 3. 冻结场景与矩阵

- 新 Disadvantage12：两组未使用距离/速度、三种高度条件、双镜像，初始 `ATA=160°`、`AA=20°`，均为 pre-merge opening-pressure 状态。
- 对手分别报告：`expert`、`end_to_end`、`independent_ppo_vpp`；禁止 pooled gate。
- 固定矩阵：`5 methods x 3 opponents x 12 scenarios = 180` strict-JSBSim records。
- 每条 episode 只允许一次合法 reset；禁止 snapshot restore、future-state injection、backend/checkpoint fallback 与 history padding。

## 4. 判定门与分流

每个候选、每个 opponent 先同时通过执行门：完整 10-step warm-up hash 与 reference 一致；20-step window 无 fallback；basis rank 为 3、condition 不超过 100；实际 pulse、实际 VPP 和 guidance 变化分别达到预注册下限；平均投影残差不超过 650 m；action clipping、saturation 与 ego crash/OOB 不超过 reference 的允许增量。

在执行门通过后，各原语还需通过唯一对应的几何门：

- lateral break：对手攻击区暴露 fraction 相对 reference `<= -0.10`。
- range extension：mean range-rate delta `>= +5 m/s`。
- vertical separation：绝对高度差增加 `>= 50 m` 且 specific-energy-height delta `>= -100 m`。

同一原语在至少两个 opponent 分别通过，才被记录为 `candidate_low_level_motif_exists`，此时只允许为该原语起草一个全新、独立 heldout 的 Plan 4 单技能 pilot。

| Plan 2 结果 | 结论 | 禁止事项 |
|---|---|---|
| 某原语在至少两个对手通过 | 现有双技能库缺少该可训练低层能力的候选证据 | 不得同时训练多技能库 |
| 所有原语无法通过执行门 | 现有 VPP-to-guidance transmission 尚未建立 | 不得训练技能或调 reward |
| 执行门通过但没有跨对手几何原语 | 当前 action envelope/horizon 下的 physical-or-behavior reachability 缺口 | 不得加步数、改幅度或改场景补跑 |
| 仅一个对手有几何改善 | opponent-dependent sensitivity | 不得声称通用技能缺口 |

## 5. 可复核入口

- Manifest builder: `scripts/build_thesis_disadvantage_capability_plan2_manifest.py`
- Config: `config/experiment/thesis_disadvantage_capability_plan2_v1.yaml`
- Motif compiler/policy: `src/uav_vpp_guidance/evaluation/tactical_basis_motif_policy.py`
- Runner: `scripts/run_thesis_disadvantage_capability_plan2.py`
- Independent verifier: `scripts/verify_thesis_disadvantage_capability_plan2.py`

该 Source 的唯一目的是回答“缺什么能力，或是否根本不是技能问题”。它不能自行证明最终胜率优势，也不会自动授权 Plan 4 训练。
