# V13 Six-Objective Intent-Controllability 结果与 Go/No-Go 决定

**Source ID：** `THESIS-FIVE-STATE-REENTRY-INTENT-CONTROLLABILITY-V13`
**运行 SHA：** `5b045ea0fcdc5163887475af3d1a1ab04fecda34`
**结果根目录：** `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_reentry_v13_intent_controllability`
**最终判定：** `no_safe_local_multiobjective_candidate_under_frozen_single_point_vpp_interface`

## 1. 完整性

- V13 在修复 runtime-template preflight 合同后完成 `8 methods x 3 opponents x 12 scenes = 288/288` 条 strict-JSBSim record；没有 `failure.json`。
- 每个 opponent 的每种方法均与 reference 使用相同 handoff hash；qualified pair 数为 expert `4`、end-to-end `5`、independent PPO/VPP `5`，满足预注册最小分母。
- 独立 verifier 从 288 个 episode ledger 重算 gate，确认 `reported_gate_matches_recomputed=true`、`passes=false`。
- 所有方法的 ego crash/OOB 相对 reference delta 均为 `0.0`；全部 640 个 expert、800 个 end-to-end、800 个 independent PPO/VPP qualified window step 均未出现 command saturation。

## 2. 动作确实进入执行链

V13 排除了“动作没有被注入”的解释。非参考方法在 qualified window 的实际 semantic VPP delta norm 如下：

| 轴组合 | expert (m) | end-to-end (m) | independent PPO/VPP (m) |
|---|---:|---:|---:|
| range only | 521.5 | 745.2 | 630.0 |
| turn only | 759.4 | 693.8 | 725.5 |
| vertical only | 293.3 | 207.0 | 244.7 |
| range + turn | 906.5 | 653.5 | 920.4 |
| range + vertical | 618.5 | 816.8 | 711.8 |
| turn + vertical | 823.0 | 732.7 | 775.4 |
| all three | 966.9 | 697.9 | 972.9 |

三轴的平均归一化命令也都非零：range 约 `0.36–0.64`、turn 约 `0.77–0.84`、vertical 的平均绝对值约 `0.52–0.73`。因此失败不是 clip、fallback、零动作或飞控饱和造成。

## 3. 六维目标的实际响应

尽管动作进入 VPP/guidance，六维 profile-error 相对 reference 的变化普遍很小，且对手依赖：

- expert：最好的 weighted error delta 仅 `-0.000275`（range+turn 或 all-three），远小于预注册 `-0.01`；AA/ATA 或 range 往往轻微恶化。
- end-to-end：range-only 的 range-rate 分量改善 `-0.01193`，AA/ATA 也改善约 `-0.006`，但 altitude 恶化 `+0.00670`，总 weighted delta 仅 `-0.00331`，不足以达到门槛。
- independent PPO/VPP：range+turn/all-three 的 range-rate 分量恶化 `+0.03097`，超过高优先级允许的 `+0.02`；其 weighted delta 为 `+0.00636`。
- vertical 轴被实际注入，但在 20-step 窗口内对六维 profile-error 的相对变化接近零；这不能被解释为垂直动作不存在，只能说明在本接口与本窗口下未形成可用的多目标收益。

没有任何 action combination 在至少两个对手同时满足：range/range-rate 非恶化、至少两项改善、任一分量不显著恶化、weighted error 改善 `>=0.01` 和安全门。因此 `local_multiobjective_candidates=[]`。

## 4. 正式决定

**No-go：** 不得基于当前 single-point 3-D VPP interface 启动终端奖励重跑、单技能或四技能训练、combat finetune、P5 PPO、P6 消融或 P7 heldout。不得在 V13 下改变 action scale、窗口长度、profile target、方法矩阵、对手或场景补跑。

**可保留的机制结论：** 当前系统安全且可复现地将 relative-frame range/turn/vertical 语义送达 VPP 和 guidance；但在冻结的 20-step 单点 VPP 接口内，存在明显的多目标分辨率/耦合限制，尚未建立可供学习利用的、跨两对手一致的安全局部下降方向。

**后续唯一合理的研发方向：** 若继续追求五态势共享意图的实用优势，必须提出新的、预注册的**时间扩展几何接口**，例如受约束的 short-horizon reference path 或 escape/re-entry corridor，而不是继续调单点 VPP、奖励、尺度或训练步数。该接口需先以新的 Source ID 做非学习 action-to-trajectory-to-six-intent transfer probe，证明存在跨两对手的一致多目标可控方向后，才可讨论训练。
