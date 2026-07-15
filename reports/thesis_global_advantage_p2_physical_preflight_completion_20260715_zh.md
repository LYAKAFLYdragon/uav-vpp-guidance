# P2-B2 Strict JSBSim Physical Preflight 完成审计

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHYSICAL-PREFLIGHT-V1`
**最终判定：** `physical_pass_phase_observability_no_go`
**执行状态：** 已一次性完成，授权已关闭；不允许重跑、补样本或调参。

## 结论

本轮证明了冻结 reference chain 能在全部 60 个几何 cell 与三个分开报告的 opponent 上完成严格 JSBSim 积分；它**没有**证明五态势包线中的 phase 顺序或 re-entry 可达性。因而，P2-B2 是“物理执行合同通过、phase 可观测性合同不通过”的混合结果，不会解锁低层技能、高层 PPO 或 formal held-out。

| 合同 | 结果 | 证据含义 |
|---|---:|---|
| Episode 完整性 | `180/180` valid | 每个 `60 cell x 3 opponent` 均持久化成功 |
| 有限 action/state | `44,571/44,571` steps | 3-D action 与 own/target 的 speed、altitude、`n_z` 均有限 |
| 严格后端 | `0` backend fallback | 本轮未回退 simple backend |
| 预测器 | `0` 非 warmup fallback | 未用 predictor fallback 掩盖记录 |
| Episode 长度 | `160--260` steps | 全部超过预注册最小 20 steps |
| 初始 phase 一致性 | `0/180` | manifest 为 `pre_merge`，首个 raw phase 均为 `post_merge` |
| phase predicate 可复算性 | `0/44,571` steps | raw telemetry 未保存 `range_m/range_rate_mps` |
| re-entry 覆盖 | 三对手均 `0` | 不能作为 re-entry 训练或性能证据 |

## 分开报告的 phase 结果

| Opponent | pre_merge steps | post_merge steps | re_entry steps | 首次 first-pass episode |
|---|---:|---:|---:|---:|
| expert | 0 | 14,742 | 0 | 25/60 |
| end_to_end | 0 | 14,612 | 0 | 17/60 |
| independent_ppo_vpp | 0 | 15,217 | 0 | 24/60 |

`first_pass_complete=true` 在全体记录中只出现于 66 个 episode，而 phase label 从首个 recorded step 起始终为 `post_merge`。这不是“对手已经证明很强/很弱”，也不是“新技能训练不足”；它表明本轮 raw contract 无法独立验证 phase tracker 是因真实初始 range、scenario application，还是 telemetry 口径而进入 `post_merge`。

冻结 manifest 联接的 `initial_state x opponent x phase` 完整 45 行矩阵见 `reports/thesis_global_advantage_p2_physical_preflight_phase_matrix_20260715.csv`。五种初始态势在三对手下都呈相同模式：仅有 post-merge step，pre-merge 与 re-entry 均为 0。

## 终端分解

| Opponent | Target crash/OOB | Ego crash/OOB | Timeout advantage | Timeout disadvantage | Timeout draw |
|---|---:|---:|---:|---:|---:|
| expert | 16 | 4 | 25 | 15 | 0 |
| end_to_end | 27 | 0 | 25 | 8 | 0 |
| independent_ppo_vpp | 11 | 2 | 25 | 19 | 3 |

这些 terminal mix 只刻画共同 frozen run-in reference 所受的对手压力，不能构成 Elo、总体 opponent 强度排序、候选方法性能或安全认证。

## 溯源与后续边界

- 原始 gate：`physical_preflight_gate.json`，SHA-256 `38a36badad27805e4f041b410caaccb2907453e85e466a640f91fe71ad7750d1`。
- 原始 run manifest：`run_manifest.json`，SHA-256 `9d8458a0bc893727b0d648879a2e8c3bc5e2fe525e5481c9c7483c689be73d9d`。
- 只读 audit：`reports/thesis_global_advantage_p2_physical_preflight_audit_20260715.json`，SHA-256 `9483812a784480259dddf327fc78ae0f9ada1b189713af405df62a434126dd18`。
- phase matrix：`reports/thesis_global_advantage_p2_physical_preflight_phase_matrix_20260715.csv`，SHA-256 `6001f5ed6991db162785283b6f0cd80043bbd615be189080676b8a8fe6ab3294`。

P2-B2 本身不得修补或重跑。唯一合理的后续是独立的 P2-B3 **设计线**：新 source ID、新场景/seed、reset receipt 与 step-0 记录、完整的 base observation/range/range-rate、PhaseTracker input/output 以及 first-pass receipt。只有该新合同在三 opponent 下独立通过，才可重新讨论 phase-dependent 单技能 feasibility；四技能库、P5 和 formal held-out 继续锁定。
