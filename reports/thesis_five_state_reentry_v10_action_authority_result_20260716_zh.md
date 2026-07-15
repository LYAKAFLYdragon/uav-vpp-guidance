# V10 3-D VPP 动作可控性结果与机制决定

**Source ID：** `THESIS-FIVE-STATE-REENTRY-VPP-ACTION-AUTHORITY-V10`
**范围：** 非学习、非论文主结果的 local action-authority probe。
**最终判定：** `no_safe_practical_range_extension_action_authority_under_current_raw_vpp_head`。

## 证据完整性

- 以全新的物理不相交 manifest 完成 `7 methods x 3 opponents x 12 scenarios = 252` 条 strict-JSBSim record。
- 三对手分别报告；所有 20-step 配对均使用相同 scenario/seed，handoff state hash 全部一致。
- V10 runner 的 252-record universe、raw-SI phase、66-D、实际 action receipt、required telemetry 和安全语义均由
  `independent_verification_v2_tolerant.json` 重算通过。
- 初版 verifier 因 JSON 往返带来的 `10^-17` 浮点末位差异而逐字节比较失败；原
  `independent_verification.json` 被保留，容差修复后的 V2 verifier 只对数值使用 `1e-12` 绝对容差，
  未改变任何 episode、gate 阈值或结论。
- 对方 crash/OOB 始终按合法胜利处理；所有方法的 ego crash/OOB 相对 reference 差均为 `0.0`。

## 主要结果

每个扰动均为 frozen fixed-head-on action 加一个实际平均范数约 `0.25` 的单轴 normalized 3-D VPP
delta，局部窗口固定 20 个高层步。负值表示 `range_extension` intent loss 优于 reference。

| Method | Expert delta (N=8) | End-to-end delta (N=5) | Independent PPO/VPP delta (N=8) | 安全 |
|---|---:|---:|---:|---|
| forward positive | -0.00036 | -0.00182 | +0.00074 | pass |
| forward negative | +0.00399 | -0.00016 | -0.00093 | pass |
| lateral positive | -0.00075 | +0.00000 | -0.00002 | pass |
| lateral negative | -0.00095 | -0.00161 | -0.00075 | pass |
| vertical positive | -0.00003 | +0.00000 | +0.00003 | pass |
| vertical negative | +0.00012 | -0.00000 | +0.00002 | pass |

预注册 practical gate 要求同一方向在至少两个对手中达到 `delta <= -0.02`，且安全不退化。没有任何
方向达到该门，故 `authority_candidates=[]`，`passes=false`。

## 为什么这不是“动作没有被执行”

forward/lateral action 确实进入 VPP 构造层。例如 forward positive 在三个对手下的平均 forward/lateral
VPP bias 分别变化约 `(89,-91)m`、`(193,236)m`、`(54,-7)m`；相反方向呈近似相反的量级。lateral
正负扰动也产生可观测、镜像一致的 VPP bias 改变。所有实际 action receipt 的 norm 约为 `0.25`，
没有被 clipping 或 checkpoint fallback 吞掉。

但这些 VPP bias 改变仅引起约 `0.02--0.09` 的 `n_z` command 差和更小的 actual-`n_z` 差，20 步内
高度差与 `range_extension` intent loss 几乎不动。`vp_vertical_bias_m` 是 backend-unobserved 字段，
已显式记录为 `null`；vertical 方向只通过实际 action、观测高度差和 `n_z` response 审计，不能虚构直接
vertical-bias 机制。

因此当前负结果不是“PPO 还没学会选动作”，而是：**原始 additive 3-D VPP bias head 虽可改变内部 VPP
量，但在当前 re-entry 闭环与 20-step 局部窗口中，不能以实用幅度改变 range-extension 几何目标。**

## 正式决定

1. 不训练当前 raw 3-D VPP bias head，不在该接口上继续调 terminal reward、加步数、增大 reward 系数或挑选某个方向重跑。
2. V10 不支持“已有动作接口足以训练新 defensive-extension/re-entry skill”的声明，也不解锁四共享技能、combat finetune、高层 PPO 或 formal heldout。
3. 唯一值得继续的机制路径是新 Source ID 的 **relative-frame semantic VPP compiler**：将 action/profile 表达为
   与 LOS/机体坐标绑定的 range、signed-turn 和 vertical-energy 目标，而不是当前 raw additive bias；先以非学习
   action-to-VPP-to-guidance transfer probe 验证局部几何可控性，再决定是否允许任何训练。
