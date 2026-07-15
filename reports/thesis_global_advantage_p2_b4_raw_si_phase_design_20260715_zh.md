# P2-B4 Raw-SI Phase Preflight 设计预注册

**拟议 Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-RAW-SI-PHASE-PREFLIGHT-V1`
**状态：** `implementation_frozen_execution_not_authorised`
**前置证据：** `P2-B3 normalized_phase_input_contract_no_go`

## 修复对象

P2-B3 已严格证明 scenario 已正确进入 JSBSim，且 tracker replay 本身一致；唯一失配是 phase tracker 读入了归一化 policy observation 的 `range_m/range_rate_mps`。P2-B4 不修改 B3，也不修改 policy observation contract，而是建立独立 runner：

```text
observation.relative_state (raw SI range_m, range_rate_mps)
    -> PhaseTracker thresholds in metres and metres/second
policy observation vector (normalised)
    -> diagnostic-only persistence; never used by PhaseTracker
```

## 冻结协议

- 新 manifest：30 条 `5 states x 3 heights x 2 mirrors`，4,000 m / 275--255 m/s 的新 package 与 14,001--14,030 的新 seeds；与 P2-B2、P2-B3、heldout240、dev30 和历史 heldout60 的物理签名不相交。
- 三个 opponent 分开运行，frozen `run_in_head_on` specialist、fresh JSBSim environment、无训练/调参/候选策略加载。
- 逐条保存 scenario application receipt、raw-SI phase input、normalized policy diagnostic、PhaseTracker 内部状态前后值、replay、first-pass、terminal 与 fallback。
- `range_m_scale=5,000`、`range_rate_mps_scale=200` 只用于确认 normalized diagnostic 与 raw-SI input 的数值一致性，绝不参与 phase threshold。

## Gate 与决策

| Gate | 通过条件 | 失败后的唯一解释方向 |
|---|---|---|
| Structural | 90/90 strict JSBSim、有限 action、无 fallback、至少 20 steps | backend/control contract |
| Scenario receipt | 90/90 requested 与 raw backend state 一致 | reset/sampler contract |
| Unit separation | 90/90 raw-SI phase input 与 normalized diagnostic 的固定比例一致 | observation unit contract |
| Phase replay | 90/90 persisted receipt 可独立回放 | phase instrumentation |
| Step-0 semantics | 90/90 manifest `pre_merge` 与 raw-SI tracker `pre_merge` 一致 | manifest-to-runtime semantics |

`re_entry` 的零覆盖仍只是一项 readout，不是本轮失败。B4 通过只说明 phase 观测合同成立；随后允许设计 phase-feasible sampler，仍不自动解锁共享技能训练、PPO 或 formal held-out。
