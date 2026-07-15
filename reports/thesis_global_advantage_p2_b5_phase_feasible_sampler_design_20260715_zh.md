# P2-B5 Phase-Feasible Sampler 设计预注册

**状态：** `design_only_not_authorised`
**首要 motif：** `disadvantage -> post_merge/re_entry -> defensive extension`

P2-B4 已消除 phase 单位错误，但显示该 motif 在 independent PPO/VPP 下没有自然 re-entry 样本。B5 不训练任何技能；它只验证连续 run-in 后的合法 handoff 能否在三个 opponent 下提供足够的真实 phase 样本。

## 方法

- 新 Source ID、新 manifest 和新 seed；不复用 B2--B4 的场景实例或 output。
- 每条 episode 只 reset 一次，由 frozen head-on reference 连续运行；达到合法 first-pass 与 raw-SI dynamic disadvantage 条件后，不 reset、不 snapshot restore、不 future-state injection 地记录 handoff。
- 持久化 reset/step-0/handoff raw-SI range、range-rate、dynamic taxonomy、phase receipt、10-frame real history、预测器状态、3-D VPP action、VPP/guidance/PID response 与 terminal。
- 三个 opponent 分开统计。没有 candidate skill、reward、PPO 或 profile 训练。

## 预注册 gate

每个 opponent 独立要求：至少 2 条 qualifying episode、20 个连续有效 target motif steps、两个 mirror sign、两个场景 signature；全部记录必须 strict JSBSim、无 fallback、无 reset/padding、phase 可回放且 66-D/3-D 均有限。

三 opponent 都通过，才允许起草单一 defensive-extension pilot 的执行授权。任一 opponent 持续为零时，结论是当前包线尚未为该 motif 建立跨 opponent 的训练数据契约；不得直接启动四共享技能训练。
