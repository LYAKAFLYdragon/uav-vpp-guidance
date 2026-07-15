# Neutral Post-Merge Pilot V1 执行前物理契约修正

**状态：** `pre-execution correction; no JSBSim pilot episode executed`

执行 runner 的最小 sampler test 发现原连续训练分布在 `range=2850 m`、`|altitude difference|=600 m` 的合法边界处，`172 deg` 的目标姿态角无法映射为有效三维航向。这是几何可实现性错误，不是训练失败或性能结果。

修正仅发生在任何 pilot episode、checkpoint 或 output root 创建之前：

- 将 own/target LOS angle support 的上界由 `172 deg` 收紧为 `165 deg`，仍严格属于 Neutral 象限，且在冻结高度/距离 support 的最坏几何下可实现。
- 将已写入预注册文字、但遗漏于 YAML 的 `remaining_opponent_noninferiority_vs_best_existing_specialist <= 0.02` 门槛补入 config。
- 训练 sampler 增加显式三维几何可行性检查；若未来 config 越过可行边界，将 fail-closed 而不是隐式裁剪角度。

因此先前以 `bc356bc` 为对象的独立设计复核只对修正前配置成立，不能作为执行授权的最终复核。修正后的 config 已重新通过设计 preflight 与 `9 passed` 合同测试；待实现提交成为新的 clean SHA 后，必须据此生成并复核一次性 authorization config。
