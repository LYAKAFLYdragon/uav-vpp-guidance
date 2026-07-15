# Neutral Post-Merge Reentry-Recovery Pilot V2 预注册

**Source ID：** `THESIS-NEUTRAL-POSTMERGE-REENTRY-RECOVERY-PILOT-V2`
**状态：** `preregistered_design_only_not_authorised`

## V1 的关系

V1 在首个 dev baseline 的 ledger 序列化前因缺失 `metadata.scenario_signature` 终止，没有任何 checkpoint、逐步 telemetry 或性能结果。因此 V2 不把 V1 作为正、负或 safety 证据；它只吸收 V1 的实现归因：配对身份不能从 runner 假设，必须由 manifest 显式提供。

## V2 的冻结改变

- 新 `dev12` 和一次性 `heldout24` 使用全新 distance/speed packages 和 seeds，并与 V1、B2--B6、`heldout240`、`dev30`、历史 `heldout60`、phase-v2 逐一零物理初态/seed 交集。
- 每个 scenario 必须含 `geometry_cell_id`、整数 `scenario_seed` 和序列化的 `pair_key = geometry_cell_id::seed=<seed>`。
- 每种方法的 paired delta 必须先验证相同 `pair_key` 与相同 handoff state hash；缺失或不一致是 contract failure，不是性能数值。
- P3 encoder、66-D/3-D VPP、`reentry_recovery + reentry_preparation`、routing disabled、三对手分开报告、50,000-step 上限、dev-only selection 和 heldout-only decision 均维持既定边界。

## 当前证据与限制

V2 design preflight 已确认 12 个 dev pair keys、24 个 heldout pair keys、输出根不存在和 `291.182 GB` 可用空间。它不运行 JSBSim、不训练、不创建 checkpoint，也不支持任何新技能有效或全五态势优越性的结论。

下一步必须先把 V2 wrapper、真实 manifest pairing test 和本配置冻结为 clean SHA，再由用户单独授权一次性执行。V2 不继承 V1 的授权。
