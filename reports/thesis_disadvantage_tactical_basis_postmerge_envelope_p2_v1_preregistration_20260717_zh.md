# Post-Merge Tactical-Basis P2 V1 预注册

**Source ID：** `THESIS-DISADVANTAGE-TACTICAL-BASIS-POSTMERGE-ENVELOPE-P2-V1`
**性质：** 非学习、离线 compiler-envelope screen，`paper_safe=false`

## 唯一问题

在 V3 已验证的同语义连续 post-merge baseline window 中，冻结 P1 temporal-reference templates 是否能在不扩展 3-D action envelope 的前提下，编译为足够大且无 box/slew 失真的参考轨迹？

## 冻结输入与范围

- 只读 V3 的 17 条逐记录合格 window：expert 5 条、end-to-end 5 条、independent PPO/VPP 7 条。
- 只使用 P1 的 8 个固定 template/sign variants，8 秒、40 步、`sin_squared_c1`、`max_action_delta=0.35`、`max_action_slew=0.10`。
- `expert` 仅作诊断：其 5 条记录仅覆盖 3 个 signature，不能参与 P2 pass gate。
- 不执行候选 action，不运行 JSBSim，不训练，不重采样，不修改模板、奖励、路由、VPP、guidance 或 PID。

## 通过门

同一个 variant 必须在 `end_to_end` 和 `independent_ppo_vpp` 各自满足：至少 4 条 window、4 个 signature、双镜像、40 步完整；box clipping = 0、slew clipping = 0、平均投影残差 `<= 50 m`、峰值实际参考位移 `>= 100 m`、basis condition `<= 100`。

多个 variant 通过时，按“最大化两对手中的最小可行样本数，再最大化最小峰值位移，再最小化最大平均残差，最后按 variant ID”唯一选择一个。若无 variant 通过，则停止 temporal-reference/shared-skill training line；若通过，只允许设计一次 selected-variant paired runtime probe。
