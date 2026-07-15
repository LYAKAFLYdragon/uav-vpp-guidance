# V16 Dynamic Reference-Path 设计预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-DYNAMIC-REFERENCE-PATH-V16`
**状态：** design-only；未授权执行或训练

V16 是对 V15 的定性接口变更：不再按固定时间把同一单点 VPP 切换 profile，而是从当前 ego/target 几何与冻结预测器的单个预测位置构造三个可枚举 anchor：`escape`、`connector`、`predicted_reentry`。active anchor 仅由观测到的 range-rate 进度选择，不使用 future simulator state、synthetic target 或改变终止/胜负语义。

第一轮必须是非学习 path-transfer probe。它需要在三个对手下逐步记录三 anchor、prediction validity、range-rate progress、所选 anchor、由此得到的 VPP/guidance/PID receipt，以及与 frozen fixed-head-on 相同 handoff hash 的配对 physical-intent 指标。训练、reward、专家数量和 VPP/guidance/PID 参数均保持冻结。

只有当该 path interface 在至少两个对手下先证明：无 fallback、anchor/VPP/guidance 三层存在非零且受限的因果差异、且没有新增 ego crash/OOB，才允许起草独立训练 pilot。该预注册不代表 V16 已建立性能优势。

## 执行前冻结的 Probe 宇宙

V16 runner 必须生成一个新的 12 场景 manifest：`head_on` 与 `crossing_entry`、三种高度差、两种镜像方向；物理包固定为初始距离 `7950 m`、我方/目标速度 `328/360 m/s`、seed 基数 `198100`。builder 必须对 V1--V15、Taxonomy30、Dev30 和 Heldout60 做 physical-signature 与 seed 双重不相交检查。

固定方法矩阵为 `frozen_fixed_head_on`、`static_range_extension` 与 `progress_triggered_reference_path`，共 `3 x 3 opponents x 12 scenes = 108` 条 strict-JSBSim record。后者只允许将当前 target、冻结 predictor 的单个预测位置和当前 range-rate 输入 V16 path core；每一步写入三 anchor、progress、selected index/profile、semantic receipt、实际 VPP offset、guidance command、actual `n_z`、saturation、prediction validity/fallback 和 terminal reason。

与 fixed-head-on 的相同 handoff hash 配对后，V16 至少要求两个对手分别具备 4 个完整 pair、非零 anchor/VPP/guidance receipt、ego crash/OOB delta `<= +0.05`，且不得恶化 escape range-rate error 超过 `+0.02`。这些是接口可达性门，不是训练或全局优势门；任何缺失、fallback、reset 或不完整 record 都 fail-closed。
