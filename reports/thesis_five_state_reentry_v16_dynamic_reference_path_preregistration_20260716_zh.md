# V16 Dynamic Reference-Path 设计预注册

**Source ID：** `THESIS-FIVE-STATE-REENTRY-DYNAMIC-REFERENCE-PATH-V16`
**状态：** design-only；未授权执行或训练

V16 是对 V15 的定性接口变更：不再按固定时间把同一单点 VPP 切换 profile，而是从当前 ego/target 几何与冻结预测器的单个预测位置构造三个可枚举 anchor：`escape`、`connector`、`predicted_reentry`。active anchor 仅由观测到的 range-rate 进度选择，不使用 future simulator state、synthetic target 或改变终止/胜负语义。

第一轮必须是非学习 path-transfer probe。它需要在三个对手下逐步记录三 anchor、prediction validity、range-rate progress、所选 anchor、由此得到的 VPP/guidance/PID receipt，以及与 frozen fixed-head-on 相同 handoff hash 的配对 physical-intent 指标。训练、reward、专家数量和 VPP/guidance/PID 参数均保持冻结。

只有当该 path interface 在至少两个对手下先证明：无 fallback、anchor/VPP/guidance 三层存在非零且受限的因果差异、且没有新增 ego crash/OOB，才允许起草独立训练 pilot。该预注册不代表 V16 已建立性能优势。
