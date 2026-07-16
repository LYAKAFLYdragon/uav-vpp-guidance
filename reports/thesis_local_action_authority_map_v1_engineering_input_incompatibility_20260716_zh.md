# Local Action Authority Map V1 工程输入不兼容归档

V1 的 28-state 设计没有进入正式脉冲矩阵。6 条 engineering smoke 显示：V2 的三个 opponent
均可从合法 reset 连续重放至历史 handoff hash 并完成 20-step telemetry；V15 的三个对应重放则
不匹配，不能按预注册规则注入动作。

这不是关于局部控制方向的正面或负面结论。V15 原运行提交为 `ff36d60a`，当前环境的
`tracking_env.py` 已与该提交存在行为差异，历史 hash 因此不可重现。V1 按其 stop rule 冻结为
`engineering_input_incompatible_before_pulse`，不覆盖 V15 artifact，也不把 V15 state 混入新因果
分析。

随后新建 V2-only Source，只保留通过同一工程 smoke 的 12 个 V2 state，并以新的 Source ID、
新 cohort 和新输出根独立执行。该收缩由可重现性条件决定，不由局部响应、胜率或任何结果挑选决定。
