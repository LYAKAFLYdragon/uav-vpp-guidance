# V19 Windowed Phase-Telemetry Audit 预注册

V19 是 V18 观测合同失败后的**全新**非学习 Source ID。它复用 V17 场景只为逐格诊断，但不会重开
V17 或 V18 的任何 gate，也不会训练或改变 controller。

关键修正是分开两个时间窗口：预测器 warm-up 中允许 `prediction_valid=false`，但仍要求
`prediction_fallback=false`；只有 history 达到 10 frame 且预测有效后，才要求完整的几何、能量、
taxonomy、VPP、guidance、PID telemetry。所有三对手分别要求 24 条 record 并精确复现 V17 handoff。

V19 通过只能说明 V17 end-to-end coverage 缺口已具备正确的可解释遥测，不能证明路径上下文、奖励、
新技能或五态势系统有效。V19 失败则冻结为连续数据合同仍未建立的证据，不得训练。
