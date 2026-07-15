# V21 Warm-up Anchor Protocol 预注册

V21 是新的非学习接口验证，不与 V17--V20 的结果合并。

协议固定为：reset 后以 `current_target` 构造 VPP；仅当 history 连续满 10 帧、预测 receipt 为
`prediction_valid=true` 且 `prediction_fallback=false` 时，一次性切换到 `predicted_target`。切换
后不允许回退；任何 prediction invalid、fallback、backend fallback、第二次切换或未记录切换原因
都会冻结该 Source ID。

环境必须通过显式 `set_runtime_vpp_anchor_mode(mode, reason)` 实现切换，在每步 info/provenance 记录
effective mode、switch count、switch step、reason 和 prediction receipt。直接修改 generator 内部属性
禁止。三对手各自报告，不 pooled；先仅做连续 phase/telemetry contract，不训练、不改变 VPP action、
guidance、PID、policy 或 reward。
