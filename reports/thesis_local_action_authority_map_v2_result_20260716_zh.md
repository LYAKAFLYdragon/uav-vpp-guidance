# Local Action Authority Map V2 结果与二分决策

**Source ID：** `THESIS-FIVE-STATE-LOCAL-ACTION-AUTHORITY-MAP-V2`  
**结果：** 有效的跨对手局部控制权威负证据

## 完整性

V2 仅使用通过 engineering replay 的 V2 historical cohort：expert 5、end-to-end 2、
independent PPO/VPP 5，共 12 个真实 handoff。完整矩阵为 reference 与三轴 `+/-0.08`、
三步 pulse、二十步观察窗，共 `84/84` strict-JSBSim 连续 replay。

独立 verifier 重算 gate 并与 runner 输出完全一致。所有 84 条 record 都满足 replay hash match、
window complete；pulse clipping、backend/prediction fallback 和 ego failure 均为零。

## 结果

没有 `local_authority_candidate` 通过预注册的“至少两个 opponent 同方向、幅度至少 0.01、
符号一致且安全”的门槛。最大单对手响应是 end-to-end 下 longitudinal x range-rate 的
`0.02921`（2/2 同号）；但 expert 的对应响应为 `-0.00321`，independent PPO/VPP 未复现该方向。
其余最大响应均低于 `0.00531`。

因此本实验区分了两个事实：

1. 当前 3-D VPP head 能安全地承受小幅对称扰动，且局部响应对 end-to-end 条件可能敏感。
2. 没有一个轴-指标方向在 expert 与 independent PPO/VPP 中同方向、足够幅度地复现，故尚不能
   作为 state-conditioned compiler 的可迁移依据。

## 正式决策

不设计、不训练 state-conditioned compiler；不调整 pulse 幅度、脉冲步数、窗口、cohort 或 gate 后
重跑 V2；停止 temporal-corridor/共享技能训练线。硕士论文应将此写为严谨的证据链：可达 handoff
状态可以连续重现，动作小扰动物理安全，但当前接口尚未展示跨对手一致的局部控制权威。

该结论不否定现有双技能层级 VPP 的论文主线，也不声称所有未来接口无效；它只排除了当前冻结
action envelope 上的短时、小幅、状态无关 pulse 作为下一步训练基础。
