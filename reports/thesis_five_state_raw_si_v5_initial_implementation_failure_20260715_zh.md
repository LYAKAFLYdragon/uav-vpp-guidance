# V5 初次执行实现失败记录（非实验结果）

## 事实

- Source ID：`THESIS-FIVE-STATE-RAW-SI-PHASE-OBSERVABILITY-V5`
- 初次授权输出：`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/five_state_raw_si_v5_observability`
- 已完成 record：`0 / 180`
- 失败位置：第一条 JSBSim episode 的 telemetry `step_sha256` 写入前。
- 直接原因：`guidance_command` 含 `numpy.ndarray`，初版哈希函数未进行 canonical 转换，抛出 `TypeError`。

## 解释边界

本次失败发生在任何 record 写入前。它不能用于判断物理可达性、对手差异、五态势覆盖、P3 编码器、VPP、制导律、PID 或固定 head-on specialist 的性能。原输出目录和 `runner_failure_manifest.json` 保留，不覆盖、不删除、不作为实验负证据。

## R1 处置

R1 仅将 ndarray、numpy scalar 与非有限 telemetry 值转换为确定性哈希表示；动作、环境、JSBSim、场景 manifest、三对手、P3 encoder、66-D contract、VPP、制导律、PID 及全部 gate 均不改变。R1 使用新的 output root，并继续禁止训练、调参和重跑后选择。
