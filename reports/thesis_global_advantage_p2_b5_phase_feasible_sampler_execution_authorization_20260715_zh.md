# P2-B5 Phase-Feasible Sampler 一次性执行授权

**Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P2-PHASE-FEASIBLE-SAMPLER-B5-R1`
**授权状态：** `open_once_not_yet_executed`
**冻结实现 commit：** `4f250ed2ea7cef394e500c080f48da598d052064`

## 唯一允许命令

```powershell
python scripts/run_thesis_global_advantage_p2_b5_phase_feasible_sampler.py --execute
```

该命令只运行 12 个独立 `disadvantage` 连续 run-in 场景与三个已冻结 opponent，共 36 条 strict-JSBSim record。每条 record 使用同一个 frozen head-on specialist，从单次合法 reset 连续运行至 first-pass，随后仅记录物理 k 到 k+1 handoff；collector 不替换 VPP action。

## 运行前不可变条件

- 工作树必须 clean，且 `4f250ed` 必须是当前提交的祖先。
- B5 config 中列出的 13 个运行相关源文件必须逐一匹配 SHA-256；manifest、runtime config、runtime registry、P3 encoder 和 head-on specialist 也必须匹配已冻结哈希。
- 输出根 `E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p2_b5_phase_feasible_sampler_r1` 必须不存在；E 盘可用空间必须不少于 120 GB。
- phase tracker 只接收 `observation.relative_state` 的 raw-SI `range_m/range_rate_mps`；归一化 policy observation 只写入诊断字段。
- 动态 taxonomy 只按 `(ATA, AA)` 计算；`ATA=160°、AA=20°` 的回归案例必须标为 `disadvantage`。

## 预注册 Gate 与停止规则

每个 opponent 分别要求至少 2 条 qualifying episode、20 个有效 target step、2 个不同场景 signature、两个镜像方向，以及 strict JSBSim、无 fallback/reset/padding、raw-SI phase replay、连续 handoff 和有限 66-D/3-D action。不得池化 opponent 结果。

无论结果正负，执行后立即关闭 `execution_permitted`。若任意 opponent 未通过，B5 固定为负证据；不得用调参、增加训练步数、更换场景、snapshot restore 或重跑同一 Source ID 修补。即使三 opponent 均通过，也只允许起草单一 defensive-extension pilot 的执行授权，不能直接训练或启动 formal held-out。
