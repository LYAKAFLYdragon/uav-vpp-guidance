# Global-Advantage P1 R3 独立环境复核预注册设计

**状态：** `design_only_not_authorised_to_run`

**拟定 Source ID：** `THESIS-GLOBAL-ADVANTAGE-V1-P1-R3-FRESH-ENV-REPRO-V1`

**唯一目的：** 判定 P1 R2 的轨迹分歧是否来自复用环境实例，或来自仍未观测到的进程级/控制器级初始状态。R3 不训练、不调参、不比较候选方法，也不产生任何性能主张。

## 1. 冻结输入与不变项

| 项目 | R3 规定 |
|---|---|
| 场景 | 冻结复用 R2 的 dev30；不接触 GLOBAL-ADVANTAGE-V1 held-out240 |
| 对手 | `expert`、`end_to_end`、`independent_ppo_vpp`，分别审计 |
| 控制 | 仅 `frozen_run_in_head_on_specialist`；无高层 routing、无 candidate policy |
| 重复 | 每个 opponent x scenario 运行 3 次，共 270 episode |
| 后端 | `jsbsim`、`strict_backend=true`，任何 fallback 立即失败 |
| 参数 | 继承 R2 runtime config、registry、P3 checkpoint 和 scenario manifest 的 SHA；任一输入 SHA 改变即不得开跑 |
| 禁止项 | 禁止训练、奖励/VPP/guidance/PID 修改、snapshot restore、future-state injection、删去不一致 cell 或更换 seed |

R2 是已冻结负证据，而不是可覆盖的旧输出。R3 必须使用新的空 output root：

`E:/uav-vpp-guidance-five-state-heldout-envelope-v1-results/global_advantage_v1_p1_r3_fresh_environment`

## 2. 关键协议：每回合独立的环境和进程

1. 父进程只读取预冻结的场景、资产与 launch plan；它不创建可复用的 `CloseRangeTrackingEnv`。
2. 每一个 `opponent x scenario x repeat` 由一个新 Python 子进程执行。子进程只构造一个 environment，运行一个回合并退出。
3. 同一 cell 的三个 repeat 仍使用 frozen 的相同 scenario/seed；`forward`、`reverse`、`mirror_interleaved` 仅改变父进程的启动顺序，不能改变子进程的输入。
4. 子进程不得读取上一回合输出，不得复用 environment、predictor、guidance/PID、opponent 或随机数生成器对象。
5. 每个 episode 完成后以临时文件写入、fsync 并原子改名；父进程仅聚合已验证 SHA-256 的文件。

这一定义同时检验环境复用泄漏与顺序相关的进程级隐状态。如果 R3 仍不一致，结论是“当前模拟/评估栈尚未建立同起点的可重复契约”，而不是“需要更多训练”。

## 3. 必须持久化的 reset 与逐步证据

每个 raw episode 必须以 JSON-safe、键排序、量化精度固定的形式写入以下内容；任意不可序列化、缺字段、NaN/Inf 或 padding 都是 contract failure。

| 时点 | 必须字段 |
|---|---|
| reset 前 | scenario signature、seed、完整 resolved config SHA、checkpoint SHA、运行时 registry SHA、Python/JSBSim 版本与子进程 command |
| reset 后 | 完整 observation vector 与 feature names、16-D base projection、history buffer、predictor state、VPP/guidance/PID/controller state、opponent state、完整 FDM state snapshot、各对象的独立 SHA-256 |
| 每个控制步 | observation/history/predictor/controller/FDM state hash、reference action、opponent action、normalized 3-D VPP action、VPP bias、guidance command、`n_z` command/actual、AoA、saturation、phase、dynamic taxonomy、terminal reason |
| 回合末 | 全部 step hash 序列、first-pass 或 terminal boundary snapshot、backend/prediction/checkpoint fallback 标记、raw artifact SHA-256 |

“完整 FDM state”不是只保存 position/speed 的摘要。实现前必须在代码中冻结一个明确 property whitelist 或可复现的 simulator-state exporter，并以单元测试证明每个声明字段都进入 canonical hash。

## 4. 预注册判定与停止规则

对每个 `opponent x scenario` cell 比较三个 repeat：

| 条件 | R3 通过要求 |
|---|---|
| 输入等价 | 完整 reset observation、history、predictor、controller、opponent 与 FDM state hash 全部一致 |
| 行为等价 | reference/opponent/VPP action hash、逐步 trajectory hash、first-pass/terminal boundary hash 与 terminal reason 全部一致 |
| 完整性 | 270/270 raw artifacts 完整，且无 backend、prediction、checkpoint 或 telemetry fallback |
| 总门槛 | 90/90 cells 同时满足以上三项 |

**Fail-closed：** 任意一个 cell 不一致、缺失字段、子进程异常或 hash 错误，R3 为 `runin_protocol_not_reproducible_do_not_train`。不得通过降低量化精度、只比较 terminal、删除 mismatch、重排场景、重跑单个 cell 或修改系统参数来获得通过。

## 5. 实现前 checklist

- [x] 新建 R3 config，显式写入所有 R2 输入 SHA、`fresh_process_per_episode=true` 与全新 output root；默认 `execution_permitted=false`。
- [x] 实现 `capture_runtime_envelope()`；不能完整导出 controller/FDM state 时必须抛错，而不是返回部分摘要。
- [x] 实现 child runner；其输入只允许一个 frozen launch record，且对子进程环境禁止 fallback。
- [x] 实现 canonical serializer；对 `numpy.ndarray`、numpy scalar、嵌套 mapping 和非有限数有单元测试。
- [x] 实现 parent aggregator；它先验证 raw artifact SHA 再生成 90 行 comparison table。
- [x] 为完整 observation、history、predictor、controller、opponent、FDM、action 与 boundary 写入字段敏感性或 fail-closed 测试。
- [x] 为 child-process 隔离写入 gate：同一 scenario 的三个 PID 必须不同，否则整个 cell 失败。
- [ ] 对 R3 实现做独立代码复核后，才可申请一次性执行授权。

## 6. R3 后的唯一分支

| R3 结果 | 允许的下一步 | 仍然禁止 |
|---|---|---|
| 90/90 通过 | 冻结 P1，申请 P2 场景/对手 capability baseline 授权 | 直接进入四技能训练、P5 或 formal held-out |
| 任一 cell 失败 | 依据完整 state hash 定位 evaluator/simulator 状态边界，并作为复现性负证据归档 | 以训练、增大样本或更换候选方法掩盖失败 |

**R2 锚点：** `run_manifest.json` SHA-256 为 `a9bfa248cc0d2d9118b470eccac7efc3b669a4f7f00bed0f105534ce25a639ca`；R2 config SHA-256 为 `bbbb4d62775006a96eafa19403333dde83e5ee9db058560686ebb526bd182558`。
