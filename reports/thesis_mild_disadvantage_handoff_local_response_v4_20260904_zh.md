# 局部动作响应采集 V4 研发记录

## 1. 目的

本次工作只验证一个问题：在真实首次交会后的交接状态上，三个 VPP 动作轴的小幅正、负变化，是否能够在严格 JSBSim 中产生可重复的局部响应。该工作不训练策略，不调整奖励、PID、滤波器或 JSBSim 参数，也不比较胜率。

## 2. 失败原因修正

此前 V2、V3 使用的源轨迹虽然编号不同，但重复回合实际在同一运行链中生成，导致部分重复编号不能由相同场景和随机种子稳定重放。因此，V2、V3 的结果不能用于正式局部响应结论。

V4 先建立独立源轨迹路线。每个“对手—场景—重复”均由新的 Python 进程运行，并由另一个新的进程使用相同场景和种子重放。只有 handoff 步号和状态哈希一致，源轨迹才被登记。

## 3. 可复现信息

| 项目 | 内容 |
|---|---|
| 源轨迹 Source ID | `THESIS-MILD-DISADVANTAGE-HANDOFF-SOURCE-TRACES-ISOLATED-V1-20260904` |
| 局部响应 Source ID | `THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V4-20260904` |
| 对手 | `expert`、`end_to_end` |
| 场景 | `x625_y200`、`x750_y250` |
| 每个组合重复数 | 3 |
| 源轨迹数 | 12 |
| 局部条件数 | 84（每条源轨迹包括基线和三轴正负脉冲） |
| 脉冲幅值 | 归一化动作单位 `±0.08` |
| 响应时刻 | handoff 后 4、12、48 步 |
| 后端 | JSBSim 1.2.3 |

源轨迹目录：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_handoff_source_traces_isolated_v1_retry3`

局部响应目录：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_handoff_local_response_v4`

离线分析目录：

`E:/uav-vpp-guidance/outputs/mild_disadvantage_handoff_local_response_v4/analysis_final`

## 4. 门禁结果

源轨迹重放 verifier：12/12 通过。局部响应 verifier：84/84 条件通过，252/252 个响应记录有效。所有条件均为严格 JSBSim，未发生 backend fallback；每回合仅有一次初始 reset；4、12、48 步响应窗口均完整。

正负动作契约审计：84/84 通过。正脉冲和负脉冲只改变指定动作轴，记录差值分别为 `+0.08` 和 `-0.08`，没有继续使用此前错误的同号负脉冲实现。

## 5. 主要结果

离线分析形成 108 条状态级局部效应、36 条对手—场景—动作轴—时刻汇总和 18 条跨对手比较。

- 在 4 步窗口，距离、距离率、攻击区角、进入角、比能和高度六项指标在两个主对手下均表现出一致方向，所有 6 个动作轴—场景组合都满足重复方向一致性。
- 在 12 步窗口，距离、距离率、攻击区角、进入角和高度仍保持跨对手同向；比能的方向出现分化，说明能量变化不宜与几何变化简单等同。
- 在 48 步窗口，动作轴之间的方向差异增大。轴 0 的几何响应保持最好，轴 1 和轴 2 的部分指标出现对手或场景相关变化。这表明短时局部控制响应可以跨对手复现，但长时域结果已经受到闭环演化影响。
- 36 条汇总记录均通过安全记录检查，未发现新增坠毁、越界、fallback 或动作裁剪问题。

## 6. 研究意义

本次工作首先修复了局部响应实验的可重复性问题，使“源 handoff 是否真实可重放”与“脉冲是否正确施加”得到分开验证。结果表明，当前控制接口并非完全没有局部几何控制权：在 handoff 后短时域，三个动作轴都能在两个主对手下产生可重复的局部响应。

因此，下一步可以进入**离线状态条件化动作序列筛选**，优先使用 4 步和 12 步的几何响应估计，避开直接把 48 步结果当作固定规律。当前证据还不足以直接声称新技能已经带来系统级收益，也不足以直接启动共享技能库训练；若继续研发，应先用这张局部响应图设计包线内的短时动作序列，再进行一次独立的非学习运行探针。

## 7. 工件清单

- 源轨迹配置：[mild_disadvantage_handoff_source_traces_isolated_v1.yaml](../config/experiment/mild_disadvantage_handoff_source_traces_isolated_v1.yaml)
- 源轨迹协调器：[run_mild_disadvantage_handoff_source_traces_isolated_v1.py](../scripts/run_mild_disadvantage_handoff_source_traces_isolated_v1.py)
- 单回合 worker：[run_mild_disadvantage_handoff_source_trace_worker_v1.py](../scripts/run_mild_disadvantage_handoff_source_trace_worker_v1.py)
- 源轨迹 verifier：[verify_mild_disadvantage_handoff_source_traces_isolated_v1.py](../scripts/verify_mild_disadvantage_handoff_source_traces_isolated_v1.py)
- 局部响应配置：[mild_disadvantage_handoff_local_response_collector_v4.yaml](../config/experiment/mild_disadvantage_handoff_local_response_collector_v4.yaml)
- 局部响应 verifier：[verify_mild_disadvantage_handoff_local_response_collector_v1.py](../scripts/verify_mild_disadvantage_handoff_local_response_collector_v1.py)
- 局部响应分析器：[analyze_mild_disadvantage_handoff_local_response_v1.py](../scripts/analyze_mild_disadvantage_handoff_local_response_v1.py)
- 源轨迹重放结果：[replay_verification.json](../outputs/mild_disadvantage_handoff_source_traces_isolated_v1_retry3/replay_verification.json)
- V4 校验结果：[verifier_result.json](../outputs/mild_disadvantage_handoff_local_response_v4/verifier_result.json)
- V4 分析报告：[local_response_analysis_report.json](../outputs/mild_disadvantage_handoff_local_response_v4/analysis_final/local_response_analysis_report.json)
- V4 敏感度图：[local_response_sensitivity_heatmap.svg](../outputs/mild_disadvantage_handoff_local_response_v4/analysis_final/local_response_sensitivity_heatmap.svg)

## 8. 当前边界

本次结果是局部响应和接口可达性证据，不是胜率证据，也不是完整技能性能证据。当前工作树仍为 dirty 状态，因此本目录应先作为研发证据保存；如需写入论文正式结果，还应在 clean SHA 下重新固定配置、源轨迹包和分析输出。
