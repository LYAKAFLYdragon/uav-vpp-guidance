# 劣势态势真实交会局部响应采集记录

## 1. 实验目的

本采集用于建立真实首次交会后 handoff 状态的局部动作响应数据。对每个状态保留基线动作，并分别加入三个 VPP 动作轴的正、负对称小脉冲，观察第 4、12 和 48 步的响应。该工作只验证数据链和局部响应，不训练新策略、不修改奖励函数、不调整 PID 或 JSBSim 参数。

## 2. 固定条件

- Source ID：`THESIS-MILD-DISADVANTAGE-HANDOFF-LOCAL-RESPONSE-COLLECTOR-V1-20260903`
- 对手：`expert`、`end_to_end`
- 场景：`x625_y200`、`x750_y250`
- 每个对手和场景重复 3 条源轨迹，共 12 条真实 handoff 轨迹
- 每条源轨迹包含 `baseline`、`axis0_plus/minus`、`axis1_plus/minus`、`axis2_plus/minus`，共 7 个条件
- 脉冲幅值：三个动作轴均为 `±0.08`
- 响应窗口：4、12、48 步
- 运行后端：严格 JSBSim

## 3. 执行修正

最初的局部响应批次在同一 Python 进程中连续 reset。虽然进程返回成功，但历史 handoff hash 只有 `2/12` 可重放，另外 10 条受到连续 reset 后 JSBSim/FCS 微小漂移影响，因此整批不接收为配对证据。

修正后新增单回合入口：

```text
scripts/run_mild_disadvantage_envelope_recovery_single_episode_v1.py
```

每个“对手 × 场景 × 重复编号”均在独立进程中生成源轨迹；随后每个“源轨迹 × 局部动作条件”也在独立进程中运行。策略、奖励、动作空间、制导、PID 和 JSBSim 模型保持不变。该执行粒度只用于消除 reset 漂移，不改变实验方法本身。

## 4. 结果与门禁

正式结果目录：

```text
outputs/mild_disadvantage_handoff_local_response_v1/
```

结果统计如下：

| 项目 | 数量或结果 |
|---|---:|
| 真实源 handoff | 12 |
| 局部动作条件 | 84 |
| 响应窗口记录 | 252 |
| handoff hash 一致 | 84/84 |
| strict JSBSim | 84/84 |
| backend fallback | 0 |
| 响应窗口 predictor fallback | 0 |
| 脉冲裁剪 | 0 |
| 响应窗口缺失 | 0 |
| 我方坠毁 | 0 |
| 我方越界 | 0 |
| 训练启动 | 否 |
| compiler runtime probe | 否 |

独立 verifier：

```text
python scripts/verify_mild_disadvantage_handoff_local_response_collector_v1.py \
  --root outputs/mild_disadvantage_handoff_local_response_v1 \
  --output outputs/mild_disadvantage_handoff_local_response_v1/verifier_result.json
```

验收结果为 `paired_local_response_gate: pass`。

## 5. 可复现工件

- 聚合条件结果：`outputs/mild_disadvantage_handoff_local_response_v1/condition_results.csv`
- 4/12/48 步响应：`outputs/mild_disadvantage_handoff_local_response_v1/local_response_records.csv`
- 逐条件原始轨迹：`outputs/mild_disadvantage_handoff_local_response_v1/paired_traces/`
- 独立源轨迹：`outputs/mild_disadvantage_handoff_local_response_v1/source_traces/`
- 顶层运行清单：`outputs/mild_disadvantage_handoff_local_response_v1/run_manifest.json`
- 集合摘要：`outputs/mild_disadvantage_handoff_local_response_v1/collection_summary.json`
- 验证结果：`outputs/mild_disadvantage_handoff_local_response_v1/verifier_result.json`
- 单回合源轨迹入口：`scripts/run_mild_disadvantage_envelope_recovery_single_episode_v1.py`

## 6. 证据边界

本批次证明了真实 handoff、对称三轴脉冲和 4/12/48 步响应数据可以在严格 JSBSim 下连续、成对地采集。它还没有判断哪个动作轴能够改善交战几何，也没有证明存在跨对手一致的控制方向。下一步只能在本批数据上进行离线反对称响应分析；在该分析完成前，不启动动作序列筛选、编译器运行时 probe、新技能训练或高层 PPO 扩展。
