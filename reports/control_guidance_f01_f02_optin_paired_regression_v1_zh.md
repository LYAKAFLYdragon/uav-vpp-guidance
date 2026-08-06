# F01/F02 显式候选模式配对回归报告

## 1. 实验目的

本次验证只检查显式 `f01_f02_remediated` 模式是否能够在严格 JSBSim 中运行，并观察修正后的制导指令是否能够传递到执行器和飞行动力学响应。`legacy` 仍是默认模式，本次结果不授权替换默认控制链，也不代表作战性能提升。

Source ID：`THESIS-CONTROL-GUIDANCE-F01-F02-OPTIN-PAIRED-REGRESSION-V1`

代码提交：`6739f3eae2f78bf925edda08e841ae7977f72f8b`

复现命令：

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_control_guidance_f01_f02_optin_paired_regression.py `
  --jsbsim-root E:\uav-vpp-guidance-thesis-five-state-v1-jsbsim-v1.1.6 `
  --output-root outputs\control_guidance_f01_f02_optin_paired_regression_v1
python scripts/verify_control_guidance_f01_f02_optin_paired_regression.py `
  --output-root outputs\control_guidance_f01_f02_optin_paired_regression_v1
```

实验为非学习运行：四个固定场景、每个场景一个 legacy 回合和一个候选回合，每回合连续运行 48 步，合计 4 对配对记录。两种模式使用相同初始状态、相同目标机和相同 JSBSim 模型；只改变 `guidance.params.remediation_mode`。

## 2. 门禁结果

| 检查项 | 结果 |
|---|---|
| 配对数量 | 4 对，全部完成 48 步 |
| 最终后端 | `jsbsim` |
| `strict_backend` | `true` |
| `backend_fallback_occurred` | `false` |
| 初始状态哈希 | 4/4 对一致 |
| 目标轨迹哈希 | 4/4 对逐步一致 |
| 每回合 reset 次数 | 1 |
| 中途 reset | 0 |
| clipping | legacy 与候选均为 0/4 回合 |
| 非有限遥测 | 未发现 |
| 独立 verifier | `pass` |
| 运行授权结论 | `runtime_modification_authorized=false` |

上述结果说明配对实验链本身有效，且候选模式没有触发后端回退、额外重置或限幅异常。

## 3. 指令和物理响应

下表为候选相对 legacy 在 48 步窗口内的最大绝对差值。`Δn_z` 为实际过载差值，`Δp` 为滚转角速度差值。

| 场景 | 原始 `n_z` 指令 | 原始滚转角速度指令 | 实际升降舵角 | 实际副翼角 | `Δn_z` | `Δp` |
|---|---:|---:|---:|---:|---:|---:|
| 迎头水平 | 0.626 | 0.001 | 0.0417 | 0.0004 | 0.548 | 0.0019 |
| 左侧交叉 | 0.626 | 0.829 | 0.0418 | 0.4556 | 1.497 | 1.623 |
| 右侧交叉、高目标 | 0.212 | 0.821 | 0.0132 | 0.4476 | 1.871 | 1.461 |
| 偏置接近 | 1.964 | 1.295 | 0.0992 | 0.3157 | 1.491 | 0.0456 |

因此，候选模式并非只改变了数学输出：在本次固定接口下，指令差异确实传递到了升降舵、副翼、实际过载和滚转角速度。

## 4. 几何响应

下表为候选相对 legacy 的配对差值，正值表示候选数值更大。指标分别取第 4、12、48 步的状态。距离单位为米，距离率单位为米每秒，角度单位为度。

| 场景 | 步数 | 距离差 | 距离率差 | 迎角差（ATA） | 目标视线角差（AA） |
|---|---:|---:|---:|---:|---:|
| 迎头水平 | 4 | +0.0000003 | +0.00004 | -0.00004 | +0.00000003 |
| 迎头水平 | 12 | -0.0017 | -0.0214 | +0.0280 | -0.00015 |
| 迎头水平 | 48 | -0.0520 | -0.1955 | -0.5765 | -0.0173 |
| 左侧交叉 | 4 | +0.00007 | -0.00010 | -0.00003 | +0.0000008 |
| 左侧交叉 | 12 | -0.00074 | -0.00676 | +0.00088 | +0.00003 |
| 左侧交叉 | 48 | -0.1697 | -0.3773 | -0.2628 | -0.0131 |
| 右侧交叉、高目标 | 4 | +0.00007 | -0.00015 | -0.00004 | +0.0000008 |
| 右侧交叉、高目标 | 12 | -0.00034 | -0.00492 | -0.00066 | +0.00001 |
| 右侧交叉、高目标 | 48 | +0.9546 | +4.7015 | +1.4178 | +0.0319 |
| 偏置接近 | 4 | +0.00014 | +0.00307 | +0.00230 | +0.0000013 |
| 偏置接近 | 12 | +0.00070 | +0.00584 | +0.0405 | +0.00012 |
| 偏置接近 | 48 | -0.3577 | -1.5529 | -1.2387 | -0.0123 |

4 步和 12 步的几何差异总体较小，48 步后不同场景出现不同方向：右侧交叉、高目标场景的距离和距离率增加，而其余场景的距离或距离率减小。由此，本次回归证明了“候选接口具有可观测的物理响应”，但没有证明存在跨场景一致的几何收益。

## 5. 结论与后续边界

1. F01/F02 候选模式已实现为显式 opt-in 路径，legacy 默认行为没有被覆盖。
2. 在四个固定场景中，候选指令、执行器实际量和 JSBSim 状态响应均可读且可重复；这完成了接口层的运行可行性验证。
3. 候选在 48 步几何结果上表现出场景依赖，当前证据不足以称为性能修复，也不足以支持共享技能训练或扩大 PPO。
4. 本次结果不改变审计门禁：`reviewer_approved=false`，`runtime_modification_authorized=false`。如需继续，只能另行登记更大场景矩阵或专门的几何闭环分析，不能把本报告升级为默认控制行为授权。

## 6. 证据文件

输出目录：`outputs/control_guidance_f01_f02_optin_paired_regression_v1/`

| 文件 | SHA-256 |
|---|---|
| `paired_records.json` | `87bbe4333e95a75b2eec9e1336958423c9a42a31fa31123048c977cf12f182b2` |
| `run_manifest.json` | `8c7545134bb164bdc4cd58a8722bb0057d21a05c349f01e2a1194a10328d353f` |
| `optin_paired_verification.json` | `350058e6877df3ce26a83d87c73d0ddfc54a853a733421e1708fab5439965e78` |

## 7. 复现输入哈希

| 输入或工具 | SHA-256 |
|---|---|
| `scripts/run_control_guidance_f01_f02_optin_paired_regression.py` | `04d41f96b3b83e27cbf7b479d078d229e5d61dac35786fefe504259ae21cc698` |
| `scripts/verify_control_guidance_f01_f02_optin_paired_regression.py` | `e09d6e998c133050a80afd4913844bb5f19c6acbcdf3c0c41d6d8177a9204f0c` |
| `tests/test_control_guidance_f01_f02_optin_paired_regression.py` | `26c2c9ac70d3daf205e735608fbf65c54f689fb1c57675fe2428788e27047067` |
| `config/guidance.yaml` | `2f78d8c65f4379ace326b6f7792ba5d21575e27f38105b66030c5e8a3daa3306` |
| `los_rate_guidance.py` | `3dabec76e0d95a4720b286f7a06d0c97f0363a35df1b0bbf8e3e19261ef7b90e` |
| `low_level_controller.py` | `3c3ae6d41349329b8b8cc039d9acef17187dfd6bd2a7f36d0653aa06eb9f3523` |
| JSBSim `f16.xml` | `2fd41b858d00eea4b53dd677ac91e1ad3772ec99136e312fa3a2e81309c0ac60` |
