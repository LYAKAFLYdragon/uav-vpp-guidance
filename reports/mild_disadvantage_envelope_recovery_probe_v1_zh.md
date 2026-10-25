# 固定包线恢复打法非学习物理预检结果

## 1. 实验标识

- Source ID：`THESIS-MILD-DISADVANTAGE-ENVELOPE-RECOVERY-PHYSICAL-PROBE-V1-20260903`
- 类型：非学习、严格 JSBSim 物理预检
- 运行目录：`outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1`
- 初始点：`(625 m, 200 m)`、`(750 m, 250 m)`
- 对手：`expert`、`end_to_end`
- 每个点、每个对手重复：3 次，共 12 条回合

本轮保持奖励函数、PID、滤波、动作空间和 JSBSim 模型不变。包线外先使用冻结的 `crossing` 专家，最多恢复 48 个高层步；若进入稳定可行区，再保持 10 步并切回固定 H0 控制器。该打法不改变状态，不注入未来状态，也不生成新的 VPP 动作。

## 2. 工程门禁

独立 verifier 结果：`ok=true`。

| 门禁项目 | 结果 |
|---|---|
| 12 条回合均有结果记录 | 通过 |
| 有限且非空的逐步遥测 | 通过 |
| 有效后端为 JSBSim | 通过 |
| 后端回退 | 未发生 |
| 每回合 reset 次数 | 1 次 |
| 训练启动 | 未启动 |
| 恢复门：每个单元至少 2/3 次进入并保持 | 未通过 |

## 3. 结果

| 对手 | 初始点 | 进入稳定可行区 | 恢复步数 | 保持步数 | 终端原因 | 坠毁 | 越界 | 回退 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| expert | `(625, 200)` | 0/3 | 48 | 0 | 越界 | 0/3 | 3/3 | 0/3 |
| expert | `(750, 250)` | 0/3 | 48 | 0 | 越界 | 0/3 | 3/3 | 0/3 |
| end_to_end | `(625, 200)` | 0/3 | 48 | 0 | 越界 | 0/3 | 3/3 | 0/3 |
| end_to_end | `(750, 250)` | 0/3 | 48 | 0 | 越界 | 0/3 | 3/3 | 0/3 |

四个单元均从第 0 步进入恢复模式，连续使用 crossing 专家 48 步，但没有一次达到稳定可行区，也没有形成 10 步保持。之后恢复预算耗尽，系统回到 H0，最终越界。逐步记录的限幅标记均为 0，说明本轮主要问题不是记录到的动作限幅或后端回退。

代表性终端状态的距离约为 `12012--12076 m`，高度约为 `3692--4313 m`。不同重复的结果一致，未观察到随机分支改变结论。

## 4. 结论

本轮通过了严格 JSBSim 运行链门禁，但没有通过包线恢复门。因此当前证据支持的结论是：

> 在当前 H0 运行接口、现有冻结 crossing 专家和给定 48 步恢复策略下，两个不可行初始点均未能恢复到固定 H0 的稳定可行包线。

这不能单独证明“缺少一个新的防御技能”，也不能证明 crossing 专家在所有接口中都没有控制能力。它只证明了当前已有专家切换打法尚不足以恢复这两个包线外状态。由于 crossing checkpoint 的历史观测契约为 18 维，而 H0 为 16 维，本轮采用了显式的 18 维运行观测和 H0 末端任务标志截断适配；因此后续若要继续，首先应做动作语义和运行配置一致性复核，而不是直接训练新技能。

## 5. 可复核文件

- 机器可读结果：[pilot_manifest.json](../outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1/pilot_manifest.json)
- 回合汇总：[episode_results.csv](../outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1/episode_results.csv)
- 独立核验：[verification.json](../outputs/mild_disadvantage_envelope_recovery_v1/physical_probe_v1/verification.json)
- 运行脚本：[run_mild_disadvantage_envelope_recovery_probe_v1.py](../scripts/run_mild_disadvantage_envelope_recovery_probe_v1.py)
- 核验脚本：[verify_mild_disadvantage_envelope_recovery_probe_v1.py](../scripts/verify_mild_disadvantage_envelope_recovery_probe_v1.py)
