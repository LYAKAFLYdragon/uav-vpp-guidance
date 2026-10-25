# 时间扩展参考轨迹离线动作序列筛选审计

Source ID：`THESIS-MILD-DISADVANTAGE-TEMPORAL-REFERENCE-OFFLINE-SCREEN-V1-20260903`

## 1. 审计范围

本次只读检查输入目录：

`outputs/mild_disadvantage_envelope_recovery_v1/matched_telemetry_v1`

该目录包含两个主对手（`expert`、`end_to_end`）在两个初始点上的连续轨迹。审计没有运行 JSBSim，没有训练，没有修改专家、奖励、PID、滤波器或动作接口，也没有启动编译器运行时探针。

复现命令：

```powershell
python -B scripts\analyze_mild_disadvantage_temporal_reference_offline_screen_v1.py `
  --input-root outputs\mild_disadvantage_envelope_recovery_v1\matched_telemetry_v1 `
  --output-root outputs\mild_disadvantage_temporal_reference_offline_screen_v1
```

## 2. 输入事实

候选包共发现 `18` 个可审计的 CSV/JSON 文件，其中 `12` 个轨迹文件含有 `state_hash_before_action` 和三轴 `action_0/action_1/action_2`。两个主对手均可从轨迹字段中分开识别，安全与终端字段也存在。

但是，这些文件是自然运行轨迹，不是同一 handoff 状态下的反事实动作实验。当前记录没有显式的成对状态编号、正负对称动作轴和幅值、无干预基线，也没有以 4、12、48 步为对象的局部反事实响应记录。

## 3. 门禁结果

| 检查项 | 结果 |
|---|---|
| 真实 handoff 状态可识别 | 未通过 |
| 动作前状态哈希存在 | 通过 |
| 成对状态身份可识别 | 未通过 |
| 三轴对称动作干预可识别 | 未通过 |
| 4/12/48 步局部响应可识别 | 未通过 |
| `expert` 与 `end_to_end` 可分开统计 | 通过 |
| 安全与终端字段完整 | 通过 |

## 4. 离线筛选判定

离线动作序列筛选资格：**不通过**。

编译器运行时 probe：**不授权，未执行**。

不能从当前文件估计局部响应矩阵，也不能把单条自然轨迹中观察到的几何变化当作动作干预的因果效果。因而当前最准确的判定是：

> 现有数据能够证明连续遥测链和动作前状态哈希已经存在，但尚未建立真实 handoff 的成对局部响应数据契约；时间扩展编译器暂不具备可筛选的物理依据。

这不是对控制器性能的判断，而是对离线证据资格的判断。

## 5. 后续入口

只有补齐以下字段后，才重新运行本筛选器：

- 对手、handoff 编号和动作前状态哈希；
- 同一 handoff 状态下的无干预基线；
- 动作轴、正负方向和动作幅值；
- 4、12、48 步的距离、距离率、ATA、AA、能量和高度响应；
- clipping、fallback、reset、terminal 及连续性信息。

契约通过后，离线筛选需进一步检查动作有限性、三轴动作范围、相邻动作变化率和投影残差。只有两个主对手都出现共同可行的动作序列，才可另行预注册并授权运行时 probe。

## 6. 结论

本轮应停止时间扩展编译器的运行时验证，不启动新技能训练、不调奖励、不调 PID、不调滤波系数。当前研究结论保持为：已确认共同经验能力包线，但尚未获得足以支持时间扩展动作序列筛选的真实成对局部响应数据。
