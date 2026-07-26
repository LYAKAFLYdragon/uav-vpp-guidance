# 任务 8 交付检查摘要

## 结果
- 审计 CLI：成功（退出码 0），重生成两份既有 allowlist 报告。
- 聚焦测试：`42 passed in 1.59s`；包含报告一致性、受保护路径 SHA256 保全、严格 Git allowlist 与端到端 CLI 检查。
- 未提交、未推送；未训练、未调参、未改 checkpoint。

## 精确执行命令与结果
```powershell
python scripts/audit_control_method_rationality.py --guidance-config config/guidance.yaml --out-md reports/control_method_rationality_audit_findings_20260725_zh.md --out-json reports/control_method_rationality_audit_findings_20260725.json
# Exit Code: 0

python -m pytest tests/test_control_method_audit.py -v
# 42 passed in 1.59s
```

## 判定与优先级
- `confirmed_defect`（5）：F01、F02、F06、F07、F16。
- `confirmed_risk`（10）：F03、F05、F08、F10、F11、F12、F13、F14、F17、F18。
- `by_design_ok`（2）：F04、F09。
- `premise_incorrect`（1）：F15。
- 优先 remediation backlog：F01 → F06 → F05 → F02 → F16 → F08 → F09 → F03 → F04 → F07 → F10 → F11 → F12 → F13 → F14 → F17 → F18。

## 前提与来源纠正
- F15：`(sin θ, cos θ)` 在 `[0, 2π)` 可唯一确定角度；30° 与 330° 的 cos 相同而 sin 不同，因此原“丢失角度幅值”前提错误，无需改动编码。
- F08：实际激活 CEM 增益空间为 **5-D**，不是 7-D；12 candidates、`elite_ratio=0.25`，故精英数为 3，10×dim 启发式为 50。
- F05：配置横向偏置是 **±800 m**，不是 500 m；在 2500 m/800 m 时分别约 17.74°/45.00°。500 m→2500 m 的 11.31°仅为假设性计算对照，不是配置值。

## SHA256、变更与 allowlist 证明
- `tests/test_control_method_audit.py` 的保全检查通过：Phase 1 SHA256 基线覆盖的 guidance、flight_control、virtual_point、gain_optimizer、指定 env 文件与全部 `config/**` 均字节一致；`config/guidance.yaml` 当前 SHA256：`2f78d8c65f4379ace326b6f7792ba5d21575e27f38105b66030c5e8a3daa3306`。
- 该测试的严格 Git changed/untracked allowlist 检查通过。交付后变更只在 `.kiro/specs/control-method-rationality-audit/**`、两份 reports、CLI、审计模块和聚焦测试；本摘要也在允许的 spec 目录内。
- 主要生成报告 SHA256：JSON `cf9196198eb17961de8fa11febbfd7b7245fb9fde93dc483c31594ad5d738134`；Markdown `d5ee49d7981ba20f7518c2775324766f7a24747a94b64839173ed4955837b2c4`。

## 结论
本审计**未改变任何控制行为、配置或 checkpoint**。所有 remediation 建议均为 advisory，必须另立 gated 任务，并遵守 `AGENTS.md` 的观测、后端、配置覆盖与测试契约。
