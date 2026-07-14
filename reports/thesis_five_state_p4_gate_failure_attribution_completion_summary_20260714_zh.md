# P4 几何门失败归因审计 — 完成总结 (GEO-20260713-R1)

**日期：** 2026-07-14
**分支：** `research/thesis-five-state-shared-intent-v1`
**性质：** 严格非训练、非调参、只读归因审计（不重训、不重评估、不修改 gate / checkpoint / model / reward，也不触碰双技能 canonical 主线）
**Spec：** `.kiro/specs/p4-geometry-gate-failure-attribution-audit/`（bugfix，requirements-first）

---

## 1. 结论（不可更改）

在既有的预注册 per-skill gate 下，**不允许**启动 combat finetune（`combat_finetune_NOT_allowed`）。本审计无权更改该结论；它只解释失败、冻结证据、纠正状态记录。combat finetune / P5 高层 PPO / P6 消融 / P7 heldout60 全部保持锁定。

---

## 2. 核心事实纠正（26/22/4 前提已被反证）

原始 spec 曾假设“declared 26 / emitted 22 / diff 4（`crossing_entry` 缺失）”。经对本机在仓库内的真实证据独立核验，该前提**不成立、已被反证、在结构上不可能**：

| 项 | 结论 |
|---|---|
| declared cells | **26**（pursuit_conversion 6、lead_intercept 6、defensive_extension 6、reentry_recovery 8） |
| emitted cells | **26**（每技能 6/6、6/6、6/6、8/8） |
| diff / extra | **0 / 0** |
| `crossing_entry` 单元 | **存在且非零**（如 `lead_intercept crossing_entry:pre_merge=1092`、`crossing_entry:post_merge=182`） |

结构性原因：`evaluate_skill_gate`（`thesis_shared_skill_geometry.py:511-546`）按 `required_states × required_phases` 的完整笛卡尔积构造 coverage，缺失组合以 `.get(..., 0)` 填 0，因此 emitted 总数恒等于 declared 总数，22-emitted 不可能出现。

**真实失败结构：**
- `geometry_phase_coverage = false`（4/4 技能）完全由**已声明但以 0 值发出**的 `post_merge`/`re_entry` 单元（`present_value_0`）驱动，而非任何缺失/absent 单元。
- `intent_progress` 三个失败：`defensive_extension` 0.5325、`lead_intercept` 0.4977、`pursuit_conversion` 0.5193（均 < 0.55）。
- `reentry_recovery` 0.5820 **通过** intent 进度，但仍因 phase-coverage 失败被阻断。
- 4/4 技能均通过 `profile_conditioning_coverage` 与 `safety`（予以保留的通过性证据）。

---

## 3. 交付物

| 文件 | 说明 |
|---|---|
| `src/uav_vpp_guidance/evaluation/p4_gate_failure_attribution.py` | 归因纯函数模块（stdlib only、无 side effect、计数自适应） |
| `scripts/analyze_p4_geometry_gate_failure_attribution.py` | CLI（2b 一致性优先、2a 对账继续、可配置路径、无硬编码盘符） |
| `reports/thesis_five_state_p4_gate_failure_attribution_audit_20260714_zh.md` | 中文归因审计叙述 + 26 行归因矩阵 |
| `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.csv` | 26 行归因矩阵（全部 `emission_status = emitted`） |
| `reports/thesis_five_state_p4_gate_failure_attribution_matrix_20260714.json` | 同数据 + `preflight` 块（26/26/0/0）+ `provenance` 块 |
| `tests/test_p4_gate_failure_attribution.py` | 冻结证据事实测试 + 审计输出测试 + 保留性测试 + 参数化纯函数测试 |
| `tests/_p4_preservation_baseline.json` | 审计前的 SHA256 基线（保留性对照） |
| `reports/thesis_five_state_shared_intent_v1_execution_checklist_20260713_zh.md` | P4 状态更新为最终负证据态 |

---

## 4. 归因矩阵与标签

- 每个已声明单元一行（26 行），三列独立状态：`train_field_status`、`dev_gate_field_status`、`emission_status`，另加 `reason_code`。
- 字段状态编码：`present_value_0` / `present_value_gt_0` / `absent`（`unverifiable`）；绝不从“缺失”推断“物理不可能”。本机全部单元均 PRESENT，主导情形为 `present_value_0`。
- 六个受控标签：`not_observed_cannot_assess`、`train_coverage_gap`、`observed_behavior_failure`、`telemetry_or_gate_observability_gap`、`mixed_or_inconclusive`、`contract_or_provenance_mismatch`。
  - `observed_behavior_failure` 在当前仅有 skill 级聚合证据下**永不**被赋予。
  - train>0 且 dev=0（如 `lead_intercept advantage:post_merge` train=2920/dev=0）→ `not_observed_cannot_assess` + `dev_coverage_gap`。
- **initial-class × phase 局限**：coverage 按场景固定 `initial_class` 记录（`thesis_shared_skill_geometry.py:451`，调用点 677/806），故只能证明 initial-class × phase 覆盖，**无法**证明 `post_merge`/`re_entry` 的动态五态势；该局限已贯穿列语义、结论与每行置信度。

---

## 5. 溯源边界（provenance）

- 审计仅覆盖仓库内现有 P4 证据；**不是**原 E: 盘 `GEO-20260713-R1` run 的完整复现。
- `provenance_unverified` 输入：**run commit SHA、`training_plan.json`、`resolved_geometry_config.json`、dev30 manifest**（均缺失）。declared cells 由 registry 代替 `training_plan.json`，该替代本身标记 `provenance_unverified`。
- 冻结溯源块与工作树 git 信息分列，绝不以工作树源冒充冻结 run 事实。
- 运行级 `contract_or_provenance_mismatch` 仅由缺失溯源输入与不可核验的历史 26/22/4 前提触发，**不**由任何 declared-vs-emitted diff 触发（diff 为 0）。
- 主机/解释器说明：本次运行使用 Python 3.11.14（本机可用）。

---

## 6. 保留性验证（未回归）

- 6 个 P4 证据文件（gate JSON、四个 per-skill summary、registry）审计前后 **SHA256 逐字节一致**。
- gate 判定仍为 `all_skills_ready_for_combat_finetune = false`；阈值仍为覆盖 `2` / intent `0.55`；P3 encoder `best.pt` SHA256 未变。
- 通过性子检查仍报告为通过（`reentry_recovery` intent 进度；四技能 profile coverage 与 safety），未被重述为失败。
- 唯一被修改的受追踪文件是执行 checklist；其余交付物均为新增文件。

---

## 7. 测试结果

命令：`python -m pytest tests/test_p4_gate_failure_attribution.py -v`（无 `--run` 标志）
结果：**31 passed**（Python 3.11.14, pytest 9.0.3）
- 冻结证据事实测试（26/26/0/0，历史前提不可复现）— PASS
- 审计输出测试（26 行矩阵、三状态列、run-level 溯源标志、结论）— PASS
- 保留性测试（SHA256 逐字节一致 + git 白名单）— PASS
- 27 个参数化纯函数用例（对账 / 字段状态 / 标签安全 / 2b 一致性 / 溯源冻结 / csv-json 一致）— PASS

CLI 运行（只读）：2b CLEAN → 2a `declared=26 emitted=26 diff=0 extra=0` → 26 行矩阵 → exit 0。

---

## 8. 下一步动作

唯一的下一步是 **P4 gate 失败归因决策（非重训、非调参）**，依据即上述归因审计产物。任何“修复 P4 让其通过”的动作（重训、放宽 gate、更换 held-out、事后调 mask）均被禁止。

> 说明：可选任务 7、8（frozen-artifact 集成测试、2b 完整性失败集成测试）按“仅必选任务”范围未执行；核心 26/26/0/0 事实已由冻结证据事实测试与 CLI 运行覆盖。
