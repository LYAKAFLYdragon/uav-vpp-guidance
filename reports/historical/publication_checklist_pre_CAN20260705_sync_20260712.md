# 可投稿标准执行 Checklist（当前仓库可执行版）

**项目**: UAV VPP Guidance  
**主线目标**: 把 VPP 从“追一个虚拟点”的控制接口，推进成“能稳定塑造空战有利几何的机动决策接口”。

## 当前 canonical paper mainline

- canonical family:
  - `oracle_task_gate`
  - `commander_post_merge_recovery3`
- canonical learned commander algorithm family:
  - PPO-based hierarchical commander
- fixed scope:
  - tasks: `head_on`, `crossing_feasible`
  - opponents: `expert`, `end_to_end`
  - backend: `jsbsim`
  - crossing AoA gate: `60 deg`

## 当前 authoritative evidence

- clean worktree:
  - `E:\uav-vpp-guidance-clean-formal-448155d`
- frozen branch / commit:
  - `codex/canonical-post-merge-recovery-manifest60-fwdneg`
  - `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`
- authoritative raw outputs currently live in the clean worktree:
  - `E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_expert_20260705\`
  - `E:\uav-vpp-guidance-clean-formal-448155d\outputs\jsbsim_hrl_comparison\oracle_vs_commander_post_merge_recovery_manifest60_jointgate_fwdneg_formal_end_to_end_20260705\`
- repo-local paper-facing entry points:
  - `reports/canonical_commander_mainline_freeze_20260705.md`
  - `outputs/diagnostics/manifest60_jointgate_fwdneg_formal_dual_split_20260705.md`
  - `reports/post_merge_recovery_route_b_evidence_package.md`

## 当前关键事实

| Opponent | Task | Oracle | Commander | Paper-facing interpretation |
|---|---:|---:|---:|---|
| `expert` | `head_on` | 0.5667 | 0.6500 | commander no longer weaker than oracle |
| `expert` | `crossing_feasible` | 0.7500 | 0.7500 | preserved |
| `end_to_end` | `head_on` | 0.9322 | 0.9000 | slightly below oracle; localized residual only |
| `end_to_end` | `crossing_feasible` | 0.7500 | 0.7500 | preserved |

Safe paper/report claims:

1. 当前 canonical family 已将 VPP 从“纯虚拟点追踪接口”推进为“带明确几何意图的机动决策接口”。
2. `expert/head_on` 上 commander 已不弱于 oracle。
3. `crossing_feasible` 在两个 opponent split 上均保持。
4. `end_to_end/head_on` 仍略低于 oracle，但残差已局部化到 3 个 oracle-only formal 场景。

Unsafe claims:

- `uniform oracle parity`
- `end_to_end/head_on >= oracle`
- `remaining hardest templates are solved`
- 把历史 DQN / Double DQN / 早期 PPO 表格当成当前 canonical 主结果

## 本版 checklist 的原则

1. 一切 paper-facing 口径都以 2026-07-05 frozen canonical evidence 为准。
2. 2026-07-03 的 loader-fix formal、Route B artifacts、manifest60 ablation 等保留为 supporting / historical context，不再作为 headline number source。
3. 历史 DQN / Double DQN / SAC / self-play lanes 只能作为 `historical/non-canonical` 补充讨论。
4. 当前主线工作是关闭 `M2` 和组装 `M3`，不是重开 commander tuning。

---

## 里程碑定义

| 里程碑 | 定义 | 验收标准 | 当前状态 |
|---|---|---|---|
| `M0` | 工程闭环完成 | 代码、tests、formal 入口、artifact pipeline 可运行 | ✅ 已完成 |
| `M1` | 主线证据定盘 | frozen canonical PPO commander mainline 建立并完成 claim boundary 冻结 | ✅ 已完成 |
| `M2` | 可投稿证据包完成 | 主结果表、图、统计、补充材料、复现说明、README、checklist、state docs 全部对齐 | ✅ 已完成 | manuscript master unified, P0 closed, PDF compiled, refs 20 |
| `M3` | 投稿包完成 | 论文正文、supplement、cover letter、reviewer-response scaffolding 完成 | ⏳ 未开始 |

---

## 当前 paper route

### 冻结主线

当前默认论文路线不再是“旧 Route B 的 commander weakness 叙事”，而是：

**positive but bounded canonical mainline**

即：

- `expert/head_on` 已修复并通过 `commander >= oracle` gate
- `crossing_feasible` 保持
- `end_to_end/head_on` 仍存在小范围局部残差
- 后续若继续攻 hardest templates，应进入 non-canonical low-level recovery lane，而不是重开当前 canonical family

### Supporting / historical context

以下内容仍可用于解释过程，但不能替代 headline 结果：

- `reports/post_merge_recovery_route_b_evidence_package.md`
- `outputs/diagnostics/post_merge_recovery_route_b_artifacts_20260703/`
- `reports/post_merge_recovery_route_b_safety_report.md`
- 历史 DQN / Double DQN / SAC / curriculum/self-play lane

---

## P0 致命任务

### P0-0：authoritative clean-worktree formal freeze

**状态**: `completed`

**目标**:

- 建立 paper-safe 的 canonical mainline 裁决，不再让旧 formal 或窄审计决定论文口径。

**验收标准**:

- clean worktree
- `paper_safe=True`
- artifact contract valid
- 有 freeze note 与 dual-split summary

**当前 authoritative source**:

- `reports/canonical_commander_mainline_freeze_20260705.md`
- `outputs/diagnostics/manifest60_jointgate_fwdneg_formal_dual_split_20260705.md`

### P0-1：paper-facing claim boundary freeze

**状态**: `completed`

**目标**:

- 冻结论文可安全声称的结果边界，避免后续写作再次漂移。

**验收标准**:

- `expert/head_on >= oracle` 被明确写入主结果表
- `end_to_end/head_on < oracle` 被明确写入主结果表
- `crossing_feasible N=4` caveat 被保留
- 历史 RL family 不混入主结果表

**当前 authoritative wording**:

- `reports/canonical_commander_mainline_freeze_20260705.md`
- `reports/paper_narrative_framework_v2.md`
- `reports/post_merge_recovery_route_b_manuscript_package.md`

### P0-2：supporting evidence closure

**状态**: `completed`

**目标**:

- 图、统计、safety、mock review、evidence package 全部可用，支撑 frozen canonical mainline。

**当前 supporting artifacts**:

- `outputs/diagnostics/post_merge_recovery_route_b_artifacts_20260703/`
- `reports/post_merge_recovery_route_b_safety_report.md`
- `reports/post_merge_recovery_route_b_mock_review.md`
- `reports/post_merge_recovery_route_b_evidence_package.md`

### P0-3：M2 manuscript sync gate

**状态**: `completed`

**目标**:

- 关闭所有 paper-facing 文档漂移，让正文、README、state docs、checklist、补充材料口径统一。

**验收结果**:

1. 摘要、正文结果表、结论只使用 2026-07-05 frozen canonical numbers。✅
2. README / STATE / LOOP / publication_checklist 与 freeze note 一致。✅
3. 论文明确写出：
   - `expert/head_on >= oracle` ✅
   - `end_to_end/head_on < oracle` ✅
   - `crossing_feasible` is preserved but low-sample (`N=4`) ✅
4. 历史 DQN / Double DQN / early PPO 表格全部显式标注 `historical/non-canonical`。✅
5. manuscript master 已统一为 `dfartv2_drones_mdpi.tex`（routeb_frozen 内容），旧稿已归档。✅
6. residual TBD 已填充真实值（steps 27/37/37, Ego crash/OOB）。✅
7. REPO_URL_TBD 已替换为真实仓库地址。✅
8. PDF 已通过 2-pass pdflatex 编译（7 pages, build_main/dfartv2_drones_mdpi.pdf）。✅
9. 参考文献从 12 条扩展到 20 条。✅

---

## P1 严重任务

### P1-1：paper-code sync pass

**状态**: `in_progress`

**目标**:

- 对正文、摘要、README、support docs 做一次数值与路径对齐审计。

**验收标准**:

- 不存在 2026-07-03 headline numbers
- 不存在把历史 DQN / PPO 表格当 canonical 的表述
- 不存在引用缺失的 authoritative output 路径

### P1-2：manuscript integration

**状态**: `completed`

**目标**:

- 把 freeze note、narrative framework、manuscript package 落进真实投稿正文。

**验收结果**:

- ✅ AST retarget 完成：主稿从 MDPI Drones 格式迁移到 Aerospace Science and Technology (AST) 格式
- ✅ 匿名主稿 `dfartv2_ast_anonymized.tex`（14 页，编译通过）
- ✅ 独立标题页 `title_page_ast.tex`（1 页，编译通过）
- ✅ Highlights `highlights.txt`（5 bullet points，每行 ≤85 字符）
- ✅ AST cover letter `cover_letter_ast.md`
- ✅ 所有内部术语已替换为标准学术语言（authoritative formal evidence → experimental validation, clean-worktree → frozen evaluation snapshot, lane → task, commander → high-level policy, specialist → frozen low-level specialist, etc.）
- ✅ 方法部分补完：state space, 3D VPP action head, command interval, 3 specialists, PPO commander, stationary induced decision process
- ✅ 结果部分嵌入 5 张图：TikZ architecture diagram + 4 frozen figures from figures_ast/
- ✅ Data Availability 匿名化，无本地盘符，无真实仓库 URL
- ✅ 参考文献 20 条全部保留
- ✅ 主稿编译通过（build_ast/dfartv2_ast_anonymized.pdf, 14 pages, 842 kB）
- ✅ Title page 编译通过（build_ast/title_page_ast.pdf, 1 page, 69 kB）

**必须完成（已达标）**:

- authoritative main results table with Wilson 95% CI ✅
- residual-boundary paragraph and Table 2 ✅
- crossing `N=4` caveat ✅
- methods technical description ✅
- Related Work 三角对位：经典 VPP guidance vs end-to-end RL vs hierarchical routing ✅
- Discussion 强调工程洞见：VPP 从追点到几何塑形接口，不改 3D 低层接口 ✅
- Claim boundaries 严格保留：expert/head_on "no longer weaker", end_to_end/head_on "still slightly below oracle", crossing N=4 caveat, residual "localized to three scenarios" ✅

### P1-3：supplement / reproducibility appendix assembly

**状态**: `in_progress`

**目标**:

- 让审稿人能顺着 freeze note、evidence package、asset manifest 追到所有关键 artifact。

**至少包含**:

- frozen SHA / branch
- authoritative output paths
- config paths
- figure index
- statistics CSV index
- safety report

### P1-4：historical contamination cleanup

**状态**: `in_progress`

**目标**:

- 把历史 RL family 从 paper mainline 中彻底降级为补充材料或 future work。

**重点对象**:

- 历史 DQN / Double DQN commander tables
- 早期 PPO `1.00/1.00/1.00/1.00` 片段
- 2026-07-03 Route B formal headline framing

---

## P2 中等任务

### P2-1：安全性与飞行包线分析

**状态**: `completed`

- `reports/post_merge_recovery_route_b_safety_report.md`
- `outputs/diagnostics/post_merge_recovery_route_b_safety_20260703/`

### P2-2：复现文档与资产索引

**状态**: `completed`

- `README.md`
- `reports/post_merge_recovery_route_b_repro_index.md`
- `reports/post_merge_recovery_route_b_asset_manifest.md`
- `scripts/run_post_merge_recovery_route_b_publication_bundle.ps1`

### P2-3：mock review / reviewer-readiness

**状态**: `completed`

- `reports/post_merge_recovery_route_b_mock_review.md`

---

## P3 轻量任务

### P3-1：期刊与 cover letter

**优先候选**:

- `Drones`
- `Aerospace Science and Technology`
- `Chinese Journal of Aeronautics`
- `Defence Technology`

### P3-2：最终润色

**目标**:

- 压缩历史叙事噪声
- 强化“VPP as maneuver-decision interface”主线
- 保持 claim boundary 克制

---

## 当前最高优先级的可执行动作

1. 同步所有 paper-facing 文档到 2026-07-05 frozen canonical wording。
2. 跑一次 paper-code sync，专门扫 historical contamination 与路径漂移。
3. 把 authoritative results table、supporting figures/stats、supplement index 迁入投稿正文与补充材料。

---

## 关键文件

- `reports/canonical_commander_mainline_freeze_20260705.md`
- `reports/paper_narrative_framework_v2.md`
- `reports/post_merge_recovery_route_b_manuscript_package.md`
- `reports/post_merge_recovery_route_b_evidence_package.md`
- `outputs/diagnostics/manifest60_jointgate_fwdneg_formal_dual_split_20260705.md`
- `README.md`
- `STATE.md`
- `LOOP.md`

---

## 当前一句话结论

当前最正确、最可执行的投稿推进方式，不是再开新实验，而是把 **2026-07-05 frozen canonical PPO commander mainline** 彻底同步进论文主文、support docs 与 supplement，确保所有 headline claims 都只锚定这组 positive but bounded evidence。
