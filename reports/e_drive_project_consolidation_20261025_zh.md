# E 盘 UAV-VPP 项目整理记录

整理日期：2026-10-25

## 目标与结果

E 盘根目录最多保留两个 `uav-vpp-guidance*` 项目文件夹：

- `E:\uav-vpp-guidance`：唯一主仓库，分支 `fix/flight-control-comparison-sync`。
- `E:\uav-vpp-guidance-archives`：本地非 Git 证据、历史目录和待审 worktree 的统一归档。

所有仿真输出、checkpoint、数据包和历史项目目录均未删除。归档移动前后按目录核对文件数和总字节数，结果一致；移动发生在同一 E 盘。CAN-20260705 原 bundle 保留在归档目录中，未重打包或改写。

## 已推送的 Git 内容

主分支远端 `origin/fix/flight-control-comparison-sync` 已核验到：

- `e23c0b92a2ef11927e48740280d76beb9f9c5037`：整理前已有提交及研究代码/配置/测试提交。
- `d34e410`：观测与预测几何、劣势态势能力边界工具及测试；定向回归 `78 passed`。
- `e23c0b9`：论文审查记录、结果报告、章节图表及复现材料。

独立研究分支只推送，不合并到主线：

- `codex/disadvantage-capability-v1`：`66fb3821a31df4d89615c52fb258d882a0618cc2`。
- `codex/exploratory-rl-algo-entry`：`eb05d43558a97705111eae781046bfc79f7fd208`。
- `codex/five-state-router-mvp-v0`：`9413834e96c60188bd8bef92c2d3177909ca53bb`。
- `research/thesis-five-state-shared-intent-v1`：`9413834e96c60188bd8bef92c2d3177909ca53bb`。

远端核验时，上述分支头与本机提交一致。推送只包含已提交历史；没有把其他 worktree 的未提交更改带入远端。

## 归档布局

所有原本位于 E 盘根目录、名称以 `uav-vpp-guidance-` 开头的结果、资产和非 worktree 文件夹，已移动到：

`E:\uav-vpp-guidance-archives\drive-folders\<原文件夹名>`

包括 `artifacts`、`asset-collection`、`evidence-results`、`thesis-evidence`、`five-state`、`noncanonical`、`option-value-router`、`ppo-routing`、`tactical-basis`、`disadvantage-capability` 等结果与资产目录。原文件夹名保留，便于按 Source ID 和旧路径辨认。

6 个 Git worktree 保留其注册关系和未提交状态，移动到：

`E:\uav-vpp-guidance-archives\working-trees\<worktree 名>`

包括 `clean-formal`、`control-guidance-remediation-integration`、`disadvantage-capability-v1`、`explore-rl`、`five-state-router-mvp-v0`、`thesis-five-state-v1`。其中 `thesis-five-state-v1` 原有的三个 Windows junction 已重指向归档内 CAN-20260705 的资产路径；目标存在性已验证。

## CCDC 与其他未提交材料

用户明确要求 CCDC 相关内容不得提交。整个 CCDC 工作目录原样保留在：

`E:\uav-vpp-guidance-archives\private-unsubmitted\ccdc_paper_vpp_aircombat`

该目录没有进入 Git 暂存、提交或推送。四个未纳入研究主线的独立脚本也保留在 `private-unsubmitted\project-local-scripts`。

## 尚未清理的内容

六个归档 worktree 仍有未提交修改或未跟踪文件。它们包含独有研究代码、配置、输出或临时试验，尚未逐项审查，故没有提交、stash、reset 或删除。主仓库已干净；后续若要删除这些嵌套 worktree，须先逐项审查并将有价值的 dirty 内容另行提交到对应分支或做可验证的本地补丁/证据包。

本次没有运行训练或 JSBSim 实验，也没有删除任何文件。
