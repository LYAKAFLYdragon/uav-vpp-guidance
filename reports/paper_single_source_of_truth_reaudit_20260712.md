# Paper Single-Source-of-Truth Re-Audit

## Scope

This re-audit verifies the B1--B9 paper-facing correction patch applied to:

- `drones/dfar_ast_final_v5.tex`
- `drones/supplementary_material_v4.tex`
- `drones/supplement_index.tex`
- `reports/post_merge_recovery_route_b_evidence_package.md`
- `reports/paper_authoritative_artifact_index_20260712.md`

It does not rerun experiments, alter policies, or promote supporting evidence
to the canonical headline comparison.

## Decision

**B1--B9 PASS.** The manuscript and supplement now use CAN-20260705 as the
exclusive source of dual-opponent headline results. Supporting RQ1 and G4
evidence is separated by source ID in the authoritative artifact index.

## Resolution Ledger

| ID | Finding before correction | Correction | Re-audit status |
|---|---|---|---|
| B1 | Three frozen specialists/policies were claimed. | Text now identifies two frozen checkpoint policies and a deterministic post-merge profile that reuses the head-on checkpoint. The PPO checkpoint is described as having two learned actions. | PASS |
| B2 | Cadence and horizon were stated as 1/120 s, 6,000 steps, and 50 s. | Methods and protocol table now state 60 Hz JSBSim integration, 0.2 s / 5 Hz routing cadence, 12-step hold, 512 high-level steps, and 102.4 s cap. | PASS |
| B3 | The PPO network was described as 256/256. | Methods now state the checkpoint-recorded 19-D input, two actions, and 128/128 hidden layers. | PASS |
| B4 | Low-level observation contracts were described as identical. | Methods now distinguish head-on 19-D and crossing-v3 18-D contracts and disclose the evaluation-time compatibility adapter. | PASS |
| B5 | Head-on training was described as a warm start, 16,384 completed steps, and 1:1 weights. | Methods now report `warm_start.enabled=false`, 32,768 configured steps, the selected 8,192-step checkpoint, and 2:1 head-on/crossing weights. | PASS |
| B6 | Crossing was described as warm-started. | The unsupported warm-start statement was removed; 65,536 steps and 1:2 head-on/crossing weights are retained. | PASS |
| B7 | The Route B package named the 2026-07-03 SHA as authoritative. | The document is now explicitly historical/supporting and directs headline claims to CAN-20260705. | PASS |
| B8 | Latency values had no indexed benchmark artifact. | The entire numerical latency and real-time-feasibility paragraph was removed. | PASS |
| B9 | G1/S3--S5 values had only short, unindexed filenames. | The tables, their narrative references, G1 folder index entry, and dependent diagnostic figures were removed from the submission-facing files. The RQ1 synthesis was renumbered to Table/Figure S3. | PASS |

## Evidence Hierarchy

- **CAN-20260705:** exclusive source of headline Oracle-versus-PPO results at
  clean commit `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`.
- **RQ1-20260711 / RQ1-SEEDS-20260711:** finite crossing-envelope preservation
  evidence only; not pooled with canonical outcomes.
- **G4-20260711:** supporting end-to-end fixed-reference counterexample only;
  retained in Supplementary Table S3 with its own source ID.
- **G1 and latency:** excluded from the current submission until raw artifacts
  and reproducible provenance are restored.

## Build Verification

The following sources were compiled twice with isolated job names, preventing
stale source-directory `.aux` files from affecting reference resolution:

| Source | Output | Result |
|---|---|---|
| `drones/dfar_ast_final_v5.tex` | `drones/build_b1b9/dfar_ast_final_v5_b1b9.pdf` | PASS, 12 A4 pages |
| `drones/supplementary_material_v4.tex` | `drones/build_b1b9/supplementary_material_v4_b1b9.pdf` | PASS, 5 A4 pages |
| `drones/supplement_index.tex` | `drones/build_b1b9/supplement_index_b1b9.pdf` | PASS, 2 A4 pages |

The second-pass logs contain no fatal errors, undefined references, label-change
warnings, or overfull boxes. The main manuscript retains several underfull-box
warnings in pre-existing dense prose and bibliography entries; these are
typographic polish items, not evidence or compilation failures.

## Remaining Non-Claims

This correction closes the B1--B9 evidence-chain defects only. It does not
claim statistical superiority on expert/head-on, broad crossing generalization,
learned recovery routing, airworthiness, or measured real-time latency.
