# AST Submission Readiness Checklist

**Current source package:** `CAN-20260705`  
**Current manuscript source:** `drones/dfar_ast_final_v5.tex`  
**Current supplement source:** `drones/supplementary_material_v4.tex` and `drones/supplement_index.tex`

## Hard Gates

### H1. Claim Discipline

- [x] Headline results use only CAN-20260705 at commit `9a9f9bf6d68560afa81560f5260b78f318dcd9b1`.
- [x] `expert/head_on` is written as “no longer weaker than Oracle,” not statistically superior.
- [x] `end_to_end/head_on` remains explicitly below Oracle.
- [x] Every canonical crossing statement retains its `N=4` caveat.
- [x] RQ1 crossing evidence is described as practical preservation against fixed head-on VPP within a finite envelope.
- [x] The manuscript does not claim task-label-free routing, learned recovery routing, airworthiness, or a certified flight envelope.

### H2. Evidence and Reproducibility

- [x] Detached code worktree is exactly `9a9f9bf6`.
- [x] CAN-20260705 artifact bundle contains copied raw expert/end-to-end formal runs, external assets, file hashes, and checkpoint shapes.
- [x] Bundle verification and exact-worktree preflight pass.
- [x] The authoritative artifact index separates headline, RQ1/RQ2/RQ3/RQ4, G4, and excluded evidence.
- [x] Historical unindexed ablation material and unproven latency figures are excluded from the current submission source.

### H3. Manuscript and Supplement

- [x] Main manuscript compiles from `dfar_ast_final_v5.tex`.
- [x] Current supplement uses Tables S1--S3 and Figure S3; it does not cite obsolete supplement numbering.
- [x] Supplementary Table S3 identifies RQ1/G4 as supporting, separate source strata.
- [x] The v5 paper-code re-audit has no B1--B9 fact conflicts.

### H4. Upload Package

- [x] `drones/submission_manifest.yaml` lists the only permitted manuscript, supplement, metadata, evidence-ledger, and figure files.
- [x] `scripts/build_submission_bundle.py build` compiles, checksums, verifies, and creates `drones/submission_dist/ast_vpp_can20260705.zip`.
- [ ] Human confirms journal-portal fields, author identities, funding, suggested reviewers, and opposed reviewers.
- [ ] Human uploads the generated archive contents according to AST portal requirements.

## Supporting Evidence Boundary

| Source | Permitted wording |
|---|---|
| RQ1 / RQ2 | Finite-envelope practical crossing preservation only. |
| RQ3 | Task-aware PPO selects crossing when bootstrap/lock are disabled. |
| RQ4 | Constrained post-merge state-conditioned re-selection; guards own recovery entry. |
| G4 | Retained counterexample; no superiority conclusion. |
| v14b | Non-canonical exploratory low-level routing; not part of this submission. |

## Final Human Check

1. Open the PDFs contained in the generated bundle and confirm visual quality.
2. Confirm the title page contains the final, authorized author and funding details.
3. Confirm every uploaded figure is exactly the file listed in `submission_manifest.yaml`.
4. Upload only after the CAN bundle verifier still passes on the same day.
