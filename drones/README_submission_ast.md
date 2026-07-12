# AST Submission Package

**Current package ID:** `ast_vpp_can20260705`  
**Build command:**

```powershell
D:\Anaconda3\envs\jsbenv\python.exe scripts\build_submission_bundle.py build
```

The command reads `drones/submission_manifest.yaml`, recompiles all LaTeX sources twice, copies only manifest-listed files, writes checksums, verifies the unpacked package, and creates:

```text
drones/submission_dist/ast_vpp_can20260705.zip
```

## Included Deliverables

| Deliverable | Source |
|---|---|
| Anonymized manuscript | `dfar_ast_final_v5.tex` -> `manuscript/dfar_ast_final_v5.pdf` |
| Title page | `title_page_ast.tex` -> `title_page/title_page_ast.pdf` |
| Supplement | `supplementary_material_v4.tex` -> `supplementary/supplementary_material_v4.pdf` |
| Supplement index | `supplement_index.tex` -> `supplementary/supplement_index.pdf` |
| Highlights and cover letter | `highlights.txt`, `cover_letter_ast.md` |
| Figures | Only the six files listed in `submission_manifest.yaml` |
| Reproducibility ledger | CAN-20260705 artifact index and B1--B9 re-audit |

## Evidence Boundary

- Headline numbers come only from `CAN-20260705`.
- Current RQ1 material is Supplementary Table/Figure S3.
- Historical unindexed ablation tables, obsolete supplement numbering, old manuscript variants, and non-canonical v14b results are not upload inputs.

## Human Portal Checklist

- [ ] Review the four generated PDFs visually.
- [ ] Confirm final author and funding details on the title page.
- [ ] Add suggested and opposed reviewers.
- [ ] Upload only package-manifest deliverables.
