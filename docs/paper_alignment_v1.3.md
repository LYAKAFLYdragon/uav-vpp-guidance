# Paper Alignment Log v1.3

> **Date**: 2026-06-14
> **Scope**: Align `paper_materials/paper.tex` and supporting sections with the
> frozen implementation (`config/canonical/`, `docs/status.md`,
> `docs/reward_audit.md`).
> **Policy**: Only `paper_materials/paper.tex` is to be aligned; the separate
> theoretical derivation report is out of scope. The `.tex` source edits are
> intentionally kept local-only and are **not** committed to version control.

---

## 1. Guid mismatch between code and paper (resolved in paper)

### §2.2 Hierarchical Architecture — rewritten

**Before (paper.tex, lines 49--60)**:
- Described an idealized translational-acceleration command
  `a_cmd = k_los λ̇ × v_own + ...`
- Claimed CEM optimized only `[k_los, k_pos, k_damp]`
- Referred to "acceleration commands" throughout

**After**:
- Equations now describe the implemented three-loop command chain:
  - `ω_cmd = k_roll ψ_err - k_damp φ_own`
  - `n_z,cmd = n_z,0 + k_los λ_el + k_pos (||p_v - p_own|| / d_scale)`
  - `δ_t,cmd = δ_t,0 + k_speed (v_err / v_scale)`
- Explicitly states the idealized acceleration model is used only as an
  analytical intuition elsewhere; actual closed-loop behavior is governed by
  the command chain above.
- References `config/canonical/guidance.yaml` for fixed parameters.

**Files changed**: `paper_materials/paper.tex`

### CEM gain space dimension

**Before**: `[k_los, k_pos, k_damp]` (3-D)

**After**: `[k_los, k_pos, k_damp, k_roll, k_speed]` (5-D), matching
`config/canonical/gain_space.yaml`. EMA-CEM is noted as the default; CEM-GD is
deprecated.

**Files changed**: `paper_materials/paper.tex`

---

## 2. Symbol table added

A new Table~\ref{tab:symbols} in §"Notation and Canonical Gain Space" lists:
- CEM-optimized gain vector `g`
- Canonical ranges for each of the 5 gains
- Fixed parameters (`α_filter`, `n_z,0`, `δ_t,0`)

The table explicitly references `config/canonical/gain_space.yaml` and
`config/canonical/guidance.yaml`.

**Files changed**: `paper_materials/paper.tex`

---

## 3. Reward description aligned with `docs/reward_audit.md`

**Before**: "sparse terminal success bonus with dense shaping terms"

**After**: "dense per-step mixture of range, aspect-angle, safety, saturation,
smoothness, turn-rate, closing-rate, and alive bonuses, plus sparse terminal
events for success, failure, and crash". Also notes potential-based shaping is
disabled in the canonical configuration and treated only as an optional
ablation.

**Files changed**: `paper_materials/paper.tex`

---

## 4. "Proof" / theorem / proposition tone adjusted

No occurrence of the word "proof" was found in the paper content (it only
appears in `IEEEtran.cls`). The following claims were softened to match
`docs/status.md`:

| Location | Before | After |
|----------|--------|-------|
| `sections/introduction.tex` | "21 propositions and theorems" | "21 propositions and formal statements ... provide physical intuition and scaling guidance rather than rigorous guarantees" |
| `sections/capture_region_analysis.tex` | "theoretical expectation in Proposition~2" | "analytical intuition captured by Proposition~2" |
| `sections/capture_region_analysis.tex` | "degeneration property predicted by Proposition~2" | "degeneration property described by Proposition~2" |
| `sections/discussion.tex` | "theoretical prediction of Proposition~2" | "analytical intuition captured by Proposition~2" |
| `sections/discussion.tex` | "as predicted by Proposition~22" | "consistent with the conceptual framework of Proposition~22" |
| `paper_materials/paper.tex` (CEM variants) | "theoretical analysis predicts" | kept, but framed as "empirical pattern supports the theoretical recommendation" (already aligned) |

---

## 5. Limitations chapter expanded with 5 honesty declarations

`paper_materials/paper.tex` §"Limitations and Future Work" now contains the
original 4 empirical limitations plus the 5 honesty declarations from
`docs/status.md`:

1. Guidance-law scope (idealized acceleration model vs. implemented command chain)
2. Reward design (dense + terminal sparse; PBS redundant per A2' ablation)
3. CEM convergence (heuristic scaling intuition, not rigorous guarantee)
4. Bilevel equilibrium (conceptual Stackelberg lens, not proved property)
5. PPO convergence (conditional under idealized assumptions)

Each declaration references `config/canonical/` or `docs/reward_audit.md`.

---

## 6. Other consistency fixes

| Issue | Fix |
|-------|-----|
| Duplicate "\subsection{Guidance Environment}" with placeholder text | Removed placeholder and duplicate header; clarified policy action vs. guidance-law commands |
| Abstract and intro referring to "acceleration commands" | Updated to "roll-rate, normal-overload, and throttle commands" |
| Discussion saying policy learns "how to generate acceleration commands" | Updated to "how to generate roll-rate, normal-overload, and throttle commands" |

---

## 7. Scope confirmation

The user confirmed that only `paper_materials/paper.tex` needs to be aligned;
the separate "theoretical derivation report" is out of scope. The paper edits
listed in this log have been applied to the local working tree but are **not**
committed or pushed, per project policy to keep `paper_materials/` changes out
of version control.

---

## 8. Files modified in this alignment

```
paper_materials/paper.tex
paper_materials/sections/introduction.tex
paper_materials/sections/discussion.tex
paper_materials/sections/capture_region_analysis.tex
docs/paper_alignment_v1.3.md  (this file)
```

## 9. Validation

- `grep` verified no remaining "acceleration commands" claims in the
  hierarchical-architecture context.
- `grep` verified no "proof" claims in paper content.
- Limitations section now contains 9 items (4 empirical + 5 honesty declarations).
- All canonical-gain references use the 5-D vector.

## 10. Next steps

1. Locate and rewrite theoretical report §2.2 (pending source file).
2. Compile `paper.tex` to verify LaTeX syntax after the equation/figure changes.
3. Re-run paper figure/table generation scripts if symbol-table cross-references
   change numbering.
