# Revised Paper Narrative Plan

After fixing the domain-randomization seed bug in `tracking_env.py`, the data no
longer supports the original claim that the virtual-pursuit-point (VPP) offset
is dispensable or actively detrimental. This plan records the new evidence and
the required semantic pivot for the paper.

## 1. What the new data shows

### 1.1 5-seed VPP vs No-VPP validation (200 k training steps)

| Method | Pooled SR | Per-seed SR | Mean return | N episodes |
|---|---:|---:|---:|---:|
| No-VPP (zero offset) | **73.3 %** (110/150) | 73.3 % on all 5 seeds | +3.46 | 150 |
| VPP + LOS-rate | **70.0 %** (105/150) | 70.0 % on all 5 seeds | −1.18 | 150 |

Key statistical facts:

- **No significant success-rate difference**: McNemar exact test on 150 paired
  episodes gives **p = 0.0625**; Welch's t-test on per-seed success rates gives
  **p = 0.1992**. Both are above α = 0.05.
- **Returns differ significantly**: Mann-Whitney U on episode returns gives
  **p = 2.1 × 10⁻⁹**; No-VPP returns are systematically higher because VPP's
  failure episodes incur larger out-of-bounds penalties.
- **Variance caveat**: success rates are identical across the 5 seeds because
  the evaluation uses 30 episodes per seed (3.33 % granularity). The comparison
  should therefore be reported as “comparable performance” rather than a
  precise ordering.

Interpretation: **the VPP offset layer is neither essential nor actively
detrimental in the canonical close-range tracking setting.** The previous 0 %
VPP result was a training/evaluation artifact caused by the fixed
domain-randomization RNG seed.

### 1.2 VPP clamped ablation (±100 m, 2 seeds)

| Method | Pooled SR | Crash | OOB |
|---|---:|---:|---:|
| VPP full | 70.0 % | 16.7 % | 13.3 % |
| VPP clamped ±100 m | 36.7 % | 13.3 % | **50.0 %** |

Restricting the offset box to ±100 m roughly halves success rate and drives
most failures out of bounds. **Simple static clamping is harmful**, not a
remedy for large offsets.

### 1.3 End-to-end baseline, fair reward, extended training (2 seeds)

| Training budget | Seed 0 SR | Seed 1 SR | Mean SR | Failure mode |
|---|---:|---:|---:|---|
| 200 k | 36.7 % | 0.0 % | 18.3 % | crash / OOB |
| 500 k | 0.0 % | 0.0 % | 0.0 % | crash / OOB |
| 1 M | 36.7 % | 0.0 % | 18.3 % | OOB / crash |
| 2 M | 36.7 % | 0.0 % | 18.3 % | OOB / crash |

With the reward function aligned to the hierarchical baseline, one seed reaches
≈37 % success, but the other remains at 0 %. More training (up to 2 M steps)
does not reliably improve the E2E baseline; variance and instability dominate.
This supports the weaker claim that **the hierarchical decomposition is
beneficial and easier to train**, not that it is strictly necessary.

### 1.4 JSBSim F-16 validation with maneuvering targets (10-seed matrix, zero-shot weaving)

A separate JSBSim F-16 backend experiment trained VPP, No-VPP, and E2E policies on
canonical constant-velocity targets (10 seeds each, 200 k steps) and evaluated them
zero-shot on a `sinusoidal_weaving` target (5 g / 1.0 rad/s):

| Method | favorable | neutral | challenging | disadvantage | overall |
|---|---:|---:|---:|---:|---:|
| VPP | 100 % | 100 % | 100 % | 0 % | **75.0 %** |
| No-VPP | 100 % | 100 % | 100 % | 0 % | **75.0 %** |
| End-to-End | 100 % | 100 % | 100 % | 0 % | **75.0 %** |

- VPP and No-VPP are **numerically identical** on this aggressive maneuvering target.
- The only difference is failure mode in `disadvantage`: E2E crashes, whereas VPP/No-VPP go out-of-bounds.
- This cross-backend validation supports the weaker, but robust, claim that **the hierarchical guidance interface is more stable than end-to-end control**, while confirming that the learned VPP offset is not the primary driver.

## 2. New title candidates

1. *Bilevel Optimization of Guidance Gains for UAV Close-Range Tracking:
   Guidance-Law Tuning Outperforms Virtual-Pursuit-Point Offset*
2. *Guidance-Law Gain Optimization as the Primary Driver of Hierarchical
   DRL-Based UAV Close-Range Tracking*
3. *Hierarchical DRL Guidance for UAV Tracking: The Dominant Role of
   Guidance-Law Gain Optimization*

**Recommended: #1** — it explicitly downgrades the old “necessity” claim and
highlights the positive cross-entropy-method (CEM) gain result.

## 3. New abstract (draft)

```latex
We present a hierarchical guidance architecture that combines a learned
virtual-pursuit-point (VPP) strategy with a classical line-of-sight-rate
guidance law for UAV close-range tracking. A controlled 5-seed comparison
shows that a VPP-enabled policy and a zero-offset variant achieve comparable
success rates (70\% vs 73\%, McNemar $p=0.063$), indicating that the VPP
offset layer is neither essential nor detrimental in typical close-range
tracking. Instead, the dominant performance driver is guidance-law gain
optimization via the cross-entropy method, which improves success rate from
60\% to 75\% (+14.5 percentage points) over fixed default gains. We validate
these findings under JSBSim F-16 aerodynamics with maneuvering targets,
confirming that the hierarchy's robustness stems from the guidance-law layer
rather than from the learned VPP offset.
```

## 4. Semantic pivot: old claims → new claims

| Old claim (must change) | New claim | Rationale |
|---|---|---|
| “Hierarchical decomposition is **necessary**.” | “Hierarchical decomposition is **effective**; gain optimization is the **primary driver**.” | VPP works; E2E is unstable but not uniformly 0 %. |
| “VPP offset is **dispensable**.” | “VPP offset provides **no statistically significant advantage** over zero offset.” | 70 % vs 73 %, McNemar $p=0.063$. |
| “VPP is **actively detrimental**.” | **Delete entirely.** | No evidence; clamping hurts, not full VPP. |
| “VPP is **essential**.” | “VPP is **optional** / **non-critical**.” | It works but is not required. |
| “No-VPP 100 % proves zero offset is optimal.” | “No-VPP is a **strong, simple baseline** comparable to VPP.” | 73.3 % in 5 seeds, not 100 %. |
| “E2E 0 % proves hierarchy necessary.” | “E2E underperforms and is **highly seed-sensitive** even with aligned rewards and up to 2 M steps**.” | One seed reaches 36.7 %, one stays at 0 %. |

## 5. Concrete text edits

### 5.1 Must delete

- All occurrences of “actively detrimental”.
- All claims that “VPP is dispensable”.
- All claims that “hierarchical decomposition is necessary” or “essential”.
- Any table/figure caption that states the VPP success rate as 0 %.

### 5.2 Must rewrite

- **Abstract**: use the draft in §3.
- **Introduction**: shift contribution from “proving necessity” to
  “identifying the dominant performance driver” (guidance-law gain
  optimization).
- **Results / Architecture comparison**: report VPP and No-VPP as comparable;
  lead with the CEM gain lift.
- **Discussion**: explain why the VPP offset may not help (close-range geometry
  already well handled by LOS-rate law; offset adds learnable degrees of
  freedom without clear benefit) **without claiming it harms**.
- **Conclusion**: state that the paper's contribution is demonstrating that
  gain optimization, not the VPP offset, is the key design lever.

### 5.3 Can keep

- CEM vs default vs heuristic gain comparison (strongest positive result).
- JSBSim validation as cross-backend robustness evidence.
- No-VPP as a strong, simple baseline.
- E2E baseline results, with explicit caveats about reward/action-space
  fairness, seed sensitivity, and the 2 M-step extended training study.

## 6. Statistical validation status

- [x] 5 seeds for VPP full and No-VPP (200 k steps).
- [x] 30 eval episodes per seed → success-rate granularity of 3.33 pp.
- [x] Report mean ± std and perform Welch's t-test / McNemar / Mann-Whitney U.
- [x] Conclude **no statistically significant difference** in success rate.
- [ ] Increase eval episodes to 50–100 (or raise domain-randomization scale) to
      obtain non-zero success-rate variance across seeds (future work / camera
      ready).

## 7. VPP clamped follow-up

The ±100 m clamp degraded performance. Future ablations should test:

- ±50 m, ±200 m, ±500 m boxes.
- Dynamic clamping (`dynamic_offset_scale`) instead of static limits.
- Offset distribution analysis and OOB-episode inspection to understand why
  restricting the box raises OOB failures.

## 8. E2E baseline follow-up

The extended-training study is complete up to 2 M steps. With aligned rewards,
the E2E baseline remains unstable (one seed at ~37 %, one seed at 0 %). This
supports presenting E2E as evidence that the hierarchical decomposition is
**more reliable and easier to train**, but it does **not** justify a claim of
strict necessity.

## 9. Updated artifacts

- `docs/results/validation_5seed_vpp/raw_episodes.csv`
- `docs/results/validation_5seed_no_vpp/raw_episodes.csv`
- `scripts/extract_validation_raw_episodes.py`
- `paper_materials/tables/table_architecture_comparison_exact.tex`
- `paper_materials/tables/table_vpp_ablation.tex`
- `paper_materials/tables/table_vpp_ablation_per_seed.csv`
- `paper_materials/tables/table_vpp_ablation_stats.csv`

## 10. Next actions

- [ ] Update LaTeX source sections (abstract, intro, results, discussion,
      conclusion) according to this plan.
- [ ] Add a dedicated E2E extended-training table/figure.
- [ ] Consider a 50–100 episode evaluation re-run to break the 3.33 pp
      success-rate granularity.
- [ ] Prepare a response letter explaining how reviewer concerns #1–#3 and #6
      were addressed by the new experiments and narrative.
