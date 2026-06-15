# Revised Paper Narrative Plan

Based on the 2-seed validation results (2026-06-15), the original paper narrative
is no longer tenable. This document records the required semantic pivot and
concrete text changes.

## What the new data shows

| Method | 2-seed SR | Interpretation |
|---|---|---|
| VPP full | 70.0% ± 0.0% | VPP **works**. Not 0%. |
| No-VPP | 73.3% ± 0.0% | Zero-offset works marginally better in this tiny sample. |
| VPP clamped ±100 m | 36.7% ± 0.0% | Severely restricting the offset box **harms** performance. |
| E2E (aligned reward, 200k) | 36.7% / 0.0% | E2E is not uniformly 0% when reward is fair. |
| E2E (aligned reward, 500k) | 0.0% / 0.0% | More steps do not obviously help in 2 seeds. |

Key insight: **VPP offset is neither essential nor actively detrimental** in the
typical close-range tracking setting. The previous 0% VPP result was an
implementation/training artifact (likely the fixed domain-randomization RNG seed
and possibly action-space/reward mismatches). The real performance driver in the
hierarchical architecture is the **guidance-law gain optimization** (CEM).

## New title candidates

1. "Bilevel Optimization of Guidance Gains for UAV Close-Range Tracking: Guidance-Law Tuning Outperforms Virtual-Pursuit-Point Offset"
2. "Guidance-Law Gain Optimization as the Primary Driver of Hierarchical DRL-Based UAV Close-Range Tracking"
3. "Hierarchical DRL Guidance for UAV Tracking: The Dominant Role of Guidance-Law Gain Optimization"

Recommended: **#1** — it explicitly downgrades the old "necessity" claim and
highlights the positive CEM result.

## New abstract (draft)

```latex
We present a hierarchical guidance architecture that combines a learned
virtual-pursuit-point (VPP) strategy with a classical line-of-sight-rate
(guidance) law for UAV close-range tracking. A controlled multi-seed comparison
shows that a VPP-enabled policy and a zero-offset variant achieve comparable
success rates (70\% vs 73\%), indicating that the VPP offset layer is neither
essential nor detrimental in typical close-range tracking. Instead, the dominant
performance driver is guidance-law gain optimization via the cross-entropy
method, which improves success rate from 60\% to 75\% (+14.5 percentage points)
over fixed default gains. We validate these findings under JSBSim F-16
aerodynamics with maneuvering targets, confirming that the hierarchy's
robustness stems from the guidance-law layer rather than from the learned VPP
offset.
```

## Semantic pivot: old claims → new claims

| Old claim (must change) | New claim | Rationale |
|---|---|---|
| "Hierarchical decomposition is **necessary**" | "Hierarchical decomposition is **effective**; gain optimization is the **primary driver**" | VPP works, E2E not definitively proven unable. |
| "VPP offset is **dispensable**" | "VPP offset provides **no statistically significant advantage** over zero offset" | 70% vs 73% within noise. |
| "VPP is **actively detrimental**" | **Delete entirely** | No evidence; clamping hurts, not full VPP. |
| "VPP is **essential**" | "VPP is **optional** / **non-critical**" | It works but is not required. |
| "No-VPP 100% proves zero offset is optimal" | "No-VPP is a strong baseline comparable to VPP" | 73.3% in 2-seed, not 100%. |
| "E2E 0% proves hierarchy necessary" | "E2E underperforms in our default config; fairness validation ongoing" | Aligned reward gave one seed 36.7%. |

## Concrete text edits required

### Must delete

- All occurrences of "actively detrimental".
- All claims that "VPP is dispensable".
- All claims that "hierarchical decomposition is necessary" or "essential".
- Any table/figure caption that states VPP success rate as 0%.

### Must rewrite

- **Abstract**: use draft above.
- **Introduction**: shift contribution from "proving necessity" to "identifying the dominant performance driver".
- **Results / Architecture comparison**: report VPP and No-VPP as comparable; emphasize CEM gain lift.
- **Discussion**: explain why VPP offset may not help (task geometry is close-range, LOS-rate law already adequate; offset adds noise) without claiming it harms.
- **Conclusion**: state that the paper's contribution is demonstrating that gain optimization, not VPP offset, is the key design lever.

### Can keep

- CEM vs default vs heuristic gain comparison (the strongest positive result).
- JSBSim validation as cross-backend robustness evidence.
- No-VPP as a strong, simple baseline.
- E2E baseline results, but with caveats about reward/action-space fairness and ongoing extended-training study.

## Statistical validation needed before rewriting

1. Run **5–10 seeds** for VPP full and No-VPP (200k steps).
2. Use **50–100 eval episodes** per seed to obtain non-zero success-rate variance.
3. Report mean ± std and perform Welch's t-test or Mann-Whitney U on per-seed success rates.
4. If p > 0.05, officially state "no statistically significant difference".
5. Compute effect size (Cohen's d) to quantify practical difference.

## VPP clamped follow-up

The ±100 m clamp degraded performance. Future ablations should try:

- ±50 m, ±200 m, ±500 m
- Dynamic clamping (`dynamic_offset_scale`)
- Analyze offset distribution and OOB episodes to understand why clamping hurts

## E2E baseline follow-up

Complete the hyperparameter sweep and extended-training study. If E2E remains at
0% after fair reward + extended steps, present it as evidence that the
hierarchical decomposition is beneficial (but not "necessary"). If E2E reaches
competitive performance, the narrative becomes "hierarchy is one viable design
among several".

## Next actions

- [ ] Finish 5–10 seed VPP vs No-VPP experiment.
- [ ] Finish E2E extended training (1M/2M) and hyperparameter sweep.
- [ ] Update all LaTeX/paper text according to this plan.
- [ ] Regenerate all tables/figures from new data.
- [ ] Prepare a response letter explaining how reviewer concerns #1–#3 and #6
      were addressed by the new experiments and narrative.
